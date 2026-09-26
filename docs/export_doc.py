import markdown
import os
import subprocess

md_path = r"d:\BlindtoBright\docs\TAI_LIEU_DU_AN_BLINDTOBRIGHT.md"
html_path = r"d:\BlindtoBright\docs\TAI_LIEU_DU_AN_BLINDTOBRIGHT.html"
pdf_path = r"d:\BlindtoBright\docs\TAI_LIEU_DU_AN_BLINDTOBRIGHT.pdf"

with open(md_path, "r", encoding="utf-8") as f:
    text = f.read()

# Chuyển đổi Markdown sang HTML
html_body = markdown.markdown(text, extensions=["tables", "fenced_code"])

# Sơ đồ quy trình 2 chiều trực quan
diagram_html = """
<div class="pipeline-container">
    <div class="pipeline-col sign-to-speech">
        <div class="pipeline-header">🔵 CHIỀU 1: THỦ NGỮ SANG TIẾNG NÓI (SIGN-TO-SPEECH)</div>
        <div class="step-card">
            <span class="step-num">1</span>
            <div><strong>Học sinh ra ký hiệu VSL</strong><br><small>Camera Laptop / ESP32-CAM thu nhận (25-30 FPS)</small></div>
        </div>
        <div class="step-arrow">↓</div>
        <div class="step-card">
            <span class="step-num">2</span>
            <div><strong>Trích xuất Khung xương (MediaPipe)</strong><br><small>76 Keypoints: 33 Pose + 42 Hands + 1 Head</small></div>
        </div>
        <div class="step-arrow">↓</div>
        <div class="step-card">
            <span class="step-num">3</span>
            <div><strong>Trích xuất Động học 9 Kênh</strong><br><small>Tọa độ (x,y,z) + Vận tốc (dx) + Gia tốc (d²x)</small></div>
        </div>
        <div class="step-arrow">↓</div>
        <div class="step-card">
            <span class="step-num">4</span>
            <div><strong>Phân loại ST-GCN + Transformer</strong><br><small>Nhận diện 472 nhãn VSL từ chuỗi 48 frame</small></div>
        </div>
        <div class="step-arrow">↓</div>
        <div class="step-card">
            <span class="step-num">5</span>
            <div><strong>Temporal Decoder & Margin Filter</strong><br><small>Lọc trạng thái Idle, độ tin cậy > 65%, biên > 8%</small></div>
        </div>
        <div class="step-arrow">↓</div>
        <div class="step-card">
            <span class="step-num">6</span>
            <div><strong>Gemini LLM Tái cấu trúc Ngữ pháp</strong><br><small>Chuyển trật tự từ VSL thành câu tiếng Việt hoàn chỉnh</small></div>
        </div>
        <div class="step-arrow">↓</div>
        <div class="step-card highlight-blue">
            <span class="step-num">7</span>
            <div><strong>Loa Nhúng ESP32 I2S (MAX98357A)</strong><br><small>Phát giọng nói tiếng Việt to rõ ra lớp học</small></div>
        </div>
    </div>

    <div class="pipeline-col speech-to-text">
        <div class="pipeline-header">🟢 CHIỀU 2: TIẾNG NÓI SANG VĂN BẢN (SPEECH-TO-TEXT)</div>
        <div class="step-card">
            <span class="step-num">1</span>
            <div><strong>Giáo viên / Bạn học nói</strong><br><small>Microphone Điện thoại hoặc Laptop</small></div>
        </div>
        <div class="step-arrow">↓</div>
        <div class="step-card">
            <span class="step-num">2</span>
            <div><strong>Lọc Tiếng ồn Silero VAD</strong><br><small>Chỉ bắt đoạn giọng nói, loại bỏ tạp âm phòng học</small></div>
        </div>
        <div class="step-arrow">↓</div>
        <div class="step-card">
            <span class="step-num">3</span>
            <div><strong>Mô hình Faster-Whisper</strong><br><small>Phiên âm giọng nói tiếng Việt thời gian thực (&lt; 600ms)</small></div>
        </div>
        <div class="step-arrow">↓</div>
        <div class="step-card highlight-green">
            <span class="step-num">4</span>
            <div><strong>Giao diện Web / Phone Trợ giảng</strong><br><small>Chữ phóng to tương phản cao + Tự động tóm tắt bài</small></div>
        </div>
        <div class="step-arrow">↓</div>
        <div class="step-card">
            <span class="step-num">5</span>
            <div><strong>Học sinh khiếm thính đọc tức thì</strong><br><small>Tiếp thu trọn vẹn lời giảng và thảo luận nhóm</small></div>
        </div>
    </div>
</div>
"""

# Sơ đồ phần cứng trực quan
hardware_html = """
<div class="hw-box">
    <div class="hw-item">
        <div class="hw-title">🔋 Nguồn Điện 5V / 2A</div>
        <div class="hw-desc">Pin sạc dự phòng hoặc adapter USB cấp nguồn toàn bộ hệ thống</div>
    </div>
    <div class="hw-connector">➔</div>
    <div class="hw-item">
        <div class="hw-title">📷 Module ESP32-CAM</div>
        <div class="hw-desc">Thu nhận luồng video cử chỉ, stream MJPEG qua Wi-Fi về Server</div>
    </div>
    <div class="hw-connector">➔</div>
    <div class="hw-item">
        <div class="hw-title">💻 Máy Chủ / Laptop Xử Lý</div>
        <div class="hw-desc">Chạy mô hình ST-GCN phân loại + Gọi Gemini LLM tạo câu & TTS</div>
    </div>
    <div class="hw-connector">➔</div>
    <div class="hw-item highlight">
        <div class="hw-title">🔊 ESP32 + MAX98357A I2S</div>
        <div class="hw-desc">Nhận HTTP PCM stream, giải mã âm thanh DAC và phát ra loa 3W</div>
    </div>
</div>
"""

# Thay thế placeholder bằng khối HTML trực quan
html_body = html_body.replace("<!-- PIPELINE_DIAGRAM_PLACEHOLDER -->", diagram_html)
html_body = html_body.replace("<!-- HARDWARE_DIAGRAM_PLACEHOLDER -->", hardware_html)

full_html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>Báo Cáo Thuyết Minh Dự Án - BlindtoBright</title>
<style>
    @page {{
        size: A4;
        margin: 18mm 16mm 18mm 16mm;
    }}
    * {{
        box-sizing: border-box;
    }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        font-size: 13.5px;
        line-height: 1.6;
        color: #1e293b;
        max-width: 860px;
        margin: 0 auto;
        padding: 20px 25px;
        background-color: #fff;
    }}
    h1 {{
        color: #0f172a;
        font-size: 20px;
        font-weight: 700;
        border-bottom: 2.5px solid #2563eb;
        padding-bottom: 8px;
        margin-top: 24px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    h2 {{
        color: #1e40af;
        font-size: 15.5px;
        font-weight: 700;
        margin-top: 22px;
        border-bottom: 1.5px solid #e2e8f0;
        padding-bottom: 5px;
        text-transform: uppercase;
    }}
    h3 {{
        color: #334155;
        font-size: 14px;
        font-weight: 600;
        margin-top: 16px;
    }}
    p, li {{
        text-align: justify;
        margin-bottom: 8px;
    }}
    ul, ol {{
        padding-left: 24px;
        margin-top: 6px;
        margin-bottom: 12px;
    }}
    li {{
        margin-bottom: 5px;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 16px 0;
        font-size: 13px;
        page-break-inside: avoid;
    }}
    th, td {{
        border: 1px solid #cbd5e1;
        padding: 8px 12px;
        text-align: left;
    }}
    th {{
        background-color: #f1f5f9;
        color: #1e293b;
        font-weight: 600;
    }}
    tr:nth-child(even) {{
        background-color: #f8fafc;
    }}
    code {{
        background-color: #f1f5f9;
        padding: 2px 6px;
        border-radius: 4px;
        font-family: "Consolas", "Courier New", monospace;
        font-size: 12.5px;
        color: #0369a1;
    }}
    blockquote {{
        border-left: 4px solid #2563eb;
        margin: 14px 0;
        padding: 8px 16px;
        background-color: #eff6ff;
        color: #1e40af;
        border-radius: 0 4px 4px 0;
    }}
    hr {{
        border: 0;
        border-top: 1px solid #e2e8f0;
        margin: 20px 0;
    }}

    /* Pipeline Flowchart Cards */
    .pipeline-container {{
        display: flex;
        gap: 16px;
        margin: 18px 0;
        page-break-inside: avoid;
    }}
    .pipeline-col {{
        flex: 1;
        border-radius: 8px;
        padding: 12px;
        display: flex;
        flex-direction: column;
        gap: 6px;
    }}
    .sign-to-speech {{
        background-color: #f0f7ff;
        border: 1.5px solid #bfdbfe;
    }}
    .speech-to-text {{
        background-color: #f0fdf4;
        border: 1.5px solid #bbf7d0;
    }}
    .pipeline-header {{
        font-weight: 700;
        font-size: 11.5px;
        text-align: center;
        padding-bottom: 6px;
        border-bottom: 1px solid rgba(0,0,0,0.08);
        margin-bottom: 4px;
    }}
    .sign-to-speech .pipeline-header {{
        color: #1d4ed8;
    }}
    .speech-to-text .pipeline-header {{
        color: #15803d;
    }}
    .step-card {{
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 8px 10px;
        font-size: 12px;
        display: flex;
        align-items: center;
        gap: 10px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }}
    .step-card strong {{
        color: #0f172a;
    }}
    .step-card small {{
        color: #64748b;
        font-size: 11px;
    }}
    .step-num {{
        width: 20px;
        height: 20px;
        border-radius: 50%;
        background-color: #e2e8f0;
        color: #334155;
        font-weight: 700;
        font-size: 11px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }}
    .highlight-blue {{
        border: 1.5px solid #3b82f6;
        background-color: #eff6ff;
    }}
    .highlight-blue .step-num {{
        background-color: #2563eb;
        color: #fff;
    }}
    .highlight-green {{
        border: 1.5px solid #22c55e;
        background-color: #f0fdf4;
    }}
    .highlight-green .step-num {{
        background-color: #16a34a;
        color: #fff;
    }}
    .step-arrow {{
        text-align: center;
        font-size: 13px;
        color: #94a3b8;
        line-height: 1;
        margin: -2px 0;
    }}

    /* Hardware Block */
    .hw-box {{
        display: flex;
        align-items: center;
        gap: 8px;
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px 12px;
        margin: 16px 0;
        page-break-inside: avoid;
    }}
    .hw-item {{
        flex: 1;
        background: #fff;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        padding: 8px 10px;
        font-size: 11.5px;
        text-align: center;
    }}
    .hw-item.highlight {{
        border-color: #3b82f6;
        background-color: #eff6ff;
    }}
    .hw-title {{
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 4px;
        font-size: 12px;
    }}
    .hw-desc {{
        color: #64748b;
        font-size: 10.5px;
        line-height: 1.4;
    }}
    .hw-connector {{
        font-size: 14px;
        color: #94a3b8;
        font-weight: bold;
    }}

    @media print {{
        body {{
            padding: 0;
            max-width: 100%;
        }}
        h1, h2, h3 {{
            page-break-after: avoid;
        }}
        table, .pipeline-container, .hw-box, blockquote {{
            page-break-inside: avoid;
        }}
    }}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""

with open(html_path, "w", encoding="utf-8") as f:
    f.write(full_html)

print("Đã tạo file HTML sạch tại:", html_path)

# Xuất ra PDF bằng Edge headless với flag --no-pdf-header-footer
cmd = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "--headless",
    "--no-pdf-header-footer",
    f"--print-to-pdf={pdf_path}",
    f"file:///{html_path.replace(os.sep, '/')}"
]

res = subprocess.run(cmd, capture_output=True, text=True)
if res.returncode == 0:
    print("Xuất file PDF THÀNH CÔNG tại:", pdf_path)
else:
    print("Lỗi khi xuất PDF:", res.stderr)
