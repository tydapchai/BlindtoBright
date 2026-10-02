# KÊ KHAI CÔNG CỤ AI, DỮ LIỆU, API, THƯ VIỆN, MÃ NGUỒN MỞ

**Dự án:** BlindToBright — Thiết bị giao tiếp 2 chiều thời gian thực cho người khiếm thính  
**Ngày kê khai:** 02/10/2026

---

## 1. CÔNG CỤ AI & MÔ HÌNH HỌC MÁY

| # | Tên công cụ / Mô hình | Nhà phát triển | Phiên bản | Giấy phép | Mục đích sử dụng trong dự án |
| :---: | :--- | :--- | :--- | :--- | :--- |
| 1 | **Google Gemini 3.5 Flash** | Google DeepMind | `gemini-3.5-flash` | Google AI Terms of Service | Ghép câu tiếng Việt tự nhiên từ chuỗi từ khóa VSL (LLM) |
| 2 | **Google Gemini 3.1 Flash Lite** | Google DeepMind | `gemini-3.1-flash-lite` | Google AI Terms of Service | Nhận dạng giọng nói tiếng Việt (Speech-to-Text / STT) |
| 3 | **Google Gemini 3.1 Flash TTS Preview** | Google DeepMind | `gemini-3.1-flash-tts-preview` | Google AI Terms of Service | Tổng hợp giọng đọc tiếng Việt (Text-to-Speech / TTS), giọng "Aoede" |
| 4 | **MediaPipe Holistic** | Google | ≥ 0.10.14 | Apache 2.0 | Trích xuất 76 điểm xương khớp cơ thể (33 Pose + 21 Tay trái + 21 Tay phải + 1 Cổ) từ khung hình camera thời gian thực |
| 5 | **CTR-GCN** (Channel-wise Topology Refinement GCN) | Tự xây dựng dựa trên kiến trúc gốc của Chen et al. (2021) | Tùy chỉnh cho VSL | Tự phát triển | Mạng đồ thị tích chập nhận diện cử chỉ Ngôn ngữ ký hiệu Việt Nam. Kiến trúc tham khảo bài báo gốc, code tự viết lại hoàn toàn để phù hợp 76 nodes / 9 kênh đặc trưng |
| 6 | **PyTorch** | Meta AI (Facebook) | ≥ 2.3 | BSD-3-Clause | Framework deep learning chạy inference mô hình CTR-GCN |

> **Ghi chú về CTR-GCN**: Kiến trúc CTR-GCN được **lấy cảm hứng** từ bài báo gốc *"Channel-wise Topology Refinement Graph Convolution for Skeleton-Based Action Recognition"* (Chen et al., ICCV 2021). Tuy nhiên, toàn bộ code mô hình trong dự án (`ctr_gcn_model.py`) được **viết lại hoàn toàn** để phù hợp với bài toán nhận diện VSL: 76 nodes (thay vì 25 nodes NTU-RGB+D), 9 kênh đặc trưng (x,y,z + vận tốc + gia tốc), và pipeline tiền xử lý riêng biệt cho ngôn ngữ ký hiệu.

---

## 2. API CLOUD (DỊCH VỤ TRỰC TUYẾN)

| # | API | Nhà cung cấp | Giao thức | Chi phí | Mục đích sử dụng |
| :---: | :--- | :--- | :--- | :--- | :--- |
| 1 | **Google Generative Language API** | Google AI | REST HTTPS (`generativelanguage.googleapis.com/v1beta`) | Miễn phí (Free tier) | Gọi 3 tính năng Gemini AI: LLM ghép câu, STT nhận dạng giọng nói, TTS tổng hợp giọng nói |

> **Cơ chế sử dụng API**: Sử dụng hệ thống xoay vòng nhiều API Key (`itertools.cycle`) để tránh bị giới hạn tốc độ (Rate Limit 429). Toàn bộ giao tiếp qua REST API chuẩn, không sử dụng SDK đóng gói.

---

## 3. THƯ VIỆN PYTHON (MÃ NGUỒN MỞ)

| # | Thư viện | Phiên bản | Giấy phép | Mục đích sử dụng |
| :---: | :--- | :--- | :--- | :--- |
| 1 | **torch** (PyTorch) | ≥ 2.3 | BSD-3-Clause | Chạy inference mô hình nhận dạng cử chỉ CTR-GCN |
| 2 | **mediapipe** | ≥ 0.10.14 | Apache 2.0 | Trích xuất xương khớp cơ thể từ khung hình camera (Holistic: Pose + Hands) |
| 3 | **opencv-python** (OpenCV) | ≥ 4.10 | Apache 2.0 | Thu nhận & xử lý khung hình camera, hiển thị giao diện trực quan (HUD) |
| 4 | **numpy** | ≥ 1.26 | BSD-3-Clause | Tính toán ma trận, xử lý mảng số, tiền xử lý dữ liệu landmarks, resample âm thanh |
| 5 | **Pillow** (PIL) | — | HPND License | Vẽ chữ tiếng Việt Unicode (có dấu) lên khung hình camera |
| 6 | **requests** | — | Apache 2.0 | Gọi HTTP API tới Google Gemini và giao tiếp HTTP với ESP32 |
| 7 | **sounddevice** | ≥ 0.5 | MIT | Thu âm giọng nói từ micro laptop (chế độ fallback khi không dùng micro ESP32) |
| 8 | **soundfile** | ≥ 0.12 | BSD-3-Clause | Đọc/ghi file âm thanh WAV |
| 9 | **flask** | ≥ 3.0 | BSD-3-Clause | Web server phục vụ giao diện hội thoại trên điện thoại (phiên bản V1) |
| 10 | **waitress** | ≥ 3.0 | ZPL 2.1 | WSGI production server cho Flask (thay thế development server) |
| 11 | **faster-whisper** | ≥ 1.0 | MIT | Nhận dạng giọng nói offline cục bộ (bản dự phòng khi không có Internet) |
| 12 | **silero-vad** | ≥ 6.2.1 | MIT | Phát hiện hoạt động giọng nói (Voice Activity Detection) cho bản offline |
| 13 | **onnxruntime** | ≥ 1.20 | MIT | Inference ONNX tốc độ cao cho bản offline |
| 14 | **pyttsx3** | ≥ 2.90 | MPL 2.0 | Tổng hợp giọng đọc offline cục bộ (bản dự phòng) |

---

## 4. THƯ VIỆN PHẦN CỨNG (FIRMWARE ESP32 - ARDUINO)

| # | Thư viện | Nhà phát triển | Giấy phép | Mục đích sử dụng |
| :---: | :--- | :--- | :--- | :--- |
| 1 | **Arduino Core for ESP32** | Espressif Systems | LGPL-2.1 | Nền tảng firmware cho vi điều khiển ESP32-S3 |
| 2 | **esp_camera** | Espressif Systems | Apache 2.0 | Điều khiển camera OV2640, stream MJPEG qua HTTP |
| 3 | **ESP_I2S** | Espressif Systems | Apache 2.0 | Giao tiếp I2S với micro INMP441 (RX) và loa MAX98357A (TX) |
| 4 | **esp_http_server** | Espressif Systems | Apache 2.0 | HTTP server đa cổng trên ESP32 (Port 80, 81, 82) |
| 5 | **WiFi** | Espressif Systems | LGPL-2.1 | Kết nối mạng Wi-Fi (STA + SoftAP fallback) |
| 6 | **Wire** | Arduino | LGPL-2.1 | Giao tiếp I2C với màn hình OLED SSD1306 |
| 7 | **Adafruit GFX Library** | Adafruit Industries | BSD License | Thư viện đồ họa nền tảng cho hiển thị OLED |
| 8 | **Adafruit SSD1306** | Adafruit Industries | BSD License | Driver điều khiển màn hình OLED SSD1306 128×64 |

---

## 5. DỮ LIỆU HUẤN LUYỆN

| # | Tên bộ dữ liệu | Nguồn gốc | Quy mô | Mục đích sử dụng |
| :---: | :--- | :--- | :--- | :--- |
| 1 | **VSL Dataset V3** | Tự thu thập & xây dựng | 10 lớp từ vựng VSL, mỗi lớp ~50-100 video cử chỉ | Huấn luyện mô hình CTR-GCN nhận diện 10 từ Ngôn ngữ ký hiệu Việt Nam |

> **Ghi chú**: Bộ dữ liệu được **tự thu thập** bằng cách quay video các cử chỉ tay của thành viên nhóm, trích xuất xương khớp bằng MediaPipe, và gán nhãn thủ công. Không sử dụng bộ dữ liệu thương mại hoặc có bản quyền bên ngoài.

---

## 6. PHẦN CỨNG NHÚNG (EMBEDDED HARDWARE)

| # | Linh kiện | Model / Chip | Nhà sản xuất | Vai trò trong hệ thống |
| :---: | :--- | :--- | :--- | :--- |
| 1 | Board vi điều khiển | **Goouuu ESP32-S3-CAM** | Goouuu / Espressif | Vi xử lý trung tâm, kết nối Wi-Fi, chạy firmware multi-server |
| 2 | Camera | **OV2640** (gắn sẵn) | OmniVision | Thu hình cử chỉ tay người dùng, stream MJPEG qua Port 81 |
| 3 | Micro kỹ thuật số | **INMP441** (I2S MEMS) | InvenSense / TDK | Thu âm giọng nói 16kHz 16-bit, stream PCM qua Port 82 |
| 4 | Mạch khuếch đại loa | **MAX98357A** (I2S DAC) | Maxim Integrated / Analog Devices | Phát giọng đọc TTS ra loa từ dữ liệu PCM nhận qua Port 80 |
| 5 | Màn hình OLED | **SSD1306** 128×64 I2C | Solomon Systech | Hiển thị câu ký hiệu (`SIGN:`) và lời nói (`MIC:`) cho người khiếm thính đọc |

---

## 7. CÔNG CỤ PHÁT TRIỂN & MÔI TRƯỜNG

| # | Công cụ | Phiên bản | Mục đích sử dụng |
| :---: | :--- | :--- | :--- |
| 1 | **Python** | 3.12 | Ngôn ngữ lập trình chính cho ứng dụng trên laptop |
| 2 | **Arduino IDE** | 2.x | Biên dịch và nạp firmware cho ESP32-S3 |
| 3 | **Git** | — | Quản lý phiên bản mã nguồn |
| 4 | **Visual Studio Code** | — | Trình soạn thảo mã nguồn chính |
| 5 | **Google AI Studio** | — | Tạo và quản lý API Key Gemini |
| 6 | **Arduino Board Package (esp32)** | ≥ 3.x | Hỗ trợ biên dịch firmware cho dòng chip ESP32-S3 |

---

## 8. TÀI LIỆU THAM KHẢO KHOA HỌC

| # | Tài liệu | Tác giả | Năm | Mức độ sử dụng |
| :---: | :--- | :--- | :---: | :--- |
| 1 | *Channel-wise Topology Refinement Graph Convolution for Skeleton-Based Action Recognition* (ICCV 2021) | Chen Y., Zhang Z., Yuan C., Li B., Deng Y., Hu W. | 2021 | Tham khảo kiến trúc mạng CTR-GCN; **code tự viết lại hoàn toàn** |
| 2 | *Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition* (AAAI 2018) | Yan S., Xiong Y., Lin D. | 2018 | Tham khảo ý tưởng ST-GCN cho mô hình dự phòng |
| 3 | *MediaPipe Holistic: Whole-body Pose, Face, and Hand Tracking* | Google Research | 2020 | Sử dụng trực tiếp pipeline trích xuất xương khớp |

---

## TÓM TẮT

| Hạng mục | Số lượng |
| :--- | :---: |
| Mô hình / Công cụ AI | 6 |
| API Cloud | 1 (Google Generative Language API — 3 tính năng) |
| Thư viện Python (mã nguồn mở) | 14 |
| Thư viện Arduino (mã nguồn mở) | 8 |
| Bộ dữ liệu | 1 (tự thu thập) |
| Linh kiện phần cứng | 5 |
| Công cụ phát triển | 6 |
| Tài liệu tham khảo khoa học | 3 |

> **Cam kết**: Toàn bộ mã nguồn ứng dụng chính (Python và Firmware C++) đều do nhóm **tự viết**. Các thư viện và framework được liệt kê ở trên đều là **mã nguồn mở** hoặc có **giấy phép sử dụng miễn phí** cho mục đích học thuật và phi thương mại.
