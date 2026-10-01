# BlindtoBright Core v1.1 (CTR-GCN / ST-GCN)

Phiên bản này hỗ trợ chạy mô hình **CTR-GCN** (đồ thị động kênh tương quan) và **ST-GCN Transformer** với input 9 kênh (tọa độ, vận tốc, gia tốc), 48 frames và 76 landmarks (33 pose + 21 tay trái + 21 tay phải + 1 cổ). Bản gốc trong `core/` không bị thay đổi.

Checkpoint mặc định: `BlindtoBright/models/best_vsl_model_ctr_gcn.pth` đi kèm `BlindtoBright/core_v1.1/label_map_472_10w.json`.

---

## 1. Chạy chương trình chính (Nhận diện & Ghép câu)

Từ thư mục workspace:

- **Chạy với Webcam laptop:**
```powershell
python BlindtoBright/core_v1.1/main.py --camera 0
```

- **Chạy với Camera ESP32, LLM dịch câu và TTS qua loa ESP32:**
```powershell
python BlindtoBright/core_v1.1/main.py --camera esp --esp-ip 10.245.192.219
```

- **Chạy với checkpoint ST-GCN cũ (nếu muốn):**
```powershell
python BlindtoBright/core_v1.1/main.py --checkpoint BlindtoBright/models/best_vsl_model.pth --labels BlindtoBright/core_v1.1/label_map_472_10w.json
```

Khi gom đủ 48 frame, mô hình dự đoán từ vựng. Người dùng bấm `[ENTER]` để chốt câu gửi qua Gemini LLM sắp xếp lại thành câu tiếng Việt tự nhiên và phát loa qua ESP32.

Các phím tắt điều khiển:
- `SPACE`: Bật / Tạm dừng nhận diện.
- `ENTER`: Chốt câu và gửi LLM dịch + phát âm thanh.
- `BACKSPACE`: Xóa từ cuối cùng vừa nhận diện nếu bị sai.
- `C`: Xóa toàn bộ câu đang ghép.
- `Q` / `ESC`: Thoát chương trình.

---

## 2. Kiểm tra model và công cụ hỗ trợ

- **Kiểm tra checkpoint model (Forward pass tensor (1, 9, 48, 76)):**
```powershell
python BlindtoBright/core_v1.1/verify_model.py
```

- **Kiểm tra độ chính xác từng từ (Debug Word):**
```powershell
python BlindtoBright/core_v1.1/debug_word.py --camera 0
# Hoac liet ke danh sach nhan:
python BlindtoBright/core_v1.1/debug_word.py --list
```

- **Nhận diện cử chỉ liên tục (Continuous Word Spotting):**
```powershell
python BlindtoBright/core_v1.1/continuous_sentence.py --camera 0
```

- **Kiểm tra luồng frame camera ESP32 (FPS, băng thông, độ trễ):**
```powershell
python BlindtoBright/core_v1.1/test_esp_frame.py
```
