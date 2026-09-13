# WLASL Communication Assistant V2

## 1. Train trên Kaggle

Import `wlasl_communication_v2.ipynb`, Add Input dataset WLASL, bật GPU và Run All.

Tải về và đặt vào các thư mục tương ứng:

- `models/best_bigru_v2.pt`
- `models/hand_landmarker.task`
- `configs/conversation.json`

## 2. Chuẩn bị môi trường

```bash
python3 -m venv .venv
source .venv/bin/activate  # Hoặc: .\.venv\Scripts\Activate.ps1 trên Windows
pip install -r requirements.txt
```

## 3. Khởi chạy ứng dụng

Chạy phiên bản chính (Gemini Cloud):
```bash
python core/main.py
```
Hoặc dùng script:
```powershell
.\run.ps1
```

Chạy phiên bản offline dự phòng (Whisper cục bộ):
```bash
python backup/app_offline.py --camera 0 --language vi
```

## 4. Điều khiển sign recognition

- SPACE: bắt đầu / kết thúc một ký hiệu
- Q hoặc ESC: thoát

Khi nhận đúng, app:

- hiện intent;
- chuyển intent thành câu hoàn chỉnh;
- đọc câu bằng TTS;
- thêm câu vào lịch sử hội thoại.

## 5. Mở note trên điện thoại

Điện thoại và laptop cùng Wi-Fi. Mở:

```text
http://IP_LAPTOP:8000
```

Trang này cho phép:

- ghi/upload audio;
- Whisper chuyển giọng nói thành chữ;
- xem lịch sử sign + speech;
- tạo tóm tắt ngắn;
- copy nội dung vào Notes trên điện thoại.

## Lưu ý

Whisper mặc định dùng model `small` CPU int8. Lần đầu chạy cần tải model.
