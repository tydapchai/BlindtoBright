# WLASL Communication Assistant V2

## 1. Train trên Kaggle

Import `wlasl_communication_v2.ipynb`, Add Input dataset WLASL, bật GPU và Run All.

Tải về:

- `best_bigru_v2.pt`
- `conversation_config.json`
- `hand_landmarker.task`

## 2. Chuẩn bị app

Đặt các file trên cùng thư mục với `app.py`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
sudo apt install espeak-ng ffmpeg
```

## 3. Chạy webcam

```bash
python app.py --camera 0 --language vi
```

IP camera:

```bash
python app.py --camera "http://PHONE_IP:8080/video" --language vi
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
