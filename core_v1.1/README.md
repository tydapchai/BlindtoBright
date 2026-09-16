# BlindtoBright ST-GCN Transformer v1.1

Ban nay chay checkpoint ST-GCN Transformer voi input 9 kenh, 48 frame va 76 node.
Ban goc trong `core/` khong bi thay doi.

## Chay

Tu thu muc workspace:

```powershell
python BlindtoBright/core_v1.1/main.py --camera 0
```

Voi camera ESP32, LLM va TTS qua loa ESP32:

```powershell
python BlindtoBright/core_v1.1/main.py --camera http://ESP_IP:81/stream --esp-ip ESP_IP
```

Khi decoder ket thuc mot cau, chuoi gloss duoc gui qua Gemini LLM de sap xep lai thanh cau tieng Viet, sau do Gemini TTS tao PCM va gui toi `http://ESP_IP/play`. Dung `--no-tts` neu chi muon xem ket qua tren man hinh.

Nhan `SPACE` de bat/tat nhan dien va `Q` de thoat. `--stride 4` chay inference moi 4 frame; giam xuong `1` neu may du manh, tang len neu CPU qua tai.

## Kiem tra model

```powershell
python BlindtoBright/core_v1.1/verify_model.py
```

Checkpoint mac dinh la `BlindtoBright/models/best_vsl_model.pth`. File nhan mac dinh duoc doc tu `VSL_pipeline/modules/label_map_472.json`.