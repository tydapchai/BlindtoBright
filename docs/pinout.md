# Sơ Đồ Chân GPIO - ESP32-S3 (Goouuu ESP32-S3-CAM)

Dự án **Blind to Bright** sử dụng board mạch **Goouuu ESP32-S3-CAM** kết hợp 4 ngoại vi chính:

---

## 1. Camera OV2640 (Gắn sẵn trên FPC slot)

| Tín hiệu Camera | Chân GPIO ESP32-S3 | Ghi chú |
| :--- | :--- | :--- |
| **XCLK** | `GPIO 15` | XCLK Clock |
| **SIOD** | `GPIO 4` | SCCB Data (I2C SDA) |
| **SIOC** | `GPIO 5` | SCCB Clock (I2C SCL) |
| **Y9 (D7)** | `GPIO 16` | Data Bit 7 |
| **Y8 (D6)** | `GPIO 17` | Data Bit 6 |
| **Y7 (D5)** | `GPIO 18` | Data Bit 5 |
| **Y6 (D4)** | `GPIO 12` | Data Bit 4 |
| **Y5 (D3)** | `GPIO 10` | Data Bit 3 |
| **Y4 (D2)** | `GPIO 8` | Data Bit 2 |
| **Y3 (D1)** | `GPIO 9` | Data Bit 1 |
| **Y2 (D0)** | `GPIO 11` | Data Bit 0 |
| **VSYNC** | `GPIO 6` | Frame Sync |
| **HREF** | `GPIO 7` | Line Sync |
| **PCLK** | `GPIO 13` | Pixel Clock |

---

## 2. Micro I2S INMP441 (I2S RX - Thu âm thanh)

| Chân INMP441 | Chân GPIO ESP32-S3 | Ghi chú |
| :--- | :--- | :--- |
| **SCK / BCLK** | `GPIO 47` | I2S Bit Clock |
| **WS / LRC** | `GPIO 48` | Word Select (Left/Right Clock) |
| **SD / DOUT** | `GPIO 1` | Serial Data Out |
| **L/R** | `GND` | Kênh trái (Left Channel) |
| **VDD** | `3.3V` | Nguồn cấp 3.3V |
| **GND** | `GND` | Mass chung |

---

## 3. Màn hình OLED SSD1306 (I2C - 128x64)

| Chân SSD1306 | Chân GPIO ESP32-S3 | Ghi chú |
| :--- | :--- | :--- |
| **SCL** | `GPIO 40` | I2C Clock |
| **SDA** | `GPIO 41` | I2C Data |
| **VCC** | `3.3V` | Nguồn cấp 3.3V |
| **GND** | `GND` | Mass chung |

---

## 4. Mạch Khuếch Đại Loa MAX98357A (I2S TX - Phát âm thanh)

| Chân MAX98357A | Chân GPIO ESP32-S3 | Ghi chú |
| :--- | :--- | :--- |
| **DIN** | `GPIO 35` | I2S Serial Data In |
| **BCLK** | `GPIO 36` | I2S Bit Clock |
| **LRC** | `GPIO 37` | Word Select (Left/Right Clock) |
| **VIN** | `5V` hoặc `3.3V` | Cấp 5V để âm lượng to nhất |
| **GND** | `GND` | Mass chung |
| **GAIN** | Không nối (hoặc nối trở) | Gain mặc định 9dB |
| **SD / EN** | Không nối | Bật mặc định |
