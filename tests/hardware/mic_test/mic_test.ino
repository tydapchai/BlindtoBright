#include <Arduino.h>
#include "ESP_I2S.h"
#include "soc/soc.h"
#include "soc/rtc_cntl_struct.h"

// ==============================================================================
// TEST MICRO I2S INMP441 CHO GOOUUU ESP32-S3
// ==============================================================================
// Sơ đồ nối dây chuẩn theo Goouuu AIoT Kit:
// - VDD -> 3V3 (3.3V)
// - GND -> GND
// - SCK -> GPIO 47 (BCLK)
// - WS  -> GPIO 48 (LRCK)
// - SD  -> GPIO 01 (DOUT / DIN)
// - L/R -> GND (Chọn kênh Mono bên Trái)
// ==============================================================================

#define I2S_SCK_PIN   47
#define I2S_WS_PIN    48
#define I2S_SD_PIN     1

#define SAMPLE_RATE   16000  // 16kHz chuẩn cho mô hình Whisper AI
#define BUFFER_SAMPLES 256

I2SClass i2s;
int32_t raw_samples[BUFFER_SAMPLES];

void setup() {
  // 1. Tắt bảo vệ sụt áp Brownout
  RTCCNTL.brown_out.ena = 0;
  RTCCNTL.brown_out.rst_ena = 0;

  Serial.begin(115200);
  delay(1500);

  Serial.println("\n==================================================");
  Serial.println("       TEST MICRO I2S (INMP441) - GOOUUU S3       ");
  Serial.println("==================================================");
  Serial.printf("[+] Chan SCK (BCLK) : GPIO %d\n", I2S_SCK_PIN);
  Serial.printf("[+] Chan WS  (LRCK) : GPIO %d\n", I2S_WS_PIN);
  Serial.printf("[+] Chan SD  (DIN)  : GPIO %d\n", I2S_SD_PIN);
  Serial.printf("[+] Tan so lay mau  : %d Hz (16-bit Mono)\n", SAMPLE_RATE);
  Serial.println("--------------------------------------------------");

  // 2. Cấu hình chân I2S
  i2s.setPins(I2S_SCK_PIN, I2S_WS_PIN, -1, I2S_SD_PIN);

  // 3. Khởi tạo bus I2S chuẩn Standard (Master RX, 32-bit slot, Mono Left)
  if (!i2s.begin(I2S_MODE_STD, SAMPLE_RATE, I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO, I2S_STD_SLOT_LEFT)) {
    Serial.println("[ERROR] Khoi tao I2S that bai! Kiem tra lai bo thu vien.");
    while (true) {
      delay(1000);
    }
  }

  Serial.println("[SUCCESS] Khoi tao I2S cho INMP441 thanh cong 100%!");
  Serial.println("\n>>> HAY NOI VAO MIC HOAC VO TAY DE XEM COT AM LUONG <<<");
  Serial.println("(Meo: Ban co the mo Tools -> Serial Plotter de xem dang song am thanh!)\n");
  delay(1000);
}

void loop() {
  // Đọc khối dữ liệu âm thanh từ INMP441
  size_t bytes_read = i2s.readBytes((char*)raw_samples, sizeof(raw_samples));

  if (bytes_read == 0) {
    delay(10);
    return;
  }

  int samples_count = bytes_read / sizeof(int32_t);
  int32_t max_peak = 0;
  int64_t sum_squares = 0;
  bool all_zeros = true;

  for (int i = 0; i < samples_count; i++) {
    // INMP441 xuất dữ liệu 24-bit căn lề trái trong khung 32-bit.
    // Dịch phải 14 bit để chuyển về chuẩn int16_t có độ khuếch đại tốt
    int16_t sample16 = (int16_t)(raw_samples[i] >> 14);

    if (sample16 != 0) {
      all_zeros = false;
    }

    int32_t abs_val = abs(sample16);
    if (abs_val > max_peak) {
      max_peak = abs_val;
    }
    sum_squares += (int64_t)sample16 * sample16;
  }

  // Tính âm lượng hiệu dụng (RMS)
  int16_t rms = sqrt(sum_squares / samples_count);

  // Cảnh báo nếu toàn số 0 (chưa tiếp xúc dây)
  if (all_zeros) {
    Serial.println("[CANH BAO] Tin hieu tat ca bang 0! Kiem tra lai chan SD, SCK, WS hoac nguon 3.3V.");
    delay(500);
    return;
  }

  // Vẽ thanh đo âm lượng trực quan (VU Meter bar)
  // Thang đo từ 0 đến 16000
  int bar_length = map(constrain(max_peak, 0, 16000), 0, 16000, 0, 30);

  Serial.print("Am luong: [");
  for (int i = 0; i < 30; i++) {
    if (i < bar_length) {
      Serial.print("=");
    } else {
      Serial.print(" ");
    }
  }
  Serial.printf("] Peak: %5d | RMS: %4d", max_peak, rms);

  if (max_peak > 3000) {
    Serial.print("  <-- Phat hien giong noi!");
  }
  Serial.println();

  delay(60); // Cập nhật thanh đo khoảng 15 lần/giây để nhìn mượt
}
