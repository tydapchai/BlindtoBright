#include <Arduino.h>
#include "esp_camera.h"

// ==============================================================================
// TEST CAMERA CHUẨN CHO GOOUUU ESP32-S3-CAM (OV2640)
// ==============================================================================

#define PWDN_GPIO_NUM     -1
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM     15
#define SIOD_GPIO_NUM      4
#define SIOC_GPIO_NUM      5

#define Y9_GPIO_NUM       16
#define Y8_GPIO_NUM       17
#define Y7_GPIO_NUM       18
#define Y6_GPIO_NUM       12
#define Y5_GPIO_NUM       10
#define Y4_GPIO_NUM        8
#define Y3_GPIO_NUM        9
#define Y2_GPIO_NUM       11
#define VSYNC_GPIO_NUM     6
#define HREF_GPIO_NUM      7
#define PCLK_GPIO_NUM     13

void setup() {
  Serial.begin(115200);
  delay(1500);

  Serial.println("\n==================================================");
  Serial.println("       TEST CAMERA GOOUUU ESP32-S3 (OV2640)       ");
  Serial.println("==================================================");

  // 1. Kiểm tra trạng thái PSRAM
  if (psramFound()) {
    Serial.printf("[+] PSRAM: HOAT DONG TOT (Dung luong: %d MB)\n", ESP.getPsramSize() / 1024 / 1024);
  } else {
    Serial.println("[!] CHÚ Ý: PSRAM chưa được bật!");
    Serial.println("    -> Hãy vào Arduino IDE: Tools -> PSRAM -> Chọn 'OPI PSRAM'");
  }
  Serial.flush();

  // 2. Cấu hình Camera Goouuu
  Serial.println("\n[-] Đang thiết lập cấu hình camera Goouuu...");
  camera_config_t config = {};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;

  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;

  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;

  config.xclk_freq_hz = 10000000; // 10MHz chuẩn ổn định cho Goouuu
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size   = FRAMESIZE_QVGA; // 320x240
  config.jpeg_quality = 12;
  config.fb_count     = 1;
  config.grab_mode    = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location  = psramFound() ? CAMERA_FB_IN_PSRAM : CAMERA_FB_IN_DRAM;

  Serial.println("[-] Đang kết nối thử với mắt OV2640...");
  Serial.flush();

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("[!] 10MHz chưa nhận (mã lỗi 0x%x), đang thử lại với 20MHz...\n", err);
    config.xclk_freq_hz = 20000000;
    err = esp_camera_init(&config);
  }

  if (err == ESP_OK) {
    Serial.println("\n==================================================");
    Serial.println(" [SUCCESS] KHOI TAO CAMERA GOOUUU THANH CONG 100%!");
    Serial.println("==================================================");

    sensor_t* s = esp_camera_sensor_get();
    if (s != nullptr) {
      Serial.printf("[+] Sensor PID: 0x%04X\n", s->id.PID);
      if (s->id.PID == OV2640_PID) {
        Serial.println("[+] Xác nhận đúng mắt camera: OV2640");
      }
    }
    Serial.println("[-] Bắt đầu chụp ảnh trong loop()...\n");
  } else {
    Serial.println("\n==================================================");
    Serial.printf(" [THẤT BẠI] esp_camera_init báo lỗi: 0x%x\n", err);
    Serial.println("==================================================");
    Serial.println("HƯỚNG XỬ LÝ PHẦN CỨNG:");
    Serial.println("1. Mặt có các VẠCH ĐỒNG VÀNG ở đuôi cáp PHẢI ÚP XUỐNG DƯỚI.");
    Serial.println("2. Sợi cáp phải được đẩy KỊCH ĐÁY vào khe.");
    Serial.println("3. Thanh lẫy đen của khe cắm phải được BẤM CHẶT xuống.");
  }
  Serial.flush();
}

unsigned long frame_count = 0;

void loop() {
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("[!] Không thể chụp ảnh (fb is NULL)");
    delay(2000);
    return;
  }

  frame_count++;
  Serial.printf(
    "[OK] Frame #%-4lu | Kích thước: %6u bytes | Độ phân giải: %dx%d (JPEG)\n",
    frame_count,
    fb->len,
    fb->width,
    fb->height
  );

  esp_camera_fb_return(fb);
  delay(1500); // Chụp mỗi 1.5 giây
}
