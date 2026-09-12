#include <Arduino.h>
#include "esp_camera.h"
#include <WiFi.h>

// Hàm này nằm trong app_httpd.cpp của ví dụ CameraWebServer chính thức.
void startCameraServer();
void setupLedFlash(int pin);

// =====================================================
// Wi-Fi
// =====================================================

const char* WIFI_SSID = "AABW3";
const char* WIFI_PASSWORD = "AGENTIC2026";

// =====================================================
// PIN CAMERA
//
// PHẢI thay bằng pinout chính xác của board.
// Không đoán pin.
// =====================================================

#define CAM_PIN_PWDN   -1
#define CAM_PIN_RESET  -1

#define CAM_PIN_XCLK   YOUR_XCLK_PIN
#define CAM_PIN_SIOD   YOUR_SIOD_PIN
#define CAM_PIN_SIOC   YOUR_SIOC_PIN

#define CAM_PIN_D7     YOUR_Y9_PIN
#define CAM_PIN_D6     YOUR_Y8_PIN
#define CAM_PIN_D5     YOUR_Y7_PIN
#define CAM_PIN_D4     YOUR_Y6_PIN
#define CAM_PIN_D3     YOUR_Y5_PIN
#define CAM_PIN_D2     YOUR_Y4_PIN
#define CAM_PIN_D1     YOUR_Y3_PIN
#define CAM_PIN_D0     YOUR_Y2_PIN

#define CAM_PIN_VSYNC  YOUR_VSYNC_PIN
#define CAM_PIN_HREF   YOUR_HREF_PIN
#define CAM_PIN_PCLK   YOUR_PCLK_PIN

// Đặt -1 nếu không biết hoặc board không có LED flash.
#define LED_GPIO_NUM   -1

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  delay(1000);

  Serial.println();
  Serial.println("Starting ESP32-S3 camera...");

  camera_config_t config = {};

  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;

  config.pin_d0 = CAM_PIN_D0;
  config.pin_d1 = CAM_PIN_D1;
  config.pin_d2 = CAM_PIN_D2;
  config.pin_d3 = CAM_PIN_D3;
  config.pin_d4 = CAM_PIN_D4;
  config.pin_d5 = CAM_PIN_D5;
  config.pin_d6 = CAM_PIN_D6;
  config.pin_d7 = CAM_PIN_D7;

  config.pin_xclk = CAM_PIN_XCLK;
  config.pin_pclk = CAM_PIN_PCLK;
  config.pin_vsync = CAM_PIN_VSYNC;
  config.pin_href = CAM_PIN_HREF;

  config.pin_sccb_sda = CAM_PIN_SIOD;
  config.pin_sccb_scl = CAM_PIN_SIOC;

  config.pin_pwdn = CAM_PIN_PWDN;
  config.pin_reset = CAM_PIN_RESET;

  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  // Thiết lập nhẹ để stream ổn định.
  if (psramFound()) {
    Serial.println("PSRAM found.");

    config.frame_size = FRAMESIZE_VGA;
    config.jpeg_quality = 12;
    config.fb_count = 2;
    config.grab_mode = CAMERA_GRAB_LATEST;
    config.fb_location = CAMERA_FB_IN_PSRAM;
  } else {
    Serial.println("PSRAM not found; using conservative settings.");

    config.frame_size = FRAMESIZE_QVGA;
    config.jpeg_quality = 14;
    config.fb_count = 1;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
    config.fb_location = CAMERA_FB_IN_DRAM;
  }

  esp_err_t camera_error = esp_camera_init(&config);

  if (camera_error != ESP_OK) {
    Serial.printf(
      "Camera init failed with error 0x%x\n",
      camera_error
    );

    Serial.println(
      "Check the camera model, ribbon cable, PSRAM and GPIO mapping."
    );

    while (true) {
      delay(1000);
    }
  }

  sensor_t* sensor = esp_camera_sensor_get();

  if (sensor != nullptr) {
    // Khởi động ở QVGA để giảm tải và tăng FPS.
    sensor->set_framesize(sensor, FRAMESIZE_QVGA);

    // Có thể chỉnh sau nếu ảnh bị ngược.
    // sensor->set_vflip(sensor, 1);
    // sensor->set_hmirror(sensor, 1);
  }

#if LED_GPIO_NUM >= 0
  setupLedFlash(LED_GPIO_NUM);
#endif

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  Serial.print("Connecting to Wi-Fi");

  const unsigned long start_time = millis();
  const unsigned long timeout_ms = 30000;

  while (
    WiFi.status() != WL_CONNECTED &&
    millis() - start_time < timeout_ms
  ) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("Wi-Fi connection failed.");

    while (true) {
      delay(1000);
    }
  }

  Serial.print("Connected. IP: ");
  Serial.println(WiFi.localIP());

  startCameraServer();

  Serial.println();
  Serial.println("Camera server started.");
  Serial.printf(
    "Control page: http://%s\n",
    WiFi.localIP().toString().c_str()
  );
  Serial.printf(
    "MJPEG stream: http://%s:81/stream\n",
    WiFi.localIP().toString().c_str()
  );
}

void loop() {
  delay(10000);

  Serial.printf(
    "Wi-Fi RSSI: %d dBm | Free heap: %u\n",
    WiFi.RSSI(),
    ESP.getFreeHeap()
  );
}
