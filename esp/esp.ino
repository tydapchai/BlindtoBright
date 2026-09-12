#include <Arduino.h>
#include "esp_camera.h"
#include <WiFi.h>
#include "esp_http_server.h"
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

// =====================================================
// Wi-Fi Configuration
// =====================================================
const char* WIFI_SSID = "AABW3";
const char* WIFI_PASSWORD = "AGENTIC2026";

// =====================================================
// PIN CAMERA (ESP32-S3-CAM / Freenove / Ai-Thinker)
// =====================================================
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

// Đặt -1 nếu không có LED flash hoặc không dùng
#define LED_GPIO_NUM      -1

// =====================================================
// HTTP STREAM SERVER
// =====================================================
#define PART_BOUNDARY "123456789000000000000987654321"
static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* _STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* _STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

static httpd_handle_t stream_httpd = NULL;
static httpd_handle_t camera_httpd = NULL;

static esp_err_t stream_handler(httpd_req_t *req) {
  camera_fb_t * fb = NULL;
  esp_err_t res = ESP_OK;
  size_t _jpg_buf_len = 0;
  uint8_t * _jpg_buf = NULL;
  char part_buf[64];

  res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
  if (res != ESP_OK) {
    return res;
  }
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

  while (true) {
    fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Camera capture failed");
      res = ESP_FAIL;
    } else {
      _jpg_buf_len = fb->len;
      _jpg_buf = fb->buf;
    }

    if (res == ESP_OK) {
      size_t hlen = snprintf(part_buf, sizeof(part_buf), _STREAM_PART, _jpg_buf_len);
      res = httpd_resp_send_chunk(req, part_buf, hlen);
    }
    if (res == ESP_OK) {
      res = httpd_resp_send_chunk(req, (const char *)_jpg_buf, _jpg_buf_len);
    }
    if (res == ESP_OK) {
      res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
    }

    if (fb) {
      esp_camera_fb_return(fb);
      fb = NULL;
      _jpg_buf = NULL;
    } else if (res != ESP_OK) {
      break;
    }

    if (res != ESP_OK) {
      break;
    }
  }

  return res;
}

static esp_err_t index_handler(httpd_req_t *req) {
  const char* html = "<!DOCTYPE html><html><head><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"><title>ESP32-S3 CAM</title></head><body style=\"text-align:center;background:#181818;color:#eee;font-family:sans-serif;\"><h2>ESP32-S3 Stream</h2><img src=\"/stream\" style=\"max-width:100%;height:auto;border-radius:8px;\"></body></html>";
  httpd_resp_set_type(req, "text/html");
  return httpd_resp_send(req, html, HTTPD_RESP_USE_STRLEN);
}

void setupLedFlash(int pin) {
#if LED_GPIO_NUM >= 0
  pinMode(pin, OUTPUT);
  digitalWrite(pin, LOW);
#endif
}

void startCameraServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;
  config.ctrl_port = 32768;

  httpd_uri_t index_uri = {
    .uri       = "/",
    .method    = HTTP_GET,
    .handler   = index_handler,
    .user_ctx  = NULL
  };

  httpd_uri_t stream_uri = {
    .uri       = "/stream",
    .method    = HTTP_GET,
    .handler   = stream_handler,
    .user_ctx  = NULL
  };

  // Khởi động server xem trực tiếp trên port 80
  if (httpd_start(&camera_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(camera_httpd, &index_uri);
    httpd_register_uri_handler(camera_httpd, &stream_uri);
  }

  // Khởi động stream server chuyên dụng trên port 81 (khớp với test.py và app.py)
  config.server_port = 81;
  config.ctrl_port = 32769;
  if (httpd_start(&stream_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(stream_httpd, &stream_uri);
  }
}

// =====================================================
// SETUP
// =====================================================
void setup() {
  // 1. TẮT BROWNOUT DETECTOR NGAY TỪ ĐẦU ĐỂ TRÁNH RESET SỤT ÁP CỔNG USB
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
  delay(100);

  Serial.begin(115200);
  Serial.setDebugOutput(true);
  delay(500);

  Serial.println();
  Serial.println("Starting ESP32-S3 camera...");

  camera_config_t config = {};

  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;

  // Gán chân camera ESP32-S3
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;

  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;

  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;

  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;

  config.xclk_freq_hz = 10000000; // 10MHz bắt buộc cho board Freenove ESP32-S3
  config.pixel_format = PIXFORMAT_JPEG;

  // Cấu hình buffer và độ phân giải dựa trên PSRAM
  if (psramFound()) {
    Serial.println("PSRAM found.");
    config.frame_size = FRAMESIZE_QVGA; // Khởi động ở QVGA để ổn định probe
    config.jpeg_quality = 10;
    config.fb_count = 2;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
    config.fb_location = CAMERA_FB_IN_PSRAM;
  } else {
    Serial.println("PSRAM not found; using conservative settings.");
    config.frame_size = FRAMESIZE_QVGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
    config.fb_location = CAMERA_FB_IN_DRAM;
  }

  esp_err_t camera_error = esp_camera_init(&config);
  if (camera_error != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", camera_error);
    Serial.println("Check the camera model, ribbon cable, PSRAM and GPIO mapping.");
    while (true) {
      delay(1000);
    }
  }

  sensor_t* sensor = esp_camera_sensor_get();
  if (sensor != nullptr) {
    // Để QVGA (320x240) để đảm bảo FPS cao và độ trễ thấp cho nhận diện bàn tay
    sensor->set_framesize(sensor, FRAMESIZE_QVGA);
    // Nếu ảnh bị lộn ngược hoặc lật gương, có thể bật các dòng sau:
    // sensor->set_vflip(sensor, 1);
    // sensor->set_hmirror(sensor, 1);
  }

  delay(500); // Tạm dừng 500ms để điện áp ổn định sau khi camera bật

#if LED_GPIO_NUM >= 0
  setupLedFlash(LED_GPIO_NUM);
#endif

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.setTxPower(WIFI_POWER_11dBm); // Hạ bớt công suất phát để bảo vệ nguồn USB

  Serial.printf("\n[-] Dang ket noi vao Wi-Fi '%s' (bang tan 2.4GHz)...\n", WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  const unsigned long start_time = millis();
  const unsigned long timeout_ms = 15000; // Đợi tối đa 15 giây

  while (WiFi.status() != WL_CONNECTED && (millis() - start_time < timeout_ms)) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("[+] Da ket noi Wi-Fi thanh cong! IP: ");
    Serial.println(WiFi.localIP());
    startCameraServer();

    Serial.println();
    Serial.println("Camera server started successfully!");
    Serial.printf("Web viewer:   http://%s/\n", WiFi.localIP().toString().c_str());
    Serial.printf("Stream (80):  http://%s/stream\n", WiFi.localIP().toString().c_str());
    Serial.printf("Stream (81):  http://%s:81/stream\n", WiFi.localIP().toString().c_str());
  } else {
    // Nếu không kết nối được Wi-Fi ngoài -> TỰ ĐỘNG PHÁT HOTSPOT RIÊNG
    Serial.println("\n[!] Khong the ket noi Wi-Fi ngoai (co the do song 5GHz hoac sai pass).");
    Serial.println("[+] CHUYEN SANG CHE DO TU PHAT HOTSPOT (SoftAP)...");

    WiFi.disconnect(true);
    delay(500);
    WiFi.mode(WIFI_AP);
    WiFi.softAP("ESP32-S3-CAMERA", "12345678");

    IPAddress apIP = WiFi.softAPIP();
    Serial.println("[SUCCESS] Da tu phat mang Wi-Fi thanh cong!");
    Serial.println("  -> Ten Wi-Fi : ESP32-S3-CAMERA");
    Serial.println("  -> Mat khau  : 12345678");
    Serial.printf("  -> Dia chi IP: %s\n", apIP.toString().c_str());

    startCameraServer();

    Serial.println();
    Serial.printf("Web viewer:   http://%s/\n", apIP.toString().c_str());
    Serial.printf("Stream (80):  http://%s/stream\n", apIP.toString().c_str());
    Serial.printf("Stream (81):  http://%s:81/stream\n", apIP.toString().c_str());
  }
}

// =====================================================
// LOOP
// =====================================================
void loop() {
  delay(10000);
  Serial.printf(
    "Wi-Fi RSSI: %d dBm | Free heap: %u\n",
    WiFi.RSSI(),
    ESP.getFreeHeap()
  );
}
