#include <Arduino.h>
#include "esp_camera.h"
#include <WiFi.h>
#include "esp_http_server.h"
#include "soc/soc.h"
#include "soc/rtc_cntl_struct.h"
#include "ESP_I2S.h"
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <math.h>

// =====================================================
// Wi-Fi Configuration
// =====================================================
const char* WIFI_SSID = "FPTU_Library";
const char* WIFI_PASSWORD = "12345678";

// ==============================================================================
// 1. PIN CAMERA OV2640 (Goouuu ESP32-S3-CAM)
// ==============================================================================
#define PWDN_GPIO_NUM     -1
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM     15
#define SIOD_GPIO_NUM      4  // SCCB Data
#define SIOC_GPIO_NUM      5  // SCCB Clock

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
#define LED_GPIO_NUM      -1

// ==============================================================================
// 2. PIN MICRO I2S INMP441 (I2S RX)
// SCK: 47 | WS: 48 | SD: 1 | L/R: GND | VDD: 3.3V
// ==============================================================================
#define MIC_I2S_SCK_PIN   47
#define MIC_I2S_WS_PIN    48
#define MIC_I2S_SD_PIN     1
#define SAMPLE_RATE    16000

// ==============================================================================
// 3. PIN SCREEN OLED I2C SSD1306 (128x64)
// SCL: 40 | SDA: 41 | VCC: 3.3V | GND: GND
// ==============================================================================
#define OLED_SCL_PIN      40
#define OLED_SDA_PIN      41
#define SCREEN_WIDTH     128
#define SCREEN_HEIGHT     64

// ==============================================================================
// 4. PIN AMPLIFIER MAX98357A (I2S TX)
// DIN: 35 | BCLK: 36 | LRC: 37 | VIN: 5V/3.3V | GND: GND
// ==============================================================================
#define SPK_I2S_DIN_PIN   35
#define SPK_I2S_BCLK_PIN  36
#define SPK_I2S_LRC_PIN   37

// =====================================================
// GLOBAL OBJECTS
// =====================================================
static I2SClass i2s_mic(I2S_NUM_1);
static I2SClass i2s_spk(I2S_NUM_0);
static httpd_handle_t control_httpd = NULL;
static httpd_handle_t camera_httpd  = NULL;
static httpd_handle_t mic_httpd     = NULL;

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);
static bool oled_present = false;
static uint8_t detected_oled_addr = 0x3C;

// =====================================================
// OLED HELPER FUNCTIONS
// =====================================================
uint8_t scan_and_init_oled() {
  pinMode(OLED_SDA_PIN, INPUT_PULLUP);
  pinMode(OLED_SCL_PIN, INPUT_PULLUP);

  // Thử cấu hình 1: SDA=41, SCL=40
  Wire.end();
  Wire.begin(OLED_SDA_PIN, OLED_SCL_PIN, 100000);
  delay(50);

  uint8_t addr_found = 0;
  for (uint8_t a = 1; a < 127; a++) {
    Wire.beginTransmission(a);
    if (Wire.endTransmission() == 0) {
      Serial.printf("[I2C SCAN] Phat hien thiet bi tai dia chi 0x%02X\n", a);
      if (a == 0x3C || a == 0x3D) {
        addr_found = a;
        break;
      }
    }
  }

  // Thử cấu hình 2: Đảo chân SDA=40, SCL=41
  if (addr_found == 0) {
    Serial.println("[I2C SCAN] Thu dao chan SDA=40, SCL=41...");
    Wire.end();
    Wire.begin(OLED_SCL_PIN, OLED_SDA_PIN, 100000);
    delay(50);
    for (uint8_t a = 1; a < 127; a++) {
      Wire.beginTransmission(a);
      if (Wire.endTransmission() == 0) {
        Serial.printf("[I2C SCAN] Phat hien thiet bi tai 0x%02X (chan dao)\n", a);
        if (a == 0x3C || a == 0x3D) {
          addr_found = a;
          break;
        }
      }
    }
  }

  if (addr_found == 0) {
    addr_found = 0x3C;
    Wire.end();
    Wire.begin(OLED_SDA_PIN, OLED_SCL_PIN, 100000);
  }

  // QUAN TRỌNG: reset=false, periphBegin=false để không bị Adafruit_SSD1306 gọi wire->begin() đè mất cấu hình chân GPIO 40/41!
  display.begin(SSD1306_SWITCHCAPVCC, addr_found, false, false);
  Serial.printf("[+] Khoi tao OLED tai dia chi 0x%02X (periphBegin=false)\n", addr_found);
  return addr_found;
}

void oled_show_status(const char* l1, const char* l2, const char* l3, const char* l4) {
  if (!oled_present) return;

  display.clearDisplay();

  // Tiêu đề viền phía trên
  display.fillRect(0, 0, 128, 14, SSD1306_WHITE);
  display.setTextColor(SSD1306_BLACK);
  display.setTextSize(1);
  display.setCursor(14, 3);
  display.println("BLIND TO BRIGHT");

  display.setTextColor(SSD1306_WHITE);
  display.setTextWrap(true);

  if (l1) {
    display.setTextSize(1);
    display.setCursor(0, 17);
    display.println(l1);
  }

  // Dòng chính (IP address): CỠ CHỮ TO SIZE 2
  if (l2) {
    display.setTextSize(2);
    display.setCursor(0, 30);
    display.println(l2);
  }

  if (l3) {
    display.setTextSize(1);
    display.setCursor(0, 52);
    display.println(l3);
  }

  display.display();
}

void oled_show_transcript(const char* label, const char* text) {
  if (!oled_present) return;

  display.clearDisplay();

  // Header bar (nền trắng chữ đen nổi bật)
  display.fillRect(0, 0, 128, 13, SSD1306_WHITE);
  display.setTextColor(SSD1306_BLACK);
  display.setTextSize(1);
  display.setCursor(4, 3);
  display.println(label ? label : "HOI THOAI");

  // Nội dung chữ nổi bật trên nền đen
  display.setTextColor(SSD1306_WHITE);
  display.setTextWrap(true);

  int len = text ? strlen(text) : 0;
  if (len <= 25) {
    // CHỮ TO SIZE 2: Cực kỳ rõ ràng, dễ nhìn từ xa
    display.setTextSize(2);
    display.setCursor(0, 18);
  } else {
    // Chữ Size 1 cho câu dài hơn
    display.setTextSize(1);
    display.setCursor(0, 16);
  }
  display.println(text ? text : "");

  display.display();
}

// =====================================================
// SPEAKER BEEP HELPER (Phát nốt nhạc to rõ)
// =====================================================
void play_startup_chime() {
  int16_t buf[256];
  float freqs[] = { 523.25, 659.25, 783.99 }; // Do, Mi, Sol
  for (int f = 0; f < 3; f++) {
    float phase_step = 2.0 * M_PI * freqs[f] / SAMPLE_RATE;
    float phase = 0.0;
    for (int i = 0; i < 128; i++) {
      int16_t sample = (int16_t)(sin(phase) * 24000.0);
      buf[i * 2]     = sample;
      buf[i * 2 + 1] = sample;
      phase += phase_step;
      if (phase >= 2.0 * M_PI) phase -= 2.0 * M_PI;
    }
    for (int rep = 0; rep < 25; rep++) {
      i2s_spk.write((const uint8_t*)buf, sizeof(buf));
    }
    delay(50);
  }
}

// =====================================================
// HTTP STREAM & CONTROL SERVER
// =====================================================
#define PART_BOUNDARY "123456789000000000000987654321"
static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* _STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* _STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

static esp_err_t stream_handler(httpd_req_t *req) {
  camera_fb_t * fb = NULL;
  esp_err_t res = ESP_OK;
  size_t _jpg_buf_len = 0;
  uint8_t * _jpg_buf = NULL;
  char part_buf[64];

  res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
  if (res != ESP_OK) return res;
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

static esp_err_t mic_stream_handler(httpd_req_t *req) {
  httpd_resp_set_type(req, "audio/L16;rate=16000;channels=1");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

  // Gửi ngay 2 byte mẫu 0 để client nhận HTTP 200 OK ngay lập tức không bị timeout
  int16_t start_hdr = 0;
  httpd_resp_send_chunk(req, (const char*)&start_hdr, sizeof(start_hdr));

  int32_t raw_buf[64];
  int16_t pcm_buf[64];

  while (true) {
    size_t bytes_read = i2s_mic.readBytes((char*)raw_buf, sizeof(raw_buf));
    if (bytes_read > 0) {
      int count = bytes_read / sizeof(int32_t);
      for (int i = 0; i < count; i++) {
        pcm_buf[i] = (int16_t)(raw_buf[i] >> 14);
      }
      esp_err_t res = httpd_resp_send_chunk(req, (const char*)pcm_buf, count * sizeof(int16_t));
      if (res != ESP_OK) {
        break;
      }
    } else {
      vTaskDelay(pdMS_TO_TICKS(5));
    }
  }
  return ESP_OK;
}

static esp_err_t record_wav_handler(httpd_req_t *req) {
  const size_t duration_sec = 3;
  const size_t total_samples = SAMPLE_RATE * duration_sec;
  const size_t pcm_size = total_samples * sizeof(int16_t);
  const size_t subchunk2_size = pcm_size;
  const size_t chunk_size = 36 + subchunk2_size;

  uint8_t wav_header[44];
  uint32_t sample_rate = SAMPLE_RATE;
  uint16_t num_channels = 1;
  uint16_t bits_per_sample = 16;
  uint32_t byte_rate = sample_rate * num_channels * (bits_per_sample / 8);
  uint16_t block_align = num_channels * (bits_per_sample / 8);

  memcpy(wav_header, "RIFF", 4);
  memcpy(wav_header + 4, &chunk_size, 4);
  memcpy(wav_header + 8, "WAVE", 4);
  memcpy(wav_header + 12, "fmt ", 4);
  uint32_t subchunk1_size = 16;
  memcpy(wav_header + 16, &subchunk1_size, 4);
  uint16_t audio_format = 1;
  memcpy(wav_header + 20, &audio_format, 2);
  memcpy(wav_header + 22, &num_channels, 2);
  memcpy(wav_header + 24, &sample_rate, 4);
  memcpy(wav_header + 28, &byte_rate, 4);
  memcpy(wav_header + 32, &block_align, 2);
  memcpy(wav_header + 34, &bits_per_sample, 2);
  memcpy(wav_header + 36, "data", 4);
  memcpy(wav_header + 40, &subchunk2_size, 4);

  httpd_resp_set_type(req, "audio/wav");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  httpd_resp_set_hdr(req, "Content-Disposition", "inline; filename=\"record_3s.wav\"");

  // Gửi header WAV trước
  httpd_resp_send_chunk(req, (const char*)wav_header, 44);

  // Stream trực tiếp theo từng chunk nhỏ, không tốn RAM
  int32_t raw_temp[64];
  int16_t pcm_chunk[64];
  size_t samples_recorded = 0;

  while (samples_recorded < total_samples) {
    size_t to_read = min((size_t)64, total_samples - samples_recorded);
    size_t bytes_read = i2s_mic.readBytes((char*)raw_temp, to_read * sizeof(int32_t));
    if (bytes_read > 0) {
      size_t count = bytes_read / sizeof(int32_t);
      for (size_t i = 0; i < count; i++) {
        pcm_chunk[i] = (int16_t)(raw_temp[i] >> 14);
      }
      samples_recorded += count;
      if (httpd_resp_send_chunk(req, (const char*)pcm_chunk, count * sizeof(int16_t)) != ESP_OK) {
        break;
      }
    } else {
      vTaskDelay(pdMS_TO_TICKS(5));
    }
  }

  httpd_resp_send_chunk(req, NULL, 0); // Kết thúc stream WAV
  return ESP_OK;
}

// Handler nhận Text từ Laptop và hiển thị ngay lên màn hình OLED
static esp_err_t oled_post_handler(httpd_req_t *req) {
  char buf[256];
  int ret = httpd_req_recv(req, buf, min((size_t)sizeof(buf) - 1, (size_t)req->content_len));
  if (ret <= 0) {
    httpd_resp_send_500(req);
    return ESP_FAIL;
  }
  buf[ret] = '\0';
  Serial.printf("[OLED RECV]: %s\n", buf);

  // Phân loại nhãn hiển thị: MIC (Lời nói) hoặc SIGN (Ký hiệu)
  if (strncmp(buf, "MIC:", 4) == 0) {
    oled_show_transcript("MICROPHONE -> CHU", buf + 4);
  } else if (strncmp(buf, "SIGN:", 5) == 0) {
    oled_show_transcript("KY HIEU -> LOA", buf + 5);
  } else {
    oled_show_transcript("THONG TIN", buf);
  }

  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  httpd_resp_send(req, "OK", HTTPD_RESP_USE_STRLEN);
  return ESP_OK;
}

// Handler phát âm thanh trực tiếp qua Loa MAX98357A
static esp_err_t play_post_handler(httpd_req_t *req) {
  if (req->content_len <= 0) {
    httpd_resp_send(req, "EMPTY", HTTPD_RESP_USE_STRLEN);
    return ESP_OK;
  }

  char buf[512];
  int remaining = req->content_len;

  while (remaining > 0) {
    int to_read = min((size_t)sizeof(buf), (size_t)remaining);
    int ret = httpd_req_recv(req, buf, to_read);
    if (ret <= 0) break;
    remaining -= ret;
    i2s_spk.write((const uint8_t*)buf, ret);
  }

  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  httpd_resp_send(req, "OK", HTTPD_RESP_USE_STRLEN);
  return ESP_OK;
}

static esp_err_t index_handler(httpd_req_t *req) {
  const char* html = 
    "<!DOCTYPE html><html><head><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
    "<title>Blind to Bright - ESP32-S3</title>"
    "<style>body{text-align:center;background:#141414;color:#f0f0f0;font-family:sans-serif;padding:20px;}"
    "img{max-width:100%;height:auto;border-radius:10px;box-shadow:0 4px 12px rgba(0,0,0,0.6);margin-bottom:15px;}"
    ".btn{display:inline-block;margin:10px;padding:12px 24px;background:#007bff;color:#fff;text-decoration:none;border-radius:8px;font-weight:bold;}"
    ".btn:hover{background:#0056b3;}</style></head><body>"
    "<h2>🌟 Blind to Bright - ESP32-S3 Camera & Audio</h2>"
    "<p>Camera Stream: Port 81 | Mic Stream: Port 82</p>"
    "<a class=\"btn\" href=\"/record.wav\" target=\"_blank\">🎙️ Thu Âm 5 Giây & Nghe Thử Mic</a>"
    "</body></html>";
  httpd_resp_set_type(req, "text/html");
  return httpd_resp_send(req, html, HTTPD_RESP_USE_STRLEN);
}

static esp_err_t status_handler(httpd_req_t *req) {
  char buf[256];
  snprintf(buf, sizeof(buf),
    "Blind to Bright Status:\n- Free Heap: %u bytes\n- PSRAM Found: %s\n- Free PSRAM: %u bytes\n- Wi-Fi RSSI: %d dBm\n- OLED: %s (0x%02X)\n",
    ESP.getFreeHeap(),
    psramFound() ? "YES" : "NO",
    ESP.getFreePsram(),
    WiFi.RSSI(),
    oled_present ? "OK" : "NOT FOUND",
    detected_oled_addr
  );
  httpd_resp_set_type(req, "text/plain");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  return httpd_resp_send(req, buf, HTTPD_RESP_USE_STRLEN);
}

void setupLedFlash(int pin) {
#if LED_GPIO_NUM >= 0
  pinMode(pin, OUTPUT);
  digitalWrite(pin, LOW);
#endif
}

void startCameraServer() {
  // 1. Control & Audio Server trên Port 80 (/oled, /play, /record.wav, /status, /)
  httpd_config_t config80 = HTTPD_DEFAULT_CONFIG();
  config80.server_port = 80;
  config80.ctrl_port = 32768;
  config80.stack_size = 8192;
  config80.max_open_sockets = 7;
  config80.lru_purge_enable = true;

  httpd_uri_t index_uri      = { .uri = "/",           .method = HTTP_GET,  .handler = index_handler,     .user_ctx = NULL };
  httpd_uri_t record_wav_uri = { .uri = "/record.wav", .method = HTTP_GET,  .handler = record_wav_handler,.user_ctx = NULL };
  httpd_uri_t status_uri     = { .uri = "/status",     .method = HTTP_GET,  .handler = status_handler,    .user_ctx = NULL };
  httpd_uri_t oled_uri       = { .uri = "/oled",       .method = HTTP_POST, .handler = oled_post_handler, .user_ctx = NULL };
  httpd_uri_t play_uri       = { .uri = "/play",       .method = HTTP_POST, .handler = play_post_handler, .user_ctx = NULL };

  if (httpd_start(&control_httpd, &config80) == ESP_OK) {
    httpd_register_uri_handler(control_httpd, &index_uri);
    httpd_register_uri_handler(control_httpd, &record_wav_uri);
    httpd_register_uri_handler(control_httpd, &status_uri);
    httpd_register_uri_handler(control_httpd, &oled_uri);
    httpd_register_uri_handler(control_httpd, &play_uri);
    Serial.println("[+] Control & Audio Server hoat dong tren Port 80 (/oled, /play, /record.wav, /status)");
  }

  // 2. Server Stream Camera rieng biet tren Port 81 (/stream)
  httpd_config_t config81 = HTTPD_DEFAULT_CONFIG();
  config81.server_port = 81;
  config81.ctrl_port = 32769;
  config81.stack_size = 4096;
  config81.max_open_sockets = 4;
  config81.lru_purge_enable = true;

  httpd_uri_t stream_uri = { .uri = "/stream", .method = HTTP_GET, .handler = stream_handler, .user_ctx = NULL };
  if (httpd_start(&camera_httpd, &config81) == ESP_OK) {
    httpd_register_uri_handler(camera_httpd, &stream_uri);
    Serial.println("[+] Camera Stream Server hoat dong tren Port 81 (/stream)");
  }

  // 3. Server Stream Micro rieng biet tren Port 82 (/mic)
  httpd_config_t config82 = HTTPD_DEFAULT_CONFIG();
  config82.server_port = 82;
  config82.ctrl_port = 32770;
  config82.stack_size = 4096;
  config82.max_open_sockets = 4;
  config82.lru_purge_enable = true;

  httpd_uri_t mic_uri = { .uri = "/mic", .method = HTTP_GET, .handler = mic_stream_handler, .user_ctx = NULL };
  if (httpd_start(&mic_httpd, &config82) == ESP_OK) {
    httpd_register_uri_handler(mic_httpd, &mic_uri);
    Serial.println("[+] Microphone Stream Server hoat dong tren Port 82 (/mic)");
  }
}

// =====================================================
// SETUP
// =====================================================
void setup() {
  // 1. TẮT BROWNOUT DETECTOR TRÊN ESP32-S3
  RTCCNTL.brown_out.ena = 0;
  RTCCNTL.brown_out.rst_ena = 0;
  delay(100);

  Serial.begin(115200);
  Serial.setDebugOutput(true);
  delay(500);

  Serial.println("\n==================================================");
  Serial.println("       BLIND TO BRIGHT - ESP32-S3 FULL SYSTEM     ");
  Serial.println("==================================================");
  Serial.printf("[+] OLED   : SCL=GPIO %d, SDA=GPIO %d\n", OLED_SCL_PIN, OLED_SDA_PIN);
  Serial.printf("[+] Mic    : SCK=GPIO %d, WS=GPIO %d, SD=GPIO %d\n", MIC_I2S_SCK_PIN, MIC_I2S_WS_PIN, MIC_I2S_SD_PIN);
  Serial.printf("[+] Speaker: DIN=GPIO %d, BCLK=GPIO %d, LRC=GPIO %d\n", SPK_I2S_DIN_PIN, SPK_I2S_BCLK_PIN, SPK_I2S_LRC_PIN);
  Serial.println("--------------------------------------------------");

  // 2. Khởi tạo Màn hình OLED (SSD1306)
  detected_oled_addr = scan_and_init_oled();
  oled_present = true; // Luôn cho phép hiển thị OLED
  Serial.printf("[+] OLED SSD1306 hoat dong tai 0x%02X!\n", detected_oled_addr);
  oled_show_status("Dang khoi dong...", "Blind to Bright", "Micro INMP441", "Loa MAX98357A");

  // 3. Khởi tạo Loa I2S MAX98357A (TX: 35, 36, 37)
  i2s_spk.setPins(SPK_I2S_BCLK_PIN, SPK_I2S_LRC_PIN, SPK_I2S_DIN_PIN, -1);
  if (i2s_spk.begin(I2S_MODE_STD, SAMPLE_RATE, I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_STEREO)) {
    Serial.println("[+] Loa MAX98357A khoi tao thanh cong!");
    play_startup_chime(); // Phát tiếng chuông khởi động kiểm tra loa ngay!
  } else {
    Serial.println("[!] Loa MAX98357A khoi tao that bai!");
  }

  // 4. Khởi tạo Micro I2S INMP441 (RX: 47, 48, 1)
  i2s_mic.setPins(MIC_I2S_SCK_PIN, MIC_I2S_WS_PIN, -1, MIC_I2S_SD_PIN);
  if (i2s_mic.begin(I2S_MODE_STD, SAMPLE_RATE, I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO, I2S_STD_SLOT_LEFT)) {
    Serial.println("[+] Micro INMP441 khoi tao thanh cong!");
  } else {
    Serial.println("[!] Micro INMP441 khoi tao that bai!");
  }

  // 5. Khởi tạo Camera OV2640
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

  config.xclk_freq_hz = 10000000;
  config.pixel_format = PIXFORMAT_JPEG;
  if (psramFound()) {
    Serial.println("[+] PSRAM phat hien! Cau hinh 2 frame buffer trong PSRAM.");
    config.frame_size   = FRAMESIZE_QVGA;
    config.jpeg_quality = 10;
    config.fb_count     = 2;
    config.fb_location  = CAMERA_FB_IN_PSRAM;
    config.grab_mode    = CAMERA_GRAB_LATEST;
  } else {
    Serial.println("[!] Khong tim thay PSRAM. Cau hinh 1 frame buffer trong DRAM.");
    config.frame_size   = FRAMESIZE_QVGA;
    config.jpeg_quality = 12;
    config.fb_count     = 1;
    config.fb_location  = CAMERA_FB_IN_DRAM;
    config.grab_mode    = CAMERA_GRAB_WHEN_EMPTY;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("[ERROR] Camera init failed: 0x%x\n", err);
    char err_msg[32];
    snprintf(err_msg, sizeof(err_msg), "CAMERA: LOI 0x%X", err);
    oled_show_status(err_msg, "Kiem tra cap vang", NULL, NULL);
  } else {
    Serial.println("[+] Camera OV2640 khoi tao thanh cong!");
  }

  // 6. Kết nối Wi-Fi
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.setTxPower(WIFI_POWER_11dBm);

  Serial.printf("\n[-] Dang ket noi Wi-Fi '%s'...\n", WIFI_SSID);
  oled_show_status("Dang ket noi Wi-Fi...", WIFI_SSID, "Vui long cho...", NULL);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  unsigned long start_time = millis();
  while (WiFi.status() != WL_CONNECTED && (millis() - start_time < 15000)) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    IPAddress ip = WiFi.localIP();
    Serial.printf("[+] Da ket noi Wi-Fi! IP: %s\n", ip.toString().c_str());
    startCameraServer();

    char ip_str[32];
    snprintf(ip_str, sizeof(ip_str), "%s", ip.toString().c_str());
    oled_show_status("Wi-Fi: KET NOI OK", ip_str, "Cam: 81 | Mic: 82", "Loa & OLED: 80");
  } else {
    Serial.println("\n[!] Khong the ket noi Wi-Fi. CHUYEN SANG SOFTAP...");
    WiFi.disconnect(true);
    delay(500);
    WiFi.mode(WIFI_AP);
    WiFi.softAP("ESP32-S3-CAMERA", "12345678");

    IPAddress apIP = WiFi.softAPIP();
    Serial.println("[+] Tu phat Wi-Fi: ESP32-S3-CAMERA (pass: 12345678)");
    Serial.printf("[+] Dia chi IP: %s\n", apIP.toString().c_str());
    startCameraServer();

    oled_show_status("WIFI: ESP32-S3-CAM", "192.168.4.1", "Cam: 81 | Mic: 82", "Pass: 12345678");
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
