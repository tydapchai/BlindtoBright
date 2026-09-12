#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include "soc/soc.h"
#include "soc/rtc_cntl_struct.h"

// =====================================================
// PIN OLED I2C SSD1306 (128x64)
// SCL: GPIO 40
// SDA: GPIO 41
// =====================================================
#define OLED_SCL_PIN   40
#define OLED_SDA_PIN   41
#define SCREEN_WIDTH  128
#define SCREEN_HEIGHT  64

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

uint8_t scan_i2c(int sda, int scl) {
  Wire.end();
  Wire.begin(sda, scl, 100000);
  delay(50);

  Serial.printf("[-] Dang quet I2C tren chan SDA: GPIO %d | SCL: GPIO %d...\n", sda, scl);
  uint8_t found_addr = 0;
  for (uint8_t addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {
      Serial.printf("    [+] TIM THAY THIET BI I2C TAI DIA CHI: 0x%02X\n", addr);
      if (addr == 0x3C || addr == 0x3D) {
        found_addr = addr;
      }
    }
  }
  return found_addr;
}

void setup() {
  RTCCNTL.brown_out.ena = 0;
  RTCCNTL.brown_out.rst_ena = 0;

  Serial.begin(115200);
  delay(1500);

  Serial.println("\n==================================================");
  Serial.println("     KIEM TRA & DEBUG MAN HINH OLED (SSD1306)     ");
  Serial.println("==================================================");
  Serial.printf("[+] Cau hinh goc: SCL=GPIO %d, SDA=GPIO %d\n", OLED_SCL_PIN, OLED_SDA_PIN);

  // 1. Quét I2C với cấu hình chuẩn (SDA=41, SCL=40)
  uint8_t oled_addr = scan_i2c(OLED_SDA_PIN, OLED_SCL_PIN);

  // 2. Nếu không thấy, thử đảo chân (SDA=40, SCL=41) phòng khi nối ngược dây
  if (oled_addr == 0) {
    Serial.println("[!] Khong thay OLED tren chan 41/40. Thu dao chan SDA=40, SCL=41...");
    oled_addr = scan_i2c(OLED_SCL_PIN, OLED_SDA_PIN);
    if (oled_addr != 0) {
      Serial.println("[Chu y] Ban dang cam nguoc: SDA la GPIO 40, SCL la GPIO 41!");
    }
  }

  if (oled_addr == 0) {
    Serial.println("\n[ERROR] KHONG TIM THAY MAN HINH OLED!");
    Serial.println("  -> Vui long kiem tra lai:");
    Serial.println("     1. Day nguon VCC da vao 3.3V (hoac 5V) chua?");
    Serial.println("     2. Day GND da cam dung GND chua?");
    Serial.println("     3. Day SDA, SCL da tiep xuc tot chua?");
    while (true) {
      delay(2000);
      scan_i2c(OLED_SDA_PIN, OLED_SCL_PIN);
    }
  }

  Serial.printf("\n[+] Khoi dong OLED voi dia chi 0x%02X bang Adafruit Driver...\n", oled_addr);
  if (!display.begin(SSD1306_SWITCHCAPVCC, oled_addr)) {
    Serial.println("[ERROR] display.begin() that bai!");
    while (true) delay(1000);
  }

  Serial.println("[SUCCESS] MAN HINH OLED DA SANG VA HOAT DONG 100%!");

  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);

  // Vẽ khung viền
  display.drawRect(0, 0, 128, 64, SSD1306_WHITE);

  // Tiêu đề nổi bật
  display.setTextSize(1);
  display.setCursor(12, 6);
  display.println("BLIND TO BRIGHT");

  display.drawLine(0, 18, 128, 18, SSD1306_WHITE);

  display.setCursor(8, 24);
  display.println("OLED: HOAT DONG TOT");

  display.setCursor(8, 36);
  display.printf("Dia chi: 0x%02X", oled_addr);

  display.setCursor(8, 48);
  display.println("ESP32-S3 Cam OK!");

  display.display();
}

int count = 0;
void loop() {
  delay(1000);
  count++;

  // Xóa vùng hiển thị dòng dưới và cập nhật bộ đếm
  display.fillRect(8, 48, 112, 12, SSD1306_BLACK);
  display.setCursor(8, 48);
  display.printf("Running: %d s", count);
  display.display();

  Serial.printf("[OLED TEST] Running: %d s\n", count);
}
