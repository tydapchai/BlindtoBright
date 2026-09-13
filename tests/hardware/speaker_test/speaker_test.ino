#include <Arduino.h>
#include "soc/soc.h"
#include "soc/rtc_cntl_struct.h"
#include "ESP_I2S.h"
#include <math.h>

// =====================================================
// PIN AMPLIFIER MAX98357A (I2S TX)
// DIN  : GPIO 35
// BCLK : GPIO 36
// LRC  : GPIO 37
// =====================================================
#define SPK_I2S_DIN_PIN   35
#define SPK_I2S_BCLK_PIN  36
#define SPK_I2S_LRC_PIN   37

#define SAMPLE_RATE       16000
#define BUFFER_SIZE         256

static I2SClass i2s_spk;
int16_t sine_buffer[BUFFER_SIZE * 2]; // Stereo (L+R)

void generate_sine(float freq, float volume = 0.3) {
  float phase_step = 2.0 * M_PI * freq / SAMPLE_RATE;
  float phase = 0.0;
  for (int i = 0; i < BUFFER_SIZE; i++) {
    int16_t val = (int16_t)(sin(phase) * 32767.0 * volume);
    sine_buffer[i * 2]     = val; // Left
    sine_buffer[i * 2 + 1] = val; // Right
    phase += phase_step;
    if (phase >= 2.0 * M_PI) phase -= 2.0 * M_PI;
  }
}

void setup() {
  RTCCNTL.brown_out.ena = 0;
  RTCCNTL.brown_out.rst_ena = 0;

  Serial.begin(115200);
  delay(1000);

  Serial.println("\n==================================================");
  Serial.println("       TEST LOA / AMPLIFIER (MAX98357A) - GOOUUU S3");
  Serial.println("==================================================");
  Serial.printf("[+] Chan DIN  (Data)  : GPIO %d\n", SPK_I2S_DIN_PIN);
  Serial.printf("[+] Chan BCLK (Clock) : GPIO %d\n", SPK_I2S_BCLK_PIN);
  Serial.printf("[+] Chan LRC  (WS)    : GPIO %d\n", SPK_I2S_LRC_PIN);
  Serial.printf("[+] Sample rate       : %d Hz (16-bit Stereo)\n", SAMPLE_RATE);
  Serial.println("--------------------------------------------------");

  // Cấu hình chân I2S TX cho MAX98357A
  i2s_spk.setPins(SPK_I2S_BCLK_PIN, SPK_I2S_LRC_PIN, SPK_I2S_DIN_PIN, -1);
  if (!i2s_spk.begin(I2S_MODE_STD, SAMPLE_RATE, I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_STEREO)) {
    Serial.println("[ERROR] Khoi tao I2S cho Loa that bai!");
    while (true) delay(1000);
  }

  Serial.println("[SUCCESS] Khoi tao I2S MAX98357A thanh cong!");
  Serial.println("[+] Bat dau phat am thanh beep test qua loa...");
}

void loop() {
  // Phat am 523Hz (not Do) trong 0.3 giay
  Serial.println(">>> Phat tieng beep (Do - 523Hz)...");
  generate_sine(523.25, 0.25);
  for (int i = 0; i < (SAMPLE_RATE * 0.3 / BUFFER_SIZE); i++) {
    i2s_spk.write((const uint8_t*)sine_buffer, sizeof(sine_buffer));
  }
  delay(100);

  // Phat am 659Hz (not Mi) trong 0.3 giay
  Serial.println(">>> Phat tieng beep (Mi - 659Hz)...");
  generate_sine(659.25, 0.25);
  for (int i = 0; i < (SAMPLE_RATE * 0.3 / BUFFER_SIZE); i++) {
    i2s_spk.write((const uint8_t*)sine_buffer, sizeof(sine_buffer));
  }
  delay(100);

  // Phat am 784Hz (not Sol) trong 0.5 giay
  Serial.println(">>> Phat tieng beep (Sol - 784Hz)...");
  generate_sine(783.99, 0.25);
  for (int i = 0; i < (SAMPLE_RATE * 0.5 / BUFFER_SIZE); i++) {
    i2s_spk.write((const uint8_t*)sine_buffer, sizeof(sine_buffer));
  }

  Serial.println("[-] Nghi 2 giay...");
  delay(2000);
}
