/* ============================================================
 * esp32_relay.ino — Kontrol 4 relay via serial dari PC
 * ============================================================
 * Relay terhubung ke pin: 5, 19, 18, 23  (relay 1..4)
 *
 * PROTOKOL SERIAL (dari PC), satu perintah per BARIS (diakhiri '\n'):
 *   <nomor_relay><status>
 *     nomor_relay : '1'..'4'
 *     status      : '1' = ON, '0' = OFF
 *   Contoh:  "31\n" -> relay 3 ON
 *            "20\n" -> relay 2 OFF
 *   Perintah khusus:
 *     "A0\n" -> semua relay OFF
 *     "A1\n" -> semua relay ON
 *
 * PENTING: Saat Python memakai port ini, JANGAN buka Serial Monitor
 *          di Arduino IDE — port hanya bisa dipakai satu program.
 * ============================================================ */

const int RELAY_PINS[4] = {5, 19, 18, 23};   // relay 1,2,3,4
const bool ACTIVE_LOW = true;   // modul relay aktif-LOW; set false jika aktif-HIGH

void setRelay(int idx, bool on) {
  if (idx < 0 || idx > 3) return;
  digitalWrite(RELAY_PINS[idx], (on ^ ACTIVE_LOW) ? HIGH : LOW);
}

void setAll(bool on) {
  for (int i = 0; i < 4; i++) setRelay(i, on);
}

void handleCommand(String cmd) {
  cmd.trim();                 // buang spasi / newline / CR
  if (cmd.length() < 2) return;

  char a = cmd.charAt(0);     // nomor relay atau 'A'
  char b = cmd.charAt(1);     // status 0/1
  bool on = (b == '1');

  if (a == 'A' || a == 'a') {
    setAll(on);
    Serial.print("ALL "); Serial.println(on ? "ON" : "OFF");
  }
  else if (a >= '1' && a <= '4') {
    int idx = a - '1';        // '1'->0 ... '4'->3
    setRelay(idx, on);
    Serial.print("RELAY "); Serial.print(idx + 1);
    Serial.print(" "); Serial.println(on ? "ON" : "OFF");
  }
}

void setup() {
  Serial.begin(115200);
  for (int i = 0; i < 4; i++) {
    pinMode(RELAY_PINS[i], OUTPUT);
    setRelay(i, false);       // mulai semua OFF
  }
  Serial.println("ESP32 siap. Format per baris: <relay 1-4><0/1>, mis '31'");
}

void loop() {
  // Baca SATU baris penuh sampai newline -> hilangkan masalah karakter sisa.
  if (Serial.available() > 0) {
    String line = Serial.readStringUntil('\n');
    handleCommand(line);
  }
}
