"""
inference.py — Klasifikasi Hewan + Kontrol 4 Relay ESP32 (serial)
=========================================================================
Fitur:
    - Pilih model: CNN saja / Transfer Learning saja / Komparasi keduanya
    - Mode kamera (real-time) & upload foto
    - Kontrol 4 relay ESP32 lewat USB serial berdasarkan kelas terdeteksi

Pemetaan kelas -> relay:
    Panda    -> TOGGLE relay 1 (kenali->ON, kenali lagi->OFF)
    Monkey   -> TOGGLE relay 2
    Cat      -> TOGGLE relay 3
    Cow      -> TOGGLE relay 4
    Elephant -> TOGGLE semua relay sekaligus

Mode komparasi: relay hanya bereaksi jika KEDUA model sepakat & confidence cukup.

Setup:
    pip install tensorflow numpy opencv-python matplotlib pyserial

Jalankan:
    1. Upload esp32_relay.ino ke ESP32 (relay di pin 5,19,18,23).
    2. TUTUP Serial Monitor Arduino (port hanya untuk 1 program).
    3. Set SERIAL_PORT di bawah (Windows "COM5" / Linux "/dev/ttyUSB0").
       Set None untuk uji tanpa hardware.
    4. python inference.py
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf

# ---------- dependensi opsional ----------
try:
    import cv2; HAVE_CV2 = True
except ImportError: HAVE_CV2 = False
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAVE_MPL = True
except ImportError: HAVE_MPL = False
try:
    import tkinter as tk
    from tkinter import filedialog
    HAVE_TK = True
except ImportError: HAVE_TK = False
try:
    import serial; HAVE_SERIAL = True
except ImportError: HAVE_SERIAL = False

# ==========================================
# 1. Konfigurasi
# ==========================================
# PENTING: urutan harus SAMA dengan class_names yang dicetak notebook saat training.
CLASS_NAMES  = ['Panda',  'Monkey', 'Cat',    'Cow',  'Elephant']
CLASS_LABELS = ['Panda',  'Monyet', 'Kucing', 'Sapi', 'Gajah']
CLASS_EMOJI  = ['🐼',      '🐵',     '🐱',     '🐮',   '🐘']
IMG_SIZE     = (224, 224)

CLASS_COLORS_BGR = {
    'Panda': (0,165,255), 'Monkey': (255,0,255), 'Cat': (255,255,0),
    'Cow': (0,255,0), 'Elephant': (0,255,255),
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CNN_PATH   = os.path.join(SCRIPT_DIR, "best_cnn.keras")
TRF_PATH   = os.path.join(SCRIPT_DIR, "best_tl.keras")

MODEL_TITLES = {"cnn": "CNN From Scratch", "trf": "MobileNetV2 Transfer Learning"}

# --- Serial / Relay ---
SERIAL_PORT    = "COM6"      # Windows "COM5" | Linux "/dev/ttyUSB0" | None = nonaktif
BAUD           = 115200
CONF_THRESHOLD = 0.70        # confidence minimal agar relay bereaksi

# Pemetaan kelas -> nomor relay (1..4). Elephant ditangani khusus (toggle ALL).
CLASS_TO_RELAY = {"Panda": 1, "Monkey": 2, "Cat": 3, "Cow": 4}

# ==========================================
# 2. Relay controller (serial)
# ==========================================
class RelayController:
    def __init__(self, port, baud):
        self.ser = None
        self.all_on = False        # status toggle untuk Dog
        self.relay_state = {1: False, 2: False, 3: False, 4: False}  # status tiap relay
        if port is None:
            print("[Relay] SERIAL_PORT=None -> kontrol relay NONAKTIF (mode uji).")
            return
        if not HAVE_SERIAL:
            print("[Relay] pyserial belum terpasang: pip install pyserial")
            return
        try:
            self.ser = serial.Serial(port, baud, timeout=1)
            time.sleep(2)          # tunggu ESP32 reset
            print(f"[Relay] Terhubung ke ESP32 di {port} @ {baud}")
        except Exception as e:
            print(f"[Relay] GAGAL buka {port}: {e}")
            print("        Cek port & pastikan Serial Monitor Arduino TERTUTUP.")

    def _send(self, cmd):
        """cmd: string 2 char, mis '31' (relay3 ON) atau 'A0' (semua OFF)."""
        if self.ser is None:
            print(f"[Relay-uji] kirim '{cmd}'")
            return
        try:
            self.ser.write((cmd + "\n").encode())
        except Exception as e:
            print(f"[Relay] gagal kirim: {e}")

    def activate_class(self, class_name):
        """Terapkan aturan relay sesuai kelas terdeteksi.
        - Panda/Monkey/Cat/Cow : TOGGLE relay miliknya (kenali->ON, kenali lagi->OFF).
        - Elephant             : TOGGLE semua relay sekaligus.
        """
        if class_name == "Elephant":
            self.all_on = not self.all_on
            self._send("A1" if self.all_on else "A0")
            # sinkronkan status semua relay
            for i in range(1, 5):
                self.relay_state[i] = self.all_on
            print(f"[Relay] Elephant -> semua relay {'ON' if self.all_on else 'OFF'}")
        elif class_name in CLASS_TO_RELAY:
            r = CLASS_TO_RELAY[class_name]
            # toggle relay milik kelas ini saja
            self.relay_state[r] = not self.relay_state[r]
            on = self.relay_state[r]
            self._send(f"{r}{'1' if on else '0'}")
            print(f"[Relay] {class_name} -> relay {r} {'ON' if on else 'OFF'}")

    def all_off(self):
        self._send("A0"); self.all_on = False
        self.relay_state = {1: False, 2: False, 3: False, 4: False}

    def close(self):
        if self.ser:
            self.all_off()
            self.ser.close()

# ==========================================
# 3. Model (lazy load + cache)
# ==========================================
_loaded = {}
def get_model(key):
    if key in _loaded: return _loaded[key]
    path = CNN_PATH if key == "cnn" else TRF_PATH
    if not os.path.exists(path):
        sys.exit(f"[ERROR] Model tidak ditemukan: {path}")
    print(f"Memuat {MODEL_TITLES[key]}...")
    _loaded[key] = tf.keras.models.load_model(path, compile=False)
    print("  selesai.")
    return _loaded[key]

# ==========================================
# 4. Prediksi
# ==========================================
def preprocess_array(img_rgb):
    img = tf.image.resize(img_rgb, IMG_SIZE)
    return tf.expand_dims(tf.cast(img, tf.float32), 0)

def predict_one(model, img_rgb):
    probs = model.predict(preprocess_array(img_rgb), verbose=0)[0]
    i = int(np.argmax(probs))
    return {"name": CLASS_NAMES[i], "label": CLASS_LABELS[i],
            "emoji": CLASS_EMOJI[i], "conf": float(probs[i]), "probs": probs}

def predict(active_models, img_rgb):
    res = {k: predict_one(m, img_rgb) for k, m in active_models.items()}
    if "cnn" in res and "trf" in res:
        res["agree"] = (res["cnn"]["name"] == res["trf"]["name"])
    return res

def decide_relay(res, keys):
    """
    Tentukan (class_name, conf) untuk relay, atau (None,_) jika tak bereaksi.
    - 1 model : pakai prediksinya.
    - komparasi: hanya jika KEDUA model sepakat.
    Selalu cek confidence >= threshold.
    """
    if len(keys) == 1:
        r = res[keys[0]]
        return (r["name"], r["conf"]) if r["conf"] >= CONF_THRESHOLD else (None, 0)
    # komparasi
    if res.get("agree"):
        conf = min(res["cnn"]["conf"], res["trf"]["conf"])
        return (res["cnn"]["name"], conf) if conf >= CONF_THRESHOLD else (None, 0)
    return (None, 0)

def print_result(res, keys):
    print("=" * 50)
    for k in keys:
        r = res[k]
        print(f"[{MODEL_TITLES[k]:30s}] -> {r['emoji']} {r['label']} ({r['conf']*100:.1f}%)")
    if "agree" in res:
        print("✅ SETUJU" if res["agree"] else "⚠️  BEDA")
    print("=" * 50)

# ==========================================
# 5. Overlay OpenCV
# ==========================================
def draw_overlay_cv2(frame, res, keys):
    out = frame.copy()
    overlay = out.copy()
    bar_h = 50 + 38 * len(keys)
    cv2.rectangle(overlay, (0,0), (out.shape[1], bar_h), (20,20,20), -1)
    cv2.addWeighted(overlay, 0.65, out, 0.35, 0, out)
    y = 30
    for k in keys:
        r = res[k]
        color = CLASS_COLORS_BGR.get(r["name"], (255,255,255))
        tag = "CNN" if k == "cnn" else "TL "
        cv2.putText(out, f"{tag}: {r['label']} ({r['conf']*100:.0f}%)",
                    (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2)
        y += 38
    if "agree" in res:
        t = "SETUJU" if res["agree"] else "BEDA"
        c = (50,220,50) if res["agree"] else (50,50,255)
        tw = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0][0]
        cv2.putText(out, t, (out.shape[1]-tw-12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, c, 2)
    return out

# ==========================================
# 6. Mode kamera
# ==========================================
def mode_kamera(active_models, relay):
    if not HAVE_CV2:
        print("[!] OpenCV belum terpasang: pip install opencv-python"); return
    keys = list(active_models.keys())
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Kamera tidak bisa dibuka."); return
    print("\nSPACE/C = prediksi & kontrol relay | Q/ESC = menu\n")
    last_res = None
    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame, 1)
        disp = draw_overlay_cv2(frame, last_res, keys) if last_res else frame.copy()
        cv2.putText(disp, "SPACE/C=prediksi  Q/ESC=menu",
                    (10, disp.shape[0]-12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200,200,200), 1)
        cv2.imshow("Klasifikasi Hewan + Relay (Q=Menu)", disp)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27): break
        if key in (ord(' '), ord('c')):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            last_res = predict(active_models, rgb)
            print_result(last_res, keys)
            cls, conf = decide_relay(last_res, keys)
            if cls is not None:
                relay.activate_class(cls)
            else:
                print("[Relay] tidak bereaksi (confidence rendah / model tak sepakat)")
    cap.release(); cv2.destroyAllWindows()

# ==========================================
# 7. Mode upload
# ==========================================
def _pick_file_tk():
    root = tk.Tk(); root.withdraw()
    p = filedialog.askopenfilename(title="Pilih Gambar Hewan",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")])
    root.destroy(); return p or ""

def mode_upload(active_models, relay):
    keys = list(active_models.keys())
    if HAVE_TK:
        print("\n[INFO] Pilih gambar..."); path = _pick_file_tk()
    else:
        path = input("Masukkan path gambar: ").strip().strip('"')
    if not path: print("[INFO] Dibatalkan."); return
    if not os.path.exists(path): print(f"[ERROR] Tak ditemukan: {path}"); return

    raw = tf.io.read_file(path)
    img_rgb = tf.image.decode_image(raw, channels=3,
                                    expand_animations=False).numpy().astype("uint8")
    print(f"[INFO] Memproses: {os.path.basename(path)}")
    res = predict(active_models, img_rgb)
    print_result(res, keys)
    cls, conf = decide_relay(res, keys)
    if cls is not None:
        relay.activate_class(cls)
    else:
        print("[Relay] tidak bereaksi (confidence rendah / model tak sepakat)")

    if HAVE_MPL:
        fig = plt.figure(figsize=(12,5))
        gs = gridspec.GridSpec(1,2,figure=fig,width_ratios=[1,1.3])
        ax = fig.add_subplot(gs[0]); ax.imshow(img_rgb); ax.axis("off")
        title = ""
        for k in keys:
            r = res[k]; tag = "CNN" if k=="cnn" else "TL "
            title += f"{tag}: {r['emoji']} {r['label']} ({r['conf']*100:.0f}%)\n"
        if "agree" in res:
            title += "✅ SETUJU" if res["agree"] else "⚠️  BEDA"
        ax.set_title(title, fontsize=11, loc="left")
        axb = fig.add_subplot(gs[1]); x = np.arange(len(CLASS_NAMES))
        col = {"cnn":"#4C8BF5","trf":"#F5A623"}
        if len(keys)==1:
            k=keys[0]; axb.bar(x, res[k]["probs"]*100, 0.6,
                               label=MODEL_TITLES[k], color=col[k])
        else:
            w=0.38
            axb.bar(x-w/2, res["cnn"]["probs"]*100, w, label="CNN", color=col["cnn"])
            axb.bar(x+w/2, res["trf"]["probs"]*100, w, label="Transfer", color=col["trf"])
        axb.set_xticks(x); axb.set_xticklabels(CLASS_LABELS, rotation=20, ha="right")
        axb.set_ylabel("Probabilitas (%)"); axb.set_ylim(0,105); axb.legend()
        axb.grid(axis="y", alpha=0.3); axb.set_title("Probabilitas per Kelas")
        plt.suptitle(f"Hasil — {os.path.basename(path)}", fontweight="bold")
        plt.tight_layout(); plt.show()

# ==========================================
# 8. Menu
# ==========================================
def pilih_model():
    while True:
        print("\n"+"="*50); print("PILIH MODEL"); print("="*50)
        print("1. CNN From Scratch saja")
        print("2. MobileNetV2 Transfer Learning saja")
        print("3. Komparasi keduanya"); print("-"*50)
        p = input("Pilihan (1/2/3): ").strip()
        if p=="1": return {"cnn": get_model("cnn")}
        if p=="2": return {"trf": get_model("trf")}
        if p=="3": return {"cnn": get_model("cnn"), "trf": get_model("trf")}
        print("[!] Pilihan tidak valid.")

def label_aktif(am):
    ks=list(am.keys())
    return "Komparasi CNN vs Transfer Learning" if len(ks)==2 else MODEL_TITLES[ks[0]]

def print_menu(am):
    print("\n"+"="*50); print("🐾  KLASIFIKASI HEWAN + RELAY")
    print(f"    Model aktif: {label_aktif(am)}"); print("="*50)
    print("Kelas: 🐼Panda 🐵Monyet 🐱Kucing 🐮Sapi 🐘Gajah")
    print("Relay (toggle): Panda=1 Monkey=2 Cat=3 Cow=4 | Elephant=semua")
    print("-"*50)
    print("1. Deteksi dari Kamera")
    print("2. Upload Foto")
    print("3. Ganti Model")
    print("4. Keluar"); print("-"*50)

def main():
    print("Memuat...")
    active = pilih_model()
    relay = RelayController(SERIAL_PORT, BAUD)
    try:
        while True:
            print_menu(active)
            p = input("Pilihan (1/2/3/4): ").strip()
            if p=="1": mode_kamera(active, relay)
            elif p=="2": mode_upload(active, relay)
            elif p=="3": active = pilih_model()
            elif p=="4": print("Program ditutup."); break
            else: print("[!] Pilihan tidak valid.")
    finally:
        relay.close()
        print("Relay dimatikan, port ditutup.")

if __name__ == "__main__":
    main()
