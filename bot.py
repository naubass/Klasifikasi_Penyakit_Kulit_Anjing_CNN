import logging
import os
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

# Logging System
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Configuration
TOKEN       = os.environ.get("TELEGRAM_BOT_TOKEN")
MODEL_PATH  = "best_model.keras"
IMG_SIZE    = (224, 224)
CLASS_NAMES = [
    "Dermatitis",        
    "Fungal_infections", 
    "Healthy",           
    "Hypersensitivity",  
    "demodicosis",       
    "ringworm",          
]

# Load Model
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded!")

# Informasi Penyakit
DISEASE_INFO = {
    "Dermatitis": {
        "emoji"      : "🔴",
        "description": "Dermatitis adalah peradangan kulit yang disebabkan oleh alergi, infeksi, atau iritasi dari bahan tertentu.",
        "advice"     : (
            "• Identifikasi dan singkirkan penyebab iritasi (deterjen, shampo, dll).\n"
            "• Hindari garukan berlebih — gunakan kalung pelindung (e-collar) bila perlu.\n"
            "• Konsultasikan ke dokter hewan untuk mendapatkan antihistamin atau kortikosteroid.\n"
            "• Mandikan anjing dengan shampo hypoallergenic.\n"
            "• Catat makanan atau lingkungan baru yang mungkin memicu reaksi."
        ),
    },
    "Fungal_infections": {
        "emoji"      : "🍄",
        "description": "Infeksi jamur pada kulit menyebabkan pengelupasan, kerontokan bulu, dan rasa gatal akibat pertumbuhan jamur berlebih.",
        "advice"     : (
            "• Jaga kulit anjing tetap kering dan bersih — jamur berkembang di tempat lembab.\n"
            "• Gunakan shampo antijamur sesuai rekomendasi dokter hewan.\n"
            "• Dokter mungkin meresepkan obat antijamur oral seperti flukonazol.\n"
            "• Cuci dan sterilkan semua aksesori anjing secara rutin.\n"
            "• Pisahkan dari hewan peliharaan lain selama pengobatan."
        ),
    },
    "Healthy": {
        "emoji"      : "✅",
        "description": "Kulit anjing terlihat sehat dan tidak menunjukkan tanda-tanda penyakit.",
        "advice"     : (
            "• Pertahankan rutinitas perawatan yang sudah baik ini!\n"
            "• Mandikan anjing secara rutin (1–2 kali seminggu).\n"
            "• Berikan makanan bergizi dan air bersih setiap hari.\n"
            "• Lakukan pemeriksaan rutin ke dokter hewan setiap 6 bulan sekali.\n"
            "• Pastikan vaksinasi dan pemberian antiparasit tetap terjadwal."
        ),
    },
    "Hypersensitivity": {
        "emoji"      : "⚠️",
        "description": "Hipersensitivitas adalah reaksi berlebihan sistem imun terhadap alergen seperti kutu, serbuk sari, atau makanan tertentu.",
        "advice"     : (
            "• Lakukan uji alergi (allergy test) di klinik hewan untuk mengetahui pemicunya.\n"
            "• Gunakan obat pencegah kutu secara rutin (spot-on, kalung anti-kutu).\n"
            "• Pertimbangkan diet eliminasi jika dicurigai alergi makanan.\n"
            "• Dokter dapat meresepkan imunoterapi atau antihistamin jangka panjang.\n"
            "• Bersihkan rumah secara rutin untuk mengurangi debu dan serbuk sari."
        ),
    },
    "demodicosis": {
        "emoji"      : "🦠",
        "description": "Demodicosis (Demodectic Mange) disebabkan oleh tungau Demodex yang mengakibatkan kerontokan bulu dan iritasi kulit.",
        "advice"     : (
            "• Segera bawa anjing ke dokter hewan untuk pemeriksaan skin scraping.\n"
            "• Hindari kontak dengan anjing lain untuk sementara waktu.\n"
            "• Dokter biasanya meresepkan obat antiparasit seperti ivermectin atau amitraz.\n"
            "• Jaga kebersihan tempat tidur dan peralatan anjing.\n"
            "• Berikan nutrisi seimbang untuk memperkuat sistem imun."
        ),
    },
    "ringworm": {
        "emoji"      : "🔵",
        "description": "Ringworm (Dermatophytosis) adalah infeksi jamur menular yang menyebabkan bercak bulat bersisik dan kebotakan pada kulit.",
        "advice"     : (
            "• ⚠️ Ringworm sangat menular ke hewan lain dan manusia — tangani dengan sarung tangan!\n"
            "• Isolasi anjing dari anggota keluarga dan hewan lain segera.\n"
            "• Konsultasi dokter hewan untuk antijamur topikal dan/atau oral.\n"
            "• Cuci semua permukaan, karpet, dan tempat tidur hewan dengan disinfektan.\n"
            "• Periksa anggota keluarga — terutama anak-anak — untuk tanda yang sama."
        ),
    },
}

# Clinic Info
CLINIC_INFO = """
🏥 *Klinik Hewan Terdekat yang Bisa Dihubungi:*

1️⃣ *Pet Care Veterinary Clinic*
   📞 (021) 123-4567
   🕐 Senin–Sabtu: 08.00–20.00

⚠️ Hasil ini bukan pengganti diagnosis dokter hewan profesional. 
Selalu konsultasikan kondisi hewan Anda ke dokter hewan.
"""

# Preprocessing Function
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)   # (1, 224, 224, 3)

# Prediksi Function
def predict(image_bytes: bytes):
    img_array = preprocess_image(image_bytes)
    probs     = model.predict(img_array, verbose=0)[0]   # (num_classes,)
    idx       = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]) * 100, probs

# Start BOT System
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Halo! Selamat datang di *DogSkin AI Bot* 🐶\n\n"
        "Bot ini dapat membantu mendeteksi kemungkinan penyakit kulit pada anjing "
        "berdasarkan foto yang kamu kirimkan.\n\n"
        "📸 *Cara pakai:*\n"
        "Kirimkan foto kulit anjing kamu langsung ke chat ini.\n\n"
        "⚠️ _Hasil analisis ini bersifat informatif dan bukan pengganti "
        "diagnosis dokter hewan profesional._",
        parse_mode="Markdown"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🆘 *Bantuan*\n\n"
        "• Kirim foto kulit anjing → bot akan menganalisis penyakitnya.\n"
        "• /start — Pesan sambutan\n"
        "• /klinik — Daftar nomor klinik hewan\n"
        "• /penyakit — Daftar penyakit yang dapat dideteksi\n"
        "• /help — Tampilkan bantuan ini",
        parse_mode="Markdown"
    )

async def klinik(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(CLINIC_INFO, parse_mode="Markdown")

async def penyakit(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = "🔬 *Penyakit yang dapat dideteksi:*\n\n"
    for name, info in DISEASE_INFO.items():
        text += f"{info['emoji']} *{name}*\n_{info['description']}_\n\n"
    await update.message.reply_text(text, parse_mode="Markdown")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔍 Sedang menganalisis gambar, harap tunggu...")

    try:
        # Ambil foto resolusi tertinggi
        photo_file = await update.message.photo[-1].get_file()
        image_bytes = await photo_file.download_as_bytearray()

        # Prediksi
        predicted_class, confidence, all_probs = predict(bytes(image_bytes))
        info = DISEASE_INFO[predicted_class]

        # Susun probabilitas semua kelas
        prob_text = "\n".join([
            f"  {'→' if CLASS_NAMES[i] == predicted_class else '  '} "
            f"{CLASS_NAMES[i]}: {all_probs[i]*100:.1f}%"
            for i in range(len(CLASS_NAMES))
        ])

        # Susun pesan hasil
        result_message = (
            f"{info['emoji']} *Hasil Analisis*\n"
            f"{'─' * 30}\n\n"
            f"🐶 *Kondisi Terdeteksi:* {predicted_class}\n"
            f"📊 *Tingkat Keyakinan:* {confidence:.2f}%\n\n"
            # f"📋 *Probabilitas Semua Kelas:*\n{prob_text}\n\n"
            f"📝 *Deskripsi:*\n{info['description']}\n\n"
            f"💡 *Saran Penanganan:*\n{info['advice']}\n\n"

            # info klinik hewan
            f"{CLINIC_INFO}"
        )

        await update.message.reply_text(result_message, parse_mode=None)

    except Exception as e:
        logger.error(f"Error saat prediksi: {e}")
        await update.message.reply_text(
            "❌ Terjadi kesalahan saat memproses gambar.\n"
            "Pastikan gambar jelas dan coba kirim ulang."
        )

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📸 Silakan kirim *foto* kulit anjing untuk dianalisis.\n"
        "Ketik /help untuk melihat panduan penggunaan.",
        parse_mode="Markdown"
    )

# Main Function
def main():
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start",    start))
    app.add_handler(CommandHandler("help",     help_command))
    app.add_handler(CommandHandler("klinik",   klinik))
    app.add_handler(CommandHandler("penyakit", penyakit))
    app.add_handler(MessageHandler(filters.PHOTO,        handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    print("Bot berjalan... tekan Ctrl+C untuk berhenti.")
    app.run_polling()

if __name__ == "__main__":
    main()