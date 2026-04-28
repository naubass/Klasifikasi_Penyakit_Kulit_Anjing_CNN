import os
import io
import numpy as np
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request

app = FastAPI(title="DogSkin AI - Klasifikasi Penyakit Kulit Anjing")

templates = Jinja2Templates(directory="templates")

# ── Konfigurasi ───────────────────────────────────────────────────────────────
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

# ── Load Model ────────────────────────────────────────────────────────────────
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded!")

# ── Informasi Penyakit ────────────────────────────────────────────────────────
DISEASE_INFO = {
    "Dermatitis": {
        "emoji": "🔴",
        "label": "Dermatitis",
        "color": "#ef4444",
        "description": "Dermatitis adalah peradangan kulit yang disebabkan oleh alergi, infeksi, atau iritasi dari bahan tertentu.",
        "advice": [
            "Identifikasi dan singkirkan penyebab iritasi (deterjen, shampo, dll).",
            "Hindari garukan berlebih — gunakan kalung pelindung (e-collar) bila perlu.",
            "Konsultasikan ke dokter hewan untuk mendapatkan antihistamin atau kortikosteroid.",
            "Mandikan anjing dengan shampo hypoallergenic.",
            "Catat makanan atau lingkungan baru yang mungkin memicu reaksi.",
        ],
    },
    "Fungal_infections": {
        "emoji": "🍄",
        "label": "Infeksi Jamur",
        "color": "#f59e0b",
        "description": "Infeksi jamur pada kulit menyebabkan pengelupasan, kerontokan bulu, dan rasa gatal akibat pertumbuhan jamur berlebih.",
        "advice": [
            "Jaga kulit anjing tetap kering dan bersih — jamur berkembang di tempat lembab.",
            "Gunakan shampo antijamur sesuai rekomendasi dokter hewan.",
            "Dokter mungkin meresepkan obat antijamur oral seperti flukonazol.",
            "Cuci dan sterilkan semua aksesori anjing secara rutin.",
            "Pisahkan dari hewan peliharaan lain selama pengobatan.",
        ],
    },
    "Healthy": {
        "emoji": "✅",
        "label": "Sehat",
        "color": "#22c55e",
        "description": "Kulit anjing terlihat sehat dan tidak menunjukkan tanda-tanda penyakit.",
        "advice": [
            "Pertahankan rutinitas perawatan yang sudah baik ini!",
            "Mandikan anjing secara rutin (1–2 kali seminggu).",
            "Berikan makanan bergizi dan air bersih setiap hari.",
            "Lakukan pemeriksaan rutin ke dokter hewan setiap 6 bulan sekali.",
            "Pastikan vaksinasi dan pemberian antiparasit tetap terjadwal.",
        ],
    },
    "Hypersensitivity": {
        "emoji": "⚠️",
        "label": "Hipersensitivitas",
        "color": "#f97316",
        "description": "Hipersensitivitas adalah reaksi berlebihan sistem imun terhadap alergen seperti kutu, serbuk sari, atau makanan tertentu.",
        "advice": [
            "Lakukan uji alergi (allergy test) di klinik hewan untuk mengetahui pemicunya.",
            "Gunakan obat pencegah kutu secara rutin (spot-on, kalung anti-kutu).",
            "Pertimbangkan diet eliminasi jika dicurigai alergi makanan.",
            "Dokter dapat meresepkan imunoterapi atau antihistamin jangka panjang.",
            "Bersihkan rumah secara rutin untuk mengurangi debu dan serbuk sari.",
        ],
    },
    "demodicosis": {
        "emoji": "🦠",
        "label": "Demodikosis",
        "color": "#a855f7",
        "description": "Demodicosis (Demodectic Mange) disebabkan oleh tungau Demodex yang mengakibatkan kerontokan bulu dan iritasi kulit.",
        "advice": [
            "Segera bawa anjing ke dokter hewan untuk pemeriksaan skin scraping.",
            "Hindari kontak dengan anjing lain untuk sementara waktu.",
            "Dokter biasanya meresepkan obat antiparasit seperti ivermectin atau amitraz.",
            "Jaga kebersihan tempat tidur dan peralatan anjing.",
            "Berikan nutrisi seimbang untuk memperkuat sistem imun.",
        ],
    },
    "ringworm": {
        "emoji": "🔵",
        "label": "Ringworm",
        "color": "#3b82f6",
        "description": "Ringworm (Dermatophytosis) adalah infeksi jamur menular yang menyebabkan bercak bulat bersisik dan kebotakan pada kulit.",
        "advice": [
            "Ringworm sangat menular ke hewan lain dan manusia — tangani dengan sarung tangan!",
            "Isolasi anjing dari anggota keluarga dan hewan lain segera.",
            "Konsultasi dokter hewan untuk antijamur topikal dan/atau oral.",
            "Cuci semua permukaan, karpet, dan tempat tidur hewan dengan disinfektan.",
            "Periksa anggota keluarga — terutama anak-anak — untuk tanda yang sama.",
        ],
    },
}

# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((256, 256), Image.BILINEAR)
    left = top = (256 - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar.")

    image_bytes = await file.read()
    img_array   = preprocess_image(image_bytes)
    probs       = model.predict(img_array, verbose=0)[0]
    idx         = int(np.argmax(probs))

    predicted_key   = CLASS_NAMES[idx]
    confidence      = float(probs[idx]) * 100
    info            = DISEASE_INFO[predicted_key]

    all_probs = [
        {
            "class": CLASS_NAMES[i],
            "label": DISEASE_INFO[CLASS_NAMES[i]]["label"],
            "prob": round(float(probs[i]) * 100, 2),
            "color": DISEASE_INFO[CLASS_NAMES[i]]["color"],
        }
        for i in range(len(CLASS_NAMES))
    ]
    all_probs.sort(key=lambda x: x["prob"], reverse=True)

    return JSONResponse({
        "predicted_class": predicted_key,
        "label"          : info["label"],
        "emoji"          : info["emoji"],
        "color"          : info["color"],
        "confidence"     : round(confidence, 2),
        "description"    : info["description"],
        "advice"         : info["advice"],
        "all_probs"      : all_probs,
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)