import numpy as np
from PIL import Image
import tensorflow as tf

MODEL_PATH = "saved_model/saved_model"  
IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "Demodicosis",
    "Dermatitis", 
    "Fungal Infections",
    "Healthy",
    "Hypersensitivity",
    "Ringworm",
]

model = tf.saved_model.load(MODEL_PATH)
infer = model.signatures['serving_default']

# Cek input & output signature
print("=== INPUT SIGNATURE ===")
for k, v in infer.structured_input_signature[1].items():
    print(f"  key: {k}, shape: {v.shape}, dtype: {v.dtype}")

print("\n=== OUTPUT SIGNATURE ===")
for k, v in infer.structured_outputs.items():
    print(f"  key: {k}, shape: {v.shape}, dtype: {v.dtype}")

img_path = "test_dog.jpg"  

# Preprocessing versi 1: resize 256 → crop 224
img = Image.open(img_path).convert("RGB")
img_256 = img.resize((256, 256), Image.BILINEAR)
left, top = (256 - 224) // 2, (256 - 224) // 2
img_crop = img_256.crop((left, top, left + 224, top + 224))
arr_v1 = np.expand_dims(np.array(img_crop, dtype=np.float32) / 255.0, axis=0)

# Preprocessing versi 2: langsung resize 224
img_224 = img.resize((224, 224), Image.BILINEAR)
arr_v2 = np.expand_dims(np.array(img_224, dtype=np.float32) / 255.0, axis=0)

# Preprocessing versi 3: resize 256 → crop 224 pakai LANCZOS
img_256b = img.resize((256, 256), Image.LANCZOS)
img_cropb = img_256b.crop((left, top, left + 224, top + 224))
arr_v3 = np.expand_dims(np.array(img_cropb, dtype=np.float32) / 255.0, axis=0)

print("\n=== HASIL PREDIKSI ===")
for label, arr in [("Resize 256→crop 224 (BILINEAR)", arr_v1),
                   ("Langsung resize 224 (BILINEAR)", arr_v2),
                   ("Resize 256→crop 224 (LANCZOS)",  arr_v3)]:
    tensor = tf.constant(arr, dtype=tf.float32)
    output = infer(tensor)
    probs  = list(output.values())[0].numpy()[0]
    idx    = int(np.argmax(probs))
    print(f"\n  [{label}]")
    for i, (cls, p) in enumerate(zip(CLASS_NAMES, probs)):
        marker = "→" if i == idx else " "
        print(f"    {marker} {cls}: {p*100:.2f}%")