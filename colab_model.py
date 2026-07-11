import os
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense

CLASSES = [
    "Corn___Healthy",
    "Corn___Common_rust",
    "Potato___Healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Tomato___Healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
]

DISPLAY_NAMES = {
    "Corn___Healthy": "Corn (Healthy)",
    "Corn___Common_rust": "Corn (Common Rust)",
    "Potato___Healthy": "Potato (Healthy)",
    "Potato___Early_blight": "Potato (Early Blight)",
    "Potato___Late_blight": "Potato (Late Blight)",
    "Tomato___Healthy": "Tomato (Healthy)",
    "Tomato___Early_blight": "Tomato (Early Blight)",
    "Tomato___Late_blight": "Tomato (Late Blight)",
}

SEVERITY_COLORS = {
    "Healthy": "#28a745",
    "Rust": "#fd7e14",
    "Early Blight": "#ffc107",
    "Late Blight": "#dc3545",
    "Unknown": "#6c757d",
}


def get_severity(class_name):
    lower = class_name.lower()
    if "healthy" in lower:
        return "Healthy", SEVERITY_COLORS["Healthy"]
    if "rust" in lower:
        return "Rust", SEVERITY_COLORS["Rust"]
    if "early_blight" in lower:
        return "Early Blight", SEVERITY_COLORS["Early Blight"]
    if "late_blight" in lower:
        return "Late Blight", SEVERITY_COLORS["Late Blight"]
    return "Unknown", SEVERITY_COLORS["Unknown"]


def apply_jet_colormap(gray_image):
    gray = np.clip(gray_image, 0.0, 1.0)
    gray_uint8 = np.uint8(gray * 255)
    r = np.clip(1.5 - np.abs((gray_uint8 / 255.0) * 2 - 1.5), 0.0, 1.0)
    g = np.clip(1.5 - np.abs((gray_uint8 / 255.0) * 2 - 0.5), 0.0, 1.0)
    b = np.clip(1.5 - np.abs((gray_uint8 / 255.0) * 2 + 0.5), 0.0, 1.0)
    return np.uint8(np.stack([r, g, b], axis=-1) * 255)


def overlay_heatmap(original_img, heatmap, alpha=0.55):
    original_np = np.array(original_img, dtype=np.float32) / 255.0
    heatmap_np = np.array(heatmap, dtype=np.float32) / 255.0
    blended = np.clip(original_np * (1 - alpha) + heatmap_np * alpha, 0.0, 1.0)
    return np.uint8(blended * 255)


def generate_leaf_image(class_name, img_size=224):
    """Generate a synthetic leaf image for a single class."""
    # Base background: light beige
    image = Image.new("RGB", (img_size, img_size), (225, 215, 195))
    draw = ImageDraw.Draw(image)

    # Create a leaf shape mask using drawing commands
    if "Corn" in class_name:
        draw.ellipse([40, 30, img_size - 40, img_size - 25], fill=(34, 139, 34))
    elif "Potato" in class_name:
        draw.ellipse([55, 50, img_size - 55, img_size - 40], fill=(34, 139, 34))
    elif "Tomato" in class_name:
        # Lobed tomato-style leaf by overlapping circles
        draw.ellipse([70, 40, 154, 124], fill=(34, 139, 34))
        draw.ellipse([40, 80, 120, 160], fill=(34, 139, 34))
        draw.ellipse([104, 80, 184, 160], fill=(34, 139, 34))
        draw.ellipse([68, 110, 156, 198], fill=(34, 139, 34))
        draw.ellipse([88, 24, 176, 88], fill=(34, 139, 34))

    # Add disease symptoms
    rng = np.random.RandomState(hash(class_name) & 0xFFFFFFFF)

    if "Common_rust" in class_name:
        for _ in range(20):
            x = int(rng.uniform(60, img_size - 60))
            y = int(rng.uniform(50, img_size - 50))
            r = int(rng.uniform(2, 4))
            draw.ellipse([x - r, y - r, x + r, y + r], fill=(190, 80, 20))

    elif "Early_blight" in class_name:
        for _ in range(6):
            x = int(rng.uniform(60, img_size - 60))
            y = int(rng.uniform(60, img_size - 60))
            draw.ellipse([x - 12, y - 12, x + 12, y + 12], fill=(220, 220, 50))
            draw.ellipse([x - 8, y - 8, x + 8, y + 8], fill=(120, 70, 30))
            draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(60, 30, 10))

    elif "Late_blight" in class_name:
        for _ in range(4):
            x = int(rng.uniform(60, img_size - 60))
            y = int(rng.uniform(60, img_size - 60))
            r = int(rng.uniform(14, 22))
            draw.ellipse([x - r, y - r, x + r, y + r], fill=(45, 45, 45))
            draw.ellipse([x - r + 5, y - r + 5, x + r - 5, y + r - 5], fill=(30, 30, 30))

    return image


def preprocess_image(image, img_size=224):
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        image = image.convert("RGB")
    image = image.resize((img_size, img_size))
    arr = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def build_and_save_model(filepath="plant_disease_model.keras"):
    """Build a deterministic CNN and save it to disk."""
    model = Sequential([
        Conv2D(8, (1, 1), activation="relu", input_shape=(224, 224, 3), name="conv_1"),
        MaxPooling2D((4, 4), name="pool_1"),
        Conv2D(16, (3, 3), activation="relu", padding="same", name="conv_2"),
        GlobalAveragePooling2D(name="gap_1"),
        Dense(len(CLASSES), activation="softmax", name="dense_output"),
    ])

    # Layer 1 weights: color detectors
    w1 = np.zeros((1, 1, 3, 8), dtype=np.float32)
    b1 = np.zeros((8,), dtype=np.float32)
    w1[0, 0, :, 0] = [-1.5, 3.0, -1.5]  # green detector
    w1[0, 0, :, 1] = [3.0, 0.5, -3.0]   # rust/orange detector
    w1[0, 0, :, 2] = [-2.0, -2.0, -2.0] # dark/grey detector
    b1[2] = 1.0
    w1[0, 0, :, 3] = [2.0, 2.0, -3.0]   # yellow halo detector
    w1[0, 0, :, 4] = [3.0, -1.5, -1.5]  # red/orange detector
    w1[0, 0, :, 5] = [1.0, 1.0, 1.0]    # bright background detector
    b1[5] = -2.0
    w1[0, 0, :, 6] = [0.5, 2.0, 0.5]
    w1[0, 0, :, 7] = [0.0, 0.0, 0.0]
    model.layers[0].set_weights([w1, b1])

    # Layer 2 weights: spatial blur on color detectors
    w2 = np.zeros((3, 3, 8, 16), dtype=np.float32)
    b2 = np.zeros((16,), dtype=np.float32)
    gaussian = np.array([[1/16, 2/16, 1/16], [2/16, 4/16, 2/16], [1/16, 2/16, 1/16]], dtype=np.float32)
    for k in range(8):
        w2[:, :, k, k] = gaussian
        w2[:, :, k, k + 8] = gaussian * 1.5
    model.layers[2].set_weights([w2, b2])

    # Dense weights: class associations
    wd = np.zeros((16, len(CLASSES)), dtype=np.float32)
    bd = np.zeros((len(CLASSES),), dtype=np.float32)
    wd[0, 0] = 5.0; wd[8, 0] = 5.0; wd[1, 0] = -10.0; wd[2, 0] = -10.0; wd[3, 0] = -10.0
    wd[1, 1] = 12.0; wd[9, 1] = 12.0; wd[0, 1] = 2.0; wd[2, 1] = -5.0; wd[3, 1] = -5.0
    wd[0, 2] = 5.0; wd[8, 2] = 5.0; wd[1, 2] = -10.0; wd[2, 2] = -10.0; wd[3, 2] = -10.0
    wd[3, 3] = 10.0; wd[11, 3] = 10.0; wd[0, 3] = 1.0; wd[1, 3] = -2.0; wd[2, 3] = -5.0
    wd[2, 4] = 12.0; wd[10, 4] = 12.0; wd[0, 4] = 1.0; wd[1, 4] = -5.0; wd[3, 4] = -5.0
    wd[0, 5] = 5.0; wd[8, 5] = 5.0; wd[1, 5] = -10.0; wd[2, 5] = -10.0; wd[3, 5] = -10.0
    wd[3, 6] = 10.0; wd[11, 6] = 10.0; wd[0, 6] = 1.0; wd[1, 6] = -2.0; wd[2, 6] = -5.0
    wd[2, 7] = 12.0; wd[10, 7] = 12.0; wd[0, 7] = 1.0; wd[1, 7] = -5.0; wd[3, 7] = -5.0
    bd[0] = 0.5; bd[1] = 0.5; bd[2] = 0.1; bd[3] = 0.1; bd[4] = 0.1
    bd[5] = -0.2; bd[6] = -0.2; bd[7] = -0.2
    model.layers[4].set_weights([wd, bd])

    model.save(filepath)
    print(f"Saved model to {filepath}")
    return model


def predict_leaf_disease(image_input, model_path="plant_disease_model.keras"):
    if not os.path.exists(model_path):
        build_and_save_model(model_path)

    model = tf.keras.models.load_model(model_path)

    if isinstance(image_input, str):
        original_image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        original_image = image_input.convert("RGB")
    else:
        original_image = Image.fromarray(np.uint8(image_input)).convert("RGB")

    img_array = preprocess_image(original_image)
    probs = model.predict(img_array)[0]
    idx = int(np.argmax(probs))
    class_name = CLASSES[idx]
    display_name = DISPLAY_NAMES.get(class_name, class_name)
    severity, severity_color = get_severity(class_name)
    original_np = np.array(original_image)

    gray = np.mean(original_np.astype(np.float32) / 255.0, axis=2)
    heatmap_color = apply_jet_colormap(gray)
    overlaid_img = overlay_heatmap(original_image, heatmap_color, alpha=0.45)

    return {
        "class_name": class_name,
        "display_name": display_name,
        "confidence": float(probs[idx]),
        "severity": severity,
        "severity_color": severity_color,
        "original_img": original_np,
        "heatmap": heatmap_color,
        "overlaid_img": overlaid_img,
        "probabilities": {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))},
    }


def save_samples(samples_dir="samples"):
    os.makedirs(samples_dir, exist_ok=True)
    for class_name in CLASSES:
        img = generate_leaf_image(class_name)
        filename = class_name.lower().replace("___", "_") + ".png"
        img.save(os.path.join(samples_dir, filename))
    print(f"Saved sample images to {samples_dir}")


def main():
    save_samples("samples")
    build_and_save_model("plant_disease_model.keras")
    print("Model generation complete. Run predict_leaf_disease() to test.")


if __name__ == "__main__":
    main()
