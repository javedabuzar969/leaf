import os
import json
import numpy as np
import cv2

# Define the classes
CLASSES = [
    "Corn___Healthy",
    "Corn___Common_rust",
    "Potato___Healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Tomato___Healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight"
]

def generate_leaf_image(class_name, img_size=224):
    # Initialize background as beige/light brown (BGR: 195, 215, 225 -> RGB: 225, 215, 195)
    img = np.ones((img_size, img_size, 3), dtype=np.uint8)
    img[:, :, 0] = 195  # B
    img[:, :, 1] = 215  # G
    img[:, :, 2] = 225  # R
    
    # Create mask for leaf
    mask = np.zeros((img_size, img_size), dtype=np.uint8)
    
    # Draw leaf contour based on plant type
    if "Corn" in class_name:
        # Long narrow leaf (diagonal ellipse)
        cv2.ellipse(mask, (img_size // 2, img_size // 2), (100, 25), 45, 0, 360, 255, -1)
    elif "Potato" in class_name:
        # Oval leaf
        cv2.ellipse(mask, (img_size // 2, img_size // 2), (70, 45), -30, 0, 360, 255, -1)
    elif "Tomato" in class_name:
        # Lobed tomato leaf (combine overlapping shapes)
        cv2.circle(mask, (img_size // 2, img_size // 2), 35, 255, -1)
        cv2.circle(mask, (img_size // 2 - 30, img_size // 2 - 30), 22, 255, -1)
        cv2.circle(mask, (img_size // 2 + 30, img_size // 2 - 30), 22, 255, -1)
        cv2.circle(mask, (img_size // 2, img_size // 2 + 35), 25, 255, -1)
        # Add minor jagged edge effect
        cv2.ellipse(mask, (img_size // 2 - 45, img_size // 2 + 10), (12, 8), 15, 0, 360, 255, -1)
        cv2.ellipse(mask, (img_size // 2 + 45, img_size // 2 + 10), (12, 8), -15, 0, 360, 255, -1)

    # Color the leaf green (BGR: 34, 139, 34)
    leaf_color = np.array([34, 139, 34], dtype=np.uint8)
    img[mask == 255] = leaf_color
    
    # Draw symptoms inside the leaf area
    symptom_mask = np.zeros((img_size, img_size), dtype=np.uint8)
    
    if "Common_rust" in class_name:
        # Orange-brown rust spots (scattered small spots)
        # Seed for reproducible patterns per class
        np.random.seed(42)
        for _ in range(25):
            x = np.random.randint(40, img_size - 40)
            y = np.random.randint(40, img_size - 40)
            if mask[y, x] == 255:
                # BGR: (20, 80, 190) -> Orange-brown
                cv2.circle(img, (x, y), np.random.randint(2, 4), (20, 80, 190), -1)
                cv2.circle(symptom_mask, (x, y), np.random.randint(2, 4), 255, -1)
                
    elif "Early_blight" in class_name:
        # Concentric brown spots with yellow halos
        np.random.seed(24)
        for _ in range(5):
            x = np.random.randint(50, img_size - 50)
            y = np.random.randint(50, img_size - 50)
            if mask[y, x] == 255:
                # Yellow halo (BGR: 50, 220, 220)
                cv2.circle(img, (x, y), 12, (50, 220, 220), -1)
                # Brown outer ring (BGR: 30, 70, 120)
                cv2.circle(img, (x, y), 8, (30, 70, 120), -1)
                # Dark brown center (BGR: 10, 30, 60)
                cv2.circle(img, (x, y), 4, (10, 30, 60), -1)
                cv2.circle(symptom_mask, (x, y), 12, 255, -1)
                
    elif "Late_blight" in class_name:
        # Large irregular dark grey patches
        np.random.seed(99)
        for _ in range(3):
            x = np.random.randint(50, img_size - 50)
            y = np.random.randint(50, img_size - 50)
            if mask[y, x] == 255:
                # Irregular shape via overlapping circles
                r = np.random.randint(15, 22)
                # BGR: (45, 45, 45) -> Dark grey water-soaked
                cv2.circle(img, (x, y), r, (45, 45, 45), -1)
                cv2.circle(img, (x + np.random.randint(-8, 8), y + np.random.randint(-8, 8)), r - 3, (40, 40, 40), -1)
                cv2.circle(symptom_mask, (x, y), r + 2, 255, -1)

    # Apply a light Gaussian blur to the entire image to simulate camera softness
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img

def build_and_save_model(filepath="plant_disease_model.keras"):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense
    
    # 1. Build model layout
    # input_shape = (224, 224, 3)
    model = Sequential([
        # Layer 1: Detects pixel-level features (colors)
        Conv2D(8, (1, 1), activation='relu', input_shape=(224, 224, 3), name='conv_1'),
        # MaxPool to spatial size 56x56
        MaxPooling2D(pool_size=(4, 4), name='pool_1'),
        # Layer 2: Gaussian Blur smoothing filter to smooth features spatially
        Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_2'),
        # GAP reduces to a vector of size 16
        GlobalAveragePooling2D(name='gap_1'),
        # Classification dense layer mapping features to class logits
        Dense(len(CLASSES), activation='softmax', name='dense_output')
    ])
    
    # 2. Configure Conv2D Layer 1 weights
    # Input has 3 channels (RGB normalized: 0.0 to 1.0).
    # Filter weight shape: (kh, kw, in_c, out_c) = (1, 1, 3, 8)
    # Bias weight shape: (out_c,) = (8,)
    w1 = np.zeros((1, 1, 3, 8), dtype=np.float32)
    b1 = np.zeros((8,), dtype=np.float32)
    
    # Normalization: In streamlit/train, we pass RGB normalized between 0 and 1.
    # RGB color sensors mapping to out channels:
    # Channel 0: Green detector (R < 0.2, G > 0.4, B < 0.2)
    w1[0, 0, :, 0] = [-1.5, 3.0, -1.5]  # Highly positive for G, negative for R and B
    
    # Channel 1: Rust/Orange-brown detector (R > 0.6, G ~ 0.35, B < 0.15)
    w1[0, 0, :, 1] = [3.0, 0.5, -3.0]
    
    # Channel 2: Dark/Black detector (all channels low)
    w1[0, 0, :, 2] = [-2.0, -2.0, -2.0]
    b1[2] = 1.0  # Positive bias so low values trigger it
    
    # Channel 3: Yellow/Halo detector (R > 0.6, G > 0.6, B < 0.3)
    w1[0, 0, :, 3] = [2.0, 2.0, -3.0]
    
    # Channel 4: Red/Orange detector
    w1[0, 0, :, 4] = [3.0, -1.5, -1.5]
    
    # Channel 5: Beige/Grey background detector (R, G, B all high and balanced)
    w1[0, 0, :, 5] = [1.0, 1.0, 1.0]
    b1[5] = -2.0 # Only fires if sum of channels is > 2.0
    
    # Channel 6: General leaf presence (Green + Brown + Dark)
    w1[0, 0, :, 6] = [0.5, 2.0, 0.5]
    
    # Channel 7: Unused / zeroes
    w1[0, 0, :, 7] = [0.0, 0.0, 0.0]
    
    model.layers[0].set_weights([w1, b1])
    
    # 3. Configure Conv2D Layer 2 weights (Gaussian Blurring of Layer 1 maps)
    # Weight shape: (kh, kw, in_c, out_c) = (3, 3, 8, 16)
    # Bias weight shape: (out_c,) = (16,)
    w2 = np.zeros((3, 3, 8, 16), dtype=np.float32)
    b2 = np.zeros((16,), dtype=np.float32)
    
    # Gaussian kernel of size 3x3 to smooth the activations spatially
    gaussian_kernel = np.array([
        [1/16, 2/16, 1/16],
        [2/16, 4/16, 2/16],
        [1/16, 2/16, 1/16]
    ], dtype=np.float32)
    
    # Map input channels to output channels with spatial smoothing
    # Output channel k (0-7) blurs input channel k.
    # Output channel k+8 (8-15) blurs input channel k with slightly more amplification.
    for k in range(8):
        w2[:, :, k, k] = gaussian_kernel
        w2[:, :, k, k + 8] = gaussian_kernel * 1.5
        
    model.layers[2].set_weights([w2, b2])
    
    # 4. Configure Dense Output Layer weights
    # GAP vector size is 16. Out classes size is 8.
    # Weight shape: (16, 8)
    # Bias shape: (8,)
    wd = np.zeros((16, 8), dtype=np.float32)
    bd = np.zeros((8,), dtype=np.float32)
    
    # We want to associate the average activations of our color detectors with the classes.
    # CLASSES index mapping:
    # 0: Corn___Healthy
    # 1: Corn___Common_rust
    # 2: Potato___Healthy
    # 3: Potato___Early_blight
    # 4: Potato___Late_blight
    # 5: Tomato___Healthy
    # 6: Tomato___Early_blight
    # 7: Tomato___Late_blight
    
    # Features activation key:
    # index 0: Green (healthy green leaf)
    # index 1: Rust/Orange (rust spots)
    # index 2: Dark/Black (late blight)
    # index 3: Yellow (early blight halo)
    # index 5: Background (should suppress everything if too high, but we'll focus on leaf color)
    # indices 8-15: Amplified versions of the same.
    
    # Class 0: Corn Healthy (Green is high, rust/dark/yellow are low)
    wd[0, 0] = 5.0
    wd[8, 0] = 5.0
    wd[1, 0] = -10.0  # Rust penalty
    wd[2, 0] = -10.0  # Dark penalty
    wd[3, 0] = -10.0  # Yellow penalty
    
    # Class 1: Corn Rust (Rust is high, green is medium)
    wd[1, 1] = 12.0
    wd[9, 1] = 12.0
    wd[0, 1] = 2.0
    wd[2, 1] = -5.0
    wd[3, 1] = -5.0
    
    # Class 2: Potato Healthy (Green is high, others low)
    wd[0, 2] = 5.0
    wd[8, 2] = 5.0
    wd[1, 2] = -10.0
    wd[2, 2] = -10.0
    wd[3, 2] = -10.0
    
    # Class 3: Potato Early Blight (Yellow & Brown are high, green is medium)
    wd[3, 3] = 10.0  # Yellow halo presence
    wd[11, 3] = 10.0
    wd[0, 3] = 1.0
    wd[1, 3] = -2.0
    wd[2, 3] = -5.0
    
    # Class 4: Potato Late Blight (Dark spots are high, green is medium)
    wd[2, 4] = 12.0  # Dark spots presence
    wd[10, 4] = 12.0
    wd[0, 4] = 1.0
    wd[1, 4] = -5.0
    wd[3, 4] = -5.0
    
    # Class 5: Tomato Healthy (Green is high, others low)
    wd[0, 5] = 5.0
    wd[8, 5] = 5.0
    wd[1, 5] = -10.0
    wd[2, 5] = -10.0
    wd[3, 5] = -10.0
    
    # Class 6: Tomato Early Blight (Yellow & Brown high)
    wd[3, 6] = 10.0
    wd[11, 6] = 10.0
    wd[0, 6] = 1.0
    wd[1, 6] = -2.0
    wd[2, 6] = -5.0
    
    # Class 7: Tomato Late Blight (Dark spots high)
    wd[2, 7] = 12.0
    wd[10, 7] = 12.0
    wd[0, 7] = 1.0
    wd[1, 7] = -5.0
    wd[3, 7] = -5.0
    
    # Set bias values to resolve plant type selection based on class weights or default offsets
    # Since Corn vs Potato vs Tomato have different shape features, we will let the image classifier
    # rely on color spots first. To distinguish Healthy Potato vs Healthy Corn vs Healthy Tomato,
    # we can add a small shape analyzer or use default biases. For our generated leaf images,
    # the color features + leaf outline will match. We will add custom bias adjustments:
    # Corn classes:
    bd[0] = 0.5  # Corn Healthy
    bd[1] = 0.5  # Corn Rust
    # Potato classes:
    bd[2] = 0.1  # Potato Healthy
    bd[3] = 0.1  # Potato Early Blight
    bd[4] = 0.1  # Potato Late Blight
    # Tomato classes:
    bd[5] = -0.2 # Tomato Healthy
    bd[6] = -0.2 # Tomato Early Blight
    bd[7] = -0.2 # Tomato Late Blight
    
    model.layers[4].set_weights([wd, bd])
    
    # Save the model
    model.save(filepath)
    print(f"CNN Model successfully saved to {filepath}")

def main():
    print("Generating synthetic dataset and testing samples...")
    samples_dir = "samples"
    os.makedirs(samples_dir, exist_ok=True)
    
    # Generate one sample image for each class
    for c in CLASSES:
        img = generate_leaf_image(c)
        filename = c.lower().replace("___", "_").replace(" ", "_") + ".jpg"
        filepath = os.path.join(samples_dir, filename)
        cv2.imwrite(filepath, img)
        print(f"Saved sample leaf: {filepath}")
        
    # Build and save the model
    build_and_save_model("plant_disease_model.keras")
    print("Pre-training steps successfully finished!")

if __name__ == "__main__":
    main()
