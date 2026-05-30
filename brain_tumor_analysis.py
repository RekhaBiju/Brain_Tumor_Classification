import os
import io
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# ── Labels ─────────────────────────────────────────────────────────────────────
# Order matches ImageFolder alphabetical sort used during training:
# glioma=0, meningioma=1, notumor=2, pituitary=3
CLASS_LABELS = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# ── Models registry ────────────────────────────────────────────────────────────
MODELS = {
    "resnet50_h5": {
        "label":    "ResNet50 (H5)",
        "desc":     "HDF5 model",
        "file":     "models/brain_tumor_classification_model.h5",
        "type":     "keras",
        "accuracy": 70.87,
        "loaded":   None,
    },
    "pytorch": {
        "label":    "PyTorch ViT",
        "desc":     "Vision Transformer",
        "file":     "models/best_model_1.pth",   # ← change to best_model.pth if needed
        "type":     "pytorch",
        "accuracy": 77.35,
        "loaded":   None,
    },
    "hybrid": {
        "label":    "Hybrid ResNet50+ViT",
        "desc":     "CNN + Transformer",
        "file":     "models/hybrid 1.h5",
        "type":     "hybrid",
        "accuracy": 97.57,
        "loaded":   None,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# PYTORCH ViT loader
# Architecture matches training notebook exactly:
#   vit-pytorch ViT(image_size=224, patch_size=32, dim=1024,
#                   depth=6, heads=16, mlp_dim=2048,
#                   dropout=0.1, emb_dropout=0.1)
# Preprocessing matches training transforms:
#   Resize(224) → ToTensor() → Normalize(ImageNet mean/std)
# ══════════════════════════════════════════════════════════════════════════════

import torch
import torch.nn as nn

try:
    from vit_pytorch import ViT as _ViT
    VIT_AVAILABLE = True
except ImportError:
    VIT_AVAILABLE = False
    print("WARNING: vit-pytorch not installed. Run: pip install vit-pytorch")


class TumorClassifierViT(nn.Module):
    """Matches training notebook definition exactly."""
    def __init__(self, num_classes=4):
        super().__init__()
        if not VIT_AVAILABLE:
            raise ImportError("vit-pytorch is required. Run: pip install vit-pytorch")
        self.vit = _ViT(
            image_size  = 224,
            patch_size  = 32,
            num_classes = num_classes,
            dim         = 1024,
            depth       = 6,
            heads       = 16,
            mlp_dim     = 2048,
            dropout     = 0.1,
            emb_dropout = 0.1,
        )

    def forward(self, x):
        return self.vit(x)


def load_pytorch_model(path):
    if not VIT_AVAILABLE:
        raise RuntimeError("vit-pytorch not installed. Run: pip install vit-pytorch")

    model = TumorClassifierViT(num_classes=4)

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    # Unwrap checkpoint dict if saved with extra keys (epoch, optimizer, etc.)
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        else:
            # Raw state_dict saved directly
            state = checkpoint
    else:
        state = checkpoint

    # strict=True: architecture must match exactly (it does — same vit-pytorch config)
    missing, unexpected = model.load_state_dict(state, strict=True)
    # If strict=True raises an error, fall back to strict=False and report
    model.eval()

    print(f"[ViT] Loaded successfully from {path}")
    if missing:
        print(f"[ViT] Missing keys  ({len(missing)}): {missing[:3]}")
    if unexpected:
        print(f"[ViT] Unexpected keys ({len(unexpected)}): {unexpected[:3]}")

    return model


# ══════════════════════════════════════════════════════════════════════════════
# KERAS — plain ResNet50 .h5
# ══════════════════════════════════════════════════════════════════════════════

def load_keras_model(path):
    import tensorflow as tf
    model = tf.keras.models.load_model(path)
    print(f"[Keras] Loaded from {path}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# HYBRID — ResNet50 + ViT Transformer (Keras functional API)
# Architecture mirrors training notebook's ResNet50ViT() exactly:
#   - ResNet50(weights="imagenet") backbone
#   - Conv2D patch embedding → BatchNorm → Reshape
#   - Positional Embedding (Embedding layer)
#   - ClassToken prepend
#   - 6× transformer_encoder blocks:
#       LN → MHA → Add → Dropout | LN → MLP(gelu) → Add → Dropout
#   - Final LN → CLS token → Dense(4) logits
# Preprocessing: raw uint8 pixel values cast to float16 (matches predict.py)
# ══════════════════════════════════════════════════════════════════════════════

def load_hybrid_model(path):
    import tensorflow as tf
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras import layers, Model

    # ── ClassToken layer — matches training predict.py exactly ────────────────
    class ClassToken(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def build(self, input_shape):
            w_init = tf.random_normal_initializer()
            self.w = self.add_weight(
                shape=(1, 1, input_shape[-1]),
                initializer=w_init,
                trainable=True,
            )

        def call(self, inputs):
            batch_size = tf.shape(inputs)[0]
            hidden_dim = self.w.shape[-1]
            cls = tf.broadcast_to(self.w, [batch_size, 1, hidden_dim])
            return tf.cast(cls, dtype=inputs.dtype)

    # ── Hyperparameters — must match training exactly ──────────────────────────
    cf = {
        "num_layers":   6,
        "hidden_dim":   64,
        "mlp_dim":      2048,
        "num_heads":    8,
        "dropout_rate": 0.2,
        "image_size":   128,
        "patch_size":   32,
        "num_patches":  16,   # (128 // 32) ** 2 = 16
        "num_channels": 3,
        "num_classes":  4,
    }

    # ── MLP block — matches training mlp() ────────────────────────────────────
    def mlp_block(x):
        x = layers.Dense(cf["mlp_dim"], activation="gelu")(x)
        x = layers.Dropout(cf["dropout_rate"])(x)
        x = layers.Dense(cf["hidden_dim"])(x)
        x = layers.Dropout(cf["dropout_rate"])(x)
        return x

    # ── Transformer encoder — matches training transformer_encoder() ───────────
    # Exact structure: LN→MHA→Add→Dropout | LN→MLP→Add→Dropout
    def transformer_encoder(x):
        skip_1 = x
        x = layers.LayerNormalization()(x)
        x = layers.MultiHeadAttention(
            num_heads=cf["num_heads"],
            key_dim=cf["hidden_dim"],
        )(x, x)
        x = layers.Add()([x, skip_1])
        x = layers.Dropout(cf["dropout_rate"])(x)

        skip_2 = x
        x = layers.LayerNormalization()(x)
        x = mlp_block(x)
        x = layers.Add()([x, skip_2])
        x = layers.Dropout(cf["dropout_rate"])(x)
        return x

    # ── Build model — mirrors training ResNet50ViT() exactly ──────────────────
    inputs = layers.Input(
        shape=(cf["image_size"], cf["image_size"], cf["num_channels"])
    )

    # ResNet50 backbone — MUST use weights="imagenet" to match training
    base = ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)
    x = base.output  # → (None, 4, 4, 2048)

    # Patch embedding
    x = layers.Conv2D(
        cf["hidden_dim"], kernel_size=cf["patch_size"], padding="same"
    )(x)
    x = layers.BatchNormalization()(x)
    _, h, w, f = x.shape
    x = layers.Reshape((h * w, f))(x)  # → (None, 16, 64)

    # Positional embedding
    positions = tf.range(start=0, limit=cf["num_patches"], delta=1)
    pos_embed = layers.Embedding(
        input_dim=cf["num_patches"], output_dim=cf["hidden_dim"]
    )(positions)
    x = layers.Add()([x, pos_embed])

    # Prepend class token
    token = ClassToken()(x)
    x = layers.Concatenate(axis=1)([token, x])  # → (None, 17, 64)

    # Transformer blocks
    for _ in range(cf["num_layers"]):
        x = transformer_encoder(x)

    # Classification head
    x = layers.LayerNormalization()(x)
    x = x[:, 0, :]                          # extract CLS token
    outputs = layers.Dense(cf["num_classes"])(x)

    model = Model(inputs, outputs)
    model.load_weights(path)

    _verify_hybrid_weights(model)
    return model


def _verify_hybrid_weights(model):
    print("\n[Hybrid] Weight load verification:")
    zeros, loaded = [], []
    for layer in model.layers:
        if layer.weights:
            nonzero = any(np.any(w.numpy() != 0) for w in layer.weights)
            (loaded if nonzero else zeros).append(layer.name)
    print(f"  ✔ Loaded: {len(loaded)} layers")
    if zeros:
        print(f"  ✘ Zero weights: {zeros}")
    print()


# ── Model cache ────────────────────────────────────────────────────────────────

def get_model(key):
    entry = MODELS[key]
    if entry["loaded"] is None:
        path = entry["file"]
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found: {path}\n"
                f"Make sure the file exists in the models/ directory."
            )
        if entry["type"] == "keras":
            entry["loaded"] = load_keras_model(path)
        elif entry["type"] == "hybrid":
            entry["loaded"] = load_hybrid_model(path)
        else:
            entry["loaded"] = load_pytorch_model(path)
    return entry["loaded"], entry["type"]


# ── Preprocessing ──────────────────────────────────────────────────────────────

def preprocess_image(file_bytes, model_type):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    if model_type == "keras":
        # ResNet50 h5: 150×150, normalized to [0,1]
        img = img.resize((150, 150))
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr[np.newaxis, ...]

    if model_type == "hybrid":
        # Hybrid: 128×128, raw float16 pixel values (matches training predict.py)
        img = img.resize((128, 128))
        arr = np.array(img, dtype=np.float16)
        return np.expand_dims(arr, axis=0)

    # pytorch ViT: matches training transforms exactly
    #   transforms.Resize((224,224)) → ToTensor() → Normalize(ImageNet)
    import torchvision.transforms as T
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),                                        # → [0,1], shape [3,H,W]
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        ),
    ])
    return transform(img).unsqueeze(0)                       # → [1, 3, 224, 224]


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    models_info = {
        k: {"label": v["label"], "desc": v["desc"], "accuracy": v.get("accuracy")}
        for k, v in MODELS.items()
    }
    return render_template("index.html", models=models_info, classes=CLASS_LABELS)


@app.route("/predict", methods=["POST"])
def predict():
    model_key = request.form.get("model_key")
    file      = request.files.get("image")

    if not file:
        return jsonify({"error": "No image uploaded"}), 400
    if model_key not in MODELS:
        return jsonify({"error": f"Unknown model key: {model_key}"}), 400

    try:
        model, model_type = get_model(model_key)
        tensor = preprocess_image(file.read(), model_type)

        if model_type == "keras":
            # ResNet50 h5 ends with softmax → already probabilities
            probs = model.predict(tensor, verbose=0)[0]
            probs = probs / probs.sum()

        elif model_type == "hybrid":
            import tensorflow as tf
            # Hybrid ends with Dense(4) no activation → raw logits
            logits = model(
                tf.convert_to_tensor(tensor, dtype=tf.float16),
                training=False
            ).numpy()[0]
            probs = softmax(logits.astype(np.float32))

        else:  # pytorch ViT
            with torch.no_grad():
                logits = model(tensor)
                probs  = torch.softmax(logits, dim=1).numpy()[0]

        idx = int(np.argmax(probs))
        return jsonify({
            "prediction": CLASS_LABELS[idx],
            "confidence": round(float(probs[idx]) * 100, 2),
            "all_scores": {
                CLASS_LABELS[i]: round(float(probs[i]) * 100, 2)
                for i in range(len(CLASS_LABELS))
            },
            "model_used": MODELS[model_key]["label"],
        })

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# ── Startup ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Brain Tumor Classification API")
    print("=" * 60)
    print("\nDependencies: pip install vit-pytorch torch torchvision")
    print("              pip install flask pillow numpy tensorflow")
    print("\nModel files:")
    for key, info in MODELS.items():
        exists = "✔ found" if os.path.exists(info["file"]) else "✘ MISSING"
        print(f"  [{exists}] {info['file']}")
    print()
    app.run(debug=True, host="0.0.0.0", port=5000)