# train_model.py
# ------------------------------
# Downloads the ImageTBAD dataset from Kaggle, extracts it, trains an EfficientNetB0-based
# classifier to detect aortic dissection, and saves the resulting model as dissectai_model.h5.
#
# Usage:
#   1. Ensure you have KAGGLE_USERNAME and KAGGLE_KEY set in your environment
#      (so that the kaggle CLI can authenticate).
#   2. Run: python train_model.py
#   3. After it finishes, dissectai_model.h5 will be in the same folder.
# ------------------------------

import os
import shutil
import subprocess
from glob import glob
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pathlib

# --------------------------------
# 0. Configuration
# --------------------------------
KAGGLE_DATASET = "xiaoweixumedicalai/imagetbad"
WORK_DIR      = os.getcwd()
DATA_DIR      = os.path.join(WORK_DIR, "data")
EXTRACT_DIR   = os.path.join(WORK_DIR, "imageTBAD_extracted")
MODEL_PATH    = os.path.join(WORK_DIR, "dissectai_model.h5")

IMG_SIZE      = (224, 224)
BATCH_SIZE    = 32
EPOCHS        = 15
AUTOTUNE      = tf.data.AUTOTUNE

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(EXTRACT_DIR, exist_ok=True)

# --------------------------------
# 1. Download & unzip from Kaggle
# --------------------------------
def download_kaggle_dataset(dataset, download_path):
    """
    Uses the Kaggle CLI to download and unzip a dataset.
    Requires KAGGLE_USERNAME and KAGGLE_KEY in environment variables.
    """
    print(f"⏬ Downloading Kaggle dataset '{dataset}' → {download_path}")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset, "-p", download_path, "--unzip"],
        check=True,
    )
    print("✅ Download complete.")


# --------------------------------
# 2. Extract multi-part zip
# --------------------------------
def extract_multipart_zip(input_dir, output_dir):
    """
    Finds all parts named imageTBAD.z01…zNN in input_dir, copies them here, and uses 7z to extract.
    """
    parts = sorted([f for f in os.listdir(input_dir) if f.startswith("imageTBAD.z")])
    if not parts:
        raise FileNotFoundError("No multi-part zip files found under 'data/'.")

    # Copy each .zNN to working dir
    for part in parts:
        src = os.path.join(input_dir, part)
        dst = os.path.join(WORK_DIR, part)
        if not os.path.exists(dst):
            shutil.copy(src, dst)

    # Extract into EXTRACT_DIR (only once)
    if not os.listdir(output_dir):
        first_part = os.path.join(WORK_DIR, parts[0])
        print(f"📦 Extracting archive from {first_part} → {output_dir}")
        subprocess.run(["7z", "x", first_part, f"-o{output_dir}"], check=True)
        print("✅ Extraction complete.")
    else:
        print("✅ Extraction directory already contains files.")


# --------------------------------
# 3. Locate Train & Val folders
# --------------------------------
def find_folder(name, root):
    """
    Recursively search for a folder named `name` (case-insensitive) under `root`.
    """
    for dirpath, dirnames, _ in os.walk(root):
        if os.path.basename(dirpath).lower() == name.lower():
            return dirpath
    return None


# --------------------------------
# 4. Build tf.data datasets
# --------------------------------
def preprocess_image(path, label):
    """
    Reads PNG image from path, resizes to IMG_SIZE, normalizes to [0,1].
    """
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = image / 255.0
    return image, label


def get_dataset(directory):
    """
    Creates a tf.data.Dataset of (image_tensor, label) from a directory where
    subfolders are class names (e.g. '0' and '1').
    """
    data_root = pathlib.Path(directory)
    all_image_paths = list(data_root.glob("*/*"))
    all_image_paths = [str(path) for path in all_image_paths]
    label_names = sorted(item.name for item in data_root.glob("*/") if item.is_dir())
    label_to_index = {name: i for i, name in enumerate(label_names)}
    all_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]

    path_ds  = tf.data.Dataset.from_tensor_slices(all_image_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(all_labels)
    ds = tf.data.Dataset.zip((path_ds, label_ds))
    ds = ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
    return ds


# --------------------------------
# 5. Build & compile model
# --------------------------------
def build_model():
    """
    Builds an EfficientNetB0-based binary classifier using mixed precision.
    """
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    base = EfficientNetB0(include_top=False, input_shape=IMG_SIZE + (3,), weights="imagenet")
    base.trainable = False

    model = models.Sequential(
        [
            base,
            layers.GlobalAveragePooling2D(),
            layers.Dense(1, activation="sigmoid", dtype="float32"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# --------------------------------
# 6. Train & fine-tune
# --------------------------------
def train_and_finetune(train_ds, val_ds):
    """
    Trains the model in two stages:
      1) Freeze backbone, train top layers (EPOCHS)
      2) Unfreeze top half of backbone, fine-tune with lower LR (EPOCHS//2)
    """
    model = build_model()
    early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    checkpoint = ModelCheckpoint("best_model.h5", monitor="val_accuracy", save_best_only=True)

    print("▶️ Stage 1: Training with frozen backbone…")
    h1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[early_stop, checkpoint],
        verbose=2,
    )

    # Unfreeze top half of backbone
    model.get_layer(index=0).trainable = True
    total_layers = len(model.layers[0].layers)
    freeze_until = total_layers // 2
    for layer in model.layers[0].layers[:freeze_until]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    print("▶️ Stage 2: Fine-tuning with unfrozen top layers…")
    h2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS // 2,
        callbacks=[early_stop, checkpoint],
        verbose=2,
    )

    model.load_weights("best_model.h5")
    loss, acc = model.evaluate(val_ds, verbose=0)
    print(f"✅ Final validation accuracy: {acc * 100:.2f}%")

    model.save(MODEL_PATH)
    print(f"✅ Saved trained model to {MODEL_PATH}")

    return h1, h2


# --------------------------------
# 7. Plot training history (optional)
# --------------------------------
def plot_history(h1, h2):
    loss = h1.history["loss"] + h2.history["loss"]
    val_loss = h1.history["val_loss"] + h2.history["val_loss"]
    acc = h1.history["accuracy"] + h2.history["accuracy"]
    val_acc = h1.history["val_accuracy"] + h2.history["val_accuracy"]

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(loss, label="train loss")
    plt.plot(val_loss, label="val loss")
    plt.legend(); plt.title("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(acc, label="train acc")
    plt.plot(val_acc, label="val acc")
    plt.legend(); plt.title("Accuracy")
    plt.show()


# --------------------------------
# 8. Main pipeline
# --------------------------------
if __name__ == "__main__":
    # 1. Download the Kaggle dataset
    download_kaggle_dataset(KAGGLE_DATASET, DATA_DIR)

    # 2. Extract the multi-part archive
    extract_multipart_zip(DATA_DIR, EXTRACT_DIR)

    # 3. Locate Train/Val folders
    train_folder = find_folder("Train", EXTRACT_DIR)
    val_folder   = find_folder("Val",   EXTRACT_DIR)
    if not train_folder or not val_folder:
        raise FileNotFoundError("Train/Val folders not found after extraction.")
    print("▶️ Using train folder:", train_folder)
    print("▶️ Using val folder:  ", val_folder)

    # 4. Build tf.data pipelines
    train_ds = get_dataset(train_folder)
    val_ds   = get_dataset(val_folder)
    train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    val_ds   = val_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    # 5–7. Train & fine-tune, then save model
    h1, h2 = train_and_finetune(train_ds, val_ds)

    # 8. (Optional) Plot training curves if running locally
    try:
        plot_history(h1, h2)
    except Exception:
        pass
