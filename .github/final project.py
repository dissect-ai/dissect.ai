#!/usr/bin/env python
# -*- coding: utf-8 -*-
# End-to-end pipeline: Kaggle download -> DICOM to PNG -> KMeans pseudo-labeling -> CNN (EfficientNetB0) training -> Streamlit interface.
# (Based on techniques from pydicom conversion:contentReference[oaicite:0]{index=0}, KMeans pseudo-labeling:contentReference[oaicite:1]{index=1}, and EfficientNet transfer learning:contentReference[oaicite:2]{index=2}.)

import os
import sys
import logging
import subprocess
import zipfile
import shutil
import numpy as np

# Set up logging for progress and errors
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 1. Install required packages
try:
    # Ensure environment has necessary libraries
    subprocess.run([sys.executable, "-m", "pip", "install", "kaggle", "pydicom", "Pillow", "numpy", "scikit-learn", "tensorflow", "streamlit"], check=True, stdout=subprocess.DEVNULL)
    logger.info("Required packages installed.")
except Exception as e:
    logger.error(f"Package installation failed: {e}")
    sys.exit(1)

# Import libraries after installation
try:
    import pydicom
    from PIL import Image
    from sklearn.cluster import KMeans
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image_dataset_from_directory
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    import pandas as pd
    import streamlit as st
except Exception as e:
    logger.error(f"Import error: {e}")
    sys.exit(1)

# Ensure TensorFlow does not allocate all GPU memory (for Colab compatibility)
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    logger.info("Configured TensorFlow GPU memory growth.")
except Exception as e:
    logger.warning(f"Could not configure GPU: {e}")

def download_dataset():
    """
    Download and unzip the ImageTBAD dataset from Kaggle.
    Requires Kaggle API credentials (username/key) to be set.
    """
    try:
        logger.info("Downloading ImageTBAD dataset from Kaggle...")
        # Ensure Kaggle API credentials are set
        kaggle_username = os.environ.get("KAGGLE_USERNAME")
        kaggle_key = os.environ.get("KAGGLE_KEY")
        kaggle_json = "kaggle.json"
        if kaggle_username and kaggle_key:
            pass  # Credentials via env variables
        elif os.path.exists(kaggle_json):
            # Copy kaggle.json to ~/.kaggle/kaggle.json for Kaggle CLI
            kaggle_dir = os.path.expanduser("~/.kaggle")
            os.makedirs(kaggle_dir, exist_ok=True)
            shutil.copy(kaggle_json, os.path.join(kaggle_dir, "kaggle.json"))
            os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)
        else:
            raise FileNotFoundError("Kaggle API credentials not found. Place kaggle.json in working directory or set KAGGLE_USERNAME/KAGGLE_KEY.")

        # Download dataset using Kaggle CLI
        dataset_name = "xiaoweixumedicalai/imagetbad"
        download_path = "./data"
        os.makedirs(download_path, exist_ok=True)
        subprocess.run(["kaggle", "datasets", "download", "-d", dataset_name, "-p", download_path], check=True)
        logger.info("Download completed.")

        # Unzip all downloaded zip files
        for fname in os.listdir(download_path):
            if fname.endswith(".zip"):
                file_path = os.path.join(download_path, fname)
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(download_path)
                logger.info(f"Extracted {fname}")
        logger.info("All archives extracted.")
    except Exception as e:
        logger.error(f"Failed to download or extract dataset: {e}")
        sys.exit(1)

def convert_dicom_to_png(source_dir, target_dir):
    """
    Converts all DICOM files in source_dir to PNG format and saves in target_dir.
    Uses pydicom to read and Pillow to save images.
    """
    try:
        os.makedirs(target_dir, exist_ok=True)
        dicom_files = []
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if file.lower().endswith(".dcm"):
                    dicom_files.append(os.path.join(root, file))
        logger.info(f"Found {len(dicom_files)} DICOM files. Converting to PNG...")

        for dcm_path in dicom_files:
            try:
                ds = pydicom.dcmread(dcm_path)
                img_array = ds.pixel_array.astype(float)
                # Normalize to 0-255 and convert to uint8:contentReference[oaicite:3]{index=3}
                img_scaled = (np.maximum(img_array, 0) / img_array.max()) * 255.0
                img_scaled = np.uint8(img_scaled)
                img = Image.fromarray(img_scaled).convert("RGB")
                # Save PNG, preserving folder structure
                rel_path = os.path.relpath(dcm_path, source_dir)
                png_path = os.path.join(target_dir, os.path.splitext(rel_path)[0] + ".png")
                os.makedirs(os.path.dirname(png_path), exist_ok=True)
                img.save(png_path)
            except Exception as inner_e:
                logger.warning(f"Failed to convert {dcm_path}: {inner_e}")
        logger.info("DICOM to PNG conversion completed.")
    except Exception as e:
        logger.error(f"Conversion error: {e}")
        sys.exit(1)

def pseudo_label_images(image_dir, n_clusters=2):
    """
    Generate pseudo-labels for images using KMeans clustering:contentReference[oaicite:4]{index=4}.
    Returns list of filepaths and corresponding cluster labels.
    """
    try:
        # Gather image file paths
        image_paths = []
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    image_paths.append(os.path.join(root, file))
        logger.info(f"Clustering {len(image_paths)} images into {n_clusters} clusters...")

        # Load images into feature vectors (resize to fixed small size for clustering)
        data = []
        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((64, 64))  # small size for faster clustering
            arr = np.array(img).flatten()
            data.append(arr)
        data = np.array(data)

        # KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(data)
        logger.info("KMeans clustering completed. Assigning pseudo-labels.")
        return image_paths, labels
    except Exception as e:
        logger.error(f"KMeans clustering failed: {e}")
        sys.exit(1)

def organize_dataset(image_paths, labels, train_dir, val_dir, test_size=0.2):
    """
    Splits the dataset into train/validation based on labels and copies images to respective class folders.
    """
    try:
        # Create label directories
        classes = np.unique(labels)
        for cls in classes:
            os.makedirs(os.path.join(train_dir, str(cls)), exist_ok=True)
            os.makedirs(os.path.join(val_dir, str(cls)), exist_ok=True)
        logger.info("Created train/val directories for each class.")

        # Split indices for train and validation
        idx = np.arange(len(image_paths))
        train_idx, val_idx = train_test_split(idx, test_size=test_size, stratify=labels, random_state=42)
        for i in train_idx:
            src = image_paths[i]
            dest = os.path.join(train_dir, str(labels[i]), os.path.basename(src))
            shutil.copy(src, dest)
        for i in val_idx:
            src = image_paths[i]
            dest = os.path.join(val_dir, str(labels[i]), os.path.basename(src))
            shutil.copy(src, dest)
        logger.info(f"Copied images to train/validation splits. Train size: {len(train_idx)}, Val size: {len(val_idx)}.")
    except Exception as e:
        logger.error(f"Organizing dataset failed: {e}")
        sys.exit(1)

def build_and_train_model(train_dir, val_dir, num_classes=2, img_size=(224,224), batch_size=16, epochs=5):
    """
    Builds an EfficientNetB0 model, trains on the dataset, and returns trained model and history.
    Fine-tuning EfficientNetB0 as per Keras example:contentReference[oaicite:5]{index=5}.
    """
    try:
        # Create datasets from directories
        train_ds = image_dataset_from_directory(train_dir, image_size=img_size, batch_size=batch_size, label_mode='int')
        val_ds = image_dataset_from_directory(val_dir, image_size=img_size, batch_size=batch_size, label_mode='int')
        class_names = train_ds.class_names
        logger.info(f"Dataset classes: {class_names}")

        # Build model
        base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=img_size+(3,))
        base_model.trainable = False
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=1e-4), 
                      loss='sparse_categorical_crossentropy', 
                      metrics=['accuracy', 
                               tf.keras.metrics.Precision(name='precision'), 
                               tf.keras.metrics.Recall(name='recall')])
        logger.info("Starting model training...")
        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
        logger.info("Model training completed.")
        # Unfreeze base model for fine-tuning
        base_model.trainable = True
        model.compile(optimizer=Adam(learning_rate=1e-5), 
                      loss='sparse_categorical_crossentropy', 
                      metrics=['accuracy', 
                               tf.keras.metrics.Precision(name='precision'), 
                               tf.keras.metrics.Recall(name='recall')])
        history_finetune = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
        logger.info("Fine-tuning completed.")
        return model, history, history_finetune, val_ds
    except Exception as e:
        logger.error(f"Model building/training failed: {e}")
        sys.exit(1)

def evaluate_and_report(model, val_ds, report_dir="report"):
    """
    Evaluates the model on validation set and generates a report with metrics.
    Saves a classification report and confusion matrix.
    """
    try:
        os.makedirs(report_dir, exist_ok=True)
        # Evaluate on validation dataset
        results = model.evaluate(val_ds, verbose=0)
        metrics_names = model.metrics_names
        eval_report = {name: value for name, value in zip(metrics_names, results)}
        logger.info(f"Evaluation results: {eval_report}")

        # Gather true labels and predictions
        y_true = np.concatenate([y for x, y in val_ds], axis=0)
        y_pred_probs = model.predict(val_ds)
        y_pred = np.argmax(y_pred_probs, axis=1)
        # Classification report
        cls_report = classification_report(y_true, y_pred, output_dict=False)
        logger.info(f"Classification Report:\n{cls_report}")
        # Save text report
        with open(os.path.join(report_dir, "classification_report.txt"), "w") as f:
            f.write(f"Evaluation metrics: {eval_report}\n\n")
            f.write(cls_report)
        # Save as CSV (output_dict)
        cls_report_dict = classification_report(y_true, y_pred, output_dict=True)
        df_report = pd.DataFrame(cls_report_dict).transpose()
        df_report.to_csv(os.path.join(report_dir, "classification_report.csv"))
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        np.savetxt(os.path.join(report_dir, "confusion_matrix.csv"), cm, delimiter=",", fmt='%d')
        logger.info("Reports saved.")
    except Exception as e:
        logger.error(f"Reporting failed: {e}")

def save_model(model, model_path="trained_model.h5"):
    """
    Saves the trained Keras model to a file.
    """
    try:
        model.save(model_path)
        logger.info(f"Model saved to {model_path}.")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")

# Main training pipeline
def train_pipeline():
    # Step 1: Download and extract dataset
    download_dataset()

    # Step 2: Convert DICOMs to PNG
    data_root = "./data"
    png_dir = "./images_png"
    convert_dicom_to_png(data_root, png_dir)

    # Step 3: Pseudo-label images with KMeans
    image_paths, labels = pseudo_label_images(png_dir, n_clusters=2)

    # Step 4: Organize into train/validation
    train_dir = "./dataset/train"
    val_dir = "./dataset/val"
    organize_dataset(image_paths, labels, train_dir, val_dir, test_size=0.2)

    # Step 5: Build and train CNN (EfficientNetB0)
    model, history, history_finetune, val_ds = build_and_train_model(train_dir, val_dir, num_classes=2, img_size=(224,224), batch_size=16, epochs=3)

    # Step 6: Evaluate and report
    evaluate_and_report(model, val_ds)

    # Step 7: Save model
    save_model(model, model_path="trained_model.h5")

    logger.info("Training pipeline completed successfully.")

# Streamlit application for predictions
def streamlit_app():
    st.title("CT Scan AI Prediction (EfficientNetB0)")
    st.write("Upload a CT scan image (PNG/JPG) to get a prediction with confidence.")

    # Load model
    try:
        model = tf.keras.models.load_model("trained_model.h5")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return

    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess image
        img_array = image.resize((224, 224))
        img_array = np.array(img_array) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        # Predict
        preds = model.predict(img_array)
        pred_label = np.argmax(preds, axis=1)[0]
        confidence = np.max(preds)
        class_name = f"Class {pred_label}"
        st.write(f"**Prediction:** {class_name} (Confidence: {confidence:.2f})")

# Orchestrate training or launch Streamlit
if __name__ == "__main__":
    # Detect if running under Streamlit or normal execution
    from streamlit import runtime
    if runtime.exists():
        # Running via `streamlit run`, launch app
        streamlit_app()
    else:
        # Run training pipeline
        train_pipeline()
        # Launch Streamlit app
        try:
            logger.info("Launching Streamlit application...")
            from streamlit.web import cli as stcli
            sys.argv = ["streamlit", "run", sys.argv[0]]
            sys.exit(stcli.main())
        except Exception as e:
            logger.error(f"Failed to launch Streamlit: {e}")
