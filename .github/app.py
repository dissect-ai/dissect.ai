import streamlit as st
from PIL import Image
import random

# Title and description
st.title("Dissect.AI - Aortic Dissection Detection Demo")
st.write("""
Upload a chest CT scan image, and our AI model will analyze it for signs of aortic dissection.
This is a demo app; the AI prediction is randomly generated for now.
""")

# File uploader
uploaded_file = st.file_uploader("Upload a chest CT scan image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Open and display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded CT Scan", use_column_width=True)

    # Show processing message
    st.write("Analyzing image with AI model...")

    # dissectai_selflearning.py
# Fully self-learning AI for detecting aortic dissection from CT scans using Kaggle data

import os
import zipfile
import random
import shutil
import subprocess
from glob import glob
import numpy as np
import pydicom
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

# ------------------------------
# 1. Convert DICOM to PNG
# ------------------------------
def convert_dicom_to_png(dicom_path, output_folder):
    ds = pydicom.dcmread(dicom_path)
    img_array = ds.pixel_array.astype(np.float32)
    img_array = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array))
    img_array = (img_array * 255).astype(np.uint8)
    img = Image.fromarray(img_array).convert("RGB")
    base_name = os.path.basename(dicom_path).replace('.dcm', '.png')
    img.save(os.path.join(output_folder, base_name))

# ------------------------------
# 2. Extract and convert
# ------------------------------
def extract_and_convert(zip_file_path, output_dir):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

    dicom_files = glob(os.path.join(output_dir, '**/*.dcm'), recursive=True)
    png_dir = os.path.join(output_dir, 'converted_images')
    os.makedirs(png_dir, exist_ok=True)

    for f in dicom_files:
        convert_dicom_to_png(f, png_dir)
    return png_dir

# ------------------------------
# 3. Generate pseudo-labels
# ------------------------------
def cluster_images(image_folder):
    imgs = []
    paths = []
    for img_path in glob(os.path.join(image_folder, '*.png')):
        img = Image.open(img_path).resize((64, 64))
        imgs.append(np.array(img).flatten())
        paths.append(img_path)
    kmeans = KMeans(n_clusters=2, random_state=42).fit(imgs)
    return list(zip(paths, kmeans.labels_))

# ------------------------------
# 4. Organize based on pseudo-labels
# ------------------------------
def organize_dataset(clustered_data, output_base='dataset'):
    os.makedirs(output_base, exist_ok=True)
    random.seed(42)

    train, val = train_test_split(clustered_data, test_size=0.2, stratify=[l[1] for l in clustered_data])
    for split_name, data in [('train', train), ('val', val)]:
        for img_path, label in data:
            class_name = f'class_{label}'
            dest_dir = os.path.join(output_base, split_name, class_name)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(img_path, os.path.join(dest_dir, os.path.basename(img_path)))

# ------------------------------
# 5. Build model
# ------------------------------
def build_model():
    base = tf.keras.applications.EfficientNetB0(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
    base.trainable = False
    model = tf.keras.Sequential([
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ------------------------------
# 6. Train and evaluate
# ------------------------------
def train_model(dataset_path, epochs=10):
    train_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255).flow_from_directory(
        os.path.join(dataset_path, 'train'),
        target_size=(224, 224), batch_size=32, class_mode='binary')

    val_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255).flow_from_directory(
        os.path.join(dataset_path, 'val'),
        target_size=(224, 224), batch_size=32, class_mode='binary')

    model = build_model()
    model.fit(train_gen, validation_data=val_gen, epochs=epochs)
    model.save('dissectai_selflearned_model.h5')

    loss, acc = model.evaluate(val_gen)
    print(f"Self-learned model accuracy: {acc * 100:.2f}%")
    return model

# ------------------------------
# 7. Download Kaggle dataset
# ------------------------------
def download_kaggle_dataset(dataset_name, download_path='data'):
    os.makedirs(download_path, exist_ok=True)
    subprocess.run([
        'kaggle', 'datasets', 'download', '-d', dataset_name, '-p', download_path, '--unzip'
    ], check=True)

# ------------------------------
# 8. Main pipeline
# ------------------------------
def main():
    dataset = 'xiaoweixumedicalai/imagetbad'
    download_kaggle_dataset(dataset, 'data')
    dicom_dir = 'data'
    dicom_zips = glob(os.path.join(dicom_dir, '*.zip'))
    for zip_file in dicom_zips:
        png_dir = extract_and_convert(zip_file, 'output')
        clustered = cluster_images(png_dir)
        organize_dataset(clustered)
        train_model('dataset')

if __name__ == '__main__':
    main()
