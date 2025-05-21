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

    # Dummy prediction logic (replace this with real AI model inference)
    prediction = random.choice(["No Aortic Dissection Detected", "Aortic Dissection Detected"])
    
    # Show result with a colored message
    if prediction == "Aortic Dissection Detected":
        st.error(f"Result: **{prediction}**")
    else:
        st.success(f"Result: **{prediction}**")
else:
    st.info("Please upload a chest CT scan image to start analysis.")
