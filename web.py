
# final prac

import os
import cv2
import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Initialize YOLO model
model = YOLO('last.pt')


# Define a function to process the image
def process_image(input_image):
    # Convert PIL image to OpenCV format
    cv_image = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)

    # Perform prediction
    results = model.predict(source=cv_image, show=False, save=True, show_labels=True, show_conf=True, conf=0.3,
                            save_txt=False, save_crop=False, line_width=2)

    # Access the first result from the list
    #result = results[0]

    # Extract the annotated image from the result
    #annotated_image = result.render()
    annotated_image = results

    # Return the annotated image
    return annotated_image


# Streamlit UI
st.title("Spruce Infected Tree Detection")
st.subheader("A Project by Christo")
st.write("Timely detection of newly infested trees is important for minimizing economic losses"
         "and effectively planning forest management activities to stop or at least slow outbreaks"
         "in Sweden forests")

# Upload image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])


# Process and display the image
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    processed_image = process_image(image)
else:
    st.write("Predicted image not found.")



# Define the base folder path
base_path = "runs/segment/"

# Find all subdirectories (predicted folders) in the base path
subdirs = [subdir for subdir in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, subdir))]
# Sort the subdirectories by converting them to integers first
subdirs = sorted(subdirs, key=lambda x: int(x.split("predict")[-1]) if x.startswith("predict") and x.split("predict")[-1].isdigit() else -1)

# If there are subdirectories (predicted folders), get the latest image path
if subdirs:
    latest_subdir = subdirs[-1]
    latest_image_path = os.path.join(base_path, latest_subdir, "image0.jpg")

    # Check if the image file exists
    if os.path.exists(latest_image_path):
        # Display the image
        st.image(latest_image_path, caption='Predicted Image', use_column_width=True)
    else:
        st.write("Predicted image not found.")
else:
    st.write("No predicted folders found.")
