import streamlit as st
from PIL import Image
import requests
from ultralytics import YOLO
from io import BytesIO
import cv2



def main():
    st.title("Image Classification App")
    st.write("Upload an image or take a photo to perform image classification.")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg"])
    take_photo = st.button("Take Photo")

    if take_photo:
        uploaded_file = st.image("", caption="Camera Feed", channels="BGR", use_column_width=True)
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            st.error("Unable to access camera.")
            return
        while True:
            ret, frame = camera.read()
            if not ret:
                st.error("Failed to capture image.")
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            uploaded_file.image(frame_rgb, channels="RGB")
            if st.button("Take Snapshot"):
                cv2.imwrite("snapshot.jpg", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                break
        camera.release()
        st.success("Snapshot saved as snapshot.jpg.")
        uploaded_file = "snapshot.jpg"
        st.warning("Sorry, capturing a photo is not supported in this environment. Please try uploading the picture")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        st.write("Performing image classification...")

        model = YOLO('/content/best_class_image.pt')
        prediction = model(image)[0]

        max_index = prediction.probs.top1
        class_names = prediction.names
        max_class_name = class_names[max_index]

        st.success("Image classification complete!")
        st.write("Classification Result:", max_class_name)

if __name__ == "__main__":
    main()
