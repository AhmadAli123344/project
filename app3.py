import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# Load your CNN model
model = tf.keras.models.load_model('D:\\update_model\\Project_Model.hdf5')

def main():
    data=[]
    st.title("Skin Disease Classification App")

    # Upload image through Streamlit
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        # Display the uploaded image
        st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

        # Resize the image to (300, 300)
        image = Image.open(uploaded_image)
        image = image.convert('L')  # Convert to grayscale
        image = np.array(image)
        resized_image = cv2.resize(image, (300,300))
        data.append(resized_image/255)
        X=np.array(data)
        st.write(X.shape)

        # Make predictions using your CNN model
        prediction = model.predict(X)

        # Display the prediction
        st.subheader("Model Prediction About your Skin Disease Type:")
        class_names = ['Acne and Rosacea', 'Atopic Dermatitis']  # Replace with your class names
        st.write(class_names[np.argmax(prediction)])
        st.subheader("For further Information Please Concern With Your Doctor/n Thanks to use our App")

if __name__ == '__main__':
    main()
