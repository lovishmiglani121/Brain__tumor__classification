import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

num_classes = 4
# Load the pre-trained model
model = load_model('Xception2.h5')
# Define class labels
class_labels = [['glioma', 'meningioma', 'notumor', 'pituitary']]
class_labels = list(range(num_classes))

# Streamlit app
st.title("Brain Tumor Classification")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Preprocess the image
    image = Image.open(uploaded_file)
    resized_image = image.resize((299, 299))
    img_array = np.array(resized_image)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Make prediction
    prediction = model.predict(img_array)

    if prediction.size > 0 and len(class_labels) > 0:
        if prediction.shape[-1] == len(class_labels):  # Check if number of classes matches prediction shape
            predicted_label_index = np.argmax(prediction)
            
            if predicted_label_index < len(class_labels):
                predicted_label = class_labels[predicted_label_index]

                # Display the image and prediction
                st.image(image, caption="Uploaded Image", use_column_width=True)
                st.write(f"Prediction: {predicted_label} ")
                
                if predicted_label == 0:
                    st.write("Glioma")
                elif predicted_label == 1:
                    st.write("Meningioma")
                elif predicted_label == 2:
                    st.write("No Tumor")
                else:
                    st.write("Pituitary Tumor")
                st.bar_chart(prediction[0])
            else:
                st.warning("Invalid predicted label index")
        else:
            st.warning("Number of classes does not match prediction shape")
    else:
        st.warning("Prediction array or class labels are empty")


# Additional content
st.sidebar.header("About")
st.sidebar.info(
    "This web app is a simple brain tumor classification tool. "
    "Upload an MRI image, and the model will predict the class of the tumor."
)

st.sidebar.header("Class Labels")
st.sidebar.text("0: Glioma\n1: Meningioma\n2: No Tumor\n3: Pituitary Tumor")
