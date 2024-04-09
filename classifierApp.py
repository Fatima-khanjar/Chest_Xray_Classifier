import streamlit as st
from keras.models import load_model
from PIL import Image
import cv2
import numpy as np

# Set title with a different color for the header
st.title('Covid Classification')
st.markdown(
    '<style>h1 {color: #004080;}</style>', 
    unsafe_allow_html=True
)

# Set header with a different color
st.subheader('Upload a chest X-ray image:')
st.markdown(
    '<p style="color:#0066cc;font-size:18px;">Please upload an X-ray image for COVID classification.</p>', 
    unsafe_allow_html=True
)

# Upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# Load classifier
model = load_model('Xray_classifier.h5')

def predict_image(model, img):
    return model.predict(img)
# Load class names
with open('Labels.txt', 'r') as f:
    lines = f.readlines()
    labels = [line.strip().split(' ')[1] for line in lines]
    f.close()

# Display image and classify
if file is not None:
    # Preprocess the image
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = img / 255.0
    img = cv2.resize(img, (100, 100))
    img = np.reshape(img, (1, 100, 100, 3))

    # Classify the image
    prediction = predict_image(model, img)
    class_index = int(round(prediction[0][0]))
    predicted_class = "Covid" if class_index == 1 else "Normal"
    confidence_score = prediction[0][0]

    # Display the result with different colors for header and subheader
    st.subheader('Prediction:')
    st.markdown(
        f'<p style="color:#004080;font-size:24px;">The X-ray is classified as: {predicted_class}</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<p style="color:#0066cc;font-size:18px;">Confidence Score: {confidence_score:.2%}</p>',
        unsafe_allow_html=True
    )
