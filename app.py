import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
import pandas as pd

# Set page configuration as the first Streamlit command
st.set_page_config(page_title="Odia Handwritten Character Recognition", page_icon="📜", layout="wide")

# Load your model
@st.cache_resource
def load_odia_model():
    return load_model('best_model_cnn.keras')

model = load_odia_model()

# Class labels (replace these with your actual 45 Odia character labels)
number_to_class = [
    'ଅ', 'ଆ', 'ଇ', 'ଈ', 'ଉ', 'ଊ', 'ଋ', 'ଏ', 'ଐ', 'ଓ', 'ଔ',
    'କ', 'ଖ', 'ଗ', 'ଘ', 'ଙ', 'ଚ', 'ଛ', 'ଜ', 'ଝ', 'ଞ', 'ଟ', 'ଠ',
    'ଡ', 'ଢ', 'ଣ', 'ତ', 'ଥ', 'ଦ', 'ଧ', 'ନ', 'ପ', 'ଫ', 'ବ', 'ଭ',
    'ମ', 'ଯ', 'ର', 'ଳ', 'ଶ', 'ଷ', 'ସ', 'ହ','ୠ','ୱ'
]  # Ensure this has 45 items

# UI Layout
st.title("Odia Handwritten Character Recognition")
st.write("Upload an image of an Odia handwritten character to get predictions.")

# File uploader UI with additional styling
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image and display with reduced size for a cleaner interface
    img = Image.open(uploaded_file).convert("RGB")
    
    # Resize for better display in the UI
    img = img.resize((100, 100))
    st.image(img, caption="Uploaded Image", use_column_width=False)

    # Preprocess image for model input
    img = img.resize((64, 64))
    img = np.array(img) / 255.0
    img = img.reshape(1, 64, 64, 3)

    # Prediction
    predictions = model.predict(img)[0]
    indices = np.argsort(predictions)[-3:][::-1]

    # Prepare data for table
    prediction_data = []
    for i in indices:
        prediction_data.append({
            'Character': number_to_class[i],
            'Probability': f"{predictions[i]*100:.2f}%"
        })
    
    # Display predictions in a table
    st.subheader("Top Predictions:")
    prediction_df = pd.DataFrame(prediction_data)
    st.table(prediction_df)
