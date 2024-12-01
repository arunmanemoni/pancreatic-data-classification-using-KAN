import streamlit as st
import torch
import numpy as np

# Import your KAN model class
from kan import KAN

# Load the pre-trained KAN model
def load_model():
    model = KAN(width=[7, 64, 128, 64, 2], grid=4, k=2)  # Use the architecture from your training
    model.load_state_dict(torch.load('model.pth'))
    model.eval()  # Set model to evaluation mode
    return model

model = load_model()

# Function to preprocess user input
def preprocess_input(sex, plasma_CA19_9, creatinine, LYVE1, REG1B, TFF1, REG1A):
    # Convert sex to binary value (0 for Female, 1 for Male)
    sex = 1 if sex == "Male" else 0
    
    # Create a NumPy array for the input features
    input_data = np.array([[sex, plasma_CA19_9, creatinine, LYVE1, REG1B, TFF1, REG1A]])
    
    # Convert to Torch tensor and normalize (if normalization was used during training)
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    return input_tensor

# Set up the page configuration
st.set_page_config(
    page_title="Cancer Classification Interface",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Page title and description
st.title("Pancreatic Cancer Prediction Interface")
st.markdown("""
Welcome to the Pancreatic Cancer Classification Interface. 
Please provide the required input parameters and click *Submit* to see the result.
""")

# Input fields
sex = st.radio("Select Sex:", options=["Male", "Female"], index=0)
plasma_CA19_9 = st.number_input("Plasma CA19_9:", min_value=0.0, step=0.1, format="%.2f")
creatinine = st.number_input("Creatinine:", min_value=0.0, step=0.1, format="%.2f")
LYVE1 = st.number_input("LYVE1:", min_value=0.0, step=0.1, format="%.2f")
REG1B = st.number_input("REG1B:", min_value=0.0, step=0.1, format="%.2f")
TFF1 = st.number_input("TFF1:", min_value=0.0, step=0.1, format="%.2f")
REG1A = st.number_input("REG1A:", min_value=0.0, step=0.1, format="%.2f")

# Submit button
if st.button("Submit"):
    # Preprocess the input
    input_tensor = preprocess_input(sex, plasma_CA19_9, creatinine, LYVE1, REG1B, TFF1, REG1A)
    
    # Make prediction
    with torch.no_grad():
        prediction = torch.argmax(model(input_tensor)).item()
        # Map the prediction to result
        result = "Positive" if prediction == 1 else "Negative"
    
    # Display the result
    st.success(f"The result is: *{result}*")
