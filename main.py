from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import random

app = FastAPI()

# Allow CORS (so frontend can communicate with backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input data schema
class MedicalData(BaseModel):
    sex: str
    plasma_CA19_9: float
    creatinine: float
    LYVE1: float
    REG1B: float
    TFF1: float
    REG1A: float

@app.post("/predict")
def predict(data: MedicalData):
    # Process the input data and use your existing model for prediction
    prediction_result = model_prediction_function(data)  # Placeholder for your model function

    # Return the prediction result
    return {"prediction": prediction_result}

# Placeholder function for model prediction
def model_prediction_function(data):
    # Implement the actual prediction logic using your existing model
    # For demonstration, returning a dummy result
    return random.choice(list("positive,negative"))  # Replace this with actual model logic

if __name__ == "_main_":
    import uvicorn
    uvicorn.run(app, host="8.0.0.0", port=8000)