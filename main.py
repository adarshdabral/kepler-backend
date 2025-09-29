from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# -------------------------------
# Model Definition (same as training)
# -------------------------------
input_size = 33
output_size = 2
hid1_size = 25
hid2_size = 10

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hid1_size)
        self.fc2 = nn.Linear(hid1_size, hid2_size)
        self.fc3 = nn.Linear(hid2_size, output_size)
    
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=-1)

# -------------------------------
# Load Model
# -------------------------------
model = Net()
model.load_state_dict(torch.load("kepler_model.pth", map_location=torch.device('cpu')))
model.eval()  # Set to evaluation mode

# -------------------------------
# FastAPI App
# -------------------------------
app = FastAPI()

# Allow CORS for frontend deployment
origins = [
    "http://localhost:3000",  # Local frontend
    "https://kepler-frontend-chi.vercel.app/"  # Replace with your Vercel URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # ["*"] for all origins, but less secure
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Request Body Schema
# -------------------------------
class PredictRequest(BaseModel):
    features: list[float]

# -------------------------------
# Endpoints
# -------------------------------
@app.get("/")
def home():
    return {"message": "Kepler Exoplanet Predictor API is running!"}

@app.post("/predict")
def predict(request: PredictRequest):
    try:
        features = np.array(request.features, dtype=np.float32)
        features_tensor = torch.from_numpy(features).unsqueeze(0)  # batch dimension
        with torch.no_grad():
            output = model(features_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
        # Map predicted class to human-readable label
        label_map = {0: "FALSE POSITIVE", 1: "CANDIDATE"}  # Adjust if needed
        return {"prediction": label_map[predicted_class]}
    except Exception as e:
        return {"error": str(e)}
