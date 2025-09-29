# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# -----------------------------
# 1️⃣ FastAPI app
# -----------------------------
app = FastAPI(title="Exoplanet Predictor API")

# Allow frontend requests (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for development only, replace with your frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# 2️⃣ Request body schema
# -----------------------------
class Features(BaseModel):
    features: list[float]  # 33 numeric features

# -----------------------------
# 3️⃣ Model architecture
# -----------------------------
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

# -----------------------------
# 4️⃣ Load model weights
# -----------------------------
MODEL_PATH = "exoplanet_model.pth"  # path to your saved state_dict
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = Net()
state_dict = torch.load(MODEL_PATH)
model.load_state_dict(state_dict)
model.eval()  # evaluation mode

# -----------------------------
# 5️⃣ Endpoints
# -----------------------------
@app.get("/")
def root():
    return {"message": "Welcome to Exoplanet Predictor API"}

@app.post("/predict")
def predict(data: Features):
    if len(data.features) != 33:
        return {"error": "Expected 33 features"}

    # Convert features to tensor
    x = torch.tensor([data.features], dtype=torch.float32)

    # Forward pass
    with torch.no_grad():
        output = model(x)
        pred_class = torch.argmax(output, dim=1).item()

    # Convert numeric to label
    label = "Exoplanet" if pred_class == 1 else "Not Exoplanet"
    return {"prediction": label}
