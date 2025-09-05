# app.py
import io
import pickle
from PIL import Image

import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse


FINAL_LATENT_DIM = 2250

class AutoEncoder_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_dim = FINAL_LATENT_DIM
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(128 * 9 * 14, self.latent_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 128 * 9 * 14),
            nn.BatchNorm1d(128 * 9 * 14),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Unflatten(1, (128, 9, 14)),

            nn.Upsample(size=(12, 28), mode='bilinear', align_corners=False),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(size=(24, 56), mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(size=(75, 113), mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(size=(150, 225), mode='bilinear', align_corners=False),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# ==== 2) FASTAPI APP + CORS ====
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==== 3) LOAD MODEL (from pickle) ====
DEVICE = "cpu"
MODEL_PATH = "models/model_state.pth"

model = AutoEncoder_CNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()



# ==== 4) HELPERS ====
def preprocess(pil_img: Image.Image) -> torch.Tensor:
    # Resize to model’s expected 150x225
    img = pil_img.convert("RGB").resize((225, 150))
    tensor = torch.tensor(list(img.getdata()), dtype=torch.float32)
    tensor = tensor.view(img.size[1], img.size[0], 3)  # H, W, C
    tensor = tensor.permute(2, 0, 1) / 255.0           # C, H, W in [0,1]
    return tensor.unsqueeze(0)                         # add batch dim

def postprocess(t: torch.Tensor) -> bytes:
    t = (t.clamp(0, 1) * 255).byte().squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = Image.fromarray(t)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


# ==== 5) ROUTES ====
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reconstruct")
async def reconstruct(file: UploadFile = File(...)):
    image_bytes = await file.read()
    pil_img = Image.open(io.BytesIO(image_bytes))
    x = preprocess(pil_img).to(DEVICE)
    with torch.no_grad():
        y = model(x)
    png_bytes = postprocess(y)
    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")
