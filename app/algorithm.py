import os
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
import torch.nn as nn
from pytorchcv.model_provider import get_model as ptcv_get_model
import torchvision.transforms as T
from facenet_pytorch import MTCNN
from efficientnet_pytorch import EfficientNet

# ────────────────────────────────
# 1) DATa PATHS
# ───────────────────────────────
BASE_DIR   = "/app"
INPUT_CSV  = os.path.join(BASE_DIR, "input.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "output.csv")
VIDEO_DIRS = [
    os.path.join(BASE_DIR, "generated_videos"),
    os.path.join(BASE_DIR, "real_videos"),
]

# ─────────────────────────────────────────────────────────────────
# 2) DEVICE & FACE DETECTOR
# ─────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(device=device, keep_all=False)

# ────────────────────────────────────────────────────────────────
# 3) MODEL setUP
model = ptcv_get_model("xception", pretrained=True)   # loads 1000-class weights
in_feats = model.output.in_features
model.output = nn.Linear(in_feats, 2)
model.to(device).eval()


from efficientnet_pytorch import EfficientNet
eff_model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=2) \
                     .to(device).eval()


# ───────────────────────────────────────────────────────────────
from efficientnet_pytorch import EfficientNet
eff_model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=2)
eff_model.to(device).eval()


# ───────────────────────────────────────────────────────────────────────────
transform = T.Compose([
    T.Resize((299, 299)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# ────────────────────────────────────────────────────────────────────────────
# FRAMe 

# ─────────────────────────────────────────────────────────────────────────────
def preprocess_frame(frame):
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    box, _ = mtcnn.detect(pil_img)
    if box is None:
        return None
    x1, y1, x2, y2 = map(int, box[0])
    face = frame[y1:y2, x1:x2]
    pil_face = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    return transform(pil_face).unsqueeze(0).to(device)

# ─────────────────────────────────────────────────────────────────────────────
    # DETECTION LOGI
# ──────────────────────────────────────────────────────────────────────────
def detect_fake(video_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    
    #for i in np.linspace(0, total - 1, num=30, dtype=int): //this is for faster GPU's
    for i in np.linspace(0, total - 1, num=10, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frm = cap.read()
        if ret:
            frames.append(frm)
    cap.release()

    
    inputs = []
    for frm in frames:
        inp = preprocess_frame(frm)
        if inp is not None:
            inputs.append(inp)
    if not inputs:
        return 1  

    
    #votes = 0                     # for faster GPU's
    #with torch.no_grad():
        for inp in inputs:
            out1       = model(inp)
            out1_flip  = model(torch.flip(inp, [-1]))
            p1 = (
                torch.softmax(out1,      dim=1)[0, 1] +
                torch.softmax(out1_flip, dim=1)[0, 1]
            ) / 2
            out2       = eff_model(inp)
            out2_flip  = eff_model(torch.flip(inp, [-1]))
            p2 = (
                torch.softmax(out2,      dim=1)[0, 1] +
                torch.softmax(out2_flip, dim=1)[0, 1]
            ) / 2

            avg_prob = (p1 + p2) / 2
            votes   += int(avg_prob >= 0.5)
    votes = 0
    with torch.no_grad():
        for inp in inputs:
            # Xception single inference
            p1 = torch.softmax(model(inp), dim=1)[0,1].item()
            # EfficientNet single inference
            p2 = torch.softmax(eff_model(inp), dim=1)[0,1].item()

            avg_prob = (p1 + p2) / 2
            votes   += int(avg_prob >= 0.5)

    #fake_ratio = votes / len(inputs)  ---------------> this one ahd %45 accuracy
    #BEST_THR = 0.55
    #return 1 if fake_ratio >= BEST_THR else 0
    fake_ratio = votes / len(inputs)
    return fake_ratio



# ─────────────────────────────────────────────────────────────────────────────
# 7) MAIN 
# ─────────────────────────────────────────────────────────────────────────────
def main():
    df = pd.read_csv(INPUT_CSV, sep=";", header=0)
    results = []
    BEST_THR = 0.55

    for fn in df["file_name"]:
        score = 1.0  # default (missing file → treat as fake)
        for d in VIDEO_DIRS:
            path = os.path.join(d, fn)
            if os.path.exists(path):
                score = detect_fake(path)
                break
        label = int(score >= BEST_THR)
        results.append((fn, label, score))

    out_df = pd.DataFrame(results, columns=["file_name", "Is_fake", "Score"])
    out_df.to_csv(OUTPUT_CSV, sep=";", index=False)
    print(f"✔ Wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
