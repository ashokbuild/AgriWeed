# 🌿 AgriWeed AI — Basic Agricultural Weed Detection & Yield Impact Estimator

A clean, lightweight, single-screen Streamlit application that:
1. **Detects & Marks Weeds vs Crops** using pretrained YOLOv8 on field images.
2. **Flags Infestation Severity** (Weed-Free, Low, Moderate, High, Severe) with counts and canopy area %.
3. **Calculates Crop Yield Impact** directly below the image based on field size and detected weeds.

---

## 📁 Why is the project folder size large?
The main project code is just **one simple script (`app.py`)**!
The folder size comes almost entirely from the local Python virtual environment (`.venv`):
- Deep learning frameworks like **PyTorch (`torch` + `torchvision`)** take ~2.5 GB of disk space to provide GPU/CPU tensor compute.
- This is completely standard for local AI applications running offline on your laptop.

---

## ⚡ Quick Start (Windows)

Open PowerShell in this folder and run:

```powershell
# 1. Activate virtual environment
.venv\Scripts\activate

# 2. Launch the app
streamlit run app.py
```

Your browser will automatically open at `http://localhost:8501`.

---

## 🎯 How It Works

1. **Select or Upload Image**:
   - Upload any JPG or PNG crop field photograph.
   - Or click **"Load Sample Maize Field"** / **"Load Sample Sugar Beet Field"** for an instant test.
2. **View Detections**:
   - 🔴 **Red Boxes**: Flagged weed competitors.
   - 🟢 **Green Boxes**: Protected crop plants.
   - Live KPI cards display total weed count, crop count, and canopy coverage %.
3. **Estimate Yield**:
   - Enter field size (acres or hectares).
   - Enter baseline yield (t/ha).
   - The app automatically suggests loss percentage from the detected weed infestation level and calculates:
     - Normal expected harvest (tonnes)
     - Potential loss from weed competition (tonnes)
     - Harvest if weeds remain unchecked (tonnes)
     - Yield recovered with weed control (tonnes)

---

## 🧪 Run Tests

To verify the yield calculations and image annotation logic:
```powershell
.venv\Scripts\pytest.exe tests -v
```
