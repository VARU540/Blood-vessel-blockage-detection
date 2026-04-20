# 🩸 Blood Vessel Blockage Detector

AI-powered medical analysis system using MLP + VAE ensemble models.

---

## 📁 Project Structure

```
blood-vessel-detector/
├── index.html          ← Frontend UI (open in browser)
├── style.css           ← Styles
├── script.js           ← Frontend logic (calls Flask API)
├── app.py              ← Flask API server
├── train_and_save.py   ← Train models & save weights (run ONCE)
├── requirements.txt    ← Python dependencies
└── model_weights/
    └── weights.pkl     ← Generated after training (auto-created)
```

---

## 🚀 Setup & Run (VS Code)

### Step 1 — Install Python dependencies
Open a terminal in VS Code and run:
```bash
pip install -r requirements.txt
```

### Step 2 — Prepare your dataset
Place your `final.csv` file in the project root folder.

The CSV must contain these columns:
- `peak_systolic_velocity`
- `resistive_index`
- `blood_flow_velocity`
- `avg_temperature`
- `temperature_difference`
- `cold_spot_area_percent`
- `heart_rate`
- `pulse_amplitude`
- `pulse_transit_time`
- `hrv`

### Step 3 — Train the models (run ONCE)
```bash
python train_and_save.py
```
This creates `model_weights/weights.pkl`. You only need to do this once.

### Step 4 — Start the Flask API server
```bash
python app.py
```
The API will start at: **http://localhost:5000**

### Step 5 — Open the frontend
Open `index.html` in your browser. You can:
- Use VS Code's **Live Server** extension (right-click → Open with Live Server)
- Or simply double-click `index.html` to open it directly in a browser

---

## 🔧 VS Code Extensions Recommended
- **Python** (ms-python.python)
- **Live Server** (ritwickdey.liveserver)

---

## ⚠️ Notes
- The Flask server must be running before using the frontend
- If the server is not running, the app falls back to a local simulation mode
- The `model_weights/` folder is created automatically during training
