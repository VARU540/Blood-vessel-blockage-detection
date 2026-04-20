"""
app.py — Flask API Server
Blood Vessel Blockage Detector Backend

Run: python app.py
API will start at: http://localhost:5000
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import os

app = Flask(__name__)
CORS(app)   # Allow frontend (HTML file) to call this API

# ─── LOAD MODELS ON STARTUP ──────────────────────────────
print("Loading model weights...")
WEIGHTS_PATH = 'model_weights/weights.pkl'

if not os.path.exists(WEIGHTS_PATH):
    raise RuntimeError(
        "weights.pkl not found!\n"
        "Please run: python train_and_save.py first"
    )

with open(WEIGHTS_PATH, 'rb') as f:
    DATA = pickle.load(f)

SCALER_MEAN = np.array(DATA['scaler_mean'])
SCALER_STD  = np.array(DATA['scaler_std'])
ALL_COLS    = DATA['all_cols']
ORIG_COLS   = DATA['orig_cols']
VAE_THRESH  = DATA['vae_threshold']

print(f"✅ Models loaded!")
print(f"   MLP test accuracy:      {DATA['metrics']['mlp_test_acc']:.2%}")
print(f"   VAE test accuracy:      {DATA['metrics']['vae_test_acc']:.2%}")
print(f"   Ensemble test accuracy: {DATA['metrics']['ens_test_acc']:.2%}")


# ─── MODEL CLASSES ────────────────────────────────────────
class MLP:
    def __init__(self, dims, drops):
        self.dims = dims; self.drops = drops; self.n = len(dims) - 1
        self.W = []; self.b = []

    def relu(self, x):    return np.maximum(0, x)
    def sigmoid(self, x): return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def load(self, data):
        self.W = [np.array(w) for w in data['W']]
        self.b = [np.array(b) for b in data['b']]

    def predict_proba(self, X):
        h = X
        for i in range(self.n):
            z = h @ self.W[i] + self.b[i]
            h = self.relu(z) if i < self.n - 1 else self.sigmoid(z)
        return h.flatten()


class VAE:
    def __init__(self):
        self.W = {}

    def load(self, data):
        for k, v in data.items():
            self.W[k] = np.array(v)

    def relu(self, x): return np.maximum(0, x)

    def encode(self, X):
        h1 = self.relu(X @ self.W['We1'] + self.W['be1'])
        h2 = self.relu(h1 @ self.W['We2'] + self.W['be2'])
        mu     = h2 @ self.W['Wmu'] + self.W['bmu']
        logvar = h2 @ self.W['Wlv'] + self.W['blv']
        return mu, logvar

    def decode(self, z):
        dh1 = self.relu(z @ self.W['Wd1'] + self.W['bd1'])
        dh2 = self.relu(dh1 @ self.W['Wd2'] + self.W['bd2'])
        return dh2 @ self.W['Wd3'] + self.W['bd3']

    def reconstruction_error(self, X):
        mu, _ = self.encode(X)       # Use mean for inference (no sampling)
        xhat  = self.decode(mu)
        return np.mean((X - xhat)**2, axis=1)


# Instantiate and load weights
mlp = MLP(DATA['mlp_dims'], DATA['mlp_drops'])
mlp.load(DATA['mlp_weights'])

vae = VAE()
vae.load(DATA['vae_weights'])


# ─── HELPER FUNCTIONS ────────────────────────────────────
def preprocess(raw: dict) -> np.ndarray:
    """Build feature vector from raw patient input dict."""
    v = raw  # shorthand

    # Engineered features
    vel_ratio  = v['peak_systolic_velocity'] / max(v['blood_flow_velocity'], 0.001)
    therm_st   = v['temperature_difference'] * v['cold_spot_area_percent']
    card_load  = v['heart_rate'] * v['pulse_amplitude']
    vasc_res   = v['resistive_index'] * v['pulse_transit_time']
    flow_eff   = v['blood_flow_velocity'] / max(v['heart_rate'], 0.001)

    vec = [
        v['peak_systolic_velocity'], v['resistive_index'],
        v['blood_flow_velocity'],    v['avg_temperature'],
        v['temperature_difference'], v['cold_spot_area_percent'],
        v['heart_rate'],             v['pulse_amplitude'],
        v['pulse_transit_time'],     v['hrv'],
        vel_ratio, therm_st, card_load, vasc_res, flow_eff
    ]

    X_raw = np.array([vec])
    X_sc  = (X_raw - SCALER_MEAN) / SCALER_STD
    return X_sc


def clinical_score(raw: dict) -> int:
    """Compute clinical score (0-12) from raw values."""
    criteria = [
        ('peak_systolic_velocity', 'above', 125,  2),
        ('resistive_index',        'above', 0.75, 2),
        ('blood_flow_velocity',    'below', 70,   2),
        ('cold_spot_area_percent', 'above', 12,   2),
        ('temperature_difference', 'above', 3.5,  1),
        ('pulse_transit_time',     'above', 0.28, 1),
        ('pulse_amplitude',        'below', 1.2,  1),
        ('hrv',                    'below', 40,   1),
    ]
    score = 0
    for feat, cond, thresh, w in criteria:
        v = raw.get(feat, 0)
        if cond == 'above' and v > thresh: score += w
        if cond == 'below' and v < thresh: score += w
    return score


def get_risk(prob: float, score: int) -> str:
    if score >= 8 or prob >= 0.75: return 'CRITICAL'
    if score >= 6 or prob >= 0.55: return 'HIGH'
    if score >= 4 or prob >= 0.35: return 'MODERATE'
    return 'LOW'


# ─── ROUTES ───────────────────────────────────────────────

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'running',
        'message': 'Blood Vessel Blockage Detection API',
        'endpoints': {
            'POST /predict': 'Get AI prediction for a patient',
            'GET  /health':  'Check server health'
        }
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'models_loaded': True,
        'mlp_accuracy':  f"{DATA['metrics']['mlp_test_acc']:.2%}",
        'vae_accuracy':  f"{DATA['metrics']['vae_test_acc']:.2%}",
        'ens_accuracy':  f"{DATA['metrics']['ens_test_acc']:.2%}",
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        body = request.get_json()
        if not body:
            return jsonify({'error': 'No JSON body received'}), 400

        # Validate required fields
        required = ['peak_systolic_velocity','resistive_index','blood_flow_velocity',
                    'avg_temperature','temperature_difference','cold_spot_area_percent',
                    'heart_rate','pulse_amplitude','pulse_transit_time','hrv']
        missing = [f for f in required if f not in body]
        if missing:
            return jsonify({'error': f'Missing fields: {missing}'}), 400

        # Cast all to float
        raw = {k: float(body[k]) for k in required}

        # Preprocess & predict
        X_sc = preprocess(raw)

        # MLP
        mlp_prob = float(mlp.predict_proba(X_sc)[0])
        mlp_pred = int(mlp_prob > 0.5)

        # VAE
        vae_err  = float(vae.reconstruction_error(X_sc)[0])
        vae_prob = float(min(vae_err / (VAE_THRESH * 2), 1.0))
        vae_pred = int(vae_err > VAE_THRESH)

        # Ensemble
        ens_prob = float(mlp_prob * 0.55 + vae_prob * 0.45)
        votes    = mlp_pred + vae_pred
        ens_pred = int(votes >= 1)

        # Clinical score
        score = clinical_score(raw)
        risk  = get_risk(ens_prob, score)

        return jsonify({
            'success': True,

            # Ensemble (final decision)
            'blockage':       ens_pred,
            'risk':           risk,
            'clinical_score': score,
            'ensemble_prob':  round(ens_prob, 4),
            'votes':          votes,

            # Individual models
            'mlp': {
                'prediction': mlp_pred,
                'probability': round(mlp_prob, 4),
            },
            'vae': {
                'prediction':          vae_pred,
                'probability':         round(vae_prob, 4),
                'reconstruction_error': round(vae_err, 6),
                'threshold':           round(VAE_THRESH, 6),
            },
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─── RUN ─────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*50)
    print("  Starting Flask API Server")
    print("  URL: http://localhost:5000")
    print("  Predict: POST http://localhost:5000/predict")
    print("  Health:  GET  http://localhost:5000/health")
    print("  Run train first: python train_and_save.py")
    print("="*50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
