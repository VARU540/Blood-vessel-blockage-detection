"""
train_and_save.py
Run this ONCE to train all models and save weights. 
Usage: python3 train_and_save.py
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

print("=" * 55)
print("  Blood Vessel Blockage — Model Training & Save")
print("=" * 55)

# ─── STEP 1: DATA PREP ───────────────────────────────────
print("\n[1/5] Loading & preparing data...")

df = pd.read_csv('final.csv')
df = df.fillna(df.median(numeric_only=True))
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    df[col] = df[col].clip(lower=Q1 - 1.5*(Q3-Q1), upper=Q3 + 1.5*(Q3-Q1))

df['velocity_ratio']      = df['peak_systolic_velocity'] / df['blood_flow_velocity'].replace(0, 0.001)
df['thermal_stress']      = df['temperature_difference'] * df['cold_spot_area_percent']
df['cardiac_load']        = df['heart_rate'] * df['pulse_amplitude']
df['vascular_resistance'] = df['resistive_index'] * df['pulse_transit_time']
df['flow_efficiency']     = df['blood_flow_velocity'] / df['heart_rate'].replace(0, 0.001)

ORIG_COLS = ['peak_systolic_velocity','resistive_index','blood_flow_velocity',
             'avg_temperature','temperature_difference','cold_spot_area_percent',
             'heart_rate','pulse_amplitude','pulse_transit_time','hrv']
ENG_COLS  = ['velocity_ratio','thermal_stress','cardiac_load','vascular_resistance','flow_efficiency']
ALL_COLS  = ORIG_COLS + ENG_COLS

X = df[ALL_COLS].values.astype(np.float64)

# Clinical scoring labels
thresholds = {
    'peak_systolic_velocity': ('above', 125,  2),
    'resistive_index':        ('above', 0.75, 2),
    'blood_flow_velocity':    ('below', 70,   2),
    'cold_spot_area_percent': ('above', 12,   2),
    'temperature_difference': ('above', 3.5,  1),
    'pulse_transit_time':     ('above', 0.28, 1),
    'pulse_amplitude':        ('below', 1.2,  1),
    'hrv':                    ('below', 40,   1),
}
score = np.zeros(len(df))
for feat, (cond, thresh, w) in thresholds.items():
    ci = ALL_COLS.index(feat)
    score += (X[:, ci] > thresh if cond == 'above' else X[:, ci] < thresh).astype(int) * w
y = (score >= 4).astype(int)

scaler = StandardScaler()
X_s = scaler.fit_transform(X)

X_train, X_temp, y_train, y_temp = train_test_split(X_s, y, test_size=0.30, random_state=42, stratify=y)
X_val,   X_test,  y_val,  y_test  = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
X_train_normal = X_train[y_train == 0]

print(f"   Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
print(f"   Blockage rate: {y.mean():.1%}")

# ─── MODEL CLASSES ────────────────────────────────────────

class MLP:
    def __init__(self, dims, drops, lr=0.001):
        np.random.seed(42)
        self.dims = dims; self.drops = drops; self.n = len(dims) - 1
        self.training = False
        self.W = [np.random.randn(dims[i], dims[i+1]) * np.sqrt(2./dims[i]) for i in range(self.n)]
        self.b = [np.zeros((1, dims[i+1])) for i in range(self.n)]
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0; self.lr = lr

    def relu(self, x):    return np.maximum(0, x)
    def drelu(self, x):   return (x > 0).astype(float)
    def sigmoid(self, x): return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def dropout(self, x, rate):
        if not self.training or rate == 0: return x, np.ones_like(x)
        mask = (np.random.rand(*x.shape) > rate).astype(float)
        return x * mask / (1 - rate), mask / (1 - rate)

    def forward(self, X):
        self.cache = []; h = X
        for i in range(self.n):
            z = h @ self.W[i] + self.b[i]
            if i < self.n - 1:
                a = self.relu(z)
                dr = self.drops[i] if i < len(self.drops) else 0
                ad, mask = self.dropout(a, dr)
                self.cache.append((h, z, a, mask)); h = ad
            else:
                a = self.sigmoid(z)
                self.cache.append((h, z, a, None)); h = a
        return h

    def bce(self, yt, yp):
        eps = 1e-9; yp = np.clip(yp, eps, 1 - eps)
        return -np.mean(yt * np.log(yp) + (1 - yt) * np.log(1 - yp))

    def backward(self, X, yt, yp):
        bsz = X.shape[0]; gW = []; gb = []
        delta = (yp - yt.reshape(-1, 1)) / bsz
        for i in reversed(range(self.n)):
            h, z, a, mask = self.cache[i]
            dW = h.T @ delta; db = delta.sum(axis=0, keepdims=True)
            gW.insert(0, dW); gb.insert(0, db)
            if i > 0:
                delta = (delta @ self.W[i].T) * self.drelu(self.cache[i-1][1])
                if self.cache[i-1][3] is not None: delta = delta * self.cache[i-1][3]
        return gW, gb

    def adam(self, gW, gb, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for i in range(self.n):
            self.mW[i] = b1*self.mW[i] + (1-b1)*gW[i]
            self.vW[i] = b2*self.vW[i] + (1-b2)*gW[i]**2
            self.W[i] -= self.lr * (self.mW[i]/(1-b1**self.t)) / (np.sqrt(self.vW[i]/(1-b2**self.t)) + eps)
            self.mb[i] = b1*self.mb[i] + (1-b1)*gb[i]
            self.vb[i] = b2*self.vb[i] + (1-b2)*gb[i]**2
            self.b[i] -= self.lr * (self.mb[i]/(1-b1**self.t)) / (np.sqrt(self.vb[i]/(1-b2**self.t)) + eps)

    def predict_proba(self, X):
        self.training = False
        return self.forward(X).flatten()

    def get_weights(self):
        return {'W': [w.tolist() for w in self.W], 'b': [b.tolist() for b in self.b]}

    def set_weights(self, data):
        self.W = [np.array(w) for w in data['W']]
        self.b = [np.array(b) for b in data['b']]


class VAE:
    def __init__(self, input_dim=15, hidden_dim=64, latent_dim=3):
        np.random.seed(42)
        self.input_dim = input_dim; self.hidden_dim = hidden_dim; self.latent_dim = latent_dim
        self.training = False
        self.We1 = np.random.randn(input_dim, hidden_dim)        * np.sqrt(2./input_dim)
        self.be1 = np.zeros((1, hidden_dim))
        self.We2 = np.random.randn(hidden_dim, hidden_dim//2)    * np.sqrt(2./hidden_dim)
        self.be2 = np.zeros((1, hidden_dim//2))
        self.Wmu = np.random.randn(hidden_dim//2, latent_dim)    * np.sqrt(2./(hidden_dim//2))
        self.bmu = np.zeros((1, latent_dim))
        self.Wlv = np.random.randn(hidden_dim//2, latent_dim)    * np.sqrt(2./(hidden_dim//2))
        self.blv = np.zeros((1, latent_dim))
        self.Wd1 = np.random.randn(latent_dim, hidden_dim//2)    * np.sqrt(2./latent_dim)
        self.bd1 = np.zeros((1, hidden_dim//2))
        self.Wd2 = np.random.randn(hidden_dim//2, hidden_dim)    * np.sqrt(2./(hidden_dim//2))
        self.bd2 = np.zeros((1, hidden_dim))
        self.Wd3 = np.random.randn(hidden_dim, input_dim)        * np.sqrt(2./hidden_dim)
        self.bd3 = np.zeros((1, input_dim))
        wnames = ['We1','be1','We2','be2','Wmu','bmu','Wlv','blv','Wd1','bd1','Wd2','bd2','Wd3','bd3']
        self.t = 0
        for w in wnames:
            setattr(self, f'm_{w}', np.zeros_like(getattr(self, w)))
            setattr(self, f'v_{w}', np.zeros_like(getattr(self, w)))

    def relu(self, x):  return np.maximum(0, x)
    def drelu(self, x): return (x > 0).astype(float)

    def encode(self, X):
        self.enc_in = X
        z1 = X @ self.We1 + self.be1; h1 = self.relu(z1)
        z2 = h1 @ self.We2 + self.be2; h2 = self.relu(z2)
        mu = h2 @ self.Wmu + self.bmu
        logvar = h2 @ self.Wlv + self.blv
        self.z1=z1; self.h1=h1; self.z2=z2; self.h2=h2
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            eps = np.random.randn(*mu.shape)
            return mu + np.exp(0.5 * logvar) * eps
        return mu

    def decode(self, z):
        self.dec_in = z
        dz1 = z @ self.Wd1 + self.bd1; dh1 = self.relu(dz1)
        dz2 = dh1 @ self.Wd2 + self.bd2; dh2 = self.relu(dz2)
        out = dh2 @ self.Wd3 + self.bd3
        self.dz1=dz1; self.dh1=dh1; self.dz2=dz2; self.dh2=dh2
        return out

    def forward(self, X):
        self.mu, self.logvar = self.encode(X)
        self.z    = self.reparameterize(self.mu, self.logvar)
        self.xhat = self.decode(self.z)
        return self.xhat

    def vae_loss(self, X, xhat, mu, logvar, beta=1.0):
        recon = -np.mean((X - xhat)**2)
        kl    = -0.5 * np.mean(1 + logvar - mu**2 - np.exp(logvar))
        return recon + beta*kl, -recon, kl

    def backward(self, X, beta=1.0):
        B = X.shape[0]
        d_out = 2*(self.xhat - X) / B
        self.dWd3 = self.dh2.T @ d_out; self.dbd3 = d_out.sum(0, keepdims=True)
        d = d_out @ self.Wd3.T * self.drelu(self.dz2)
        self.dWd2 = self.dh1.T @ d; self.dbd2 = d.sum(0, keepdims=True)
        d = d @ self.Wd2.T * self.drelu(self.dz1)
        self.dWd1 = self.dec_in.T @ d; self.dbd1 = d.sum(0, keepdims=True)
        dz = d @ self.Wd1.T
        dmu = dz + beta * self.mu / B
        dlogvar = dz * (self.z - self.mu) * 0.5 + beta*0.5*(np.exp(self.logvar) - 1)/B
        self.dWmu = self.h2.T @ dmu; self.dbmu = dmu.sum(0, keepdims=True)
        self.dWlv = self.h2.T @ dlogvar; self.dblv = dlogvar.sum(0, keepdims=True)
        dh2 = (dmu @ self.Wmu.T + dlogvar @ self.Wlv.T) * self.drelu(self.z2)
        self.dWe2 = self.h1.T @ dh2; self.dbe2 = dh2.sum(0, keepdims=True)
        dh1 = dh2 @ self.We2.T * self.drelu(self.z1)
        self.dWe1 = self.enc_in.T @ dh1; self.dbe1 = dh1.sum(0, keepdims=True)

    def adam_step(self, b1=0.9, b2=0.999, eps=1e-8, lr=0.001):
        self.t += 1
        wnames = ['We1','be1','We2','be2','Wmu','bmu','Wlv','blv','Wd1','bd1','Wd2','bd2','Wd3','bd3']
        for wn in wnames:
            g = getattr(self, f'd{wn}')
            m = b1*getattr(self, f'm_{wn}') + (1-b1)*g
            v = b2*getattr(self, f'v_{wn}') + (1-b2)*g**2
            setattr(self, f'm_{wn}', m); setattr(self, f'v_{wn}', v)
            w = getattr(self, wn)
            setattr(self, wn, w - lr*(m/(1-b1**self.t)) / (np.sqrt(v/(1-b2**self.t)) + eps))

    def reconstruction_error(self, X):
        self.training = False
        xhat = self.forward(X)
        return np.mean((X - xhat)**2, axis=1)

    def get_weights(self):
        wnames = ['We1','be1','We2','be2','Wmu','bmu','Wlv','blv','Wd1','bd1','Wd2','bd2','Wd3','bd3']
        return {w: getattr(self, w).tolist() for w in wnames}

    def set_weights(self, data):
        for k, v in data.items():
            setattr(self, k, np.array(v))


# ─── STEP 2: TRAIN MLP ────────────────────────────────────
print("\n[2/5] Training MLP (70 epochs)...")
mlp = MLP([15, 128, 64, 32, 1], [0.3, 0.2, 0.0])
best_val_loss = float('inf')
best_weights  = None

for ep in range(70):
    mlp.training = True
    idx = np.random.permutation(len(X_train))
    Xs, ys = X_train[idx], y_train[idx]
    for s in range(0, len(X_train), 32):
        Xb = Xs[s:s+32]; yb = ys[s:s+32]
        yh = mlp.forward(Xb)
        gW, gb = mlp.backward(Xb, yb, yh)
        mlp.adam(gW, gb)
    mlp.training = False
    val_preds = mlp.predict_proba(X_val)
    val_loss  = mlp.bce(y_val, val_preds)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_weights  = mlp.get_weights()
    if (ep+1) % 10 == 0:
        val_acc = ((val_preds > 0.5) == y_val).mean()
        print(f"   Epoch {ep+1:3d} | val_loss={val_loss:.4f} | val_acc={val_acc:.1%}")

mlp.set_weights(best_weights)
test_preds = mlp.predict_proba(X_test)
test_acc   = ((test_preds > 0.5) == y_test).mean()
print(f"   ✅ MLP Test Accuracy: {test_acc:.2%}")

# ─── STEP 3: TRAIN VAE ────────────────────────────────────
print("\n[3/5] Training VAE (90 epochs on normal patients only)...")
vae = VAE(15, 64, 3)
best_vae_loss = float('inf')
best_vae_weights = None

for ep in range(90):
    vae.training = True
    idx = np.random.permutation(len(X_train_normal))
    Xs  = X_train_normal[idx]
    ep_loss = 0
    for s in range(0, len(X_train_normal), 32):
        Xb   = Xs[s:s+32]
        xhat = vae.forward(Xb)
        loss, _, _ = vae.vae_loss(Xb, xhat, vae.mu, vae.logvar)
        vae.backward(Xb); vae.adam_step()
        ep_loss += loss
    if ep_loss < best_vae_loss:
        best_vae_loss    = ep_loss
        best_vae_weights = vae.get_weights()
    if (ep+1) % 15 == 0:
        print(f"   Epoch {ep+1:3d} | loss={ep_loss:.4f}")

vae.set_weights(best_vae_weights)
vae.training = False

# VAE threshold
all_normal_X = X_s[y == 0]
normal_errors = vae.reconstruction_error(all_normal_X)
vae_threshold = float(normal_errors.mean() + 2 * normal_errors.std())

vae_preds = (vae.reconstruction_error(X_test) > vae_threshold).astype(int)
vae_acc   = (vae_preds == y_test).mean()
print(f"   ✅ VAE Test Accuracy: {vae_acc:.2%} | Threshold: {vae_threshold:.6f}")

# ─── STEP 4: EVALUATE ENSEMBLE ────────────────────────────
print("\n[4/5] Evaluating ensemble...")
mlp_p = mlp.predict_proba(X_test)
vae_p = np.clip(vae.reconstruction_error(X_test) / (vae_threshold * 2), 0, 1)
ens_p = mlp_p * 0.55 + vae_p * 0.45
ens_preds = (ens_p > 0.5).astype(int)
ens_acc   = (ens_preds == y_test).mean()
print(f"   ✅ Ensemble Test Accuracy: {ens_acc:.2%}")

# ─── STEP 5: SAVE EVERYTHING ──────────────────────────────
print("\n[5/5] Saving model weights & scaler...")
os.makedirs('model_weights', exist_ok=True)

save_data = {
    'mlp_weights':   mlp.get_weights(),
    'mlp_dims':      [15, 128, 64, 32, 1],
    'mlp_drops':     [0.3, 0.2, 0.0],
    'vae_weights':   vae.get_weights(),
    'vae_threshold': vae_threshold,
    'scaler_mean':   scaler.mean_.tolist(),
    'scaler_std':    scaler.scale_.tolist(),
    'all_cols':      ALL_COLS,
    'orig_cols':     ORIG_COLS,
    'metrics': 
        'mlp_test_acc': float(test_acc),
        'vae_test_acc': float(vae_acc),
        'ens_test_acc': float(ens_acc),
    }
}

with open('model_weights/weights.pkl', 'wb') as f:
    pickle.dump(save_data, f)

print(f"   ✅ Saved: model_weights/weights.pkl")
print("\n" + "=" * 55)
print("  TRAINING COMPLETE!")
print(f"  MLP:      {test_acc:.2%}")
print(f"  VAE:      {vae_acc:.2%}")
print(f"  Ensemble: {ens_acc:.2%}")
print("=" * 55)
print("\nNext step: python3 app.py")
