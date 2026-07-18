import nest_asyncio
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from scipy.signal import butter, lfilter
from collections import deque

# --- 1. BIOMETRIC PIPELINE ENGINES ---

class SynapseSignalProcessor:
    def __init__(self, sampling_rate=250):
        self.sampling_rate = sampling_rate

    def _butter_bandpass(self, lowcut=1.0, highcut=40.0, order=4):
        nyq = 0.5 * self.sampling_rate
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def clean_and_extract_features(self, raw_data):
        b, a = self._butter_bandpass()
        filtered_data = lfilter(b, a, raw_data)
        
        if np.max(np.abs(filtered_data)) > 150.0:
            return None
        
        window = np.hanning(len(filtered_data))
        windowed_data = filtered_data * window
        
        fft_vals = np.abs(np.fft.rfft(windowed_data))
        fft_freqs = np.fft.rfftfreq(len(windowed_data), 1.0 / self.sampling_rate)
        
        alpha_mask = (fft_freqs >= 8) & (fft_freqs <= 12)
        beta_mask = (fft_freqs > 12) & (fft_freqs <= 30)
        
        alpha_power = np.sum(fft_vals[alpha_mask])
        beta_power = np.sum(fft_vals[beta_mask])
        
        if alpha_power == 0: 
            alpha_power = 0.001
        
        return {
            "Alpha_Power": alpha_power,
            "Beta_Power": beta_power,
            "Beta_Alpha_Ratio": beta_power / alpha_power
        }


class CGENEngine:
    def __init__(self, baseline_window=20):
        self.assist_level = 0
        self.baseline_history = deque(maxlen=baseline_window)

    def evaluate_and_intervene(self, features):
        if features is None:
            print("[CGEN Warning] Telemetry chunk discarded: High physical artifact noise detected.")
            return

        current_ratio = features["Beta_Alpha_Ratio"]
        
        if len(self.baseline_history) < 5:
            self.baseline_history.append(current_ratio)
            print(f"[CGEN Calibrating] Gathering user baseline... Current Ratio: {current_ratio:.2f}")
            return

        history_arr = np.array(self.baseline_history)
        mean_baseline = np.mean(history_arr)
        std_baseline = np.std(history_arr) if np.std(history_arr) > 0.001 else 0.01
        
        z_score = (current_ratio - mean_baseline) / std_baseline
        self.baseline_history.append(current_ratio)
        
        print(f"[Synapse Telemetry] Ratio: {current_ratio:.2f} | Baseline Avg: {mean_baseline:.2f} | Deviation: {z_score:.2f}")

        if z_score > 1.5:
            print("\n[CGEN Triggered] Elevated cognitive stress detected relative to personal baseline!")
            self._deploy_cognitive_assist()
        elif z_score < -0.5:
            print("\n[CGEN Triggered] User baseline indicates a calm, focused flow state.")
            self._lock_down_environment()
        else:
            print("\n[CGEN State] Cognitive load stable within nominal baseline bounds.")

    def _deploy_cognitive_assist(self):
        self.assist_level += 1
        print("[CGEN ACTION] Suppressing desktop alerts & simplifying interface clutter.")

    def _lock_down_environment(self):
        self.assist_level = 0
        print("[CGEN ACTION] Dedicating peak processing resources to active workspace.")


# --- 2. WEB API BACKEND ---

app = FastAPI(title="CGEN-NeuroSync API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

processor = SynapseSignalProcessor(sampling_rate=250)
cgen = CGENEngine()

class EEGPayload(BaseModel):
    raw_signals: list[float]

@app.get("/")
def home():
    return {"status": "CGEN-NeuroSync Online", "version": "1.0.0-Beta"}

@app.post("/analyze")
async def analyze_stream(payload: EEGPayload):
    features = processor.clean_and_extract_features(np.array(payload.raw_signals))
    
    if features is None:
        return {
            "status": "discarded",
            "message": "High noise artifact detected",
            "assist_level": cgen.assist_level
        }
    
    cgen.evaluate_and_intervene(features)
    
    return {
        "status": "processed",
        "ratio": float(features["Beta_Alpha_Ratio"]),
        "alpha_power": float(features["Alpha_Power"]),
        "beta_power": float(features["Beta_Power"]),
        "assist_level": cgen.assist_level
    }

if __name__ == "__main__":
    print("\n[SYSTEM] Starting CGEN-NeuroSync Web Server on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)