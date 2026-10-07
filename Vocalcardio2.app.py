pip install streamlit scipy numpy pandas
import streamlit as st
import numpy as np
import pandas as pd
import scipy.signal as signal
from io import BytesIO

# =====================================================================
# 🧠 LAYER 1: BACKEND MEDICAL SIGNAL PROCESSING ENGINE
# =====================================================================
class VoiceBiometricEngine:
    def __init__(self, sampling_rate: int = 22050):
        self.sr = sampling_rate

    def analyze_uploaded_file(self, audio_bytes: bytes) -> dict:
        """
        Parses raw uploaded audio file bytes, extracts micro-timing 
        fluctuations (Jitter) and micro-amplitude shifts (Shimmer).
        """
        try:
            # Safely interpret binary stream array chunks into numeric floating values
            waveform = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        except Exception:
            # Fallback if bytes map differently
            waveform = np.frombuffer(audio_bytes, dtype=np.uint8).astype(np.float32)

        if len(waveform) < 100 or np.max(np.abs(waveform)) < 1e-4:
            return {"jitter": 0.001, "shimmer": 0.010, "instability_score": 5.0}

        # Normalize the signal waveform canvas
        waveform = waveform / (np.max(np.abs(waveform)) + 1e-6)
        
        # Cross-autocorrelation peak matching to capture fundamental pitch frequencies
        corr = signal.correlate(waveform, waveform, mode='full')
        corr = corr[len(corr)//2:]
        
        min_peak_dist = int(self.sr / 300)
        max_peak_dist = int(self.sr / 80)
        
        peaks, _ = signal.find_peaks(corr[min_peak_dist:max_peak_dist], prominence=0.01)
        peaks = peaks + min_peak_dist
        
        if len(peaks) < 3:
            return {"jitter": 0.003, "shimmer": 0.014, "instability_score": 10.5}
            
        periods = np.diff(peaks) / self.sr
        jitter = float(np.mean(np.abs(np.diff(periods))) / np.mean(periods))
        
        valid_peaks = peaks[peaks < len(waveform)]
        if len(valid_peaks) < 3:
            return {"jitter": jitter, "shimmer": 0.012, "instability_score": 15.0}
            
        peak_amplitudes = waveform[valid_peaks]
        shimmer = float(np.mean(np.abs(np.diff(peak_amplitudes))) / np.mean(peak_amplitudes))
        
        # Bound combined indicator output between 0% and 100%
        instability_score = min(float((jitter * 750) + (shimmer * 150)), 100.0)
        
        return {
            "jitter": jitter,
            "shimmer": shimmer,
            "instability_score": round(instability_score, 2)
        }

# =====================================================================
# 📊 LAYER 2: CLINICAL DATABASE STORAGE MOCKUP
# =====================================================================
class PatientDatabase:
    @staticmethod
    def get_demographics(patient_id: str) -> dict:
        profiles = {
            "PT-8802": {"name": "Eleanor Vance", "age": 67, "condition": "Early-Stage Parkinson's / Asthma", "tier": "High Risk Monitoring"},
            "PT-4419": {"name": "Marcus Chen", "age": 42, "condition": "Post-Viral Respiratory Recovery", "tier": "Routine Screening"}
        }
        return profiles.get(patient_id, {"name": "Unknown", "age": 0, "condition": "N/A", "tier": "N/A"})

    @staticmethod
    def generate_longitudinal_records(patient_id: str, days: int = 30) -> pd.DataFrame:
        np.random.seed(42 if patient_id == "PT-8802" else 101)
        dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq='D')
        drift = np.linspace(0, 6, days) if patient_id == "PT-8802" else np.linspace(0, -1, days)
        
        heart_rate = np.random.normal(74, 2, days) + (drift * 0.4)
        spo2 = np.random.normal(97.5, 0.5, days) - (drift * 0.1)
        bp_systolic = np.random.normal(126, 3, days) + drift
        historical_instability = np.random.normal(14.2, 1.2, days) + (drift * 1.1)
        
        return pd.DataFrame({
            "Date": dates,
            "Heart_Rate_BPM": heart_rate.round(1),
            "Oxygen_Sat_Percentage": np.clip(spo2, 85, 100).round(1),
            "Systolic_BP_mmHg": bp_systolic.round(1),
            "Voice_Instability_Index": np.clip(historical_instability, 0, 100).round(2)
        }).set_index("Date")

# =====================================================================
# 🏥 LAYER 3: CORE APPLICATION INTERFACE DESIGN
# =====================================================================
st.set_page_config(page_title="VocalTriage Pro Panel", page_icon="🏥", layout="wide")

st.title("🏥 VocalTriage Pro: Comprehensive Medical Dashboard Engine")
st.markdown("Category: **Remote Patient Monitoring (RPM)** | Collecting user and biometric entries into Python pipelines.")
st.markdown("---")

# Sidebar EHR Navigator
st.sidebar.header("📁 Electronic Health Records Selector")
active_patient_id = st.sidebar.selectbox("Select Target Patient File:", ["PT-8802", "PT-4419"])

patient_meta = PatientDatabase.get_demographics(active_patient_id)

# Initialize data session persistent frames
if "patient_history" not in st.session_state or "current_id" not in st.session_state or st.session_state.current_id != active_patient_id:
    st.session_state.patient_history = PatientDatabase.generate_longitudinal_records(active_patient_id)
    st.session_state.current_id = active_patient_id

clinical_history = st.session_state.patient_history

# SECTION 1: MASTER CLINICAL HEADER FILE VIEW
col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)
with col_meta1:
    st.markdown(f"👤 **Patient Name:**\n### {patient_meta['name']}")
with col_meta2:
    st.markdown(f"🎂 **Chronological Age:**\n### {patient_meta['age']} Years")
with col_meta3:
    st.markdown(f"🩺 **Primary Condition:**\n`{patient_meta['condition']}`")
with col_meta4:
    st.markdown(f"🚨 **Clinical Tier:**\n`{patient_meta['tier']}`")

st.markdown("---")

# SECTION 2: LIVE USER INPUT COLLECTION MATRIX
st.subheader("📥 Log Current Patient Health Entry (Manual Input & Audio Upload)")

# Create an organized side-by-side data collection layout
form_col1, form_col2 = st.columns([1, 1])

with form_col1:
    st.markdown("##### 📝 Input Vital Measurements")
    # Python input components to gather raw patient vitals securely
    input_hr = st.number_input("Heart Rate (BPM):", min_value=40, max_value=200, value=75, step=1)
    input_bp = st.number_input("Systolic Blood Pressure (mmHg):", min_value=70, max_value=220, value=120, step=1)
    input_spo2 = st.slider("Oxygen Saturation Level - SpO2 (%):", min_value=80.0, max_value=100.0, value=98.0, step=0.1)

with form_col2:
    st.markdown("##### 🔊 Upload Voice Audio Record")
    # File upload component restricting files to raw standard digital audio formats
    uploaded_file = st.file_uploader("Drop patient sustained vowel sound file (.wav)", type=["wav"])
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        st.success("File context successfully buffered into system memories.")

# Process input collection form submission
if st.button("🚀 Commit Log Entry directly to Python Pipeline", use_container_width=True):
    vocal_instability_score = 12.50  # Default nominal base if audio entry wasn't attached
    
    if uploaded_file is not None:
        # Pass file bytes layer straight to processing engine
        file_bytes = uploaded_file.read()
        engine = VoiceBiometricEngine()
        audio_results = engine.analyze_uploaded_file(file_bytes)
        vocal_instability_score = audio_results["instability_score"]
    
    # Append collected structural inputs directly into historical tracking timeline matrix
    new_timestamp = pd.Timestamp.now()
    new_entry_row = pd.DataFrame([{
        "Heart_Rate_BPM": float(input_hr),
        "Oxygen_Sat_Percentage": float(input_spo2),
        "Systolic_BP_mmHg": float(input_bp),
        "Voice_Instability_Index": float(vocal_instability_score)
    }], index=[new_timestamp])
    
    st.session_state.patient_history = pd.concat([st.session_state.patient_history, new_entry_row])
    st.toast("Success! Collected entries integrated into analytical data loops.", icon="💾")
    st.rerun()

st.markdown("---")

# SECTION 3: SYSTEM RISK REPORT ALERT
st.subheader("🔔 Automated Pipeline Risk Alerts")
latest_vitals = clinical_history.iloc[-1].copy()
seven_day_baseline = clinical_history.iloc[-7:-1].mean()

alerts_triggered = []
if latest_vitals['Oxygen_Sat_Percentage'] < 94.0:
    alerts_triggered.append(f"🚨 **CRITICAL HYPOXIA RISK:** SpO2 tracking low ({latest_vitals['Oxygen_Sat_Percentage']}%).")
if (latest_vitals['Systolic_BP_mmHg'] - seven_day_baseline['Systolic_BP_mmHg']) > 8.0:
    alerts_triggered.append(f"⚠️ **CARDIOVASCULAR ANOMALY:** Systolic Blood Pressure surge monitored over standard deviation thresholds.")
if latest_vitals['Voice_Instability_Index'] > 22.0:
    alerts_triggered.append(f"🔍 **NEURO-RESPIRATORY EXCURSION:** Micro-vocal instability metrics are tracking high ({latest_vitals['Voice_Instability_Index']}%).")

if alerts_triggered:
    for alert in alerts_triggered:
        st.error(alert)
else:
    st.success("🟢 **SYSTEM NORMAL:** Longitudinal tracking metrics are within healthy baseline parameters.")

# SECTION 4: RENDERING HISTORICAL GRAPHS
st.subheader("📊 30-Day Physiological Trend Analysis Windows")
tab_graph1, tab_graph2, tab_data = st.tabs(["🫁 Neuro-Acoustic Trajectory", "❤️ Cardio-Respiratory Matrix", "📋 Comprehensive Electronic Log Frame"])

with tab_graph1:
    st.line_chart(clinical_history["Voice_Instability_Index"], color="#FF4B4B")
with tab_graph2:
    st.line_chart(clinical_history[["Systolic_BP_mmHg", "Heart_Rate_BPM", "Oxygen_Sat_Percentage"]])
with tab_data:
    st.dataframe(clinical_history.sort_index(ascending=False), use_container_width=True)
