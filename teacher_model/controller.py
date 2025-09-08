# controller.py (Final, Robust Version)

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from loguru import logger
from rich.traceback import install
import scienceplots

# --- Setup ---
plt.style.use(['science', 'no-latex'])
logger.remove()
logger.add("logs/controller_simulation.log", rotation="5 MB", level="INFO")
logger.add(lambda msg: print(msg, end=""), format="{message}", level="INFO")
install(show_locals=False, word_wrap=True, width=120)

# --- Load controller artifacts ---
logger.info("Loading controller artifacts...")
ideal_curves = joblib.load('ideal_curves.joblib')
median_curves = joblib.load('median_curves.joblib')
model = joblib.load('teacher_model/models/xgb_clutch_profile_model.joblib')
feature_scaler = joblib.load('teacher_model/models/curve_feature_scaler.joblib')

# --- Controller & Simulation Settings ---
BITE_POINT = 25
RELEASE_THRESH = 75
CLUTCH_ENGAGED = 0
CLUTCH_DISENGAGED = 100
DEFAULT_FALLBACK_CLUSTER = 2
MAX_CLUTCH_SPEED = 300.0 

# --- Load and Prepare Simulation Data ---
logger.info("Loading and preparing simulation data...")
df = pd.read_csv(r'data\man_raw\gear_1_manual_params.csv')

df.rename(columns={
    'Clutch housing Temp': 'tmpCltActTC',
    'EngTrq_Cval_PT': 'EngTrq',
    'EngSpd_Cval_PT': 'EngSpd',
    'VehSpd_Cval_PT': 'Calc_VehSpd'
}, inplace=True)

if 'CurrGr' not in df.columns:
    df['CurrGr'] = 1

# --- Simulation Loop ---
logger.info("Starting controller simulation...")
simulated_clutch_pos = []
in_modulation_zone = False
predicted_cluster_for_event = None
last_clutch_pos = CLUTCH_ENGAGED
last_time = df['Time'].iloc[0]

for idx, row in df.iterrows():
    # --- State & Time Update ---
    pedal_pos = row['Clutch_Pedal_Travel']
    current_time = row['Time']
    dt = current_time - last_time
    target_clutch_pos = CLUTCH_ENGAGED # Default command

    # --- Event Detection Logic ---
    if BITE_POINT < pedal_pos < RELEASE_THRESH and not in_modulation_zone:
        in_modulation_zone = True
        logger.info(f"Event started at Time: {current_time:.2f}s. Predicting profile...")
        
        initial_conditions = {
            'EngTrq_mean': row['EngTrq'], 'EngTrq_std': 0, 'EngTrq_q25': row['EngTrq'], 'EngTrq_q75': row['EngTrq'], 'EngTrq_first': row['EngTrq'],
            'EngSpd_mean': row['EngSpd'], 'EngSpd_std': 0, 'EngSpd_q25': row['EngSpd'], 'EngSpd_q75': row['EngSpd'], 'EngSpd_first': row['EngSpd'],
            'tmpCltActTC_mean': row['tmpCltActTC'], 'tmpCltActTC_std': 0, 'tmpCltActTC_q25': row['tmpCltActTC'], 'tmpCltActTC_q75': row['tmpCltActTC'], 'tmpCltActTC_first': row['tmpCltActTC'],
            'CurrGr_mean': row['CurrGr'], 'CurrGr_std': 0, 'CurrGr_q25': row['CurrGr'], 'CurrGr_q75': row['CurrGr'], 'CurrGr_first': row['CurrGr'],
            'Calc_VehSpd_mean': row['Calc_VehSpd'], 'Calc_VehSpd_std': 0, 'Calc_VehSpd_q25': row['Calc_VehSpd'], 'Calc_VehSpd_q75': row['Calc_VehSpd'], 'Calc_VehSpd_first': row['Calc_VehSpd'],
            'Duration': 1.0 
        }
        
        feature_df = pd.DataFrame([initial_conditions], columns=model.feature_names_in_)
        predicted_cluster_for_event = model.predict(feature_df)[0]
        logger.info(f"Predicted Profile for this event: Cluster {predicted_cluster_for_event}")

    # --- Robust Controller Actuation Logic ---
    if pedal_pos <= BITE_POINT:
        target_clutch_pos = CLUTCH_ENGAGED
        in_modulation_zone = False
    elif pedal_pos >= RELEASE_THRESH:
        target_clutch_pos = CLUTCH_DISENGAGED
        in_modulation_zone = False
    elif in_modulation_zone and predicted_cluster_for_event is not None:
        # Use ideal curve with a safe fallback to a median curve
        reference_curve = ideal_curves.get(predicted_cluster_for_event, median_curves.get(DEFAULT_FALLBACK_CLUSTER))
        
        region_span = RELEASE_THRESH - BITE_POINT
        mod_progress = (pedal_pos - BITE_POINT) / region_span
        
        curve_indices = np.linspace(0, 1, len(reference_curve))
        target_clutch_pos = np.interp(mod_progress, curve_indices, reference_curve)
    
    # --- Rate Limiter (Physical Actuator Constraint) ---
    max_change = MAX_CLUTCH_SPEED * dt
    # Move towards the target, but no faster than the max allowed speed
    if target_clutch_pos > last_clutch_pos:
        current_clutch_pos = min(target_clutch_pos, last_clutch_pos + max_change)
    else:
        current_clutch_pos = max(target_clutch_pos, last_clutch_pos - max_change)

    simulated_clutch_pos.append(current_clutch_pos)
    # Update state for the next iteration
    last_clutch_pos = current_clutch_pos
    last_time = current_time

logger.info("Simulation complete. Generating plot...")

# --- Plot the result ---
plt.figure(figsize=(14, 7))
plt.plot(df['Time'], simulated_clutch_pos, label='Simulated Controller Output', linewidth=2.5, color='royalblue')
plt.plot(df['Time'], df['Clutch_Pedal_Travel'], '--', label='Driver Pedal Input', alpha=0.7, color='gray')
plt.xlabel('Time (s)')
plt.ylabel('Clutch Position (%)')
plt.title('Simulated Clutch Controller with Rate Limiting')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
