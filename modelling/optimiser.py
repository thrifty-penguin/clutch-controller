# optimize_and_plot.py

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from loguru import logger
from rich.traceback import install
import scienceplots

def estimate_driveline_jerk(event_df: pd.DataFrame) -> float:
    """
    Estimates the total driveline jerk for a single engagement event based on
    the rate of change of transmitted torque. This is a more physically
    accurate measure of shift quality.

    Args:
        event_df: The full, un-aggregated DataFrame for a single event. 
                  It must contain 'EngTrq', 'ClutchCval', and 'EventTime'.

    Returns:
        The total integrated jerk for the event.
    """
    # Ensure the dataframe is sorted by time for accurate gradient calculation
    event_df = event_df.sort_values(by='EventTime')
    
    # Model the torque transmitted through the clutch at each timestep.
    # Assumes a linear relationship between clutch position and torque capacity.
    # ClutchCval = 100 (disengaged) -> 0% torque
    # ClutchCval = 0 (engaged) -> 100% torque
    transmitted_torque = event_df['EngTrq'].values * (1 - event_df['ClutchCval'].values / 100.0)
    
    # Jerk is proportional to the rate of change of torque.
    # We calculate the derivative of transmitted torque with respect to time.
    torque_derivative = np.gradient(transmitted_torque, event_df['EventTime'].values)
    
    # The total jerk for the event is the sum of the absolute values of these changes.
    total_jerk = np.sum(np.abs(torque_derivative))
    
    return total_jerk

def estimate_wear_from_event(event_df: pd.DataFrame, gear_ratios: dict, final_drive: float, wheel_radius_m: float) -> float:
    gear = event_df['CurrGr'].iloc[0]
    ratio = gear_ratios.get(gear, 1.0) * final_drive
    wheel_ang_vel = (event_df['Calc_VehSpd'] * (1000/3600)) / wheel_radius_m
    input_shaft_speed = wheel_ang_vel * ratio
    engine_speed = event_df['EngSpd'] * (np.pi / 30)
    slip_speed = np.abs(engine_speed - input_shaft_speed)
    power_loss = event_df['EngTrq'] * slip_speed
    total_energy = np.trapezoid(power_loss, x=event_df['EventTime'])
    return total_energy

def main():
    TEST_SET_PATH = r'data\amt_train_test\g90amt_testset.csv'
    TRAIN_SET_PATH = r'data\amt_train_test\g90amt_trainset.csv'
    MODEL_PATH = r'teacher_model\models/xgb_clutch_profile_model.joblib'
    SCALER_PATH = r'teacher_model\models/curve_feature_scaler.joblib'
    GEAR_RATIOS = {1: 6.696, 2: 3.806, 3: 2.289, 4: 1.480, 5: 1.000, 6: 0.728, -1: 13.862}
    FINAL_DRIVE_RATIO = 3.45 
    WHEEL_RADIUS_M = 0.35
    JERK_WEIGHT = 0.6
    WEAR_WEIGHT = 0.4

    logger.info("Loading model, scaler, and test data...")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    df1 = pd.read_csv(TEST_SET_PATH)

    df2 = pd.read_csv(TRAIN_SET_PATH)

    df_test = pd.concat([df2, df1], ignore_index=True)

    from modelling.train_model import feature_vectors, aggregator
    event_vectors = feature_vectors(df_test, 'EngagementEvents', 'ClutchCval', 'EventTime')
    event_params = aggregator(df_test, 'EngagementEvents', ['EngTrq', 'EngSpd', 'tmpCltActTC', 'CurrGr', 'Calc_VehSpd'], 'EventTime')

    logger.info("Predicting cluster profiles for test set...")
    event_params.dropna(inplace=True)
    predictions = model.predict(event_params)
    event_params['predicted_cluster'] = predictions

    logger.info("Calculating Jerk and Wear KPIs for each event...")
    costs = []
    for event_id in event_params.index:
        if event_id not in event_vectors: continue
        curve = event_vectors[event_id]

        event_df = df_test[df_test['EngagementEvents'] == event_id]
        wear_cost = estimate_wear_from_event(event_df, GEAR_RATIOS, FINAL_DRIVE_RATIO, WHEEL_RADIUS_M)
        jerk_cost = estimate_driveline_jerk(event_df)
        costs.append({'event_id': event_id, 'jerk_cost': jerk_cost, 'wear_cost': wear_cost})
    cost_df = pd.DataFrame(costs).set_index('event_id')

    cost_df['jerk_norm'] = (cost_df['jerk_cost'] - cost_df['jerk_cost'].min()) / (cost_df['jerk_cost'].max() - cost_df['jerk_cost'].min())
    cost_df['wear_norm'] = (cost_df['wear_cost'] - cost_df['wear_cost'].min()) / (cost_df['wear_cost'].max() - cost_df['wear_cost'].min())
    cost_df['total_cost'] = (JERK_WEIGHT * cost_df['jerk_norm']) + (WEAR_WEIGHT * cost_df['wear_norm'])
    event_params = event_params.join(cost_df)

    logger.info("Finding ideal and median curves for each predicted cluster...")
    median_curves = {}
    ideal_curves = {}
    cluster_statistics = {}

    for cluster_id in sorted(event_params['predicted_cluster'].unique()):
        cluster_events = event_params[event_params['predicted_cluster'] == cluster_id]
        if len(cluster_events) < 1: continue
        cluster_vectors = [event_vectors[i] for i in cluster_events.index if i in event_vectors]
        if not cluster_vectors: continue
        median_curves[cluster_id] = np.median(cluster_vectors, axis=0)

        best_event_id = cluster_events['total_cost'].idxmin()
        ideal_curves[cluster_id] = event_vectors[best_event_id]

        # Save stats for controller (optional)
        cluster_statistics[cluster_id] = {
            'member_ids': list(cluster_events.index),
            'mean_jerk': cluster_events['jerk_cost'].mean(),
            'mean_wear': cluster_events['wear_cost'].mean(),
            'min_cost_event_id': int(best_event_id)
        }

    # Save curves and stats for controller use
    joblib.dump(median_curves, "median_curves.joblib")
    joblib.dump(ideal_curves, "ideal_curves.joblib")
    joblib.dump(cluster_statistics, "cluster_stats.joblib")
    logger.info("Saved median_curves.joblib, ideal_curves.joblib, cluster_stats.joblib.")

    logger.info("Generating comparison plot...")
    plt.figure(figsize=(16, 9))
    num_clusters = len(median_curves)
    colors = plt.cm.viridis(np.linspace(0, 1, num_clusters))
    for i, cluster_id in enumerate(median_curves.keys()):
        plt.plot(median_curves[cluster_id], color=colors[i], linestyle='--', label=f'Cluster {cluster_id} Median')
        plt.plot(ideal_curves[cluster_id], color=colors[i], label=f'Cluster {cluster_id} Ideal')
    plt.title('Median (Typical) vs. Ideal (Low-Cost) Engagement Curves per Predicted Profile')
    plt.xlabel('Normalized Time Points')
    plt.ylabel('ClutchCval')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ideal_vs_median_curves.png", dpi=1200)
    plt.show()

if __name__ == '__main__':
    logger.remove()
    logger.add("logs/optimization.log", rotation="5 MB", level="INFO")
    logger.add(lambda msg: print(msg, end=""), format="{message}", level="INFO")
    install(show_locals=False, word_wrap=True, width=120)
    plt.style.use(['science', 'no-latex'])
    main()
