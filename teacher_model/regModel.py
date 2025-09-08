# train_model.py

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import optuna
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks
from loguru import logger
from rich.traceback import install
from scipy.interpolate import CubicSpline
import scienceplots

def feature_vectors(df: pd.DataFrame, identifier_col: str, target_col: str, time_col: str, points_per_curve: int = 100) -> dict:
    logger.info("Starting feature vector creation...")
    event_vectors = {}
    events = df.groupby(identifier_col)
    logger.info(f'{len(events)} events found in the dataset')
    
    for event_num, event in events:
        y = event[target_col].values
        x = event[time_col].values
        
        if len(x) < 4:
            logger.warning(f'Event {event_num} has less than 4 data points, skipping...')
            continue
            
        try:
            spline = CubicSpline(x, y, bc_type='natural')
            x_new = np.linspace(0, x.max(), points_per_curve)
            y_new = spline(x_new)
            event_vectors[event_num] = y_new
        except Exception as e:
            logger.error(f"Could not process event {event_num} for spline fitting: {e}")
            
    logger.success(f"Successfully created {len(event_vectors)} feature vectors.")
    return event_vectors

def aggregator(df: pd.DataFrame, identifier_col: str, input_features: list, time_col: str) -> pd.DataFrame:
    logger.info("Aggregating initial event parameters...")
    aggregations = {
        feature: ['mean', 'std', ('q25', lambda x: x.quantile(0.25)), ('q75', lambda x: x.quantile(0.75)), 'first'] 
        for feature in input_features
    }
    
    event_parameters_df = df.groupby(identifier_col).agg(aggregations)
    event_parameters_df.columns = ['_'.join(col).strip() for col in event_parameters_df.columns.values]
    
    event_durations = df.groupby(identifier_col)[time_col].max()
    event_parameters_df['Duration'] = event_durations
    
    logger.success(f"Aggregated event parameters with shape: {event_parameters_df.shape}")
    return event_parameters_df

def extract_curve_features(y: np.ndarray) -> list:
    return [
        np.mean(y), np.median(y), np.std(y), np.min(y), np.max(y),
        skew(y), kurtosis(y), y[0], y[-1], y[-1] - y[0],
        np.trapz(y), len(find_peaks(y)[0])
    ]

def cluster(event_vectors: dict, max_k: int = 15) -> tuple:
    logger.info("Starting feature-based clustering process...")
    event_ids = list(event_vectors.keys())
    
    logger.info("Extracting descriptive features from curve vectors...")
    feature_list = [extract_curve_features(vec) for vec in event_vectors.values()]
    
    logger.info("Scaling features for clustering...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(feature_list)
    
    best_score = -1.0
    optimal_k = 2
    
    logger.info(f"Determining optimal k by searching up to {max_k} clusters...")
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(features_scaled)
        score = silhouette_score(features_scaled, kmeans.labels_)
        logger.debug(f"For k={k}, silhouette score is {score:.4f}")
        if score > best_score:
            best_score = score
            optimal_k = k
            
    logger.info(f"Optimal k found: {optimal_k} with score: {best_score:.4f}")
    
    logger.info("Running final clustering with optimal k...")
    final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10).fit(features_scaled)
    final_label_map = dict(zip(event_ids, final_kmeans.labels_))
    
    logger.success("Clustering complete.")
    return final_label_map, scaler

def train_tune(X_train, y_train, X_val, y_val, n_trials=50) -> xgb.XGBClassifier:
    logger.info("--- Starting Hyperparameter Tuning with Optuna ---")
    
    weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    sample_weights = y_train.map(dict(enumerate(weights))).values

    def objective(trial):
        params = {
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'random_state': 42,
            'num_class': len(np.unique(y_train)),
            'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True)
        }
        
        model = xgb.XGBClassifier(**params, use_label_encoder=False, early_stopping_rounds=15)
        model.fit(X_train, y_train, sample_weight=sample_weights, eval_set=[(X_val, y_val)], verbose=False)
        
        return accuracy_score(y_val, model.predict(X_val))

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    logger.info(f"Optuna study complete. Best validation accuracy: {study.best_value:.4f}")
    logger.info(f"Best hyperparameters: {study.best_params}")
    
    logger.info("--- Training Final Model with Optimal Hyperparameters ---")
    final_model = xgb.XGBClassifier(**study.best_params).fit(X_train, y_train, sample_weight=sample_weights)
    
    return final_model

def main():
    # --- Configuration ---
    TRAIN_SET_PATH = r'data\amt_train_test\g90amt_trainset.csv'
    MODEL_SAVE_PATH = r'teacher_model/models/xgb_clutch_profile_model.joblib'
    SCALER_SAVE_PATH = r'teacher_model/models/curve_feature_scaler.joblib'
    INPUT_FEATURES = ['EngTrq', 'EngSpd', 'tmpCltActTC', 'CurrGr', 'Calc_VehSpd']
    TIME_COL = 'EventTime'
    TARGET_COL = 'ClutchCval'
    IDENTIFIER_COL = 'EngagementEvents'
    df1 = pd.read_csv(r'data\amt_train_test\g90amt_testset.csv')
    df2 = pd.read_csv(TRAIN_SET_PATH)
    df_train = pd.concat([df2, df1], ignore_index=True)
    event_vectors = feature_vectors(df_train, IDENTIFIER_COL, TARGET_COL, TIME_COL)
    event_params = aggregator(df_train, IDENTIFIER_COL, INPUT_FEATURES, TIME_COL)
    
    cluster_labels, feature_scaler = cluster(event_vectors)
    
    event_params['cluster_label'] = event_params.index.map(cluster_labels)
    event_params.dropna(inplace=True)
    event_params['cluster_label'] = event_params['cluster_label'].astype(int)
    
    X = event_params.drop('cluster_label', axis=1)
    y = event_params['cluster_label']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    final_model = train_tune(X_train, y_train, X_val, y_val, n_trials=50)
    
    # --- Save Artifacts ---
    joblib.dump(final_model, MODEL_SAVE_PATH)
    logger.success(f"Trained model saved to: {MODEL_SAVE_PATH}")
    joblib.dump(feature_scaler, SCALER_SAVE_PATH)
    logger.success(f"Feature scaler saved to: {SCALER_SAVE_PATH}")

if __name__ == '__main__':
    logger.remove()
    logger.add("logs/training_pipeline.log", rotation="5 MB", level="INFO")
    logger.add(lambda msg: print(msg, end=""), format="{message}", level="INFO")
    install(show_locals=False, word_wrap=True, width=120)
    
    main()
