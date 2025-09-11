
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import optuna
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks
from loguru import logger
from rich.traceback import install
from rich.progress import Progress
from scipy.interpolate import CubicSpline


def extract_curve_features(y: np.ndarray) -> list:
    return [
        np.mean(y), np.median(y), np.std(y), np.min(y), np.max(y),
        skew(y), kurtosis(y), y[0], y[-1], y[-1] - y[0],
        np.trapezoid(y), len(find_peaks(y)[0])
    ]


def cluster(splines: dict, max_k: int = 15) -> tuple:
    logger.info('Starting feature-based clustering process...')
    event_ids = list(splines.keys())

    logger.info('Extracting descriptive features from curve vectors...')
    feature_list = [extract_curve_features(vec) for vec in splines.values()]

    logger.info('Scaling features for clustering...')
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(feature_list)

    best_score = -1.0
    optimal_k = 2

    logger.info(f'Determining optimal k by searching up to {max_k} clusters...')
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(features_scaled)
        score = silhouette_score(features_scaled, kmeans.labels_)
        logger.debug(f'For k={k}, silhouette score is {score:.4f}')
        if score > best_score:
            best_score = score
            optimal_k = k

    logger.info(f'Optimal k found: {optimal_k} with score: {best_score:.4f}')

    logger.info('Running final clustering with optimal k...')
    final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10).fit(features_scaled)
    final_label_map = dict(zip(event_ids, final_kmeans.labels_))

    logger.success('Clustering complete.')
    return final_label_map, scaler


def train_tune(X_train, y_train, X_val, y_val, n_trials=50) -> xgb.XGBClassifier:
    logger.info('Starting Hyperparameter Tuning...')

    weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    sample_weights = y_train.map(dict(enumerate(weights))).values

    def objective(trial):
        params = {
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'random_state': 42,
            'num_class': len(np.unique(y_train)),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True)
        }

        logger.info(f'Trial {trial.number}: Testing parameters: {params}')

        model = xgb.XGBClassifier(**params, use_label_encoder=False, early_stopping_rounds=15)
        model.fit(X_train, y_train, sample_weight=sample_weights, eval_set=[(X_val, y_val)], verbose=False)

        return accuracy_score(y_val, model.predict(X_val))

    study = optuna.create_study(direction='maximize', study_name='Clutch Clustering')
    study.optimize(objective, n_trials=n_trials)

    logger.info(f'Optuna study complete. Best validation accuracy: {study.best_value:.4f}')
    logger.info(f'Best hyperparameters: {study.best_params}')

    logger.info('Training Final Model with Optimal Hyperparameters...')
    final_model = xgb.XGBClassifier(**study.best_params).fit(X_train, y_train, sample_weight=sample_weights)

    return final_model


def main():
    logger.info('Starting Clustering and Model Training Step...')

    dump_dir = 'artifacts'

    event_params_path = os.path.join(dump_dir, 'event_params.csv')
    splines_path = os.path.join(dump_dir, 'splines.npy')
    scaler_path = os.path.join(dump_dir, 'feature_scaler.joblib')
    model_save_path = os.path.join(dump_dir,'model.joblib')

    if not (os.path.exists(event_params_path) and os.path.exists(splines_path)):
        logger.critical('Required preprocessed files not found. Run preprocessing first.')
        return

    event_params = pd.read_csv(event_params_path, index_col=0)
    splines = np.load(splines_path, allow_pickle=True).item()

    # Run clustering here
    cluster_labels, feature_scaler = cluster(splines)

    event_params['cluster_label'] = event_params.index.map(cluster_labels)
    event_params.dropna(inplace=True)
    event_params['cluster_label'] = event_params['cluster_label'].astype(int)

    X = event_params.drop('cluster_label', axis=1)
    y = event_params['cluster_label']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    final_model = train_tune(X_train, y_train, X_val, y_val, n_trials=50)

    joblib.dump(final_model, model_save_path)
    logger.success(f'Trained model saved to: {model_save_path}')


if __name__ == '__main__':
    install(show_locals=False, word_wrap=True, width=120)
    with Progress() as progress:
        logger.remove()
        logger.add('logs/train_model.log', rotation='5 MB', level='INFO')
        task = progress.add_task('[cyan] Training Clutch Classification Model...', total=None)
        main()
        progress.update(task, completed=progress.tasks[0].total)
        print(f'Model Training Complete') 
        logger.success('Model Training Complete')
