import numpy as np
import pandas as pd
import os
from scipy.interpolate import CubicSpline
from loguru import logger
from rich.progress import Progress
from rich.traceback import install

def create_splines(df: pd.DataFrame, identifier_col: str, target_col: str, time_col: str, points_per_curve: int = 100) -> dict:

    splines = {}
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
            splines[event_num] = y_new
            logger.info(f'Event {event_num} successfully processed into spline.')
        except Exception as e:
            logger.error(f'Could not process event {event_num} for spline fitting: {e}')

    logger.success(f'Successfully created {len(splines)} splines.')
    return splines

def aggregator(df: pd.DataFrame, identifier_col: str, input_featureset: list, time_col: str) -> pd.DataFrame:
    logger.info('Aggregating initial event parameters...')
    aggregations = {
        feature: ['mean', 'std', ('q25', lambda x: x.quantile(0.25)), ('q75', lambda x: x.quantile(0.75)), 'first']
        for feature in input_featureset
    }

    logger.info(f'Using aggregations: {aggregations}')

    event_parameters_df = df.groupby(identifier_col).agg(aggregations)
    event_parameters_df.columns = ['_'.join(col).strip() for col in event_parameters_df.columns.values]

    event_durations = df.groupby(identifier_col)[time_col].max()
    event_parameters_df['Duration'] = event_durations

    logger.success(f'Aggregated event parameters with shape: {event_parameters_df.shape}')
    return event_parameters_df

def main():
    logger.info('Starting spline creation...')
    source_dir = 'data/amt_train_test'
    os.makedirs(source_dir, exist_ok=True)

    dump_dir = 'artifacts'
    os.makedirs(dump_dir, exist_ok=True)

    source_files = os.listdir(source_dir)
    source_files = [f for f in source_files if f.endswith('.csv')]

    if len(source_files) == 0:
        logger.critical(f'No valid .csv files found at {source_dir}.')
        return
    else:
        logger.info(f'{len(source_files)} file(s) loaded from {source_dir}')

    df_combo = pd.DataFrame()

    for file_name in source_files:
        df_loaded = pd.read_csv(os.path.join(source_dir, file_name))
        df_combo = pd.concat([df_combo, df_loaded], ignore_index=True)
        logger.info(f'Data file loaded: {file_name}.')

    input_featureset = ['EngTrq', 'EngSpd', 'tmpCltActTC', 'CurrGr', 'Calc_VehSpd']
    time_col = 'EventTime'
    target_col = 'ClutchCval'
    identifier_col = 'EngagementEvents'

    splines = create_splines(df_combo, identifier_col, target_col, time_col)
    event_params = aggregator(df_combo, identifier_col, input_featureset, time_col)

    # Save processed data
    event_params_path = os.path.join(dump_dir, 'event_params.csv')
    splines_path = os.path.join(dump_dir, 'splines.npy')

    event_params.to_csv(event_params_path)
    np.save(splines_path, splines)

    logger.success(f'Preprocessed data saved: event_params to {event_params_path}, splines to {splines_path}')

if __name__ == '__main__':
    with Progress() as progress:
        logger.remove()
        logger.add("logs/create_splines.log", rotation="5 MB", level="INFO")

        install(show_locals=True, word_wrap=True, width=120)

        task = progress.add_task("[cyan]Creating splines...", total=None)
        main()
        progress.update(task, completed=progress.tasks[0].total)
        print("Splines fitted successfully.") 
        logger.success("Splines fitted successfully.")

