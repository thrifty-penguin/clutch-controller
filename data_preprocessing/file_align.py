#Time aligning all vehicle parameters from each test
#Use ONLY AFTER running file_conv.py
#Pipeline Step 2 of 8

import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from rich.traceback import install
from rich.progress import Progress
from loguru import logger

def time_align(test_name : str, source_dir : str, time_step: float = 0.01) -> pd.DataFrame:
    '''
    Aligns all vehicle parameters from a given test to a common time vector using linear interpolation.

    Parameters:
        test_name (str): Name of the test to be processed.
        source_dir (str): Directory containing the test CSV files.
        time_step (float): Time step for the common time vector in seconds.

    Returns:
        pd.DataFrame: A DataFrame containing the time-aligned vehicle parameters.
    '''
    source_dir = f'data/amt_csv/{test_name}'
    os.makedirs(source_dir, exist_ok=True)

    logger.info(f"Starting time alignment for {test_name} in {source_dir}.")

    params = os.listdir(source_dir)
    params = [f for f in params if f.endswith('.csv')]

    if not params:
        logger.warning(f"No .csv files found in {source_dir}. Skipping alignment.")
        return
    
    logger.info(f"Found {len(params)} parameter files to process.")

    df_lst=list()
    start = float('inf')
    end = float('-inf')

    for param in params:
        df=pd.read_csv(os.path.join(source_dir, param))
        if df.empty:
            logger.warning(f".csv file {param} is empty. Skipping alignment.")
            continue
        logger.info(f"Processing file: {param}")
        if 'Time' not in df.columns:
            logger.warning(f"'Time' column not found in {param}. Skipping alignment.")
            continue

        df.sort_values(by='Time', inplace=True)
        start = min(start, df['Time'].min())
        end = max(end, df['Time'].max())
        df_lst.append(df)

    logger.info(f"Time range for alignment: {start} to {end} seconds.")
    if start == float('inf') or end == float('-inf'):
        logger.error("No valid time range found for alignment. Exiting.")
        return
    
    time = np.arange(start, end, time_step)
    aligned_df = pd.DataFrame({'Time': time})
    logger.info(f"Creating time-aligned DataFrame with {len(time)} time points.")

    for i, df in enumerate(df_lst):
        data_col = df.columns[-1]
        
        INTERP_FUNC = interp1d(df['Time'], df[data_col], kind='linear', 
                    bounds_error=False, fill_value='extrapolate')
        logger.info(f"Interpolating data for {data_col} from file {params[i]}.")

        if data_col.__contains__('Calculated'):
            data_label = data_col.split('_')[1]
        elif data_col.__contains__('CAN1'):
            data_label = data_col.split('_')[6]
        elif data_col.__contains__('SpdR'):
            data_label = str(data_col.split('_')[3]+data_col.split('_')[4])
        elif data_col.__contains__('Clutch_'):
            data_label = str(data_col.split('_')[3]+data_col.split('_')[4])
        else:
            data_label = data_col.split('_')[3]
        
        aligned_df[f'{data_label}'] = INTERP_FUNC(time)
        aligned_df.dropna(inplace=True)
        logger.info(f"Added interpolated data for {data_col} to time-aligned DataFrame for {test_name}.")

    logger.info(f"Time alignment completed for {test_name}. Total rows in aligned DataFrame: {len(aligned_df)}.")

    if aligned_df.empty:
        logger.error(f"Time-aligned DataFrame for {test_name} is empty. Exiting alignment.")
        return None
    
    aligned_df['Time']=aligned_df['Time'].round(decimals=3)

    return aligned_df
    
def main():
    logger.info("Starting the time-alignment of individual tests.")

    source_dir = 'data/amt_csv'
    logger.info(f"Input directory set to {source_dir}.")
    os.makedirs(source_dir, exist_ok=True)

    dump_dir = 'data/amt_aligned'
    os.makedirs(dump_dir, exist_ok=True)
    logger.info(f"Output directory set to {dump_dir}.")

    tests = os.listdir(source_dir)

    time_step = 0.01 # seconds
    logger.info(f"Time step set to {time_step} seconds.")

    for test_name in tests:
        df = time_align(test_name, source_dir, time_step)
        if df is not None:
            dumpPath = os.path.join(dump_dir, f'g90amt_{test_name}.csv')
            df.to_csv(dumpPath, index=False)
            logger.success(f"Time-aligned data for {test_name} saved to {dumpPath}.")
        else:
            logger.warning(f"Skipping alignment for {test_name} due to errors.")

if __name__ == "__main__":
    with Progress() as progress:
        logger.remove()
        logger.add("logs/file_align.log", rotation="5 MB", level="INFO")

        install(show_locals=True, word_wrap=True, width=120)

        task = progress.add_task("[cyan]Time-Aligning all Tests...", total=None)
        main()
        progress.update(task, completed=progress.tasks[0].total)
        print("All tests time-aligned successfully.") 
        logger.success("All tests time-aligned successfully.")