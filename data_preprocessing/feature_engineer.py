#calculates ClutchSlipPower, OutShaftSpd and VehSpd for all files
#Use ONLY AFTER running file_align.py
#Pipeline Step 3 of 7

import os
import pandas as pd
import numpy as np
from rich.traceback import install
from rich.progress import Progress
from loguru import logger

def calc_params(test_name: str, source_dir: str ='data/amt_aligned', gearratios: dict = {1:1}) -> pd.DataFrame:
    df=pd.read_csv(f'{source_dir}/{test_name}')
    logger.info(f"Calculating vehicle parameters for {test_name}.")
    col=df.columns
    if 'InShaftSpd' in col:
        df['CurrGr'] = df['CurrGr'].astype(int)
        df['Calc_SlipPower'] = (df['EngTrq']) * (df['EngSpd']-df['InShaftSpd']) * (2*np.pi/60)/ 1000
        df['Calc_OutShaftSpd'] = df['InShaftSpd'] / df['CurrGr'].map(gearratios)
        df['Calc_OutShaftSpd'] = df['Calc_OutShaftSpd'].replace([np.nan, np.inf, -np.inf],0)
        df['Calc_VehSpd'] = (3/25)*(np.pi)*(0.107)*df['Calc_OutShaftSpd'] # 0.107 is assumed tire radius
    return df

def main():
    G90_gearratios={
    1 : 6.696,
    2 : 3.806,
    3 : 2.289,
    4 : 1.480,
    5 : 1.000,
    6 : 0.728,
    -1: 13.862,
    0 : 0
    }
    logger.info("Starting Data Time-Alignment.")

    source_dir = 'data/amt_aligned'
    logger.info(f"Source directory set to {source_dir}.")
    os.makedirs(source_dir, exist_ok=True)

    dump_dir = 'data/amt_processed'
    os.makedirs(dump_dir, exist_ok=True)
    logger.info(f"Dump directory set to {dump_dir}.")

    tests = os.listdir(source_dir)
    tests = [f for f in tests if f.endswith('.csv')]
    if len(tests) == 0:
        logger.critical(f'No valid .csv files found at {source_dir}.')
        return None
    else:
        logger.info(f'{len(tests)} file(s) loaded from {source_dir}')
    
    for test_name in tests:
        df = calc_params(test_name,source_dir,G90_gearratios)
        df.to_csv(f'{dump_dir}/{test_name.split('.')[0]}_processed.csv',index=False)
        logger.success(f"Processed file saved to {dump_dir}/{test_name.split('.')[0]}_processed.csv")
    return dump_dir


if __name__ == "__main__":
    with Progress() as progress:
        logger.remove()
        logger.add("logs/feature_engineer.log", rotation="5 MB", level="INFO")

        install(show_locals=True, word_wrap=True, width=120)
        task = progress.add_task("[cyan]Processing Files and Calculating Vehicle Parameters...", total=None)
        dump_dir=main()
        progress.update(task, completed=progress.tasks[0].total)
        print(f"Vehicle Parameter Calculations Complete.\nFiles saved to {dump_dir}.") 
        logger.success("Vehicle Parameter Calculations Complete.")