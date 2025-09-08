#Convert MATLAB .mat files to .csv files
#Pipeline Step 1 of 7

import h5py
import numpy as np
import os
import pandas as pd
from rich.traceback import install
from rich.progress import Progress
from loguru import logger

def mat_to_csv(mat_name: str, source_dir: str = 'data/amt_raw', dump_dir: str = 'data/amt_csv') -> None:
    testdata={}
    mat_path = os.path.join(source_dir, mat_name)
    logger.info(f'Processing {mat_path}.')

    with h5py.File(mat_path, 'r') as f:
        for column in f:
            if f[column].shape[0] == 1:
                fil=np.array(f[column][()])
                testdata[f'{column}'] = fil[0]

                logger.info(f'{column} extracted from {mat_path}.')

    headers=testdata.keys()

    for i in range(0,12):
        tim=f't{i}'
        param_lst={}
        name_lst =[]
        if tim in headers:
            param_lst['Time']= testdata[tim]
            for name in headers:
                if name.__contains__(tim) and name != tim and name not in name_lst:
                    param_lst[name] = testdata[name]
                    param_lst= pd.DataFrame(param_lst)
                    test_dir = f'{dump_dir}/{mat_name.split(".")[0]}'
                    os.makedirs(test_dir, exist_ok=True)
                    param_lst.to_csv(os.path.join(test_dir, f'{name}.csv'), index=False)
                    name_lst.append(name)
                    logger.success(f"File saved to {test_dir}/{name}.csv")

def main():
    logger.info("Starting the conversion of .mat files to .csv files.")

    source_dir = 'data/amt_raw'
    os.makedirs(source_dir, exist_ok=True)
    logger.info(f"Source directory set to {source_dir}.")

    dump_dir = 'data/amt_csv'
    os.makedirs(source_dir, exist_ok=True)
    logger.info(f"Dump directory set to {dump_dir}.")

    mat_files = os.listdir(source_dir)
    mat_files = [f for f in mat_files if f.endswith('.mat')]

    if len(mat_files) == 0:
        logger.critical(f'No valid .mat files found at {source_dir}.')
        return None
    else:
        logger.info(f'{len(mat_files)} file(s) loaded from {source_dir}')

    for mat_name in mat_files:
        mat_to_csv(mat_name, source_dir,dump_dir)

if __name__ == "__main__":
    with Progress() as progress:
        logger.remove()
        logger.add("logs/file_conv.log", rotation="5 MB", level="INFO")

        install(show_locals=True, word_wrap=True, width=120)

        task = progress.add_task("[cyan]Processing files...", total=None)
        main()
        progress.update(task, completed=progress.tasks[0].total)
        print("All files converted successfully.")
        logger.success("All files converted successfully.")
