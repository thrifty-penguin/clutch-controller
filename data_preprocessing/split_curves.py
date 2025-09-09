#Isolates clutch engagement events and engagement curves
#Splits data into train and test sets
#Use ONLY AFTER running feature_engineer.py
#Pipeline Step 4 of 7

import pandas as pd
from rich.traceback import install
from rich.progress import Progress
from loguru import logger
import os

def find_curves(files: list, source_dir: str) -> pd.DataFrame:
    if not files:
        logger.error('No files were provided to process.')
        return pd.DataFrame()
    
    full_curve_lst = []
    offset = 0

    for file_name in files:
        if not os.path.exists(f'{source_dir}/{file_name}'):
            logger.error(f'File not found: {file_name}')
            continue
            
        df = pd.read_csv(f'{source_dir}/{file_name}')
        logger.info(f'Loaded {file_name}')

        if 'ClutchCval' not in df.columns:
            logger.warning(f'File {file_name} does not have Clutch Engagement Data. Skipping.')
            continue

        if 'ClutchStat' not in df.columns:
            curves = df[(df['ClutchCval'] > 10) & (df['ClutchCval'] < 85)].copy()
        else:
            df['ClutchStat'] = df['ClutchStat'].astype(int)
            engaging = df['ClutchStat'] == 1
            group = (engaging != engaging.shift()).cumsum()
            lengths = engaging.groupby(group).transform('sum')
            mask = (df['ClutchStat'] == 1) & (lengths >= 3)
            curves = df[mask].copy()
        
        if curves.empty:
            logger.warning(f'No engagement events found in {file_name}.')
            continue

        curves = curves.reset_index(drop=True)
        time_diff = curves['Time'].diff().fillna(0)
        
        curves['EngagementEvents'] = (time_diff > 0.015).cumsum() + offset
        
        full_curve_lst.append(curves)
        
        offset = curves['EngagementEvents'].max() + 1

        logger.success(f'Identified {curves['EngagementEvents'].nunique()} engagement events in {file_name}.')
        
    return pd.concat(full_curve_lst, ignore_index=True) if full_curve_lst else pd.DataFrame()

def main():
    logger.info('Identifying Clutch Engagement/Disengagement Events...')
    source_dir = 'data/amt_processed'
    logger.info(f'Source directory set to {source_dir}.')
    os.makedirs(source_dir, exist_ok=True)

    dump_dir = 'data/amt_train_test'
    os.makedirs(dump_dir, exist_ok=True)
    logger.info(f'Dump directory set to {dump_dir}.')

    tests = os.listdir(source_dir)
    tests = [f for f in tests if f.endswith('.csv')]
    if len(tests) == 0:
        logger.critical(f'No valid .csv files found at {source_dir}.')
        return None
    else:
        logger.info(f'{len(tests)} file(s) loaded from {source_dir}')

    test_files = [tests[0]]
    logger.info(f'Test file selected: {test_files}')
    train_files = tests[1:]
    logger.info(f'Training files selected: {train_files}')

    train_set = find_curves(train_files, source_dir)
    test_set = find_curves(test_files, source_dir)

    if train_set.empty or test_set.empty:
        logger.error('No engagement curves found in either train/test files.')
        return None
    
    train_set.to_csv(os.path.join(dump_dir,'g90amt_train_set.csv'), index=False)
    test_set.to_csv(os.path.join(dump_dir,'g90amt_test_set.csv'), index=False)
    

    return dump_dir

if __name__ =='__main__':
    with Progress() as progress:
        logger.remove()
        logger.add('logs/split_curves.log', rotation='5 MB', level='INFO')


        install(show_locals=True, word_wrap=True, width=120)
        task = progress.add_task('[cyan] Identifying Clutch Engagement/Disengagement Events...', total=None)
        dump_dir=main()
        progress.update(task, completed=progress.tasks[0].total)
        print(f'Engagement Recognition complete.\nFiles saved to {dump_dir}.') 
        logger.success('Engagement Recognition complete.')