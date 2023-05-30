from pathlib import Path

import wandb
import pandas as pd
from tqdm import tqdm


EPOCH = 'Training Epoch'
COCO_AP = 'COCO AP'
COCO_AP_BIN = 'COCO AP (binary clf)'
RECOG_AP = 'mAP'


def get_sweep_results(
    sweep_id, project="experiments", entity="jpdr", lazy=False,
    config_keys=[]
):
    res_path = Path(f'sweep_{sweep_id}.pkl')
    if res_path.exists() and lazy:
        return pd.read_pickle(res_path)

    api = wandb.Api()

    results = []

    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    runs = sweep.runs

    for run in tqdm(runs, leave=False):
        df_history = run.history(keys=[
            "COCO_recog/AP@[0.50:0.95]",
            "epoch",
        ])
        df_history = pd.concat([
            df_history,
            run.history(keys=[
                "COCO_detect/AP@[0.50:0.95]",
                "epoch",
            ])
        ])
        df_history = pd.concat([
            df_history,
            run.history(keys=[
                "AggPRCurve/mAP",
                "epoch",
            ])
        ])
        df_history['val_fold'] = run.config['k_fold_val_fold']

        for k in config_keys:
            df_history[k] = run.config[k]

        results.append(df_history)

    df = pd.concat(results, ignore_index=True)
    df = df.rename(columns={
        'epoch': EPOCH,
        'COCO_recog/AP@[0.50:0.95]': COCO_AP,
        "COCO_detect/AP@[0.50:0.95]": COCO_AP_BIN,
        "AggPRCurve/mAP": RECOG_AP,
    })

    df.to_pickle(res_path)

    return df