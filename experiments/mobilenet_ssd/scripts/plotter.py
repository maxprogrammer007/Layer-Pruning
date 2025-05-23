# experiments/mobilenet_ssd/scripts/plotter.py
import os
import json
import pandas as pd
import matplotlib.pyplot as plt

def make_tradeoff_plots(log_dir, out_dir):
    """
    Reads all JSON logs in `log_dir`, plots FPS vs mAP,
    and writes a single tradeoff.png into `out_dir`.
    """
    records = []
    for fn in os.listdir(log_dir):
        if fn.endswith('.json'):
            path = os.path.join(log_dir, fn)
            data = json.load(open(path))
            data['method'] = fn.replace('.json','')
            records.append(data)
    df = pd.DataFrame(records)

    plt.figure()
    plt.scatter(df['FPS'], df['mAP'])
    plt.xlabel('FPS')
    plt.ylabel('mAP')
    plt.title('mAP vs FPS Tradeoff')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'tradeoff.png')
    plt.savefig(out_path)
    plt.close()
    print(f"[plotter] saved tradeoff plot to {out_path}")

def aggregate_results(log_dir, out_csv):
    """
    Reads all JSON logs in `log_dir` and writes a summary CSV to `out_csv`.
    """
    records = []
    for fn in os.listdir(log_dir):
        if fn.endswith('.json'):
            path = os.path.join(log_dir, fn)
            data = json.load(open(path))
            data['method'] = fn.replace('.json','')
            records.append(data)
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[plotter] saved summary CSV to {out_csv}")
