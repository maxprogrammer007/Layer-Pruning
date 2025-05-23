#!/usr/bin/env python3
import os
import argparse
import json

from scripts.config_loader    import load_config
from scripts.evaluate         import run_evaluation
from scripts.shap_pruning     import compute_shap_scores, apply_prune as shap_apply
from scripts.l1_pruning       import compute_l1_scores, apply_prune as l1_apply
from scripts.random_pruning   import apply_random_prune
from scripts.plotter          import make_tradeoff_plots, aggregate_results

def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(
        description="Run pruning experiment on MobileNet SSD"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to config.yaml"
    )
    args = parser.parse_args()

    # 1. Load config
    cfg = load_config(args.config)

    # 2. Make sure logging folders exist
    log_dir  = cfg["logging"]["log_dir"]
    plot_dir = cfg["logging"]["plots_dir"]
    res_dir  = cfg["logging"]["results_dir"]
    ensure_dirs(log_dir, plot_dir, res_dir)

    # 3. Baseline evaluation
    baseline_json = os.path.join(log_dir, "baseline.json")
    run_evaluation(cfg, out_json=baseline_json)

    # 4. SHAP-based pruning
    shap_scores = compute_shap_scores(cfg)
    shap_mask   = shap_apply(cfg, shap_scores)
    # save mask for record
    with open(os.path.join(log_dir, "shap_mask.json"), "w") as f:
        json.dump(shap_mask, f, indent=2)
    # evaluate pruned model
    cfg_shap = {**cfg, "model_mask": shap_mask}
    shap_json = os.path.join(log_dir, "shap_pruned.json")
    run_evaluation(cfg_shap, out_json=shap_json)

    # 5. L1-norm pruning
    l1_scores = compute_l1_scores(cfg)
    l1_mask   = l1_apply(cfg, l1_scores)
    with open(os.path.join(log_dir, "l1_mask.json"), "w") as f:
        json.dump(l1_mask, f, indent=2)
    cfg_l1   = {**cfg, "model_mask": l1_mask}
    l1_json = os.path.join(log_dir, "l1_pruned.json")
    run_evaluation(cfg_l1, out_json=l1_json)

    # 6. Random pruning
    rnd_mask = apply_random_prune(cfg, ratio=cfg["prune"]["ratio"])
    with open(os.path.join(log_dir, "random_mask.json"), "w") as f:
        json.dump(rnd_mask, f, indent=2)
    cfg_rand   = {**cfg, "model_mask": rnd_mask}
    rand_json = os.path.join(log_dir, "random_pruned.json")
    run_evaluation(cfg_rand, out_json=rand_json)

    # 7. Generate plots & summary
    make_tradeoff_plots(log_dir, plot_dir)
    summary_csv = os.path.join(res_dir, "summary.csv")
    aggregate_results(log_dir, summary_csv)

    print(f"\n✅ Experiment complete. Check:\n"
          f"  • Baseline & pruned logs → {log_dir}\n"
          f"  • Tradeoff plots       → {plot_dir}\n"
          f"  • Summary CSV          → {summary_csv}")

if __name__ == "__main__":
    main()
