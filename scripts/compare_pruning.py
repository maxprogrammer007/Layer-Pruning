# scripts/compare_pruning.py

from scripts.evaluate import evaluate_model
from scripts.shap_pruning import compute_shap_scores, prune_model_by_shap
from scripts.pruning_methods import l1_norm_prune, random_prune
from scripts.logger import save_log

def compare_all_methods(model, config, dataloader, model_name, device="cuda"):
    results = {}

    # === Baseline (no pruning) ===
    print("\n📊 Evaluating baseline (no pruning)...")
    base_metrics = evaluate_model(model, dataloader, device)
    results["baseline"] = base_metrics
    base_metrics["method"] = "baseline"
    save_log(base_metrics, model_name + "_baseline", processed=True)

    # === SHAP Pruning ===
    print("\n🔍 SHAP-based pruning...")
    shap_scores = compute_shap_scores(model, dataloader, config, device=device)
    model_shap, removed_shap = prune_model_by_shap(model, shap_scores, threshold=0.2)
    shap_metrics = evaluate_model(model_shap, dataloader, device)
    shap_metrics["method"] = "shap"
    shap_metrics["layers_removed"] = len(removed_shap)
    save_log(shap_metrics, model_name + "_shap", processed=True)
    results["shap"] = shap_metrics

    # === L1-Norm Pruning ===
    print("\n📉 L1-norm-based pruning...")
    model_l1, removed_l1 = l1_norm_prune(model, config, dataloader, threshold=0.1, device=device)
    l1_metrics = evaluate_model(model_l1, dataloader, device)
    l1_metrics["method"] = "l1_norm"
    l1_metrics["layers_removed"] = len(removed_l1)
    save_log(l1_metrics, model_name + "_l1", processed=True)
    results["l1_norm"] = l1_metrics

    # === Random Pruning ===
    print("\n🎲 Random pruning...")
    model_rand, removed_rand = random_prune(model, config, fraction=0.3)
    rand_metrics = evaluate_model(model_rand, dataloader, device)
    rand_metrics["method"] = "random"
    rand_metrics["layers_removed"] = len(removed_rand)
    save_log(rand_metrics, model_name + "_random", processed=True)
    results["random"] = rand_metrics

    return results
