import json
import os
import math
from collections import Counter

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from scipy.optimize import minimize
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load configuration file
with open("config.json", "r") as f:
    CFG = json.load(f)

# Config variables
MODEL_ID = CFG["model_id"]
CALIB_SET_SIZE = CFG["calibration_size"]
TEST_SET_SIZE = CFG["test_size"]
NUM_SAMPLES_MAIN = CFG["num_samples_main"]
TEMP_MAIN = CFG["temperature_main"]
BETA_MAIN = CFG["beta_main"]
RESULTS_DIR = CFG["results_dir"]
PLOTS_DIR = CFG["plots_dir"]
HALLUCINATION_PROMPTS = CFG["hallucination_prompts"]

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Load model + tokenizer
def load_model_and_tokenizer(model_name: str):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        dtype=torch.float16,
    )
    model.eval()
    print("Model ready.\n")
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer(MODEL_ID)

# Build MCQ prompt format
def build_mcq_prompt(question: str, options):
    lines = [f"Question: {question}"]
    for idx, opt in enumerate(options):
        lines.append(f"{chr(65 + idx)}. {opt}")
    lines.append("The answer is:")
    return "\n".join(lines)

# Sample answer letters from model
def sample_mcq_choices(question, options, n_samples=5, temperature=0.8):
    prompt = build_mcq_prompt(question, options)
    device = model.device

    # Token IDs for " A", " B", " C", " D"
    choice_token_ids = [
        tokenizer.encode(" " + chr(65 + i), add_special_tokens=False)[0]
        for i in range(len(options))
    ]

    sampled_letters = []
    prob_matrix = []

    for _ in range(n_samples):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]

        # Softmax restricted to choice tokens
        scaled_logits = logits[choice_token_ids] / temperature
        probs = torch.softmax(scaled_logits, dim=0)
        p = probs.detach().cpu().numpy()

        prob_matrix.append(p)
        idx = torch.multinomial(probs, 1).item()
        sampled_letters.append(chr(65 + idx))

    return sampled_letters, np.array(prob_matrix)

# Compute DST belief and ignorance
def build_ds_belief(prob_matrix, letter_samples, num_options, beta=0.9):
    mean_probs = prob_matrix.mean(axis=0)
    counts = Counter(letter_samples)
    modal = counts.most_common(1)[0][1]  # most frequent letter count
    rho = modal / len(letter_samples)

    weighted = rho * mean_probs.sum()
    alpha = min(1.0, beta / weighted) if weighted > 0 else 1.0

    belief = {}
    for i in range(num_options):
        ltr = chr(65 + i)
        belief[ltr] = alpha * mean_probs[i] * rho

    ignorance = max(0.0, 1.0 - sum(belief.values()))
    plaus = {k: belief[k] + ignorance for k in belief}

    return belief, plaus, ignorance

# ECE calculation
def expected_calibration_error(conf, corr, n_bins=10):
    conf = np.asarray(conf)
    corr = np.asarray(corr)

    bins = np.linspace(0, 1, n_bins + 1)
    N = len(conf)
    ece = 0

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf < hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / N) * abs(corr[mask].mean() - conf[mask].mean())

    return ece

# Stabilize probabilities
def smooth_probs(p):
    eps = 1e-6
    p = p + eps
    return p / p.sum(axis=1, keepdims=True)

# Temperature fitting
def fit_temperature(val_probs, val_labels):
    val_probs = smooth_probs(val_probs)

    def objective(params):
        T = params[0]
        log_p = np.log(val_probs) / T
        exp = np.exp(log_p)
        scaled = exp / exp.sum(axis=1, keepdims=True)
        row = np.arange(len(val_labels))
        return -np.log(scaled[row, val_labels] + 1e-12).mean()

    res = minimize(
        objective,
        x0=[1.0],
        bounds=[(0.2, 5.0)],
        method="L-BFGS-B",
    )
    return res.x[0]

# Apply learned temperature
def apply_temperature(p, T):
    p = smooth_probs(p)
    log_p = np.log(p) / T
    exp = np.exp(log_p)
    return exp / exp.sum(axis=1, keepdims=True)

# Compute coverage vs accuracy curve
def accuracy_coverage_from_ignorance(ignorance, correctness, n_points=25):
    order = np.argsort(ignorance)
    corr_sorted = correctness[order]

    cov = []
    acc = []

    for frac in np.linspace(0.05, 1.0, n_points):
        k = int(len(corr_sorted) * frac)
        subset = corr_sorted[:k]
        cov.append(frac)
        acc.append(subset.mean())

    return np.array(cov), np.array(acc)

# Load MMLU calibration + test splits
def load_mmlu_splits():
    print("Loading MMLU test split...")
    ds = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=42)

    calib = ds.select(range(CALIB_SET_SIZE))
    test = ds.select(range(CALIB_SET_SIZE, CALIB_SET_SIZE + TEST_SET_SIZE))

    return calib, test

calib_ds, test_ds = load_mmlu_splits()

# Evaluate a dataset split using DST
def eval_split_with_dst(dataset, n_samples=5, temperature=0.8, beta=0.9):
    preds = []
    correct = []
    ignorance_vals = []
    belief_conf = []
    mean_probs = []
    true_idx_arr = []

    for ex in tqdm(dataset, desc="Evaluating"):
        q = ex["question"]
        options = ex["choices"]
        true_idx = int(ex["answer"])
        true_letter = chr(65 + true_idx)

        samples, prob_matrix = sample_mcq_choices(
            q, options, n_samples=n_samples, temperature=temperature
        )

        belief, plaus, ign = build_ds_belief(
            prob_matrix,
            samples,
            len(options),
            beta=beta,
        )

        pred_letter = max(belief, key=belief.get)

        preds.append(pred_letter)
        correct.append(pred_letter == true_letter)
        ignorance_vals.append(ign)
        belief_conf.append(belief[pred_letter])
        mean_probs.append(prob_matrix.mean(axis=0))
        true_idx_arr.append(true_idx)

    return {
        "pred": np.array(preds),
        "correct": np.array(correct),
        "ignorance": np.array(ignorance_vals),
        "belief_conf": np.array(belief_conf),
        "mean_probs": np.stack(mean_probs),
        "true_idx": np.array(true_idx_arr),
    }

# Reliability diagram
def plot_reliability(raw_conf, raw_corr,
                     ts_conf, ts_corr,
                     dst_conf, dst_corr,
                     n_bins=10,
                     path=None):

    combined = np.concatenate([raw_conf, ts_conf, dst_conf])
    combined = np.clip(combined, 1e-6, 1 - 1e-6)
    bins = np.quantile(combined, np.linspace(0, 1, n_bins + 1))

    def compute_stats(conf, corr):
        avg_conf = []
        avg_acc = []
        for i in range(len(bins) - 1):
            lo, hi = bins[i], bins[i + 1]
            mask = (conf >= lo) & (conf < hi)
            if mask.sum() == 0:
                avg_conf.append(np.nan)
                avg_acc.append(np.nan)
            else:
                avg_conf.append(conf[mask].mean())
                avg_acc.append(corr[mask].mean())
        return np.array(avg_conf), np.array(avg_acc)

    rc, ra = compute_stats(raw_conf, raw_corr)
    tc, ta = compute_stats(ts_conf, ts_corr)
    dc, da = compute_stats(dst_conf, dst_corr)

    # Simple moving average smoothing
    def smooth(y, window=2):
        y2 = y.copy()
        for i in range(len(y)):
            lo = max(0, i - window)
            hi = min(len(y), i + window + 1)
            y2[i] = np.nanmean(y[lo:hi])
        return y2

    ra = smooth(ra)
    ta = smooth(ta)
    da = smooth(da)

    plt.figure(figsize=(7, 7))
    plt.plot([0, 1], [0, 1], "--", color="steelblue", label="Perfect calibration")

    plt.plot(rc, ra, "o-", label="Raw softmax")
    plt.plot(tc, ta, "s-", label="Temp-scaled")
    plt.plot(dc, da, "^-", label="DST belief(pred)")

    plt.xlabel("Confidence")
    plt.ylabel("Empirical accuracy")
    plt.title("Reliability Diagram (Smoothed)")
    plt.grid(True)
    plt.legend()
    if path:
        plt.savefig(path, bbox_inches="tight")
    plt.close()

# Ablation runs
def run_ablation_experiments():
    print("\nRunning ablations (N, beta)...")

    configs = [
        (1, 0.7),
        (1, 0.9),
        (3, 0.7),
        (3, 0.9),
        (5, 0.7),
        (5, 0.9),
    ]

    rows = []

    for N, b in configs:
        print(f"  Ablation N={N}, beta={b}")
        res = eval_split_with_dst(
            test_ds,
            n_samples=N,
            temperature=TEMP_MAIN,
            beta=b,
        )
        acc = res["correct"].mean()
        ign = res["ignorance"].mean()
        ece = expected_calibration_error(
            conf=res["mean_probs"].max(axis=1),
            corr=res["correct"]
        )
        rows.append({"N": N, "beta": b, "acc": acc, "ign": ign, "ece": ece})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "ablations.csv"), index=False)
    print("Ablations saved to results/ablations.csv")
    return df

# All plots wrapper
def make_all_plots(test_res, ts_probs, ts_corr, ablations):
    os.makedirs(PLOTS_DIR, exist_ok=True)

    raw_conf = test_res["mean_probs"].max(axis=1)
    raw_corr = test_res["correct"]
    dst_conf = test_res["belief_conf"]
    dst_corr = test_res["correct"]

    ts_conf = ts_probs.max(axis=1)

    plot_reliability(
        raw_conf, raw_corr,
        ts_conf, ts_corr,
        dst_conf, dst_corr,
        path=os.path.join(PLOTS_DIR, "reliability.png"),
    )

    # Accuracy–coverage curve
    cov, acc = accuracy_coverage_from_ignorance(
        test_res["ignorance"], test_res["correct"]
    )
    plt.figure(figsize=(7, 5))
    plt.plot(cov, acc, "o-")
    plt.xlabel("Coverage (fraction answered)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy–Coverage Curve (DST, threshold on m(Θ))")
    plt.grid(True)
    plt.savefig(os.path.join(PLOTS_DIR, "accuracy_coverage.png"), bbox_inches="tight")
    plt.close()

    print(f"Plots saved to {PLOTS_DIR}")

# Hallucination demo
def hallucination_check():
    print("\nQUALITATIVE HALLUCINATION CHECK\n")

    prompts = [
        "What is the capital of the fictional country Eldoria?",
        "Who won the 2023 Nobel Prize in Mathematics?",
    ]

    choices = ["A", "B", "C", "D"]

    for p in prompts:
        print(f"Prompt: {p}\n")
        inputs = tokenizer(p, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50)
        raw = tokenizer.decode(out[0], skip_special_tokens=True)
        print("Raw model output:\n" + raw + "\n")

        letters, prob_matrix = sample_mcq_choices(p, choices, n_samples=5)
        belief, plaus, ign = build_ds_belief(prob_matrix, letters, 4, beta=0.9)

        print(f"DST Ignorance m(Θ): {ign:.4f}")
        print(f"DST Beliefs: {belief}\n")
        print("=" * 80 + "\n")

# Main experiment pipeline
def main_experiment():
    print("\nRunning calibration split...")
    calib = eval_split_with_dst(
        calib_ds,
        n_samples=NUM_SAMPLES_MAIN,
        temperature=TEMP_MAIN,
        beta=BETA_MAIN,
    )

    print("\nRunning test split...")
    test_res = eval_split_with_dst(
        test_ds,
        n_samples=NUM_SAMPLES_MAIN,
        temperature=TEMP_MAIN,
        beta=BETA_MAIN,
    )

    # Core metrics
    acc = test_res["correct"].mean()
    ign = test_res["ignorance"].mean()
    ece_raw = expected_calibration_error(
        test_res["mean_probs"].max(axis=1),
        test_res["correct"],
    )
    corr_ign, pval = spearmanr(test_res["ignorance"], ~test_res["correct"])

    print("\nDST MAIN RESULTS")
    print(f"Accuracy:                 {acc:.4f}")
    print(f"Mean ignorance m(Θ):      {ign:.4f}")
    print(f"ECE (raw softmax):        {ece_raw:.4f}")
    print(f"Spearman(ignorance,error): {corr_ign:.4f} (p={pval:.3e})")

    print("\nFitting temperature on calibration set...")
    T = fit_temperature(calib["mean_probs"], calib["true_idx"])
    print(f"Optimal temperature T*:   {T:.4f}")

    # Temperature-scaled baseline
    ts_probs = apply_temperature(test_res["mean_probs"], T)
    ts_pred_idx = ts_probs.argmax(axis=1)
    ts_corr = ts_pred_idx == test_res["true_idx"]
    ece_ts = expected_calibration_error(ts_probs.max(axis=1), ts_corr)

    print("\nTEMPERATURE SCALING BASELINE")
    print(f"Accuracy (TS):            {ts_corr.mean():.4f}")
    print(f"ECE (temp-scaled):        {ece_ts:.4f}")

    # Ablations + plots + qualitative examples
    ablations = run_ablation_experiments()
    make_all_plots(test_res, ts_probs, ts_corr, ablations)
    hallucination_check()

if __name__ == "__main__":
    main_experiment()
