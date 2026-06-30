from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "images" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


ABLATION = [
    {
        "method": "Base LLM",
        "forget_success": 42.1,
        "adv_robustness": 35.7,
        "open_safety": 48.6,
        "mmlu_accuracy": 63.8,
        "utility_retention": 100.0,
        "neighbor_utility": 100.0,
        "score": 56.2,
    },
    {
        "method": "PoRT-only",
        "forget_success": 67.4,
        "adv_robustness": 64.1,
        "open_safety": 63.0,
        "mmlu_accuracy": 63.5,
        "utility_retention": 95.6,
        "neighbor_utility": 93.8,
        "score": 72.6,
    },
    {
        "method": "Adapter-only",
        "forget_success": 72.8,
        "adv_robustness": 68.9,
        "open_safety": 70.5,
        "mmlu_accuracy": 62.8,
        "utility_retention": 94.1,
        "neighbor_utility": 92.5,
        "score": 76.0,
    },
    {
        "method": "Adapter+Judge",
        "forget_success": 78.6,
        "adv_robustness": 77.4,
        "open_safety": 76.2,
        "mmlu_accuracy": 62.4,
        "utility_retention": 92.7,
        "neighbor_utility": 91.6,
        "score": 80.8,
    },
    {
        "method": "+Adv variants",
        "forget_success": 80.3,
        "adv_robustness": 79.3,
        "open_safety": 77.8,
        "mmlu_accuracy": 62.1,
        "utility_retention": 92.2,
        "neighbor_utility": 91.0,
        "score": 81.9,
    },
    {
        "method": "+Belief negatives",
        "forget_success": 81.6,
        "adv_robustness": 79.9,
        "open_safety": 80.1,
        "mmlu_accuracy": 61.9,
        "utility_retention": 91.9,
        "neighbor_utility": 90.8,
        "score": 82.8,
    },
    {
        "method": "Full SAFE-PoRT",
        "forget_success": 82.8,
        "adv_robustness": 80.3,
        "open_safety": 81.1,
        "mmlu_accuracy": 61.6,
        "utility_retention": 91.0,
        "neighbor_utility": 89.2,
        "score": 83.7,
    },
]


SOTA = [
    {
        "method": "PoRT",
        "forget_success": 80.6,
        "adv_robustness": 79.0,
        "open_safety": 75.1,
        "mmlu_accuracy": 63.2,
        "utility_retention": 92.3,
        "neighbor_utility": 88.5,
        "score": 82.0,
    },
    {
        "method": "LLM Beliefs",
        "forget_success": 82.0,
        "adv_robustness": 78.8,
        "open_safety": 80.4,
        "mmlu_accuracy": 61.3,
        "utility_retention": 90.4,
        "neighbor_utility": 88.2,
        "score": 83.1,
    },
    {
        "method": "SAFE-PoRT",
        "forget_success": 82.8,
        "adv_robustness": 80.3,
        "open_safety": 81.1,
        "mmlu_accuracy": 62.1,
        "utility_retention": 91.0,
        "neighbor_utility": 89.2,
        "score": 83.7,
    },
]


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#2E4053",
            "axes.labelcolor": "#1B2631",
            "xtick.color": "#1B2631",
            "ytick.color": "#1B2631",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_ablation_chart(df):
    colors = {
        "forget_success": "#005A8C",
        "adv_robustness": "#1F7A8C",
        "open_safety": "#5B3F8C",
        "mmlu_accuracy": "#2E7D32",
        "score": "#D17A00",
    }
    fig, ax = plt.subplots(figsize=(12.8, 6.2))
    x = range(len(df))
    for col, label in [
        ("forget_success", "Forget success"),
        ("adv_robustness", "Adversarial robustness"),
        ("open_safety", "Open-ended safety"),
        ("mmlu_accuracy", "MMLU accuracy"),
        ("score", "SAFE-PoRT score"),
    ]:
        ax.plot(x, df[col], marker="o", linewidth=2.5, label=label, color=colors[col])
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["method"], rotation=20, ha="right")
    ax.set_ylim(30, 90)
    ax.set_ylabel("Metric value (%)")
    ax.set_title("Ablation trend when adding SAFE-PoRT components", fontsize=14, weight="bold")
    ax.grid(axis="y", color="#D5DBDB", linewidth=0.8)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "safe_port_ablation_trend.png", dpi=220)
    plt.close(fig)


def save_sota_chart(df):
    metrics = ["forget_success", "adv_robustness", "open_safety", "mmlu_accuracy", "neighbor_utility", "score"]
    labels = ["Forget", "Adv robust", "Open safety", "MMLU", "Neighbor", "Overall"]
    colors = ["#0B4F6C", "#1F7A8C", "#5B3F8C"]
    fig, ax = plt.subplots(figsize=(11.8, 6.0))
    width = 0.24
    base_x = range(len(metrics))
    for i, (_, row) in enumerate(df.iterrows()):
        xs = [x + (i - 1) * width for x in base_x]
        ax.bar(xs, [row[m] for m in metrics], width=width, label=row["method"], color=colors[i])
    ax.set_xticks(list(base_x))
    ax.set_xticklabels(labels)
    ax.set_ylim(58, 92)
    ax.set_ylabel("Metric value (%)")
    ax.set_title("Comparison with representative SOTA methods", fontsize=14, weight="bold")
    ax.grid(axis="y", color="#D5DBDB", linewidth=0.8)
    ax.legend(loc="upper left", ncols=3, frameon=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "safe_port_sota_comparison.png", dpi=220)
    plt.close(fig)


def save_tradeoff_chart(ablation_df, sota_df):
    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    safety = (ablation_df["forget_success"] + ablation_df["adv_robustness"] + ablation_df["open_safety"]) / 3
    ax.plot(safety, ablation_df["mmlu_accuracy"], color="#005A8C", linewidth=2.2, marker="o")
    ablation_offsets = {
        "+Adv variants": (-66, 8),
        "+Belief negatives": (8, -2),
        "Full SAFE-PoRT": (8, -12),
    }
    for i, row in ablation_df.iterrows():
        offset = ablation_offsets.get(row["method"], (4, 4))
        ax.annotate(row["method"], (safety.iloc[i], row["mmlu_accuracy"]), fontsize=8, xytext=offset, textcoords="offset points")
    sota_safety = (sota_df["forget_success"] + sota_df["adv_robustness"] + sota_df["open_safety"]) / 3
    ax.scatter(sota_safety, sota_df["mmlu_accuracy"], marker="s", color="#B03A2E", s=70, label="SOTA references")
    sota_offsets = {
        "PoRT": (6, -10),
        "LLM Beliefs": (6, -10),
        "SAFE-PoRT": (6, 8),
    }
    for i, row in sota_df.iterrows():
        ax.annotate(row["method"], (sota_safety.iloc[i], row["mmlu_accuracy"]), fontsize=8, xytext=sota_offsets[row["method"]], textcoords="offset points")
    ax.set_xlabel("Safety average: forget + adversarial + open-ended (%)")
    ax.set_ylabel("MMLU accuracy after unlearning (%)")
    ax.set_title("Safety-MMLU trade-off", fontsize=14, weight="bold")
    ax.set_xlim(38, 86)
    ax.set_ylim(60.5, 64.5)
    ax.grid(True, color="#D5DBDB", linewidth=0.8)
    ax.legend(loc="lower left", frameon=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "safe_port_tradeoff.png", dpi=220)
    plt.close(fig)


def main():
    setup_style()
    ablation_df = pd.DataFrame(ABLATION)
    sota_df = pd.DataFrame(SOTA)
    ablation_df.to_csv(OUT_DIR / "safe_port_ablation_metrics.csv", index=False)
    sota_df.to_csv(OUT_DIR / "safe_port_sota_metrics.csv", index=False)
    save_ablation_chart(ablation_df)
    save_sota_chart(sota_df)
    save_tradeoff_chart(ablation_df, sota_df)
    print(OUT_DIR)


if __name__ == "__main__":
    main()
