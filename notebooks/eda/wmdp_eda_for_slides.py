from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datasets import load_from_disk


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "dataset" / "WMDP"
FIG_DIR = ROOT / "slide" / "images" / "eda"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DOMAINS = ["bio", "chem", "cyber"]
VARIANTS = ["original", "noise_prefix", "composite"]
DOMAIN_LABELS = {
    "bio": "Bio",
    "chem": "Chem",
    "cyber": "Cyber",
}
VARIANT_LABELS = {
    "original": "Original",
    "noise_prefix": "Noise prefix",
    "composite": "Composite",
}


def word_count(text):
    if not isinstance(text, str):
        return 0
    return len(re.findall(r"\b\w+\b", text))


def choice_word_count(choices):
    if choices is None:
        return 0
    return sum(word_count(str(choice)) for choice in list(choices))


def load_wmdp():
    frames = []

    for domain in DOMAINS:
        path = DATA_ROOT / "original" / f"wmdp-{domain}" / "test-00000-of-00001.parquet"
        df = pd.read_parquet(path)
        df["variant"] = "original"
        df["domain"] = domain
        frames.append(df)

    for variant in ["noise_prefix", "composite"]:
        for domain in DOMAINS:
            ds = load_from_disk(str(DATA_ROOT / variant / domain))["test"]
            df = ds.to_pandas()
            df["variant"] = variant
            df["domain"] = domain
            frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    df["domain_label"] = df["domain"].map(DOMAIN_LABELS)
    df["variant_label"] = df["variant"].map(VARIANT_LABELS)
    df["answer_label"] = df["answer"].map({0: "A", 1: "B", 2: "C", 3: "D"})
    df["question_words"] = df["question"].map(word_count)
    df["choice_words"] = df["choices"].map(choice_word_count)
    df["full_question_words"] = df.get("full_question", pd.Series([None] * len(df))).map(word_count)
    df["prompt_words"] = df["full_question_words"]
    original_mask = df["variant"].eq("original")
    df.loc[original_mask, "prompt_words"] = (
        df.loc[original_mask, "question_words"] + df.loc[original_mask, "choice_words"]
    )
    df["noisy_prefix_words"] = df.get("noisy_prefix", pd.Series([None] * len(df))).map(word_count)
    return df


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=220, bbox_inches="tight")
    plt.close()


def plot_domain_counts(df):
    counts = (
        df.groupby(["variant_label", "domain_label"])
        .size()
        .reset_index(name="records")
    )
    plt.figure(figsize=(8.6, 4.8))
    ax = sns.barplot(
        data=counts,
        x="domain_label",
        y="records",
        hue="variant_label",
        order=["Bio", "Chem", "Cyber"],
        hue_order=["Original", "Noise prefix", "Composite"],
        palette=["#2c7fb8", "#7fcdbb", "#edf8b1"],
    )
    ax.set_title("WMDP size by domain and variant")
    ax.set_xlabel("Domain")
    ax.set_ylabel("Number of records")
    for container in ax.containers:
        ax.bar_label(container, fontsize=8, padding=2)
    ax.legend(title="Variant", loc="upper left")
    savefig("wmdp_domain_variant_counts.png")


def plot_answer_distribution(df):
    base = df[df["variant"].eq("original")]
    counts = (
        base.groupby(["domain_label", "answer_label"])
        .size()
        .reset_index(name="records")
    )
    totals = counts.groupby("domain_label")["records"].transform("sum")
    counts["percent"] = counts["records"] / totals * 100

    pivot = counts.pivot(index="domain_label", columns="answer_label", values="percent")
    pivot = pivot.loc[["Bio", "Chem", "Cyber"], ["A", "B", "C", "D"]]

    plt.figure(figsize=(7.3, 4.4))
    ax = sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        cbar_kws={"label": "% of records"},
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_title("Answer label distribution in WMDP original")
    ax.set_xlabel("Correct answer")
    ax.set_ylabel("Domain")
    savefig("wmdp_answer_distribution_heatmap.png")


def plot_question_lengths(df):
    base = df[df["variant"].eq("original")]
    plt.figure(figsize=(8.2, 4.8))
    ax = sns.boxplot(
        data=base,
        x="domain_label",
        y="question_words",
        order=["Bio", "Chem", "Cyber"],
        color="#9ecae1",
        showfliers=False,
    )
    sns.stripplot(
        data=base.sample(min(len(base), 800), random_state=7),
        x="domain_label",
        y="question_words",
        order=["Bio", "Chem", "Cyber"],
        color="#08519c",
        alpha=0.22,
        size=2,
    )
    ax.set_title("Question length distribution (original WMDP)")
    ax.set_xlabel("Domain")
    ax.set_ylabel("Question length (words)")
    savefig("wmdp_question_length_boxplot.png")


def plot_prompt_lengths(df):
    plt.figure(figsize=(9.2, 4.8))
    ax = sns.boxplot(
        data=df,
        x="domain_label",
        y="prompt_words",
        hue="variant_label",
        order=["Bio", "Chem", "Cyber"],
        hue_order=["Original", "Noise prefix", "Composite"],
        palette=["#2c7fb8", "#7fcdbb", "#edf8b1"],
        showfliers=False,
    )
    ax.set_title("Prompt length distribution by variant")
    ax.set_xlabel("Domain")
    ax.set_ylabel("Prompt length (words)")
    ax.legend(title="Variant", loc="upper left")
    savefig("wmdp_prompt_length_by_variant.png")


def plot_harmful_position(df):
    comp = df[df["variant"].eq("composite")].dropna(subset=["harmful_position"]).copy()
    comp["harmful_position"] = comp["harmful_position"].astype(int)
    plt.figure(figsize=(8.5, 4.6))
    ax = sns.histplot(
        data=comp,
        x="harmful_position",
        hue="domain_label",
        hue_order=["Bio", "Chem", "Cyber"],
        multiple="stack",
        bins=30,
        palette=["#2c7fb8", "#7fcdbb", "#edf8b1"],
    )
    ax.set_title("Position of harmful component in composite prompts")
    ax.set_xlabel("Harmful position index")
    ax.set_ylabel("Number of records")
    savefig("wmdp_composite_harmful_position.png")


def write_summary(df):
    summary = (
        df.groupby(["variant", "domain"])
        .agg(
            records=("answer", "size"),
            median_question_words=("question_words", "median"),
            p90_question_words=("question_words", lambda x: x.quantile(0.90)),
            median_prompt_words=("prompt_words", "median"),
            p90_prompt_words=("prompt_words", lambda x: x.quantile(0.90)),
            median_prefix_words=("noisy_prefix_words", "median"),
        )
        .reset_index()
    )
    summary.to_csv(FIG_DIR / "wmdp_eda_summary.csv", index=False)
    print(summary.to_string(index=False))

    answer_counts = (
        df[df["variant"].eq("original")]
        .groupby(["domain", "answer_label"])
        .size()
        .unstack(fill_value=0)
    )
    answer_counts.to_csv(FIG_DIR / "wmdp_answer_counts_original.csv")


def main():
    sns.set_theme(style="whitegrid", font_scale=1.0)
    df = load_wmdp()
    write_summary(df)
    plot_domain_counts(df)
    plot_answer_distribution(df)
    plot_question_lengths(df)
    plot_prompt_lengths(df)
    plot_harmful_position(df)
    print(f"Saved figures to: {FIG_DIR}")


if __name__ == "__main__":
    main()
