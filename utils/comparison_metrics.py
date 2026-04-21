import pandas as pd
from scipy import stats
import numpy as np
import pickle


METRICS = [
    "Total time",
    "Average time",
    "Avg Congestion delay",
    "Avg path length",
    "Avg Path Efficiency",
    "Exit utilisation",
]


def extract_results(path):
    with open(path, "rb") as f:
        results = pickle.load(f)

    return results


def extract_metrics(results: dict, algorithm: str) -> dict:
    """Extract metric arrays across seeds for one algorithm."""
    evals = results[algorithm]["evals"]
    return {metric: np.array([e.metrics[metric] for e in evals]) for metric in METRICS}


def build_results_table(baseline_files: dict) -> pd.DataFrame:
    """
    Builds a summary table with mean ± std for each metric,
    both algorithms, across all three cities.
    Also runs Mann-Whitney U test on Total time.
    """
    rows = []

    for city, path in baseline_files.items():
        results = extract_results(path)

        ga_metrics = extract_metrics(results, "ga")
        greedy_metrics = extract_metrics(results, "greedy")

        # wilcoxon on each metric individually
        p_values = {}
        for metric in METRICS:
            try:
                _, p = stats.wilcoxon(ga_metrics[metric], greedy_metrics[metric])
                p_values[metric] = p
            except ValueError:
                # all differences are zero - algorithms identical on this metric
                p_values[metric] = 1.0

        for metric in METRICS:
            rows.append(
                {
                    "City": city,
                    "Metric": metric,
                    "GA Mean": ga_metrics[metric].mean(),
                    "GA Std": ga_metrics[metric].std(),
                    "Greedy Mean": greedy_metrics[metric].mean(),
                    "Greedy Std": greedy_metrics[metric].std(),
                    "p-value": p_values[metric],
                }
            )

    df = pd.DataFrame(rows)
    return df


def format_results_table(df: pd.DataFrame) -> pd.DataFrame:
    display = df.copy()
    display["GA"] = display.apply(lambda r: f"{r['GA Mean']:.2f} ± {r['GA Std']:.2f}", axis=1)
    display["Greedy"] = display.apply(
        lambda r: f"{r['Greedy Mean']:.2f} ± {r['Greedy Std']:.2f}", axis=1
    )
    display["p"] = display["p-value"].apply(lambda p: f"{p:.3f}{'*' if p < 0.05 else ''}")
    return display[["City", "Metric", "GA", "Greedy", "p"]].pivot(
        index="Metric", columns="City", values=["GA", "Greedy", "p"]
    )


def cohens_d(a, b):
    diff = a - b
    return diff.mean() / diff.std()


def extract_cohens_d(files: dict):
    for city, path in files.items():
        results = extract_results(path)
        ga_times = np.array([e.metrics["Total time"] for e in results["ga"]["evals"]])
        greedy_times = np.array([e.metrics["Total time"] for e in results["greedy"]["evals"]])
        d = cohens_d(ga_times, greedy_times)
        _, p = (
            stats.wilcoxon(ga_times, greedy_times)
            if not all(ga_times == greedy_times)
            else (None, 1.0)
        )
        print(f"{city}: d={d:.2f}, p={p:.3f}")


def calculate_poa(baseline_files: dict) -> pd.DataFrame:
    rows = []
    for city, path in baseline_files.items():
        results = extract_results(path)

        ga_times = np.array([e.metrics["Total time"] for e in results["ga"]["evals"]])
        greedy_times = np.array([e.metrics["Total time"] for e in results["greedy"]["evals"]])

        ga_mean = ga_times.mean()
        greedy_mean = greedy_times.mean()
        poa = greedy_mean / ga_mean  # > 1 means GA wins, < 1 means greedy wins

        rows.append(
            {
                "Topology": city,
                "GA Mean": ga_mean,
                "Greedy Mean": greedy_mean,
                "Ratio (Greedy/GA)": round(poa, 3),
                "Winner": "GA" if poa > 1 else "Greedy",
            }
        )

    return pd.DataFrame(rows).set_index("Topology")
