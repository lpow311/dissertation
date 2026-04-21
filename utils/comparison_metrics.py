import pandas as pd
from scipy import stats
import numpy as np
import pickle
import matplotlib.pyplot as plt


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


def build_delta_table(files_a: dict, files_b: dict, label_a: str, label_b: str) -> pd.DataFrame:
    """Compare mean metrics between two experiments for each algorithm and city."""
    rows = []
    for city in files_a.keys():
        results_a = extract_results(files_a[city])
        results_b = extract_results(files_b[city])

        for algorithm in ["ga", "greedy"]:
            for metric in METRICS:
                mean_a = np.mean([e.metrics[metric] for e in results_a[algorithm]["evals"]])
                mean_b = np.mean([e.metrics[metric] for e in results_b[algorithm]["evals"]])
                delta = mean_b - mean_a  # positive = walking speed made it worse
                rows.append(
                    {
                        "City": city,
                        "Algorithm": algorithm.upper(),
                        "Metric": metric,
                        f"{label_a} Mean": round(mean_a, 2),
                        f"{label_b} Mean": round(mean_b, 2),
                        "Delta": round(delta, 2),
                    }
                )

    return pd.DataFrame(rows)


def plot_dispersion_all_cities(baseline_files: dict, n: int = 3) -> None:
    ga_colour = "#378ADD"
    greedy_colour = "#1D9E75"
    cities = list(baseline_files.keys())

    fig, axes = plt.subplots(1, len(cities), figsize=(5 * len(cities), 4))

    def count_agents_within_x(timestep_dict: dict, timestep: int, x: int) -> int:
        count = 0
        for agent_id, positions in timestep_dict.items():
            if timestep >= len(positions):
                continue
            steps_from_start = len(set(positions[: timestep + 1])) - 1
            if steps_from_start <= x:
                count += 1
        return count

    for col, city in enumerate(cities):
        ax = axes[col]
        results = extract_results(baseline_files[city])

        max_t = max(
            len(positions)
            for algorithm in ["ga", "greedy"]
            for ev in results[algorithm]["evals"]
            for positions in ev.chromosome.get_timesteps()[0].values()
        )
        timesteps = np.arange(max_t)

        for algorithm, colour, label in [
            ("ga", ga_colour, "GA"),
            ("greedy", greedy_colour, "Greedy"),
        ]:
            seed_curves = []
            for ev in results[algorithm]["evals"]:
                timestep_dict = ev.chromosome.get_timesteps()[0]
                curve = [count_agents_within_x(timestep_dict, t, n) for t in timesteps]
                seed_curves.append(curve)

            seed_curves = np.array(seed_curves)
            mean = seed_curves.mean(axis=0)
            std = seed_curves.std(axis=0)

            ax.plot(timesteps, mean, color=colour, linewidth=1.5, label=label)
            ax.fill_between(timesteps, mean - std, mean + std, alpha=0.15, color=colour)

        ax.set_title(f"{city} Topology")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Agents within 3 nodes of start" if col == 0 else "")
        ax.grid(True, alpha=0.3)

        if col == 0:
            ax.legend()

    plt.suptitle(f"Agent Dispersion: Agents Within {n} Nodes of Start — GA vs Greedy", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_metrics_by_walking_speed(
    walking_files: dict, metrics: list = None, metric_labels: list = None
) -> None:
    """
    Bar chart of average time and congestion delay per walking speed category,
    GA vs Greedy, per topology.
    """
    cities = list(walking_files.keys())
    speed_categories = [1, 2, 3]
    if metrics is None:
        metrics = ["path_time", "congestion_score"]
        metric_labels = ["Average Time", "Congestion Delay"]

    ga_colour = "#378ADD"
    greedy_colour = "#1D9E75"

    fig, axes = plt.subplots(len(metrics), len(cities), figsize=(5 * len(cities), 4 * len(metrics)))

    for row, (metric, label) in enumerate(zip(metrics, metric_labels)):
        for col, city in enumerate(cities):
            ax = axes[row, col]
            results = extract_results(walking_files[city])

            x = np.arange(len(speed_categories))
            width = 0.35

            for i, (algorithm, colour, alg_label) in enumerate(
                [("ga", ga_colour, "GA"), ("greedy", greedy_colour, "Greedy")]
            ):
                # aggregate across seeds
                speed_means = []
                speed_stds = []

                for speed in speed_categories:
                    seed_values = []
                    for ev in results[algorithm]["evals"]:
                        # get metric values for agents of this speed
                        agent_values = [
                            getattr(ev.chromosome, metric)[agent_id]
                            for agent_id, agent in ev.solution.items()
                            if agent.characteristics["walking"] == speed
                        ]
                        if agent_values:
                            seed_values.append(np.mean(agent_values))

                    speed_means.append(np.mean(seed_values))
                    speed_stds.append(np.std(seed_values))

                offset = (i - 0.5) * width
                bars = ax.bar(
                    x + offset,
                    speed_means,
                    width,
                    label=alg_label,
                    color=colour,
                    alpha=0.8,
                    yerr=speed_stds,
                    capsize=4,
                )

            ax.set_title(f"{city} — {label}")
            ax.set_xticks(x)
            ax.set_xticklabels(["Speed 1\n(Fast)", "Speed 2\n(Med)", "Speed 3\n(Slow)"])
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3, axis="y")

            if row == 0 and col == 0:
                ax.legend()

    plt.suptitle("Metrics by Walking Speed Category: GA vs Greedy", fontsize=14)
    plt.tight_layout()
    plt.show()
