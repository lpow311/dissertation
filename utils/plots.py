import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import networkx as nx
from collections import Counter

################################ Notebook 0 & 1 ################################


def get_traversal(traversals, u, v):
    return traversals.get((u, v), traversals.get((v, u), 0))


def plot_multiple_solutions(
    weighted_cities: dict, pos: dict, starts: list, exits: list, plot_title: str
) -> None:

    fig, axes = plt.subplots(1, len(weighted_cities), figsize=(24, 8))

    for ax, name in zip(axes, weighted_cities):
        G_weighted = weighted_cities[name]
        traversals = nx.get_edge_attributes(G_weighted, "traversals")
        max_traversals = max(traversals.values()) if traversals else 1

        edge_widths = [
            1 + 8 * (get_traversal(traversals, u, v) / max_traversals)
            for u, v in G_weighted.edges()
        ]

        node_colours = [
            "#6bbc8d" if n in starts else "#da5f70" if n in exits else "#5fb8d6"
            for n in G_weighted.nodes()
        ]

        nx.draw_networkx(
            G_weighted,
            pos,
            ax=ax,
            node_color=node_colours,
            node_size=500,
            font_size=8,
            font_color="black",
            width=edge_widths,
        )

        labels = {(u, v): str(w) for (u, v), w in traversals.items() if w > 0}
        nx.draw_networkx_edge_labels(G_weighted, pos, edge_labels=labels, font_size=7, ax=ax)

        ax.set_title(name, fontsize=14, fontweight="bold")
        ax.axis("off")

    plt.suptitle(plot_title, fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_traversal_weights(G_weighted, pos, starts, exits, title):
    traversals = nx.get_edge_attributes(G_weighted, "traversals")
    max_traversals = max(traversals.values()) if traversals else 1

    edge_widths = [
        1 + 10 * (get_traversal(traversals, u, v) / max_traversals) for u, v in G_weighted.edges()
    ]

    starts, exits = set(starts), set(exits)
    node_colours = [
        "#85c19e" if n in starts else "#e09ca5" if n in exits else "#87c3d6"
        for n in G_weighted.nodes()
    ]

    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx(
        G_weighted,
        pos,
        ax=ax,
        node_color=node_colours,
        node_size=500,
        font_size=8,
        font_color="black",
        width=edge_widths,
    )

    # Annotate edges with traversal counts (only non-zero)
    labels = {(u, v): str(w) for (u, v), w in traversals.items() if w > 0}
    nx.draw_networkx_edge_labels(G_weighted, pos, edge_labels=labels, font_size=7, ax=ax)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def add_traversal_weights(G: nx.Graph, agents: dict) -> nx.Graph:
    G_weighted = G.copy()

    # Initialise all edges with 0 traversal count
    nx.set_edge_attributes(G_weighted, 0, "traversals")

    # Count how many agents traverse each edge
    edge_counts = Counter()
    for agent in agents.values():
        if agent.path:
            for u, v in zip(agent.path[:-1], agent.path[1:]):
                edge_counts[frozenset((u, v))] += 1

    # Apply counts to graph
    for u, v, data in G_weighted.edges(data=True):
        data["traversals"] = edge_counts.get(frozenset((u, v)), 0)

    return G_weighted


################################ Notebook 2 ################################


def extract_metric(evolution_stats, metric_key, component):
    """Extract a metric across all seeds and generations."""
    all_seeds = []
    for seed_stats in evolution_stats:
        seed_values = []
        for evolution_num, results in seed_stats.generation_stats.items():
            if component is not None:
                seed_values.append(results[component][metric_key])
            else:
                seed_values.append(results[metric_key])
        all_seeds.append(seed_values)
    return np.array(all_seeds)  # shape: (n_seeds, n_generations)


def plot_parent_selection_stats(evolution_stats, city_name: str, method: str = "Roulette"):
    cv_all = extract_metric(evolution_stats, "cv", "parent_selection")
    dominance_all = extract_metric(evolution_stats, "dominance", "parent_selection")
    best_share_all = extract_metric(evolution_stats, "best_share", "parent_selection")

    evolutions = range(cv_all.shape[1])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = [
        (cv_all, "CV (%)", "Coefficient of Variation", 5, 30),
        (dominance_all, "Dominance (x)", "Dominance Ratio", 1.5, 10),
        (best_share_all, "Share (%)", "Best Chromosome Wheel Share", None, 20),
    ]

    for ax, (values, ylabel, title, lower, upper) in zip(axes, metrics):
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0)

        # Mean line
        ax.plot(
            evolutions, mean, color="#3498db", linewidth=2, marker="o", markersize=4, label="Mean"
        )

        # Std band
        ax.fill_between(
            evolutions, mean - std, mean + std, color="#3498db", alpha=0.2, label="±1 std"
        )

        # Individual seed lines (faint)
        for seed_vals in values:
            ax.plot(evolutions, seed_vals, color="#3498db", alpha=0.1, linewidth=0.8)

        # Thresholds
        if lower:
            ax.axhline(
                lower, color="#2ecc71", linestyle="--", linewidth=1.5, label=f"Lower ({lower})"
            )
        if upper:
            ax.axhline(
                upper, color="#e74c3c", linestyle="--", linewidth=1.5, label=f"Upper ({upper})"
            )

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Generation")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    formatted = city_name.replace("_", " ").title()
    fig.suptitle(
        f"Suitability for {formatted} — {len(evolution_stats)} Seeds",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


def plot_survivor_selection(evolution_stats, city_name: str):
    parent_count = extract_metric(evolution_stats, "parent_count", "survivor_selection")
    child_count = extract_metric(evolution_stats, "child_count", "survivor_selection")
    parent_fitness = extract_metric(evolution_stats, "parent_fitness", "survivor_selection")
    child_fitness = extract_metric(evolution_stats, "child_fitness", "survivor_selection")
    try:
        passthrough = extract_metric(
            evolution_stats, "identical_children", "survivor_selection"
        )  # passthrough
    except KeyError:
        passthrough = extract_metric(
            evolution_stats, "passthrough", "survivor_selection"
        )  # passthrough

    evolutions = range(parent_count.shape[1])

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # ── Plot 1: Counts ───────────────────────────────────────────────────────
    ax = axes[0]
    for values, label, colour in [
        (parent_count, "Parents", "#da5f70"),
        (child_count, "Children", "#6bbc8d"),
        (passthrough, "Passthrough", "#e8a838"),
    ]:
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0)
        ax.plot(evolutions, mean, linewidth=2, marker="o", markersize=4, label=label, color=colour)
        ax.fill_between(evolutions, mean - std, mean + std, alpha=0.2, color=colour)
        for seed_vals in values:
            ax.plot(evolutions, seed_vals, color=colour, alpha=0.1, linewidth=0.8)

    ax.set_title("Survivor Counts for Per Generation", fontsize=11, fontweight="bold")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    # ── Plot 2: Fitness ──────────────────────────────────────────────────────
    ax = axes[1]
    for values, label, colour in [
        (parent_fitness, "Best Parent Fitness", "#da5f70"),
        (child_fitness, "Best Child Fitness", "#6bbc8d"),
    ]:
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0)
        ax.plot(evolutions, mean, linewidth=2, marker="o", markersize=4, label=label, color=colour)
        ax.fill_between(evolutions, mean - std, mean + std, alpha=0.2, color=colour)
        for seed_vals in values:
            ax.plot(evolutions, seed_vals, color=colour, alpha=0.1, linewidth=0.8)

    ax.set_title("Best Fitness Per Generation", fontsize=11, fontweight="bold")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    title = f"Survivor Selection Analysis for {city_name.replace('_', ' ').title()}"
    fig.suptitle(f"{title} — {len(evolution_stats)} Seeds", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_covergence(evolution_stats_per_city):
    fig, axes = plt.subplots(1, len(evolution_stats_per_city), figsize=(18, 6))

    for ax, (city_name, evolution_stats) in zip(axes, evolution_stats_per_city.items()):
        fitnesses = extract_metric(evolution_stats, "fitnesses", None)

        best = np.array([[min(f) for f in seed] for seed in fitnesses])
        worst = np.array([[max(f) for f in seed] for seed in fitnesses])
        average = np.array([[np.mean(f) for f in seed] for seed in fitnesses])

        evolutions = range(best.shape[1])

        for values, label, colour in [
            (best, "Best", "#6bbc8d"),
            (average, "Average", "#3498db"),
            (worst, "Worst", "#da5f70"),
        ]:
            mean = np.mean(values, axis=0)
            std = np.std(values, axis=0)

            ax.plot(
                evolutions, mean, linewidth=2, marker="o", markersize=4, label=label, color=colour
            )
            ax.fill_between(evolutions, mean - std, mean + std, alpha=0.2, color=colour)

        ax.set_title(city_name.replace("_", " ").capitalize(), fontsize=12, fontweight="bold")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness" if ax == axes[0] else "")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Fitness Convergence Across Cities", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def extract_step(step, timing_stats):
    return np.array([exp["ga_components"][step] for exp in timing_stats])


def plot_fitness_vs_time(evolution_stats, timing_stats, title=""):
    import pandas as pd

    # ── Extract fitness ──────────────────────────────────────────────────────
    fitnesses = extract_metric(evolution_stats, "fitnesses", None)
    best = np.array([[min(f) for f in seed] for seed in fitnesses])
    best_mean = np.mean(best, axis=0)

    parent_t = extract_step("parent", timing_stats)
    crossover_t = extract_step("crossover", timing_stats)
    mutation_t = extract_step("mutation", timing_stats)
    survivor_t = extract_step("survivor", timing_stats)

    total_per_gen = parent_t + crossover_t + mutation_t + survivor_t

    parent_mean = np.mean(parent_t, axis=0)
    crossover_mean = np.mean(crossover_t, axis=0)
    mutation_mean = np.mean(mutation_t, axis=0)
    survivor_mean = np.mean(survivor_t, axis=0)
    total_mean = np.mean(total_per_gen, axis=0)

    # ── Fitness improvement (flipped so positive = better) ───────────────────
    fitness_improvement = -np.diff(best_mean)  # positive = improvement
    fitness_improvement = np.insert(fitness_improvement, 0, 0)
    smoothed_improvement = (
        pd.Series(fitness_improvement).rolling(window=10, center=True, min_periods=1).mean()
    )

    n_gen = min(best_mean.shape[0], total_mean.shape[0])
    evolutions = np.arange(n_gen)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # ── Plot A: Time per component ───────────────────────────────────────────
    ax = axes[0]
    ax.bar(evolutions, parent_mean[:n_gen], label="Parent Selection", color="#3498db")
    ax.bar(
        evolutions,
        crossover_mean[:n_gen],
        bottom=parent_mean[:n_gen],
        label="Crossover",
        color="#6bbc8d",
    )
    ax.bar(
        evolutions,
        mutation_mean[:n_gen],
        bottom=parent_mean[:n_gen] + crossover_mean[:n_gen],
        label="Mutation",
        color="#e8a838",
    )
    ax.bar(
        evolutions,
        survivor_mean[:n_gen],
        bottom=parent_mean[:n_gen] + crossover_mean[:n_gen] + mutation_mean[:n_gen],
        label="Survivor",
        color="#da5f70",
    )

    ax.set_title("Time Per Component Per Generation", fontsize=12, fontweight="bold")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Time (s)")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    # ── Plot B: Smoothed fitness improvement vs time cost ────────────────────
    ax1 = axes[1]
    ax2 = ax1.twinx()

    ax1.plot(
        evolutions,
        smoothed_improvement[:n_gen],
        color="#6bbc8d",
        linewidth=2,
        marker="o",
        markersize=3,
        label="Fitness Improvement (smoothed)",
    )
    ax1.axhline(0, color="#6bbc8d", linestyle="--", linewidth=1, alpha=0.5)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness Improvement (smoothed)", color="#6bbc8d")
    ax1.tick_params(axis="y", labelcolor="#6bbc8d")

    ax2.plot(
        evolutions,
        total_mean[:n_gen],
        color="#e8a838",
        linewidth=2,
        marker="s",
        markersize=3,
        label="Time Cost (s)",
    )
    ax2.set_ylabel("Time Cost Per Generation (s)", color="#e8a838")
    ax2.set_ylim(0, max(total_mean) * 1.2)
    ax2.tick_params(axis="y", labelcolor="#e8a838")

    # Fix phantom legend entries
    lines = [l for l in ax1.get_lines() + ax2.get_lines() if not l.get_label().startswith("_")]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", fontsize=8)

    ax1.set_title("Fitness Improvement vs Time Cost", fontsize=12, fontweight="bold")
    ax1.spines[["top"]].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
