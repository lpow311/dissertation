import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import random

# ── Fixed parameters across ALL cities ───────────────────────────────────────
N_NODES = 25
N_EXITS = 3
N_STARTS = 5
N_AGENTS = 50
NODE_CAP = 3
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Edge probability ranges that tend to produce each category
# Low density  → sparse graph → constrained
# Mid density  → moderate
# High density → flexible
EDGE_PROB = {
    "Constrained": (0.08, 0.15),
    "Moderate": (0.16, 0.25),
    "Flexible": (0.30, 0.45),
}


# ── Metrics ───────────────────────────────────────────────────────────────────
def calculate_metrics(G, exits, num_agents=N_AGENTS, node_capacity=NODE_CAP):
    betweenness = nx.betweenness_centrality(G)
    max_bc = max(betweenness.values())
    art_points = list(nx.articulation_points(G))
    connectivity = nx.node_connectivity(G)

    path_lengths = []
    for node in G.nodes():
        if node not in exits:
            lengths = [
                nx.shortest_path_length(G, node, e) for e in exits if nx.has_path(G, node, e)
            ]
            if lengths:
                path_lengths.append(min(lengths))

    avg_path = round(np.mean(path_lengths), 2) if path_lengths else float("inf")
    exit_ratio = round(len(exits) / num_agents, 3)

    # Bottleneck capacity ratio — uses expected flow through worst node
    bottleneck_node = max(betweenness, key=betweenness.get)
    expected_flow = max_bc * num_agents
    bn_cap_ratio = round(node_capacity / expected_flow, 3) if expected_flow > 0 else float("inf")

    # Classification
    if max_bc > 0.4 and connectivity <= 2:
        category = "Constrained"
    elif max_bc < 0.25 and connectivity >= 3 and len(art_points) == 0:
        category = "Flexible"
    else:
        category = "Moderate"

    return {
        "max_betweenness": round(max_bc, 3),
        "articulation_points": len(art_points),
        "avg_path_length": avg_path,
        "node_connectivity": connectivity,
        "exit_ratio": exit_ratio,
        "bottleneck_capacity_ratio": bn_cap_ratio,
        "bottleneck_node": bottleneck_node,
        "category": category,
    }


# ── City generator ────────────────────────────────────────────────────────────
def try_generate_city(edge_prob, rng_seed):
    """
    Attempt to build one valid random city.
    Returns (G, exits, starts) or None if the graph doesn't meet basic requirements.
    """
    G = nx.erdos_renyi_graph(N_NODES, edge_prob, seed=rng_seed)

    # Must be connected
    if not nx.is_connected(G):
        return None

    # Relabel to strings
    G = nx.relabel_nodes(G, {i: f"Node_{i}" for i in G.nodes()})
    nodes = list(G.nodes())

    # Pick exits on high-degree nodes (realistic — exits tend to be accessible)
    degrees = dict(G.degree())
    sorted_deg = sorted(degrees, key=degrees.get, reverse=True)

    # Spread exits: pick from top-degree nodes but not adjacent to each other
    exits = []
    for candidate in sorted_deg:
        if len(exits) >= N_EXITS:
            break
        if not any(G.has_edge(candidate, e) for e in exits):
            exits.append(candidate)

    if len(exits) < N_EXITS:
        return None

    # Pick starts: low-degree nodes far from exits (opposite side of graph)
    exit_set = set(exits)
    non_exit = [n for n in nodes if n not in exit_set]
    low_deg = sorted(non_exit, key=lambda n: degrees[n])
    starts = low_deg[:N_STARTS]

    # Verify all starts can reach all exits
    for s in starts:
        for e in exits:
            if not nx.has_path(G, s, e):
                return None

    return G, exits, starts


def generate_city_set(target_category, n=3, max_attempts=500):
    """
    Generate n cities that classify as target_category.
    Uses edge probability ranges appropriate for each category.
    """
    lo, hi = EDGE_PROB[target_category]
    cities = []
    attempt = 0

    while len(cities) < n and attempt < max_attempts:
        edge_prob = random.uniform(lo, hi)
        result = try_generate_city(edge_prob, rng_seed=attempt + 1000)

        if result is not None:
            G, exits, starts = result
            metrics = calculate_metrics(G, exits)

            if metrics["category"] == target_category:
                cities.append(
                    {
                        "graph": G,
                        "exits": exits,
                        "starts": starts,
                        "metrics": metrics,
                        "id": f"{target_category}_{len(cities) + 1}",
                    }
                )
                print(
                    f"  Found {target_category} city {len(cities)}/{n} "
                    f"(attempt {attempt}, edge_prob={edge_prob:.3f})"
                )

        attempt += 1

    if len(cities) < n:
        print(
            f"  WARNING: Only found {len(cities)}/{n} {target_category} cities "
            f"after {max_attempts} attempts"
        )

    return cities


# ── Visualisation ─────────────────────────────────────────────────────────────
def visualise_city(city_dict, ax):
    G = city_dict["graph"]
    exits = city_dict["exits"]
    starts = city_dict["starts"]
    metrics = city_dict["metrics"]
    city_id = city_dict["id"]

    pos = nx.spring_layout(G, seed=42)

    colours = []
    sizes = []
    for node in G.nodes():
        if node in exits:
            colours.append("#2ecc71")
            sizes.append(700)
        elif node in starts:
            colours.append("#3498db")
            sizes.append(700)
        elif node == metrics["bottleneck_node"]:
            colours.append("#e74c3c")
            sizes.append(900)
        else:
            colours.append("#f39c12")
            sizes.append(400)

    nx.draw(
        G,
        pos,
        node_color=colours,
        node_size=sizes,
        with_labels=True,
        font_size=5,
        ax=ax,
        font_weight="bold",
    )

    legend = [
        mpatches.Patch(color="#3498db", label="Start"),
        mpatches.Patch(color="#2ecc71", label="Exit"),
        mpatches.Patch(color="#e74c3c", label="Bottleneck"),
        mpatches.Patch(color="#f39c12", label="Corridor"),
    ]
    ax.legend(handles=legend, fontsize=7, loc="upper right")
    ax.set_title(
        f"{city_id}\n"
        f"BC={metrics['max_betweenness']} | "
        f"Conn={metrics['node_connectivity']} | "
        f"ArtPts={metrics['articulation_points']} | "
        f"AvgPath={metrics['avg_path_length']}",
        fontsize=8,
    )


def visualise_single_city(G, exits, starts, city_id="City", ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    pos = nx.spring_layout(G, seed=42)

    colours = []
    sizes = []
    for node in G.nodes():
        if node in exits:
            colours.append("#2ecc71")
            sizes.append(700)
        elif node in starts:
            colours.append("#3498db")
            sizes.append(700)
        else:
            colours.append("#f39c12")
            sizes.append(400)

    nx.draw(
        G,
        pos,
        node_color=colours,
        node_size=sizes,
        with_labels=True,
        font_size=5,
        ax=ax,
        font_weight="bold",
    )

    legend = [
        mpatches.Patch(color="#3498db", label="Start"),
        mpatches.Patch(color="#2ecc71", label="Exit"),
        mpatches.Patch(color="#e74c3c", label="Bottleneck"),
        mpatches.Patch(color="#f39c12", label="Corridor"),
    ]
    ax.legend(handles=legend, fontsize=7, loc="upper right")
    ax.set_title(city_id.replace("_", " ").capitalize(), fontsize=8)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    all_cities = {}

    for category in ["Constrained", "Moderate", "Flexible"]:
        print(f"\nGenerating {category} cities...")
        all_cities[category] = generate_city_set(category, n=3)

    # ── Save pickles ──────────────────────────────────────────────────────────
    print("\nSaving pickles...")
    for category, cities in all_cities.items():
        for city in cities:
            path = f"graphs/{city['id'].lower().replace(' ', '_')}.pkl"
            with open(path, "wb") as f:
                pickle.dump(city, f)
            print(f"  Saved {path}")

    # ── Plot: 3 rows (categories) x 3 cols (cities) ───────────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(22, 18))
    fig.suptitle(
        "Generated Cities — Constrained / Moderate / Flexible (3 each)",
        fontsize=14,
        fontweight="bold",
    )

    for row, category in enumerate(["Constrained", "Moderate", "Flexible"]):
        for col, city in enumerate(all_cities[category]):
            visualise_city(city, axes[row][col])

    plt.tight_layout()
    plt.savefig("images/generated_cities.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nSaved generated_cities.png")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n\nSUMMARY OF ALL GENERATED CITIES")
    header = f"{'City':<22} {'Category':<14} {'MaxBC':>8} {'ArtPts':>8} {'AvgPath':>9} {'Conn':>6} {'BnRatio':>9}"
    print(header)
    print("-" * 80)
    for category, cities in all_cities.items():
        for city in cities:
            m = city["metrics"]
            print(
                f"{city['id']:<22} {m['category']:<14} "
                f"{m['max_betweenness']:>8} {m['articulation_points']:>8} "
                f"{m['avg_path_length']:>9} {m['node_connectivity']:>6} "
                f"{m['bottleneck_capacity_ratio']:>9}"
            )
