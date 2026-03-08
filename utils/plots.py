import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import networkx as nx
from collections import Counter


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
