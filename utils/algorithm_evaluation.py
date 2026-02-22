import networkx as nx
import numpy as np
import plotly.graph_objects as go
import pickle


def create_city(
    num_start_nodes: int = 3,
    num_exit_nodes: int = 3,
    num_connection_nodes: int = 10,
    potential_connections: int = 2,
    num_bottlenecks: int = 0,
    bottleneck_positions: str = "mixed",  # "exits", "starts", "mid", "mixed"
    seed: int = 42,
    save_path: str = None,
) -> tuple:

    rng = np.random.default_rng(seed)

    G = nx.Graph()

    # Add nodes with types — same as original
    start_nodes = [(f"Start_{i}", {"type": "start"}) for i in range(1, num_start_nodes + 1)]
    exit_nodes = [(f"Exit_{i}", {"type": "exit"}) for i in range(1, num_exit_nodes + 1)]
    connection_nodes = [
        (f"Connection_{i}", {"type": "connection"}) for i in range(1, num_connection_nodes + 1)
    ]

    G.add_nodes_from(start_nodes)
    G.add_nodes_from(exit_nodes)
    G.add_nodes_from(connection_nodes)

    start_names = [s for s, _ in start_nodes]
    exit_names = [e for e, _ in exit_nodes]
    connection_names = [c for c, _ in connection_nodes]

    # Connect start nodes to connection nodes — same as original
    for s in start_names:
        num_connections = rng.integers(2, 5)
        targets = rng.choice(
            connection_names, size=min(num_connections, len(connection_names)), replace=False
        )
        for t in targets:
            G.add_edge(s, t)

    # Connect connection nodes to other connection nodes or exits — same as original
    for c in connection_names:
        possible_targets = [n for n in connection_names + exit_names if n != c]
        num_connections = rng.integers(1, potential_connections + 1)
        targets = rng.choice(
            possible_targets, size=min(num_connections, len(possible_targets)), replace=False
        )
        for t in targets:
            G.add_edge(c, t)

    # Ensure all exits connected — same as original
    for e in exit_names:
        if len(list(G.neighbors(e))) == 0:
            connection = rng.choice(connection_names)
            G.add_edge(e, connection)

    # Add bottlenecks
    bottleneck_names = []
    if num_bottlenecks > 0:
        G, bottleneck_names = add_bottlenecks(
            G=G,
            num_bottlenecks=num_bottlenecks,
            bottleneck_positions=bottleneck_positions,
            start_names=start_names,
            exit_names=exit_names,
            connection_names=connection_names,
            rng=rng,
        )

    if save_path:
        pickle.dump(G, open(save_path, "wb"))

    return G, start_names, exit_names, bottleneck_names


def add_bottlenecks(
    G: nx.Graph,
    num_bottlenecks: int,
    bottleneck_positions: str,
    start_names: list,
    exit_names: list,
    connection_names: list,
    rng: np.random.Generator,
) -> tuple:

    bottleneck_names = []

    # Decide which nodes to insert bottlenecks near
    if bottleneck_positions == "exits":
        targets = [(e, "exit") for e in exit_names[:num_bottlenecks]]
    elif bottleneck_positions == "starts":
        targets = [(s, "start") for s in start_names[:num_bottlenecks]]
    elif bottleneck_positions == "mid":
        betweenness = nx.betweenness_centrality(G)
        sorted_nodes = sorted(
            [n for n in betweenness if n in connection_names], key=betweenness.get, reverse=True
        )
        targets = [(n, "mid") for n in sorted_nodes[:num_bottlenecks]]
    else:  # mixed
        n_each = max(1, num_bottlenecks // 3)
        betweenness = nx.betweenness_centrality(G)
        sorted_nodes = sorted(
            [n for n in betweenness if n in connection_names], key=betweenness.get, reverse=True
        )
        targets = (
            [(e, "exit") for e in exit_names[:n_each]]
            + [(s, "start") for s in start_names[:n_each]]
            + [(n, "mid") for n in sorted_nodes[:n_each]]
        )

    for target_node, position in targets:
        bottleneck_name = f"Bottleneck_{target_node}"
        bottleneck_names.append(bottleneck_name)

        neighbours = list(G.neighbors(target_node))
        G.add_node(bottleneck_name, type="bottleneck")

        if position == "exit":
            # Insert bottleneck between network and exit
            G.add_edge(bottleneck_name, target_node)
            for neighbour in neighbours:
                G.add_edge(neighbour, bottleneck_name)
                G.remove_edge(neighbour, target_node)

        elif position == "start":
            # Insert bottleneck between start and network
            G.add_edge(target_node, bottleneck_name)
            for neighbour in neighbours:
                G.add_edge(bottleneck_name, neighbour)
                G.remove_edge(target_node, neighbour)

        elif position == "mid":
            # Restrict high betweenness node to only 2 connections
            if len(neighbours) > 2:
                to_remove = rng.choice(neighbours, size=len(neighbours) - 2, replace=False)
                for neighbour in to_remove:
                    if G.has_edge(target_node, neighbour):
                        G.remove_edge(target_node, neighbour)

    return G, bottleneck_names


def remove_bottlenecks(G: nx.Graph, bottleneck_names: list) -> nx.Graph:
    G_clean = G.copy()

    for bottleneck in bottleneck_names:
        if bottleneck in G_clean.nodes():
            neighbours = list(G_clean.neighbors(bottleneck))
            # Reconnect all neighbours directly
            for i, n1 in enumerate(neighbours):
                for n2 in neighbours[i + 1 :]:
                    G_clean.add_edge(n1, n2)
            G_clean.remove_node(bottleneck)

    return G_clean


def visualise_city(G: nx.Graph, title: str = "City Graph", save_path: str = None):
    pos = nx.spring_layout(G, seed=42)

    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, line=dict(width=1, color="#888"), hoverinfo="none", mode="lines"
    )

    node_x, node_y, node_text, node_colors = [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

        if node.startswith("Start"):
            node_colors.append("blue")
        elif node.startswith("Exit"):
            node_colors.append("green")
        elif node.startswith("Bottleneck"):
            node_colors.append("red")  # red so bottlenecks are visually obvious
        else:
            node_colors.append("orange")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        hoverinfo="text",
        marker=dict(size=10, color=node_colors, line_width=2),
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=title,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
        ),
    )

    if save_path:
        fig.write_image(save_path)

    fig.show()


if __name__ == "__main__":
    # Original city - no bottlenecks
    G, starts, exits, _ = create_city(
        num_start_nodes=3,
        num_exit_nodes=3,
        num_connection_nodes=10,
        num_bottlenecks=0,
        seed=42,
        save_path="city.pickle",
    )

    # Same city with mixed bottlenecks
    G_bottleneck, starts, exits, bottleneck_names = create_city(
        num_start_nodes=3,
        num_exit_nodes=3,
        num_connection_nodes=10,
        num_bottlenecks=3,
        bottleneck_positions="mixed",
        seed=42,
    )

    # Remove bottlenecks for comparison
    G_clean = remove_bottlenecks(G_bottleneck, bottleneck_names)

    # Visualise — bottleneck nodes shown in red
    visualise_city(G_bottleneck, title="City With Bottlenecks")
    visualise_city(G_clean, title="City Without Bottlenecks")
