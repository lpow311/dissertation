def create_city(
    num_nodes: int,
    num_exits: int,
    num_starts: int,
    num_bottlenecks: int,  # total bottlenecks to place
    bottleneck_positions: str = "mixed",  # "exits", "starts", "mid", "mixed"
    seed: int = 42,
) -> nx.Graph:

    rng = np.random.default_rng(seed)

    # Create base graph
    G = nx.random_graphs.barabasi_albert_graph(num_nodes, 2, seed=seed)
    mapping = {i: f"Connection_{i}" for i in G.nodes()}
    G = nx.relabel_nodes(G, mapping)
    nodes = list(G.nodes())

    # Add starts
    starts = []
    for i in range(num_starts):
        start_name = f"Start_{i+1}"
        connect_to = nodes[rng.integers(0, len(nodes))]
        G.add_edge(start_name, connect_to)
        starts.append(start_name)

    # Add exits
    exits = []
    for i in range(num_exits):
        exit_name = f"Exit_{i+1}"
        connect_to = nodes[rng.integers(0, len(nodes))]
        G.add_edge(connect_to, exit_name)
        exits.append(exit_name)

    # Place bottlenecks based on position strategy
    bottleneck_nodes = []

    if bottleneck_positions == "exits":
        targets = [(exits[i], "exit") for i in range(min(num_bottlenecks, len(exits)))]
    elif bottleneck_positions == "starts":
        targets = [(starts[i], "start") for i in range(min(num_bottlenecks, len(starts)))]
    elif bottleneck_positions == "mid":
        # Pick random mid-network nodes with high betweenness
        betweenness = nx.betweenness_centrality(G)
        sorted_nodes = sorted(betweenness, key=betweenness.get, reverse=True)
        targets = [(n, "mid") for n in sorted_nodes[:num_bottlenecks]]
    else:  # mixed
        n_each = num_bottlenecks // 3
        targets = (
            [(exits[i], "exit") for i in range(min(n_each, len(exits)))]
            + [(starts[i], "start") for i in range(min(n_each, len(starts)))]
            + [(nodes[i], "mid") for i in range(n_each)]
        )

    for target_node, position in targets:
        bottleneck_name = f"Bottleneck_{target_node}"
        bottleneck_nodes.append(bottleneck_name)

        if position == "exit":
            # Insert bottleneck between network and exit
            neighbours = list(G.neighbors(target_node))
            G.add_node(bottleneck_name)
            G.add_edge(bottleneck_name, target_node)
            for neighbour in neighbours:
                G.add_edge(neighbour, bottleneck_name)
                G.remove_edge(neighbour, target_node)

        elif position == "start":
            # Insert bottleneck between start and network
            neighbours = list(G.neighbors(target_node))
            G.add_node(bottleneck_name)
            G.add_edge(target_node, bottleneck_name)
            for neighbour in neighbours:
                G.add_edge(bottleneck_name, neighbour)
                G.remove_edge(target_node, neighbour)

        elif position == "mid":
            # Insert bottleneck by removing all but 2 edges from high betweenness node
            neighbours = list(G.neighbors(target_node))
            if len(neighbours) > 2:
                # Keep only 2 connections, remove the rest
                to_remove = rng.choice(neighbours, size=len(neighbours) - 2, replace=False)
                for neighbour in to_remove:
                    G.remove_edge(target_node, neighbour)

    return G, starts, exits, bottleneck_nodes


def remove_bottlenecks(G: nx.Graph, bottleneck_nodes: list) -> nx.Graph:
    G_clean = G.copy()

    for bottleneck in bottleneck_nodes:
        if bottleneck in G_clean.nodes():
            neighbours = list(G_clean.neighbors(bottleneck))
            # Reconnect all neighbours directly to each other
            for i, n1 in enumerate(neighbours):
                for n2 in neighbours[i + 1 :]:
                    G_clean.add_edge(n1, n2)
            G_clean.remove_node(bottleneck)

    return G_clean
