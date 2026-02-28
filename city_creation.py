import networkx as nx
import numpy as np
import plotly.graph_objects as go
import pickle
import os

# TODO: not really happy with this but lets test the idea first.


# =============================================================================
# PARAMETERS
# =============================================================================

GRAPH_SAVE_PATH = "graphs/"
IMAGE_SAVE_PATH = "images/"

CITY_NAME = "open_small"
SEED = 42

NUM_NODES_PER_ROOM = 3  # how many nodes in each room (not including start or exit nodes).
INTERNAL_DENSITY = 2  # how connected the nodes are to each other within a room.
EXIT_CONNECTIONS = 2  # how many nodes the start nodes connect to.
START_CONNECTIONS = 2  # how many nodes the exit nodes connect to.

BOTTLENECK_CORRIDORS = 1  # number of edges directly between start and exit room.
OPEN_CORRIDORS = 3  # number of edges between start/exit and transit rooms.

# =============================================================================


def create_city(
    seed: int = 42,
    bottleneck_corridors: int = 1,
    open_corridors: int = 3,
    exit_connections: int = 3,
    start_connections: int = 3,
):
    """
    Creates a simple room-based city:

        [StartRoom_A] --bottleneck_corridors-- [ExitRoom_A]
              |                                      |
         open_corridors                         open_corridors
              |                                      |
           [TransitRoom] ----open_corridors---- [ExitRoom_B]
              |
         open_corridors
              |
        [StartRoom_B] --bottleneck_corridors-- [ExitRoom_B]

    Short route (Start -> Exit directly) is bottlenecked.
    Long route (Start -> Transit -> Exit) is open.

    Each exit node connects to multiple internal room nodes (exit_connections)
    so the exit itself is not an unintended bottleneck.
    """
    rng = np.random.default_rng(seed)
    G = nx.Graph()

    starts = []
    exits = []

    rooms = {
        "StartRoom_A": {"type": "start", "num_starts": 2, "num_exits": 0},
        "StartRoom_B": {"type": "start", "num_starts": 2, "num_exits": 0},
        "TransitRoom": {"type": "transit", "num_starts": 0, "num_exits": 0},
        "ExitRoom_A": {"type": "exit", "num_starts": 0, "num_exits": 1},
        "ExitRoom_B": {"type": "exit", "num_starts": 0, "num_exits": 1},
    }

    room_nodes = {}

    for room_name, room_config in rooms.items():
        conn_nodes = []

        # Add connection nodes
        for i in range(1, NUM_NODES_PER_ROOM + 1):
            node = f"Connection_{room_name}_{i}"
            G.add_node(node, type="connection", room=room_name)
            conn_nodes.append(node)

        # Ring for guaranteed internal connectivity
        for i in range(len(conn_nodes)):
            G.add_edge(conn_nodes[i], conn_nodes[(i + 1) % len(conn_nodes)])

        # Extra random internal edges
        for node in conn_nodes:
            others = [n for n in conn_nodes if n != node]
            num_extra = min(INTERNAL_DENSITY - 1, len(others))
            if num_extra > 0:
                targets = rng.choice(others, size=num_extra, replace=False)
                for t in targets:
                    G.add_edge(node, t)

        # Start nodes — connect to multiple internal nodes so start itself
        # is not an unintended bottleneck
        for i in range(1, room_config["num_starts"] + 1):
            room_idx = room_name.replace("StartRoom_", "")
            node = f"Start_{room_idx}_{i}"
            G.add_node(node, type="start", room=room_name)

            num_start_conn = min(start_connections, len(conn_nodes))
            start_targets = rng.choice(conn_nodes, size=num_start_conn, replace=False)
            for t in start_targets:
                G.add_edge(node, t)

            starts.append(node)

        # Exit nodes — connect to multiple internal nodes so exit itself
        # is not an unintended bottleneck
        for i in range(1, room_config["num_exits"] + 1):
            room_idx = room_name.replace("ExitRoom_", "")
            node = f"Exit_{room_idx}_{i}"
            G.add_node(node, type="exit", room=room_name)

            num_exit_conn = min(exit_connections, len(conn_nodes))
            exit_targets = rng.choice(conn_nodes, size=num_exit_conn, replace=False)
            for t in exit_targets:
                G.add_edge(node, t)

            exits.append(node)

        room_nodes[room_name] = conn_nodes

    # Connect rooms via corridors
    corridors = [
        ("StartRoom_A", "ExitRoom_A", bottleneck_corridors),
        ("StartRoom_B", "ExitRoom_B", bottleneck_corridors),
        ("StartRoom_A", "TransitRoom", open_corridors),
        ("StartRoom_B", "TransitRoom", open_corridors),
        ("TransitRoom", "ExitRoom_A", open_corridors),
        ("TransitRoom", "ExitRoom_B", open_corridors),
    ]

    for room_from, room_to, num_edges in corridors:
        from_nodes = room_nodes[room_from]
        to_nodes = room_nodes[room_to]
        n = min(num_edges, len(from_nodes), len(to_nodes))
        from_picks = rng.choice(from_nodes, size=n, replace=False)
        to_picks = rng.choice(to_nodes, size=n, replace=False)
        for f, t in zip(from_picks, to_picks):
            G.add_edge(f, t)

    return G, starts, exits


def visualise_city(G: nx.Graph, title: str = "City", save_path: str = None):
    """
    Visualisation with room bounding boxes drawn as rectangles.
    Blue = Start, Green = Exit, Orange = Connection

    Room rectangle colours:
        StartRoom   = light blue
        ExitRoom    = light green
        TransitRoom = light yellow
    """
    PADDING = 0.08

    ROOM_COLOURS = {
        "start": "rgba(173, 216, 230, 0.3)",
        "exit": "rgba(144, 238, 144, 0.3)",
        "transit": "rgba(255, 255, 180, 0.3)",
    }
    ROOM_BORDER = {
        "start": "rgba(70, 130, 180, 0.8)",
        "exit": "rgba(34, 139, 34, 0.8)",
        "transit": "rgba(200, 180, 0, 0.8)",
    }

    pos = nx.spring_layout(G, seed=42)

    # Group nodes by room
    room_positions = {}
    room_types = {}

    for node, data in G.nodes(data=True):
        room = data.get("room")
        if room is None:
            continue
        x, y = pos[node]
        if room not in room_positions:
            room_positions[room] = []
        room_positions[room].append((x, y))

        if "Start" in room:
            room_types[room] = "start"
        elif "Exit" in room:
            room_types[room] = "exit"
        else:
            room_types[room] = "transit"

    # Build rectangle shapes
    shapes = []
    annotations = []

    for room_name, coords in room_positions.items():
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]

        x0 = min(xs) - PADDING
        x1 = max(xs) + PADDING
        y0 = min(ys) - PADDING
        y1 = max(ys) + PADDING

        room_type = room_types.get(room_name, "transit")
        fill_colour = ROOM_COLOURS[room_type]
        border_colour = ROOM_BORDER[room_type]

        shapes.append(
            dict(
                type="rect",
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                line=dict(color=border_colour, width=2),
                fillcolor=fill_colour,
                layer="below",
            )
        )

        annotations.append(
            dict(
                x=x0 + 0.01,
                y=y1 - 0.01,
                text=f"<b>{room_name}</b>",
                showarrow=False,
                xanchor="left",
                yanchor="top",
                font=dict(size=10, color=border_colour),
            )
        )

    # Edges
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, line=dict(width=1, color="#888"), hoverinfo="none", mode="lines"
    )

    # Nodes
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
            shapes=shapes,
            annotations=annotations,
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
        ),
    )

    if save_path:
        fig.write_image(save_path)

    fig.show()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    os.makedirs(GRAPH_SAVE_PATH, exist_ok=True)
    os.makedirs(IMAGE_SAVE_PATH, exist_ok=True)

    G, starts, exits = create_city(
        seed=SEED,
        bottleneck_corridors=BOTTLENECK_CORRIDORS,
        open_corridors=OPEN_CORRIDORS,
        exit_connections=EXIT_CONNECTIONS,
        start_connections=START_CONNECTIONS,
    )

    # Save pickle to graphs/
    pickle.dump(
        {"graph": G, "starts": starts, "exits": exits},
        open(os.path.join(GRAPH_SAVE_PATH, f"{CITY_NAME}.pickle"), "wb"),
    )

    # Save image to images/
    visualise_city(
        G,
        title=CITY_NAME,
        save_path=os.path.join(IMAGE_SAVE_PATH, f"{CITY_NAME}.png"),
    )

    print(f"Starts: {starts}")
    print(f"Exits:  {exits}")
