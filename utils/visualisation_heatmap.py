import networkx as nx
import plotly.graph_objects as go
from collections import defaultdict


class CityEvacuationHeatmap:
    def __init__(self, solution, city: nx.Graph):
        """
        solution: object with .agents dictionary, each agent has .path
        city: NetworkX graph
        exit_nodes: list of exit nodes
        """
        self.solution = solution
        self.city = city.graph
        self.exit_nodes = city.exits

        self.agents = self.solution.agents
        self.paths = {agent: info.path for agent, info in self.agents.items()}

        # Compute congestion
        self.edge_congestion = self.compute_edge_congestion()
        self.node_counts_over_time = self.compute_node_counts_per_timestep()

    def compute_edge_congestion(self):
        """Total number of agents passing through each edge (undirected)"""
        counts = defaultdict(int)
        for path in self.paths.values():
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                edge = tuple(sorted((u, v)))
                counts[edge] += 1
        return counts

    def compute_node_counts_per_timestep(self):
        """Number of agents at each node per timestep"""
        max_steps = max(len(p) for p in self.paths.values())
        node_counts_over_time = []

        for t in range(max_steps):
            counts = {n: 0 for n in self.city.nodes()}
            for path in self.paths.values():
                node = path[t] if t < len(path) else path[-1]
                counts[node] += 1
            node_counts_over_time.append(counts)
        return node_counts_over_time

    def animate_solution(self):
        pos = nx.spring_layout(self.city, seed=42)
        max_steps = len(self.node_counts_over_time)

        # --- build static edges ---
        self.edge_traces = self.build_edge_traces(pos)

        # --- build initial nodes (time 0) ---
        initial_nodes = self.build_node_trace(pos, self.node_counts_over_time[0])
        initial_positions = self.edge_traces + [initial_nodes]

        # --- build frames for animation ---
        frames = [
            go.Frame(data=self.edge_traces + [self.build_node_trace(pos, counts)], name=str(t))
            for t, counts in enumerate(self.node_counts_over_time)
        ]

        # --- show figure ---
        fig = go.Figure(data=initial_positions, frames=frames)
        fig.update_layout(
            title="City Evacuation Heatmap",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="y"),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[None, {"frame": {"duration": 600, "redraw": True}}],
                        )
                    ],
                )
            ],
            sliders=[
                dict(
                    steps=[
                        dict(
                            method="animate",
                            args=[[str(t)], {"frame": {"duration": 0, "redraw": True}}],
                            label=str(t),
                        )
                        for t in range(max_steps)
                    ]
                )
            ],
        )
        fig.show()

    def build_edge_traces(self, pos):
        """Create static edges with shading by total traffic"""
        max_edge = max(self.edge_congestion.values(), default=1)
        edge_traces = []
        for u, v in self.city.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            key = tuple(sorted((u, v)))
            load = self.edge_congestion.get(key, 0)
            shade = 150 + int(105 * load / max_edge)  # darker = more traffic
            color = f"rgb({shade},{shade},{shade})"
            edge_traces.append(
                go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode="lines",
                    line=dict(width=3, color=color),
                    hoverinfo="text",
                    text=f"Edge load: {load}",
                    showlegend=False,
                )
            )
        return edge_traces

    def build_node_trace(self, pos, counts):
        """Build a Scatter trace for nodes at a given timestep"""
        max_count = max(counts.values()) if counts else 1
        node_x, node_y, node_sizes, node_colors, node_symbols, hover_text = [], [], [], [], [], []

        for n in self.city.nodes():
            node_x.append(pos[n][0])
            node_y.append(pos[n][1])
            node_sizes.append(6 + 15 * counts[n] / max_count)
            hover_text.append(f"{n}<br>Agents: {counts[n]}")

            if n in self.exit_nodes:
                node_colors.append("red")
                node_symbols.append("x")
            elif n.startswith("Start"):
                node_colors.append("green")
                node_symbols.append("square")
            else:
                node_colors.append(counts[n])
                node_symbols.append("circle")

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            marker=dict(
                size=node_sizes,
                color=node_colors,
                symbol=node_symbols,
                colorscale="Turbo",
                showscale=True,
                colorbar=dict(title="Agents at node"),
            ),
            hoverinfo="text",
            text=hover_text,
            name="nodes",
        )
        return node_trace
