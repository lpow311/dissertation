import networkx as nx
import plotly.graph_objects as go
from collections import defaultdict


class VisualiseCityEvacuationAgents:
    def __init__(self, solution, city: nx.Graph):
        """
        solution: object with .agents dictionary, each agent has .path
        city: NetworkX graph
        exit_nodes: list of nodes considered exits
        """
        self.solution = solution
        self.city = city.graph
        self.exit_nodes = city.exits

        self.agents = self.solution.agents
        self.paths = {agent: info.path for agent, info in self.agents.items()}

        # congestion metrics
        self.node_congestion = self.compute_node_congestion()
        self.edge_congestion = self.compute_edge_congestion()

    def compute_node_congestion(self):
        counts = {n: 0 for n in self.city.nodes()}
        for path in self.paths.values():
            for node in path:
                counts[node] += 1
        return counts

    def compute_edge_congestion(self):
        counts = defaultdict(int)
        for path in self.paths.values():
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                edge = tuple(sorted((u, v)))
                counts[edge] += 1
        return counts

    def animate_solution(self):
        pos = nx.spring_layout(self.city, seed=42)
        max_steps = max(len(p) for p in self.paths.values())

        # --- static city traces ---
        self.edge_traces, self.node_trace = self.build_city_traces(pos)

        # --- initial agent positions ---
        initial_agents = self.build_initial_agent_traces(pos)
        initial_positions = self.edge_traces + [self.node_trace] + initial_agents

        # --- animation frames ---
        frames = self.build_frames(pos, max_steps)

        # --- combine and display ---
        fig = go.Figure(data=initial_positions, frames=frames)
        fig.update_layout(
            title="City Evacuation Simulation",
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

    def build_city_traces(self, pos):
        """Create edge and node traces"""
        # --- edges ---
        max_edge = max(self.edge_congestion.values(), default=1)
        edge_traces = []
        for u, v in self.city.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            key = tuple(sorted((u, v)))
            load = self.edge_congestion.get(key, 0)
            shade = 150 + int(105 * load / max_edge)  # 150-255 gray
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

        # --- nodes ---
        max_node = max(self.node_congestion.values(), default=1)
        node_sizes = []
        node_colors = []
        node_x = []
        node_y = []
        hover_text = []
        node_symbols = []

        for n in self.city.nodes():
            node_x.append(pos[n][0])
            node_y.append(pos[n][1])
            visits = self.node_congestion[n]
            node_sizes.append(6 + 15 * visits / max_node)
            if n in self.exit_nodes:
                node_colors.append("red")
                node_symbols.append("x")
            elif n.startswith("Start"):
                node_colors.append(visits)
                node_symbols.append("square")
            else:
                node_colors.append(visits)
                node_symbols.append("circle")

            hover_text.append(f"{n}<br>Visits: {visits}")

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            marker=dict(
                size=node_sizes,
                color=node_colors,
                colorscale="Turbo",
                showscale=True,
                symbol=node_symbols,
                colorbar=dict(title="Node congestion"),
            ),
            hoverinfo="text",
            text=hover_text,
            name="nodes",
        )

        return edge_traces, node_trace

    def build_initial_agent_traces(self, pos):
        traces = []
        for agent_id, path in self.paths.items():
            x, y = pos[path[0]]
            traces.append(
                go.Scatter(
                    x=[x],
                    y=[y],
                    mode="markers",
                    marker=dict(size=12, color="blue"),
                    name=f"Agent {agent_id}",
                )
            )
        return traces

    def build_frames(self, pos, max_steps):
        frames = []
        for t in range(max_steps):
            agent_traces = []
            for agent_id, path in self.paths.items():
                node = path[t] if t < len(path) else path[-1]
                x, y = pos[node]
                agent_traces.append(
                    go.Scatter(
                        x=[x],
                        y=[y],
                        mode="markers",
                        marker=dict(size=12, color="blue"),
                        showlegend=False,
                    )
                )
            frame_data = self.edge_traces + [self.node_trace] + agent_traces
            frames.append(go.Frame(data=frame_data, name=str(t)))
        return frames
