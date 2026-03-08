import networkx as nx
import matplotlib.pyplot as plt
import pickle
import numpy as np


class CityCreation:

    def __init__(self, G: nx.Graph, city_name: str, metrics: dict = {}) -> None:
        self.city = G
        self.city_name = city_name

        self.starts = ["S1", "S2", "S3", "S4"]
        self.exits = ["E1", "E2", "E3", "E4"]

        self.pos = {}

        self.metrics = metrics

    def save_graph_pickle(self):
        path = f"graphs/{self.city_name.lower().replace(' ', '_')}.pkl"
        city = {
            "G": self.city,
            "starts": self.starts,
            "exits": self.exits,
            "metrics": self.metrics,
            "pos": self.pos,
        }
        with open(path, "wb") as f:
            pickle.dump(city, f)

    @staticmethod
    def read_graph_pickle(name: str) -> tuple:
        with open(f"graphs/{name}.pkl", "rb") as f:
            city = pickle.load(f)

        G = city["graph"]
        exits = city["exits"]
        starts = city["starts"]
        metrics = city["metrics"]
        pos = city["pos"]
        return G, exits, starts, metrics, pos

    def calculate_city_metrics(self) -> dict:
        metrics = {
            "mean_path_length": self.mean_path_lengths(),
            "max_betweeness": self.maximum_node_betweeness(),
            "edge_density": nx.density(self.city),
            "critical_edge": self.critical_edge_use(),
        }
        if self.metrics == {}:
            self.metrics = metrics

        return metrics

    def critical_edge_use(self) -> tuple:
        edge_bc = nx.edge_betweenness_centrality_subset(
            self.city, sources=self.starts, targets=self.exits, normalized=True
        )
        top_edge = max(edge_bc, key=edge_bc.get)
        return (top_edge, edge_bc[top_edge])

    def maximum_node_betweeness(self) -> float:
        betweenness = nx.betweenness_centrality_subset(
            self.city, sources=self.starts, targets=self.exits, normalized=True
        )
        return max(betweenness.values())

    def mean_path_lengths(self) -> float:
        path_lengths = []
        for s in self.starts:
            for e in self.exits:
                try:
                    path_lengths.append(nx.shortest_path_length(self.city, s, e))
                except nx.NetworkXNoPath:
                    pass

        return np.mean(path_lengths)

    def plot_city(
        self,
        pos: dict | None = None,
        save: bool = False,
        show_paths: bool = False,
    ):
        starts, exits = set(self.starts), set(self.exits)
        node_colours = [
            "#6bbc8d" if n in starts else "#da5f70" if n in exits else "#5fb8d6"
            for n in self.city.nodes()
        ]

        if pos is None:
            pos = nx.spring_layout(self.city, seed=42)

        self.pos = pos

        path_edges = self.shortest_path_edges() if show_paths else set()

        edge_widths = [
            4.0 if (u, v) in path_edges or (v, u) in path_edges else 1.0
            for u, v in self.city.edges()
        ]

        fig, ax = plt.subplots(figsize=(6, 6))
        nx.draw_networkx(
            self.city,
            pos,
            ax=ax,
            node_color=node_colours,
            node_size=500,
            font_size=8,
            font_color="black",
            edge_color="#474f50",
            width=edge_widths,
        )
        ax.set_title(self.city_name, fontsize=14, fontweight="bold")
        ax.axis("off")
        plt.tight_layout()

        if save:
            plt.savefig(f"images/{self.city_name.replace(' ', '_').lower()}.png")
        plt.show()

        if self.metrics != {}:
            self.print_metrics()

    def print_metrics(self) -> None:
        metric_list = [
            f"Avg Path Length     → {self.metrics['mean_path_length']:.3}",
            f"Max Betweeness      → {self.metrics['max_betweeness']:.3}",
            f"Edge Density        → {self.metrics['edge_density']:.3}",
            f"Critical Edge       → {self.metrics['critical_edge'][0]} ({self.metrics['critical_edge'][1]:.3})",
        ]

        metric_string = "\n".join(metric_list)
        print(metric_string)

    def shortest_path_edges(self) -> set:
        path_edges = set()

        for s in self.starts:
            # Find the minimum distance to any exit
            min_dist = min(nx.shortest_path_length(self.city, s, e) for e in self.exits)

            # Get ALL exits at that minimum distance (handles tied exits)
            closest_exits = [
                e for e in self.exits if nx.shortest_path_length(self.city, s, e) == min_dist
            ]

            # Get all shortest paths to all closest exits
            for e in closest_exits:
                for path in nx.all_shortest_paths(self.city, s, e):
                    edges = set(zip(path[:-1], path[1:]))
                    path_edges.update(edges)

        return path_edges
