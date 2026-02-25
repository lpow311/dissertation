import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from IPython.display import Image, display


from typing import Tuple


class Environment:
    """
    Creates a class object to store city information and load in the
    created graph e.g. start and end points.
    """

    def __init__(self, city_name: str):
        self.city_name = city_name

        self.pickle_obj = self.create_graph(city_name)
        self.graph = self.pickle_obj["graph"]
        self.starts, self.num_starts = self.pickle_obj["starts"], len(self.pickle_obj["starts"])
        self.exits, self.num_exits = self.pickle_obj["exits"], len(self.pickle_obj["exits"])

        self.congestion_amount = 5  # TODO: come back and pick something better for this...

        self.distances_to_exits = {
            node: {exit: nx.shortest_path_length(self.graph, node, exit) + 1 for exit in self.exits}
            for node in self.graph.nodes
        }

        self.bottleneck_score = self.calculate_bottleneck_scores()

    def create_graph(self, city_name: str) -> nx.Graph:
        G = pickle.load(open(f"graphs/{city_name}.pickle", "rb"))
        return G

    def extract_node_types(self, node_type: str) -> Tuple[list, int]:
        filtered_nodes = [
            n for n, attr in self.graph.nodes(data=True) if attr.get("type") == node_type
        ]
        num_nodes = len(filtered_nodes)
        return filtered_nodes, num_nodes

    def __str__(self):
        nx.draw(
            self.graph, with_labels=True, node_color="lightblue", node_size=800, font_weight="bold"
        )
        plt.show(block=True)

    def summarise_city(self) -> None:
        formatted_name = self.city_name.replace("_", " ").capitalize()
        print(
            f"{formatted_name} has {self.num_starts} starting points and {self.num_exits} exit points with a bottleneck score of {self.bottleneck_score['max']:.3f}"
        )
        display(Image(f"images/{self.city_name}.png"))

    def calculate_bottleneck_scores(self) -> dict:
        """
        - Max betweenness — identifies if there's one critical node everything flows through
        - Mean betweenness — overall bottleneck pressure across the network
        - Std betweenness — high std means uneven network, some nodes much more critical than others
        - Bottleneck nodes — specific nodes more than one std above mean, useful for visualisation
        """
        betweenness = nx.betweenness_centrality(self.graph)

        values = list(betweenness.values())

        return {
            "max": max(values),  # single worst bottleneck
            "mean": np.mean(values),  # average bottleneck pressure
            "std": np.std(values),  # how unevenly distributed
            "bottleneck_nodes": {  # nodes above threshold
                node: score
                for node, score in betweenness.items()
                if score > np.mean(values) + np.std(values)
            },
        }

    def start_node_metrics(self) -> dict:
        # TODO
        avg_exit_distance = {}

        for start in self.starts:
            distances_to_exits = {
                exit: nx.shortest_path_length(self.graph, start, exit) for exit in self.exits
            }
            avg_exit_distance[start] = np.mean(list(distances_to_exits.values()))

        return avg_exit_distance
