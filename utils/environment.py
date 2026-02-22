import pickle
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

        self.graph = self.create_graph(city_name)
        self.starts, self.num_starts = self.extract_node_types(node_type="start")
        self.exits, self.num_exits = self.extract_node_types(node_type="exit")

        self.congestion_amount = 5  # TODO: come back and pick something better for this...

        self.distances_to_exits = {
            node: {exit: nx.shortest_path_length(self.graph, node, exit) for exit in self.exits}
            for node in self.graph.nodes
        }

        self.bottleneck_score = self.calculate_bottleneck_score()

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
            f"{formatted_name} has {self.num_starts} starting points and {self.num_exits} exit points with a bottleneck score of {self.bottleneck_score}"
        )
        display(Image(f"images/{self.city_name}.png"))

    def calculate_bottleneck_score(self) -> float:
        return 1
