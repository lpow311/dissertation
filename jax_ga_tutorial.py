import jax
import jax.numpy as jnp

from jax import random

import networkx as nx
import matplotlib.pyplot as plt


class Environment:

    def __init__(self):
        self.graph = self.create_graph()
        self.starts = ["C", "B"]
        self.exits = ["E", "F", "G"]

    def create_graph(self) -> nx.Graph:
        G = nx.Graph()

        edges = [
            ("E", "A"),
            ("C", "A"),
            ("B", "A"),
            ("C", "D"),
            ("D", "B"),
            ("D", "F"),
            ("D", "G"),
        ]
        G.add_edges_from(edges)
        return G

    def __str__(self):
        nx.draw(
            self.graph, with_labels=True, node_color="lightblue", node_size=800, font_weight="bold"
        )
        plt.show(block=True)


city = Environment()


num_agents = 3
population_size = 10

# example chromosone <- maybe?
agent1_path1 = ["C", "D", "F"]
agent2_path1 = ["C", "A", "B", "D", "G"]
agent3_path1 = ["B", "A", "E"]


class Chromosome:
    def __init__(self, paths):
        self.paths = paths
        self.fitness = None

    def calculate_fitness(self) -> None:
        # Super basic assumption of 1 step = 1 time unit.
        path_length = [len(path) for path in self.paths]
        total_evac_time = max(path_length)
        self.fitness = total_evac_time


def crossover(parent1, parent2):
    agents = list(parent1.keys())
    cut = random.randint(1, len(agents) - 1)

    child1, child2 = {}, {}
    for i, agent in enumerate(agents):
        if i < cut:
            child1[agent] = parent1[agent]
            child2[agent] = parent2[agent]
        else:
            child1[agent] = parent2[agent]
            child2[agent] = parent1[agent]

    return child1, child2


def repair_path(path, graph):
    exits = graph.exits

    current_location = path[-1]
    if current_location in exits:
        return path

    while current_location not in exits:
        neighbours = graph[current_location]
        next_location = random.choice(neighbours)
        path.append(next_location)
        current_location = next_location

    return path


def mutation(chromosome, graph, mutation_rate: 0.3):
    for agent, path in chromosome.items():
        if random.random() < mutation_rate:
            change_location = random.randint(1, len(path) - 1)
            current_location = path[change_location]
            neighbours = graph[current_location]
            new_location = random.choice(neighbours)

            new_partial = path[: change_location + 1] + [new_location]

            repaired_path = repair_path(new_partial, graph)
            chromosome[agent] = repaired_path

    return chromosome


class Agent:
    # Class to define agent properties e.g. behaviours/traits.
    def __init__(self, name: str, city: nx.Graph):
        self.name = name

        self.start_points = city.starts
        self.start_location = random.choice(self.start_points)
        self.path = []

    def random_path(self) -> list:
        path = [self.start_location]


# chromosome = Chromosome(
#     paths={"agent1": agent1_path1, "agent2": agent2_path1, "agent3": agent3_path1}
# )
