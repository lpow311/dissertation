import jax
import jax.numpy as jnp

from jax import random

import networkx as nx
import matplotlib.pyplot as plt

import itertools


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


class Agent:
    # Class to define agent properties e.g. behaviours/traits.
    def __init__(self, name: str, city: nx.Graph, path_seed: int):
        self.path_seed = path_seed
        self.name = name

        self.city = city
        self.key = random.PRNGKey(path_seed)

        self.start_location = self.select_start_point()
        self.path = self.generate_random_path()

        self.path_history = []

    def select_start_point(self) -> str | int:
        start_points = self.city.starts
        self.key, sub_key = random.split(self.key)
        idx = random.randint(sub_key, (), 0, len(start_points))
        return start_points[int(idx)]

    def generate_random_path(self) -> list:
        # TODO: finish this.
        # TODO: they can go back on themselves at the minute ...
        path = [self.start_location]
        current_location = self.start_location

        while current_location not in self.city.exits:
            neighbours = list(self.city.graph.neighbors(current_location))

            self.key, sub_key = random.split(self.key)
            idx = random.randint(sub_key, (), 0, len(neighbours))
            next_location = neighbours[int(idx)]

            path.append(next_location)
            current_location = next_location

        return path

    def update_current_path(self, new_path) -> None:
        self.path_history.append(self.path)
        self.path = new_path


class Chromosome:
    # chromosome = Chromosome(
    #     paths={"agent1": agent1_path1, "agent2": agent2_path1, "agent3": agent3_path1}
    # )
    def __init__(self, paths: dict[str, list]):
        # paths = {'agent1': [1, 2, 3]}
        self.paths = paths
        self.fitness = None

    def calculate_fitness(self) -> None:
        # Super basic assumption of 1 step = 1 time unit.
        path_length = [len(path.path) for path in self.paths.values()]
        total_evac_time = max(path_length)
        self.fitness = total_evac_time


class GeneticAlgorithm:

    def __init__(self, key) -> None:
        self.key = key

    def crossover(self, parents):
        parent1, parent2 = parents
        agents = list(parent1.paths)

        # FIXME - carry on from here.

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

    def repair_path(self, path, graph):
        exits = graph.exits

        current_location = path[-1]
        if current_location in exits:
            return path

        while current_location not in exits:
            neighbours = graph[current_location]

            self.key, sub_key = random.split(self.key)
            idx = random.randint(sub_key, (), 0, len(neighbours))
            next_location = neighbours[idx]
            path.append(next_location)
            current_location = next_location

        return path

    def mutation(self, chromosome, graph, mutation_rate=0.3):
        for agent, path in chromosome.items():
            if random.random() < mutation_rate:

                self.key, sub_key = random.split(self.key)

                change_location = random.randint(sub_key, 1, len(path) - 1)
                current_location = path[change_location]
                neighbours = graph[current_location]
                new_location = random.choice(neighbours)

                new_partial = path[: change_location + 1] + [new_location]

                repaired_path = self.repair_path(new_partial, graph)
                chromosome[agent] = repaired_path

        return chromosome

    @staticmethod
    def find_parent(percent, population, cumulative_fitness):
        for ind, cf in zip(population, cumulative_fitness):
            if percent <= cf:
                return ind

    def selection(self, population):
        # tournament / roulette wheel.
        [chromosome.calculate_fitness() for chromosome in population]
        fitness = [1 / (chromosome.fitness + 0.00001) for chromosome in population]
        total_fitness = sum(fitness)

        scaled_fitness = [f / total_fitness for f in fitness]
        cumulative_fitness = list(itertools.accumulate(scaled_fitness))

        self.key, sub_key = random.split(self.key)
        parent_percentages = random.uniform(minval=0, maxval=1, shape=len(population), key=sub_key)

        parents = []
        for percent in parent_percentages:
            parents.append(self.find_parent(percent, population, cumulative_fitness))

        return parents


city = Environment()

num_agents = 3
population_size = 10

# TODO: generate the population
population = []
random_seed_count = 1
for i in range(population_size):
    paths = {}
    for agent in range(num_agents):
        paths[agent] = Agent(
            name=f"Agent{agent}_Population{i}", city=city, path_seed=random_seed_count
        )

        # TODO: need to ensure each agent has different random seed for each population.
        # TODO: google this more...
        random_seed_count += 1

    chromosome = Chromosome(paths=paths)
    chromosome.calculate_fitness()
    population.append(chromosome)


key = random.PRNGKey(12345)
# TODO: Selection, Cross over, mutation loop.
ga = GeneticAlgorithm(key=key)
parents = ga.selection(population=population)
parent_pairs = chunks = list(zip(parents[::2], parents[1::2]))

children = []
for pair in parent_pairs:
    parent_children = ga.crossover(parents=pair)
    children += parent_children


print("pause")
