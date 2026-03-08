import networkx as nx
from jax import random
from jax.random import PRNGKey

from collections import defaultdict

from utils.environment import Environment

import numpy as np


class Agent:

    def __init__(
        self,
        name: str,
        city: Environment,
        characteristics: dict,
        key_manager,
        start_point: str | None = None,
        path: list | None = None,
        shortest_path: list | None = None,
        preferred_exits: list | None = None,
    ) -> None:
        self.name = name
        self.city = city
        self.key_manager = key_manager

        self.characteristics = characteristics
        self.speed = characteristics["walking_speed"]

        if start_point is None:
            self.start_point = self.select_start_point()
        else:
            self.start_point = start_point

        if path is None:
            self.path = self.generate_random_path()
        else:
            self.path = path

        if shortest_path is None:
            self.shortest_path_length = self.calculate_shortest_path()
        else:
            self.shortest_path_length = shortest_path

        if preferred_exits is None:
            # TODO: can update this later for local vs visitor.
            self.preferred_exits = self.city.exits
        else:
            self.preferred_exits = preferred_exits

    def select_start_point(self) -> str | int:
        start_point_idx = random.randint(
            key=self.key_manager.next_key(), shape=(), minval=0, maxval=self.city.num_starts
        )
        return self.city.starts[start_point_idx]

    def calculate_shortest_path(self) -> int:
        exit_path_lengths = [
            nx.shortest_path_length(self.city.graph, source=self.start_point, target=exit_option)
            for exit_option in self.city.exits
        ]
        # This calculates edges whereas I manually do node count.
        shortest_length = min(exit_path_lengths) + 1
        return shortest_length

    def generate_random_path(self) -> list:
        current_location = self.start_point
        path = [current_location]

        city_graph = self.city.graph

        while current_location not in self.city.exits:
            neighbours = list(city_graph.neighbors(current_location))

            sub_key = self.key_manager.next_key()

            next_location_idx = random.randint(
                key=sub_key, shape=(), minval=0, maxval=len(neighbours)
            )
            current_location = neighbours[next_location_idx]
            path.append(current_location)

        path = self.__remove_loops_from_path(path=path)
        return path

    def update_path(self, path: list) -> None:
        self.path = self.__remove_loops_from_path(path=path)

    @staticmethod
    def __remove_loops_from_path(path: list) -> list:
        seen = {}
        cleaned = []
        for node in path:
            if node in seen:
                # loop detected - cut back to first occurrence
                cut_idx = seen[node]
                cleaned = cleaned[:cut_idx]
                # Remove stale entries from seen for nodes no longer in cleaned
                seen = {n: i for n, i in seen.items() if i < cut_idx}
            seen[node] = len(cleaned)
            cleaned.append(node)
        return cleaned

    def copy_agent(self, new_path: list | None):
        if new_path is None:
            new_path = self.path
        else:
            new_path = self.__remove_loops_from_path(path=new_path)

        return Agent(
            name=self.name,
            city=self.city,
            characteristics=self.characteristics,
            key_manager=self.key_manager,
            start_point=self.start_point,
            path=new_path,
            shortest_path=self.shortest_path_length,
            preferred_exits=self.preferred_exits,
        )


class Chromosome:

    def __init__(
        self,
        agents: dict,
        key_manager,
        params: dict = {"congestion": False, "human": False, "fitness": "max"},
    ) -> None:
        self.params = params
        self.key_manager = key_manager

        self.agents = self.deep_copy_agents(agents=agents)
        self.num_agents = len(self.agents)
        self.fitness = None
        self.fitness_calc = params["fitness"]

        self.congestion_score = []
        self.path_lengths = []
        self.path_times = []

    def deep_copy_agents(self, agents: dict) -> dict:
        agents_copy = {}
        for agent_num, agent in agents.items():
            new_agent = agent.copy_agent(new_path=None)
            agents_copy[agent_num] = new_agent

        return agents_copy

    def calculate_fitness(self) -> None:
        """
        Evacuation time = Path Length * Walking Speed + Congestion Delays
        Loop Score = How unique the path journey is (i.e. avoid going in loops)
        Fitness = Loop- and distance-weighted inverse evacuation time

        “How close agents’ realised travel times are to their theoretical shortest paths,
        accounting for inefficiencies such as looping and congestion.”
        """
        node_occupancy = self.calculate_node_congestion()
        congestion_score = []
        path_lengths = []

        fitnesses = []
        for agent in self.agents.values():
            length = len(agent.path)
            path_lengths.append(length)

            congestion_delay = self.calculate_agent_congestion_delay(agent, node_occupancy)
            congestion_score.append(congestion_delay)

            path_timesteps = length * agent.speed if self.params["walking"] else length
            path_fitness = path_timesteps
            if self.params["congestion"]:
                path_fitness += congestion_delay

            fitnesses.append(path_fitness)

        self.fitness = self.__calculate_chromosome_fitness(fitnesses)

        # Store for use later
        self.path_lengths = path_lengths
        self.congestion_score = congestion_score
        self.path_times = fitnesses

    def __calculate_chromosome_fitness(self, fitnesses: list) -> float:
        if self.fitness_calc == "mean":
            return float(np.mean(fitnesses))
        elif self.fitness_calc == "median":
            return float(np.median(fitnesses))
        else:
            return float(np.max(fitnesses))

    def calculate_agent_congestion_delay(self, agent: Agent, occupancy: defaultdict) -> int:
        delay = 0
        for t, node in enumerate(agent.path):
            capacity = agent.city.congestion_amount
            if occupancy[t][node] > capacity:
                delay += occupancy[t][node] - capacity

        return delay

    def calculate_node_congestion(self) -> defaultdict:
        """
        Weakly time-dependent (non-causal) congestion - It is not fully dynamic, and delays
        do not propagate forward.
        i.e. So if an agent is delayed at time t, the model still assumes it arrives at t+1 next.
        """
        occupancy = defaultdict(lambda: defaultdict(int))

        for agent in self.agents.values():
            for t, node in enumerate(agent.path):
                occupancy[t][node] += 1

        return occupancy

    def calculate_average_path(self) -> float:
        lengths = []
        for agent in self.agents.values():
            path = agent.path
            lengths.append(len(path))

        return np.mean(lengths)


class PopulationCreation:

    def __init__(self, pop_size: int, num_agents: int, city: Environment, key_manager) -> None:
        self.pop_size = pop_size
        self.num_agents = num_agents

        self.city = city
        self.human_traits = {"walking_speeds": [], "panic": []}

        self.key_manager = key_manager

    def create_initial_population(self, simulation_params: dict) -> list:
        """
        Creates the initial population of chromosomes to be used, each has its
        own unique seed for reproduction of "random" probabilities.
        """
        agents, human_traits = self.initialise_agents(simulation_params)

        population = []
        for _ in range(self.pop_size):
            chromosome = Chromosome(
                agents=agents,
                params=simulation_params,
                key_manager=self.key_manager,
            )
            chromosome.calculate_fitness()

            population.append(chromosome)

        return population, human_traits

    def initialise_agents(self, simulation_params: dict) -> dict:
        """
        Creates a set of agents to use in the modelling ensuring the agent
        characteristics are the same across the different chromosomes.
        """
        agents = {}

        for agent in range(self.num_agents):
            walking_speed = self.extract_walking_speed(
                key=self.key_manager.next_key(), params=simulation_params
            )

            default_characteristics = {
                "walking_speed": walking_speed,  # How many time steps it takes to move 1 node.
            }

            agents[agent] = Agent(
                name=f"Agent{agent}",
                city=self.city,
                characteristics=default_characteristics,
                key_manager=self.key_manager,
            )

        return agents, self.human_traits

    def extract_walking_speed(self, key: PRNGKey, params: dict) -> int:
        if not params["walking"]:
            self.human_traits["walking_speeds"].append(1)
            return 1

        walking_speeds = [1, 2, 3]  # TODO: this is sooo basic but for now is fine.
        speed_idx = random.randint(key, shape=(), minval=0, maxval=len(walking_speeds))
        walking_speed = walking_speeds[speed_idx]

        self.human_traits["walking_speeds"].append(walking_speed)

        return walking_speed
