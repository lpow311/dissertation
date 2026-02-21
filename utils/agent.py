import networkx as nx
from jax import random
from collections import defaultdict

from utils.environment import Environment

import numpy as np


class Agent:

    def __init__(self, name: str, seed: int, city: Environment, characteristics: dict) -> None:
        self.name = name

        self.seed = seed
        self.key = random.PRNGKey(seed)

        self.city = city
        self.start_point = self.select_start_point()

        self.path = None

        self.characteristics = characteristics
        self.speed = characteristics["walking_speed"]

        # TODO: this only works on static stuff for the minute e.g. no congestion.
        self.shortest_path_length = self.calculate_shortest_path()

    def select_start_point(self) -> str | int:
        start_point_idx = random.randint(
            key=self.split_key(), shape=(), minval=0, maxval=self.city.num_starts
        )
        return self.city.starts[start_point_idx]

    def split_key(self) -> random.PRNGKey:
        self.key, sub_key = random.split(self.key)
        return sub_key

    def calculate_shortest_path(self) -> int:
        exit_path_lengths = [
            nx.shortest_path_length(
                self.city.graph,
                source=self.start_point,
                target=exit_option,
                # , weight='weight'
            )
            for exit_option in self.city.exits
        ]
        shortest_length = min(exit_path_lengths)
        return shortest_length

    def generate_random_path(self, chromosome_key: random.PRNGKey) -> list:
        current_location = self.start_point
        path = [current_location]

        city_graph = self.city.graph

        while current_location not in self.city.exits:
            neighbours = list(city_graph.neighbors(current_location))

            chromosome_key, sub_key = random.split(chromosome_key)

            # TODO: might want to stop it going back on itself...
            next_location_idx = random.randint(
                key=sub_key, shape=(), minval=0, maxval=len(neighbours)
            )
            current_location = neighbours[next_location_idx]
            path.append(current_location)

        self.path = path
        return path, chromosome_key


class Chromosome:

    def __init__(
        self, agents: dict, seed: int, params: dict = {"congestion": False, "human": False}
    ) -> None:
        self.seed = seed
        self.params = params

        if type(seed) == int:
            self.key = random.PRNGKey(seed)
        else:

            self.key = seed

        self.agents = self.deep_copy_agents(agents=agents)
        self.num_agents = len(self.agents)
        self.fitness = None
        self.congestion_score = None

    def deep_copy_agents(self, agents: dict) -> dict:
        agents_copy = {}
        for agent_num, agent in agents.items():
            chromosome_agent_key = self.split_key()

            new_agent = Agent(agent.name, agent.seed, agent.city, agent.characteristics)
            new_agent.generate_random_path(chromosome_key=chromosome_agent_key)

            agents_copy[agent_num] = new_agent

        return agents_copy

    def split_key(self) -> random.PRNGKey:
        self.key, sub_key = random.split(self.key)
        return sub_key

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

        score = 0
        for agent in self.agents.values():
            path = agent.path

            congestion_delay = self.calculate_agent_congestion_delay(agent, node_occupancy)
            congestion_score.append(congestion_delay)

            path_timesteps = len(path) * agent.speed if self.params["walking"] else len(path)
            path_fitness = path_timesteps
            if self.params["congestion"]:
                path_fitness += congestion_delay

            unique_nodes = len(set(path))
            loop_score = unique_nodes / len(path)
            distance_score = agent.shortest_path_length / path_fitness

            # Needs to add to one for my brain...
            score += 0.2 * loop_score + 0.8 * distance_score
            
            
            # c1 = [3,4,5,6] = 18 / 4
            # c2 = [1, 2, 3, 10] = 16 / 4
        # TODO: look at this.
        # Average path score
        # min, max or median (look at distributions)!
        self.fitness = score / self.num_agents
        self.congestion_score = congestion_score

    def calculate_agent_congestion_delay(self, agent: Agent, occupancy: defaultdict) -> int:
        delay = 0
        for t, node in enumerate(agent.path):
            capacity = agent.city.congestion_amount
            if occupancy[t][node] > capacity:
                delay += occupancy[t][node] - capacity

        return delay

    def calculate_node_congestion(self) -> defaultdict:s
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

    def __init__(self, pop_size: int, num_agents: int, city: Environment):
        self.pop_size = pop_size
        self.num_agents = num_agents

        self.city = city
        self.human_traits = {"walking_speeds": [], "panic": []}

    def create_initial_population(self, simulation_params: dict) -> list:
        """
        Creates the initial population of chromosomes to be used, each has its
        own unique seed for reproduction of "random" probabilities.
        """
        agents, human_traits = self.initialise_agents(simulation_params)

        chromosome_seed_multipler = 6724
        population = []
        for chromosome_num in range(self.pop_size):
            chromosome = Chromosome(
                agents=agents,
                seed=chromosome_num * chromosome_seed_multipler,
                params=simulation_params,
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
        agent_seed_multiplier = 1235

        for agent in range(self.num_agents):
            agent_seed = agent_seed_multiplier * agent

            walking_speed = self.extract_walking_speed(
                agent_seed=agent_seed, params=simulation_params
            )
            panic = self.extract_panic(agent_seed=agent_seed, params=simulation_params)

            default_characteristics = {
                "walking_speed": walking_speed,  # How many time steps it takes to move 1 node.
                "altruism": 0,  # Probability the agent will stop at a node to help others.
                "vunerability": 0,  # Probability the agent will have to stop at a node due to an "issue"
                "familarity": 0,  # Not sure what this is yet.
                "panic": panic,  # mutation parameter addition rate, if the agent panics its more likely to pick a random path.
            }

            agents[agent] = Agent(
                name=f"Agent{agent}",
                seed=agent_seed,
                city=self.city,
                characteristics=default_characteristics,
            )

        return agents, self.human_traits

    def extract_panic(self, agent_seed: int, params: dict) -> float:
        if not params["panic"]:
            self.human_traits["panic"].append(0)
            return 0

        panic = random.beta(random.PRNGKey(agent_seed), a=2.0, b=5.0, shape=())

        self.human_traits["panic"] = panic
        return panic

    def extract_walking_speed(self, agent_seed: int, params: dict) -> int:
        if not params["walking"]:
            self.human_traits["walking_speeds"].append(1)
            return 1

        walking_speeds = [1, 2, 3]  # TODO: this is sooo basic but for now is fine.
        speed_idx = random.randint(
            random.PRNGKey(agent_seed), shape=(), minval=0, maxval=len(walking_speeds)
        )
        walking_speed = walking_speeds[speed_idx]

        self.human_traits["walking_speeds"].append(walking_speed)

        return walking_speed
