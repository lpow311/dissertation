import networkx as nx
from jax import random
from jax.random import PRNGKey
import numpy as np


from collections import defaultdict

from utils.environment import Environment


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

        path = self.remove_loops_from_path(path=path)
        return path

    def update_path(self, path: list) -> None:
        self.path = self.remove_loops_from_path(path=path)

    @staticmethod
    def remove_loops_from_path(path: list) -> list:
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
            new_path = self.remove_loops_from_path(path=new_path)

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
        initialisation: bool = False,
    ) -> None:
        self.params = params
        self.key_manager = key_manager

        self.agents = self.deep_copy_agents(agents=agents, initialisation=initialisation)
        self.num_agents = len(self.agents)
        self.fitness = None
        self.fitness_calc = params["fitness"]

        self.congestion_score = []
        self.path_lengths = []
        self.path_times = []

        self.agent_time_paths = {}

    def deep_copy_agents(self, agents: dict, initialisation: bool) -> dict:
        agents_copy = {}
        for agent_num, agent in agents.items():
            new_agent = agent.copy_agent(new_path=None)
            if initialisation:
                new_agent.path = new_agent.generate_random_path()

            agents_copy[agent_num] = new_agent

        return agents_copy

    def calculate_fitness(self) -> None:
        """
        Calculates the path fitness for each of the agents and then
        calculates the chrosomosome fitness based off the parameter
        input i.e. max, mean, median, ...
        """
        # TODO: haven't considered walking speed....
        agent_times = self.get_timesteps()
        self.agent_time_paths = agent_times

        # path time will be the equivalent of fitness as if not congestion
        # then it just considers path length (ignoring the walking speed as need to sort...)
        self.path_time = [len(path) for path in agent_times.values()]
        self.path_lengths = [len(set(path)) for path in agent_times.values()]
        self.congestion_score = list(np.array(self.path_time) - np.array(self.path_lengths))

        self.fitness = self.calculate_chromosome_fitness(self.path_time)

    def get_timesteps(self) -> dict:
        city_nodes = self.agents[0].city.graph.nodes
        if not self.params["congestion"]:
            return {a: agent.path for a, agent in self.agents.items()}

        capacity = self.agents[0].city.congestion_amount
        times, delays, current = self.setup_timesteps()
        node_queue = {n: [] for n in city_nodes}

        while True:
            nodes = self.get_agent_positions(city_nodes, current)
            nodes_with_agents = {node: agents for node, agents in nodes.items() if agents}

            if not nodes_with_agents:
                break

            for node, agents in nodes_with_agents.items():
                # Exit nodes are never congested — pass all agents through immediately
                if "E" in node:
                    for agent in agents:
                        times[agent].append(node)
                        current[agent] += 1
                        delays[agent] = False
                        if agent in node_queue[node]:
                            node_queue[node].remove(agent)
                    continue

                if len(agents) > capacity:
                    queue = node_queue[node]

                    # Queued agents always have priority; fill remaining slots randomly
                    if queue:
                        priority_agents = list(queue)
                        remaining_slots = capacity - len(priority_agents)
                        arriving_agents = [a for a in agents if a not in queue]

                        if remaining_slots > 0 and arriving_agents:
                            extra = self.sample_without_replacement(
                                self.key_manager.next_key(),
                                arriving_agents,
                                min(remaining_slots, len(arriving_agents)),
                            )
                            first_agents = priority_agents + extra
                        else:
                            first_agents = priority_agents[:capacity]
                    else:
                        first_agents = self.sample_without_replacement(
                            self.key_manager.next_key(), agents, capacity
                        )

                    stuck_agents = [a for a in agents if a not in first_agents]

                    for agent in first_agents:
                        times[agent].append(node)
                        current[agent] += 1
                        delays[agent] = False
                        if agent in node_queue[node]:
                            node_queue[node].remove(agent)

                    for agent in stuck_agents:
                        times[agent].append(node)
                        delays[agent] = True
                        if agent not in node_queue[node]:  # prevent duplicate queue entries
                            node_queue[node].append(agent)

                else:
                    for agent in agents:
                        times[agent].append(node)
                        current[agent] += 1
                        delays[agent] = False
                        if agent in node_queue[node]:
                            node_queue[node].remove(agent)

            if all(current[a] >= len(self.agents[a].path) for a in current):
                break

        return times

    def add_exits_to_paths(self, nodes: dict, times: dict) -> dict:
        exit_nodes = {n: a for n, a in nodes.items() if "E" in n}
        for node, agents in exit_nodes.items():
            for agent in agents:
                times[agent].append(node)

        return times

    @staticmethod
    def sample_without_replacement(key, items: list, num_samples: int) -> list:
        indices = random.choice(key, a=len(items), shape=(num_samples,), replace=False)
        return [items[i] for i in indices]

    def setup_timesteps(self) -> tuple:
        times, delays, current = {}, {}, {}
        for a in self.agents:
            times[a] = []
            delays[a] = False
            current[a] = 0

        return times, delays, current

    def get_agent_positions(self, city_nodes: list, current: dict) -> dict:
        nodes = {n: [] for n in city_nodes}
        for a, pos in current.items():
            if pos < len(self.agents[a].path):
                node = self.agents[a].path[pos]
                nodes[node].append(a)

        return nodes

    def calculate_chromosome_fitness(self, fitnesses: list) -> float:
        vals = {
            "mean": float(np.mean(fitnesses)),
            "median": float(np.median(fitnesses)),
            "max": float(np.max(fitnesses)),
            "min": float(np.min(fitnesses)),
        }

        if "-" in self.fitness_calc:
            metric1, metric2 = self.fitness_calc.split("-")
            alpha = self.params.get("alpha", 1)

            return (alpha * vals[metric1]) + ((1 - alpha) * vals[metric2])
        else:
            return vals[self.fitness_calc]

    def calculate_average_path(self) -> float:
        lengths = []
        for agent in self.agents.values():
            path = agent.path
            lengths.append(len(path))

        return np.mean(lengths)


class PopulationCreation:

    def __init__(
        self,
        pop_size: int,
        num_agents: int,
        city: Environment,
        attributes: dict[str, list],
        key_manager,
    ) -> None:
        self.pop_size = pop_size
        self.num_agents = num_agents

        self.city = city
        self.attributes = attributes

        self.key_manager = key_manager

    def create_initial_population(self, simulation_params: dict) -> list:
        """
        Creates the initial population of chromosomes to be used, each has its
        own unique seed for reproduction of "random" probabilities.
        """
        agents = self.initialise_agents(simulation_params)

        population = []
        for _ in range(self.pop_size):
            chromosome = Chromosome(
                agents=agents,
                params=simulation_params,
                key_manager=self.key_manager,
                initialisation=True,
            )
            chromosome.calculate_fitness()

            population.append(chromosome)

        return population, self.attributes

    def initialise_agents(self, simulation_params: dict) -> dict:
        """
        Creates a set of agents to use in the modelling ensuring the agent
        characteristics are the same across the different chromosomes.
        """
        agents = {}

        for agent in range(self.num_agents):
            walking_speed = self.attributes["walking_speed"][agent]
            start_point = self.attributes["starts"][agent]

            default_characteristics = {
                "walking_speed": walking_speed,  # How many time steps it takes to move 1 node.
            }

            agents[agent] = Agent(
                name=f"Agent{agent}",
                city=self.city,
                characteristics=default_characteristics,
                key_manager=self.key_manager,
                start_point=start_point,
            )

        return agents


def generate_agent_options(options: list, num_agents: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    return list(rng.choice(options, size=num_agents, replace=True))
