import networkx as nx
from jax import random
from jax.random import PRNGKey
import numpy as np


from collections import defaultdict

from utils.environment import Environment


class Agent:
    """
    Agent class which is used to define the agent characteristics and
    keep track of the current path and shortest path.
    """

    def __init__(
        self,
        name: str,
        city: Environment,
        characteristics: dict,
        key_manager,
        shortest_path: list,
        start_point: str,
        path: list | None = None,
        preferred_exits: list | None = None,
    ) -> None:
        self.name = name
        self.city = city
        self.key_manager = key_manager

        self.characteristics = characteristics
        self.speed = characteristics["walking"]
        self.compliant = bool(characteristics["compliance"])

        self.start_point = start_point
        self.shortest_path = shortest_path
        self.shortest_path_length = len(self.shortest_path)

        if not self.compliant:
            self.path = self.shortest_path
        elif path is None and self.compliant:
            self.path = self.generate_random_path()
        else:
            self.path = path

        if preferred_exits is None:
            # TODO: can update this later for local vs visitor.
            self.preferred_exits = self.city.exits
        else:
            self.preferred_exits = preferred_exits

    def generate_random_path(self) -> list:
        """
        Generate a random path for an agent from the pre-defined
        starting node.
        """
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
        """
        Function for use outside the class to update the
        path paramter whilst also removing any loops.
        """
        self.path = self.remove_loops_from_path(path=path)

    @staticmethod
    def remove_loops_from_path(path: list) -> list:
        """
        Removes loops from agent paths - if an agent visits a node
        twice then the nodes between those two points are removed.
        e.g. A-B-C-A-D-E becomes A-D-E.
        """

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
        """
        Create a new agent which copies all the same agent characteristics
        either with the same path (when new_path is None [default]) or with
        a new path (for new paths the loops are also removed).
        """

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
            shortest_path=self.shortest_path,
            preferred_exits=self.preferred_exits,
        )


class Chromosome:
    """
    Chromosome class to keep track of the agents for each
    solution alongside the experiement conditions.
    """

    def __init__(
        self,
        agents: dict,
        key_manager,
        compliance: list,
        params: dict = {"congestion": False, "human": False, "fitness": "max"},
        initialisation: bool = False,
    ) -> None:
        self.params = params
        self.key_manager = key_manager

        self.agents = self.deep_copy_agents(agents=agents, initialisation=initialisation)
        self.num_agents = len(self.agents)
        self.compliance = compliance

        self.fitness = None
        self.fitness_calc = params["fitness"]

        self.congestion_score = []
        self.path_lengths = []
        self.path_time = []

        self.agent_time_paths = {}

    def deep_copy_agents(self, agents: dict, initialisation: bool) -> dict:
        """
        Copy all agents in a chromosome to create a new chromosome.
        """
        agents_copy = {}
        for agent_num, agent in agents.items():
            new_agent = agent.copy_agent(new_path=None)
            if initialisation and agent.compliant:
                new_agent.path = new_agent.generate_random_path()

            agents_copy[agent_num] = new_agent

        return agents_copy

    def calculate_fitness(self) -> None:
        """
        Calculates the path fitness for each of the agents and then
        calculates the chrosomosome fitness based off the parameter
        input i.e. max, mean, median, ...

        As long as you note in your dissertation that congestion delay
        includes both true congestion and speed-induced blocking it's
        a legitimate simplification.
        """
        agent_times, congestion_delays = self.get_timesteps()
        self.agent_time_paths = agent_times

        self.path_time = [len(path) for path in agent_times.values()]
        self.path_lengths = [len(set(path)) for path in agent_times.values()]
        self.start_delays = [a.characteristics["delay_start"] for a in self.agents.values()]

        self.congestion_score = [congestion_delays[a] for a in agent_times]
        self.walking_delay = list(
            np.array(self.path_time)
            - np.array(self.path_lengths)
            - np.array(self.congestion_score)
            - np.array(self.start_delays)
        )

        self.fitness = self.calculate_chromosome_fitness(self.path_time)

    def get_timesteps(self) -> dict:
        """
        For all agents take their route and calculate the timesteps across
        the simulation considering congestion, walking speed and delayed
        starts.
        -> When the congestion parameter is turned off then the
            nodes have capacity of number of agents + 10.
        """

        city_nodes = self.agents[0].city.graph.nodes
        capacity = (
            self.agents[0].city.congestion_amount
            if self.params["congestion"]
            else (self.num_agents + 10)
        )

        speeds = {a: int(v.characteristics["walking"]) for a, v in self.agents.items()}
        times, delays, current, walking, congestion, start_delays = self.setup_timesteps()

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
                    delayed_agents = [a for a in agents if start_delays[a] > 0]

                    # Queued agents always have priority; fill remaining slots randomly
                    if queue:
                        priority_agents = [a for a in queue if a not in delayed_agents]
                        remaining_slots = capacity - len(priority_agents)
                        arriving_agents = [
                            a for a in agents if (a not in queue) and (a not in delayed_agents)
                        ]

                        if remaining_slots > 0 and arriving_agents:
                            extra = self.sample_without_replacement(
                                self.key_manager.next_key(),
                                arriving_agents,
                                min(remaining_slots, len(arriving_agents)),
                            )
                            first_agents = priority_agents + extra
                        else:
                            first_agents = self.sample_without_replacement(
                                self.key_manager.next_key(), priority_agents, capacity
                            )
                    else:
                        active_agents = [a for a in agents if a not in delayed_agents]
                        if len(active_agents) > 0:
                            first_agents = self.sample_without_replacement(
                                self.key_manager.next_key(),
                                active_agents,
                                min(len(active_agents), capacity),
                            )
                        else:
                            first_agents = []

                    stuck_agents = [
                        a for a in agents if (a not in first_agents) and (a not in delayed_agents)
                    ]

                    for agent in delayed_agents:
                        start_delays[agent] -= 1
                        times[agent].append(node)

                    for agent in first_agents:
                        times[agent].append(node)
                        walking[agent] += 1
                        if walking[agent] > speeds[agent]:
                            current[agent] += 1
                            walking[agent] = 1
                        delays[agent] = False
                        if agent in node_queue[node]:
                            node_queue[node].remove(agent)

                    for agent in stuck_agents:
                        times[agent].append(node)
                        delays[agent] = True
                        congestion[agent] += 1
                        if agent not in node_queue[node]:  # prevent duplicate queue entries
                            node_queue[node].append(agent)

                else:
                    for agent in agents:
                        if start_delays[agent] > 0:
                            start_delays[agent] -= 1
                            times[agent].append(node)
                        else:
                            times[agent].append(node)
                            walking[agent] += 1
                            if walking[agent] > speeds[agent]:
                                current[agent] += 1
                                walking[agent] = 1
                            delays[agent] = False
                            if agent in node_queue[node]:
                                node_queue[node].remove(agent)

            if all(current[a] >= len(self.agents[a].path) for a in current):
                break

        return times, congestion

    def setup_timesteps(self) -> tuple:
        """
        Create the storing options for the timestep calculations
        """
        times, delays, current, walking, congestion, start_delays = {}, {}, {}, {}, {}, {}
        for a in self.agents:
            times[a] = []
            delays[a] = False
            current[a] = 0
            walking[a] = 1
            congestion[a] = 0
            start_delays[a] = self.agents[a].characteristics["delay_start"]

        return times, delays, current, walking, congestion, start_delays

    @staticmethod
    def sample_without_replacement(key, items: list, num_samples: int) -> list:
        """
        Randomly selecti `num_samples` from a list of items without replacement.
        """
        indices = random.choice(key, a=len(items), shape=(num_samples,), replace=False)
        return [items[i] for i in indices]

    def get_agent_positions(self, city_nodes: list, current: dict) -> dict:
        """
        Identify the current position of each agent and update each node
        accordingly. This allows us to calculate congestion delays.
        """
        nodes = {n: [] for n in city_nodes}
        for a, pos in current.items():
            if pos < len(self.agents[a].path):
                node = self.agents[a].path[pos]
                nodes[node].append(a)

        return nodes

    def calculate_chromosome_fitness(self, fitnesses: list) -> float:
        """
        Calculate the chromosome fitness.
        """

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
        """
        Calaculate the average agent path length for the chromosome.
        """
        lengths = []
        for agent in self.agents.values():
            path = agent.path
            lengths.append(len(path))

        return np.mean(lengths)


class PopulationCreation:
    """
    Population creation class for each genetic algorithm run.
    """

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
        agents = self.initialise_agents()

        population = []
        for _ in range(self.pop_size):
            chromosome = Chromosome(
                agents=agents,
                params=simulation_params,
                key_manager=self.key_manager,
                initialisation=True,
                compliance=self.attributes["compliance"],
            )
            chromosome.calculate_fitness()

            population.append(chromosome)

        return population, self.attributes

    def initialise_agents(self) -> dict:
        """
        Creates a set of agents to use in the modelling ensuring the agent
        characteristics are the same across the different chromosomes.
        """
        agents = {}

        for agent in range(self.num_agents):
            attributes = {
                "walking": self.attributes["walking"][agent],
                "delay_start": self.attributes["delay_start"][agent],
                "compliance": self.attributes["compliance"][agent],
            }
            shortest_path = self.agent_shortest_paths(
                city=self.city, start_node=self.attributes["starts"][agent], pick_first=False
            )

            agents[agent] = Agent(
                name=f"Agent{agent}",
                city=self.city,
                characteristics=attributes,
                key_manager=self.key_manager,
                start_point=self.attributes["starts"][agent],
                shortest_path=shortest_path,
            )

        return agents

    def agent_shortest_paths(self, city: Environment, start_node: str, pick_first: bool):
        """
        Calculates an agents shortest path - should match that seen in the agent class.
        # TODO: if change this add it greedy class as well.
        """
        # Find minimum distance across all exits
        min_length = float("inf")

        for exit_node in city.exits:
            try:
                path_length = nx.shortest_path_length(city.graph, start_node, exit_node)
                if path_length < min_length:
                    min_length = path_length
            except nx.NetworkXNoPath:
                continue

        all_paths = []
        for exit_node in city.exits:
            try:
                if nx.shortest_path_length(city.graph, start_node, exit_node) == min_length:
                    all_paths.extend(nx.all_shortest_paths(city.graph, start_node, exit_node))
            except nx.NetworkXNoPath:
                continue

        if pick_first:
            return all_paths[0]
        else:
            path_idx = random.randint(
                key=self.key_manager.next_key(), shape=(), minval=0, maxval=len(all_paths)
            )
            return all_paths[path_idx]


def generate_agent_options(
    options: list,
    num_agents: int,
    seed: int,
    on: bool,
    default: float = 0,
    weights: list | None = None,
) -> list:
    """
    Generate a list of characteristics for each agent. If this characteristic is
    not turned `on` then the `default` value is used, otherwise the options are
    randomly selected according to the weights provided.
    """
    if not on:
        return [default] * num_agents
    rng = np.random.default_rng(seed)
    return list(rng.choice(options, size=num_agents, replace=True, p=weights))
