import networkx as nx
from utils.agent import Agent, Chromosome
from jax import random
from utils.environment import Environment


class Greedy:

    def __init__(self, population: list, simulation_params: dict, key_manager) -> None:
        self.population = population
        self.agents = population[0].agents

        self.params = simulation_params
        self.key_manager = key_manager

    def solve(self, pick_first: bool = False) -> dict:
        for i, agent in self.agents.items():
            self.agents[i].path = self.agent_shortest_paths(
                agent.city, agent.start_point, pick_first
            )

        return self.agents

    def turn_into_chromosome_for_evaluation(self) -> None:
        chromosome = Chromosome(
            agents=self.agents,
            params=self.params,
            key_manager=self.key_manager,
            compliant_agents=[0] * len(self.agents),
        )
        chromosome.calculate_fitness()

        return chromosome

    def agent_shortest_paths(self, city: Environment, start_node: str, pick_first: bool):
        # TODO: if change this add it population class as well.
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
