import networkx as nx
from utils.agent import Agent, Chromosome


class Greedy:

    def __init__(self, population: list, simulation_params: dict, key_manager) -> None:
        self.population = population
        self.agents = population[0].agents

        self.params = simulation_params
        self.key_manager = key_manager

    def solve(self) -> dict[int, Agent]:
        for i, agent in self.agents.items():
            self.agents[i].path = self.agent_shortest_paths(
                agent.city,
                agent.start_point,
            )

        return self.agents

    def turn_into_chromosome_for_evaluation(self) -> None:
        chromosome = Chromosome(
            agents=self.agents, params=self.params, key_manager=self.key_manager
        )
        chromosome.calculate_fitness()

        return chromosome

    def agent_shortest_paths(self, city: nx.Graph, start_node: str):
        best_path = None
        best_length = float("inf")

        # Evaluate all exits
        for exit_node in city.exits:
            try:
                path_length = nx.shortest_path_length(city.graph, start_node, exit_node)
                if path_length < best_length:
                    best_length = path_length
                    best_path = nx.shortest_path(city.graph, start_node, exit_node)
            except nx.NetworkXNoPath:
                continue  # skip if no path exists

        return best_path
