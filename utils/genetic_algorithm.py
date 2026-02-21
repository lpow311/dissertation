from jax import random
from jax import numpy as jnp
import networkx as nx

from utils.agent import Chromosome, Agent


class GeneticAlgorithm:

    def __init__(self, exit_criteria: dict, pop_size: int, key_manager, num_agents: int) -> None:

        self.exit_criteria = exit_criteria
        self.population_size = pop_size
        self.num_agents = num_agents

        self.key_manager = key_manager

        ####
        self.crossover_probability = 0.8  # 0.6 and 0.9
        self.mutation_probability = 0.05
        self.epsilon = 0.2

    def generate_uniform_probability(self) -> float:
        return random.uniform(key=self.key_manager.next_key(), shape=(), minval=0, maxval=1)

    def __generate_integer(self, max_val: int, min_val: int = 0) -> int:
        return random.randint(
            key=self.key_manager.next_key(), shape=(), minval=min_val, maxval=max_val
        )

    ################# GENERAL FUNCTIONALITY #################

    def extract_termination_criteria(self, evolution: int):
        # TODO: this is very basic needs improving.
        return evolution >= self.exit_criteria["max_evolutions"]

    ################# PARENT SELECTION #################

    def parent_selection(self, population: list) -> list:
        # TODO: currently allows for duplication in parent selection
        roulette_parents = self.__roulette_selection(population=population)
        return roulette_parents

    def __roulette_selection(self, population: list) -> list:
        fitness = [chromosome.fitness for chromosome in population]

        inverted_fitness = [1 / f for f in fitness]
        total_fitness = sum(inverted_fitness)
        probabilities = [f / total_fitness for f in inverted_fitness]

        parents = []
        for _ in range(len(population)):
            pick_prob = self.generate_uniform_probability()

            parent = self.__extract_cumulative_chromosome(
                population=population, probabilities=probabilities, prob=pick_prob
            )
            parents.append(parent)

        return parents

    def __extract_cumulative_chromosome(self, population: list, probabilities: list, prob: float):
        cumulative = 0
        for chromosome, probability in zip(population, probabilities):
            cumulative += probability
            if cumulative >= prob:
                return chromosome

    ################# CROSSOVER #################

    def chromosome_crossover(self, parents: list) -> list:
        parent_pairs = self.__extract_parent_pairs(parents=parents)
        params = parents[0].params

        children = []
        for parent1, parent2 in parent_pairs:
            prob = self.generate_uniform_probability()

            if prob <= self.crossover_probability:
                child1_agents, child2_agents = self.__single_point_crossover(
                    parent1=parents[parent1], parent2=parents[parent2]
                )
                children.append(
                    Chromosome(agents=child1_agents, key_manager=self.key_manager, params=params)
                )
                children.append(
                    Chromosome(agents=child2_agents, key_manager=self.key_manager, params=params)
                )
            else:
                children.append(parents[parent1])
                children.append(parents[parent2])

        return children

    def __extract_parent_pairs(self, parents: list) -> list:
        subkey = self.key_manager.next_key()
        shuffled = random.permutation(subkey, jnp.array(range(len(parents))))
        pairs = list(zip(shuffled[0::2], shuffled[1::2]))
        return pairs

    def __single_point_crossover(self, parent1: Chromosome, parent2: Chromosome) -> list:
        child1_agents, child2_agents = {}, {}

        for agent in range(self.num_agents):

            parent1_path = parent1.agents[agent].path
            parent2_path = parent2.agents[agent].path

            # Ignore the start and end point for each path.
            overlap = set(parent1_path[1:-1]).intersection(parent2_path[1:-1])
            if not overlap:
                child1_path, child2_path = parent1_path, parent2_path
            else:
                overlap_node_idx = self.__generate_integer(max_val=len(overlap))
                overlap_node = list(overlap)[overlap_node_idx]

                idx_parent1 = parent1_path.index(overlap_node)
                idx_parent2 = parent2_path.index(overlap_node)

                child1_path = parent1_path[: idx_parent1 + 1] + parent2_path[idx_parent2 + 1 :]
                child2_path = parent2_path[: idx_parent2 + 1] + parent1_path[idx_parent1 + 1 :]

            child1_agents[agent] = parent1.agents[agent].copy_agent(new_path=child1_path)
            child2_agents[agent] = parent1.agents[agent].copy_agent(new_path=child2_path)

        return child1_agents, child2_agents

    ################# MUTATION #################

    def agent_mutation(self, children: list) -> list:
        mutated_children = []
        for chromosome in children:
            for agent_name, agent in chromosome.agents.items():
                prob = self.generate_uniform_probability()

                if prob <= self.mutation_probability:
                    mutated_agent = self.__agent_exit_path_mutation(agent=agent)
                    chromosome.agents[agent_name] = mutated_agent

            mutated_children.append(chromosome)

        return mutated_children

    def __agent_exit_path_mutation(self, agent: Agent) -> Agent:
        # Decide which exit the agent will now go to.
        new_exit = self.__select_exit(agent=agent)

        # Pick the point on the path they switch at.
        path = agent.path
        partial_point = self.__generate_integer(min_val=1, max_val=len(path) - 1)
        mutation_node = path[partial_point]

        end_path = self.epsilon_greedy_path_selection(agent, mutation_node, new_exit)
        new_path = path[:partial_point] + end_path

        agent.update_path(new_path)
        return agent

    def __select_exit(self, agent: Agent) -> str:
        exits = agent.preferred_exits
        exit_idx = self.__generate_integer(max_val=len(exits))
        return exits[exit_idx]

    def epsilon_greedy_path_selection(self, agent: Agent, start: str, end: str) -> list:
        path = [start]
        current_node = start
        visited = {start}
        graph = agent.city.graph

        while current_node not in agent.city.exits:
            neigbours = list(graph.neighbors(current_node))

            unvisited = [n for n in neigbours if n not in visited]
            if not unvisited:
                unvisited = neigbours

            prob = self.generate_uniform_probability()
            if prob < self.epsilon:
                # pick greedy neighbour
                best_node = min(unvisited, key=lambda n: agent.city.distances_to_exits[n][end])
                current_node = best_node
            else:
                neighbour_idx = self.__generate_integer(max_val=len(unvisited))
                current_node = unvisited[neighbour_idx]

            path.append(current_node)
            visited.add(current_node)

        return path

    ################# SELECTION #################

    def survivor_selection(self, children: list, old_population: list, keep_best: bool) -> list:
        [c.calculate_fitness() for c in children]

        if keep_best:
            [c.calculate_fitness() for c in old_population]
            sorted_old = sorted(old_population, key=lambda c: c.fitness)
            n_elite = max(1, int(0.05 * len(old_population)))
            elite = sorted_old[:n_elite]

            sorted_children = sorted(children, key=lambda c: c.fitness)
            remaining_slots = len(children) - n_elite

            new_population = elite + sorted_children[:remaining_slots]
        else:
            new_population = [c for c in children]

        return new_population
