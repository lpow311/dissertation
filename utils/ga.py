from jax import random

from utils.parent_selection import ParentSelection
from utils.crossover import CrossOver
from utils.mutation import Mutation


class GeneticAlgorithm(ParentSelection, CrossOver, Mutation):
    
    def __init__(self, exit_criteria: dict, pop_size: int, seed: int, num_agents: int):
        ParentSelection.__init__(self, population_size=pop_size)
        CrossOver.__init__(self, num_agents=num_agents)
        Mutation.__init__(self)
        
        self.exit_criteria = exit_criteria
        
        self.population_size = pop_size
        
        self.seed = seed
        self.key = random.PRNGKey(seed)
    
    def split_key(self) -> random.PRNGKey:
        self.key, sub_key = random.split(self.key)
        return sub_key
    
    def extract_termination_criteria(self, evolution: int):
        # TODO: this is very basic needs improving.
        return evolution >= self.exit_criteria['max_evolutions']
    
    def parent_selection(self, population: list, num_parents: int) -> list:
        # TODO: currently allows for duplication (might want this, might not...)    
    
        elite_parents = self.elite_selection(population=population)
        roulette_parents, self.key = self.roulette_selection(
            population=population, key=self.key, num_parents=num_parents
        )
        return elite_parents + roulette_parents
    
    def chromosome_crossover(self, parents: list) -> list:
        children, self.key = self.perform_crossover(parents=parents, key=self.key)
        return children
    
    def agent_mutation(self, children: list) -> list:
        mutated_children= self.partial_path_mutation(children=children)
        return mutated_children
    
    def survivor_selection(self, children: list, old_population: list, keep_best: bool) -> list:
        if keep_best:
            potential_population = children + old_population
            [c.calculate_fitness() for c in potential_population]
            new_population = sorted(potential_population, key=lambda c: c.fitness, reverse=True)[:self.population_size]
        else:
            [c.calculate_fitness() for c in old_population]
            fill_gap = self.population_size - len(children)
            elite_old_population = sorted(old_population, key=lambda c: c.fitness, reverse=True)[:fill_gap]
            new_population = children + elite_old_population
        
        return new_population
