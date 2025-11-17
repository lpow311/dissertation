from jax import random

from utils.parent_selection import ParentSelection


class GeneticAlgorithm(ParentSelection):
    
    def __init__(self, exit_criteria: dict, pop_size: int, seed: int):
        super(ParentSelection, self).__init__(population_size=pop_size)
        
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