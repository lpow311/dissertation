from jax import random



class ParentSelection:
    
    def __init__(self, population_size: int):
        self.population_size = population_size
        
        self.elite_percentage = 0.05
        self.elite_num = max(1, int(self.elite_percentage * self.population_size))
                
    def roulette_selection(self, population: list, num_parents: int, key: random.PRNGKey) -> list:
        roulette_parents = num_parents - self.elite_num
        
        fitness = [chromosome.fitness for chromosome in population]
        total_fitness = sum(fitness)
        probabilities = [f / total_fitness for f in fitness]
        
        parents = []
        for _ in range(roulette_parents):
            
            key, sub_key = random.split(key)
            pick_prob = random.uniform(key=sub_key, shape=(), minval=0, maxval=1)
            chromosome = self.extract_cumulative_chromosome(
                population=population, probabilities=probabilities, pick_prob=pick_prob
            )
            parents.append(chromosome)
    
        return parents    
    
    def extract_cumulative_chromosome(self, population: list, probabilities: list, pick_prob: float):
        cumulative = 0
        for chromosome, probability in zip(population, probabilities):
            cumulative += probability
            if cumulative >= pick_prob:
                return chromosome
        
    def elite_selection(self, population: list) -> list:
        sorted_population = sorted(
            population, key=lambda chromosome: chromosome.fitness, reverse=True
        )
        return sorted_population[:self.elite_num]
