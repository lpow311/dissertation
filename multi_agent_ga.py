from utils.environment import Environment
from utils.agent import Agent, PopulationCreation, Chromosome
from utils.ga import GeneticAlgorithm



def simulate_evacuation(city: Environment, num_agents: int, population_size: int, algorithm_seed: int):
    #1 Generate the population
    creator = PopulationCreation(city=city, pop_size=population_size, num_agents=num_agents)
    population = creator.create_initial_population()
    
    #2 Set up
    exit_criteria = {'max_evolutions': 100}
    ga = GeneticAlgorithm(
        exit_criteria=exit_criteria, pop_size=population_size, seed=algorithm_seed
    )
    
    evolution = 1
    terminate = ga.extract_termination_criteria(evolution=evolution)
    
    while not terminate:
        #3 Parent Selection
        parents = ga.parent_selection(population=population, num_parents=population_size)
    
        #4 Crossover
        
        #5 Mutation
        
        #6 Survivor Selection
         
        #7 Termination    
        evolution += 1
        terminate = ga.extract_termination_criteria(evolution=evolution)
        
        
    




if __name__ == "__main__":
    city = Environment(city_name='small_city_graph')

    simulate_evacuation(
        city=city,
        num_agents=2,
        population_size=2,
        algorithm_seed=29
    )