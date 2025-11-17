import networkx as nx
from jax import random

from utils.environment import Environment

class Agent:
    
    def __init__(self, name: str, seed: int, city: Environment) -> None:
        self.name = name
        
        self.seed = seed
        self.key = random.PRNGKey(seed)
        
        self.city = city
        self.start_point = self.select_start_point()
        
        self.path = None
        
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
                target=exit_option
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
        return path
    
    
class Chromosome:
    
    def __init__(self, agents: dict, seed: int) -> None:
        self.seed = seed
        self.key = random.PRNGKey(seed)
        
        self.agents = self.deep_copy_agents(agents=agents)
        self.num_agents =  len(self.agents)
        self.fitness = None
        
    def deep_copy_agents(self, agents: dict) -> dict:
        agents_copy = {}
        for agent_num, agent in agents.items():
            chromosome_agent_key = self.split_key()
            
            new_agent = Agent(agent.name, agent.seed, agent.city)
            new_agent.generate_random_path(chromosome_key=chromosome_agent_key)
            
            agents_copy[agent_num] = new_agent

        return agents_copy

    def split_key(self) -> random.PRNGKey:
        self.key, sub_key = random.split(self.key)
        return sub_key

    def calculate_fitness(self) -> None:
        score = 0
        for agent in self.agents.values():
            path = agent.path
            
            path_length = len(path)
            unique_nodes = len(set(path))
            loop_score = unique_nodes / path_length
            distance_score = agent.shortest_path_length / path_length
            
            # Needs to add to one for my brain...
            score += (0.5*loop_score + 0.5*distance_score)
        
        self.fitness = score / self.num_agents


class PopulationCreation:
    
    def __init__(self, pop_size: int, num_agents: int, city: Environment):
        self.pop_size = pop_size
        self.num_agents = num_agents
        
        self.city = city
        
    def create_initial_population(self) -> list:
        agents = self.initialise_agents()
        
        chromosome_seed_multipler = 6724
        population = []
        for chromosome_num in range(self.pop_size):
            chromosome = Chromosome(
                agents=agents,
                seed=chromosome_num*chromosome_seed_multipler
            )
            chromosome.calculate_fitness()
            
            population.append(chromosome)
            
        return population
        
    def initialise_agents(self) -> dict:
        # TODO: when add agent behaviour we need to make sure it transfers.
        agents = {}
        agent_seed_multiplier = 1234
        
        for agent in range(self.num_agents):
            agents[agent] = Agent(
                name=f'Agent{agent}',
                seed=agent_seed_multiplier*agent,
                city=self.city
            )
        
        return agents
    

if __name__ == "__main__":
    creator = PopulationCreation(pop_size=2, num_agents=2)
    creator.create_initial_population()