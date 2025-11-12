import jax
import jax.numpy as jnp

from jax import random
import numpy as np

import networkx as nx
import matplotlib.pyplot as plt

import itertools
import pickle


class Environment:

    def __init__(self, city_name: str):
        self.graph = self.create_graph(city_name)
        self.starts = self.extract_node_types(node_type='start')
        self.exits = self.extract_node_types(node_type='exit')

    def create_graph(self, city_name: str) -> nx.Graph:
        G = pickle.load(open(f'{city_name}.pickle', 'rb'))
        return G
    
    def extract_node_types(self, node_type: str) -> list:
        filtered_nodes = [n for n, attr in self.graph.nodes(data=True) if attr.get('type') == node_type]
        return filtered_nodes

    # def create_graph(self) -> nx.Graph:
    #     # TODO: how are you creating more "cities"
    #     G = nx.Graph()

    #     edges = [
    #         ("E", "A"),
    #         ("C", "A"),
    #         ("B", "A"),
    #         ("C", "D"),
    #         ("D", "B"),
    #         ("D", "F"),
    #         ("D", "G"),
    #     ]
    #     G.add_edges_from(edges)
    #     return G

    def __str__(self):
        nx.draw(
            self.graph, with_labels=True, node_color="lightblue", node_size=800, font_weight="bold"
        )
        plt.show(block=True)


class Agent:
    # Class to define agent properties e.g. behaviours/traits.
    def __init__(self, name: str, city: nx.Graph, path_seed: int):
        self.path_seed = path_seed
        self.name = name

        self.city = city
        self.key = random.PRNGKey(path_seed)

        self.start_location = self.select_start_point()
        self.path = self.generate_random_path()

        self.path_history = []
        
        # Test characteristic
        # TODO: what should the mutation rate even be?
        self.panic = 0.1
        # TODO: might want to add stuff later so safer
        self.mutation_rate = self.panic 

    def select_start_point(self) -> str | int:
        # TODO - stop this from being overwritten once done once.
        start_points = self.city.starts
        self.key, sub_key = random.split(self.key)
        idx = random.randint(sub_key, (), 0, len(start_points))
        return start_points[int(idx)]

    def generate_random_path(self) -> list:
        # TODO: finish this.
        # TODO: they can go back on themselves at the minute ...
        path = [self.start_location]
        current_location = self.start_location

        while current_location not in self.city.exits:
            neighbours = list(self.city.graph.neighbors(current_location))

            self.key, sub_key = random.split(self.key)
            idx = random.randint(sub_key, (), 0, len(neighbours))
            next_location = neighbours[int(idx)]

            path.append(next_location)
            current_location = next_location

        return path

    def update_current_path(self, new_path) -> None:
        self.path_history.append(self.path)
        self.path = new_path


class Chromosome:
    # chromosome = Chromosome(
    #     paths={"agent1": agent1_path1, "agent2": agent2_path1, "agent3": agent3_path1}
    # )
    def __init__(self, agents: dict[str, list]):
        # paths = {'agent1': [1, 2, 3]}
        self.agents = agents
        self.fitness = None

    def calculate_fitness(self) -> None:
        # Super basic assumption of 1 step = 1 time unit.
        path_length = [len(agent.path) for agent in self.agents.values()]
        total_evac_time = max(path_length)
        self.fitness = total_evac_time


class GeneticAlgorithm:

    def __init__(self, key) -> None:
        self.key = key

    def crossover(self, parents):
        parent1, parent2 = parents
        agents = parent1.agents

        # FIXME - carry on from here.
        self.key, sub_key = random.split(self.key)
        # Not starting at zero because that's the start and don't want the full length as can't crossover at the end either.
        cut = random.randint(minval=1, maxval=len(agents) - 1, key=sub_key, shape=())

        child1, child2 = {}, {}
        for i, agent_name in enumerate(agents):
            if i < cut:
                child1[agent_name] = parent1.agents[agent_name]
                child2[agent_name] = parent2.agents[agent_name]
            else:
                child1[agent_name] = parent2.agents[agent_name]
                child2[agent_name] = parent1.agents[agent_name]
                
        child1_chromosome = Chromosome(agents=child1)
        child2_chromosome = Chromosome(agents=child2)

        return child1_chromosome, child2_chromosome

    def repair_path(self, path, city, agent_key):
        exits = city.exits

        current_location = path[-1]
        if current_location in exits:
            return path, agent_key

        while current_location not in exits:
            neighbours = list(city.graph[current_location])

            agent_key, agent_sub_key = random.split(agent_key)
            idx = random.randint(agent_sub_key, (), 0, len(neighbours))
            next_location = neighbours[idx]
            path.append(next_location)
            current_location = next_location

        return path, agent_key

    def mutation(self, chromosome, city):
        mutation_occured = False
        for agent_name, agent_obj in chromosome.agents.items():
            
            agent_obj.key, agent_sub_key = random.split(agent_obj.key)
            mutation_probability = random.uniform(agent_sub_key, shape=(), minval=0.0, maxval=1.0)
            
            # TODO: later some agents will have different mutation rates e.g. panic.
            if mutation_probability < agent_obj.mutation_rate:
                mutation_occured = True
            
                # if random_number < hestinancy_probability:
                #     go to same step
                # else:
                agent_obj.key, agent_sub_key = random.split(agent_obj.key)

                path = agent_obj.path
                change_location = random.randint(key=agent_sub_key, minval=1, maxval=len(path) - 1, shape=())
                current_location = path[change_location]
                neighbours = list(city.graph[current_location])
                
                agent_obj.key, agent_sub_key = random.split(agent_obj.key)
                # TODO I DONT WANT IT TO GO IMMEDIATELY BACK ON SELF (hestinancy if decide to do it can do in fitness function see chaos above.)
                new_location_idx = random.randint(key=agent_sub_key, shape=(), minval=0, maxval=len(neighbours))
                new_location = neighbours[new_location_idx]

                new_partial = path[: change_location + 1] + [new_location]

                repaired_path, agent_obj.key = self.repair_path(new_partial, city, agent_obj.key)
                chromosome.agents[agent_name].path = repaired_path

        return chromosome, mutation_occured

    @staticmethod
    def find_parent(percent, population, cumulative_fitness):
        for chromosome, fitness in zip(population, cumulative_fitness):
            if percent <= fitness:
                return chromosome

    def selection(self, population):
        # roulette wheel.
        # TODO: don't think i need to do this but check before remove.
        [chromosome.calculate_fitness() for chromosome in population]
        
        # 1 over as need shorter lengths to be considered "fitter"
        fitness = [1 / chromosome.fitness for chromosome in population]
        total_fitness = sum(fitness)

        scaled_fitness = [f / total_fitness for f in fitness]
        cumulative_fitness = list(itertools.accumulate(scaled_fitness))

        self.key, sub_key = random.split(self.key)
        parent_percentages = random.uniform(minval=0, maxval=1, shape=len(population), key=sub_key)

        parents = []
        for percent in parent_percentages:
            parents.append(self.find_parent(percent, population, cumulative_fitness))

        return parents


city = Environment(city_name='small_city_graph')

num_agents = 10
population_size = 20

# TODO: generate the population
population = []
random_seed_count = 1
for i in range(population_size):
    agent_paths = {}
    for agent in range(num_agents):
        agent_paths[agent] = Agent(
            name=f"Agent{agent}_Chromosome{i}", path_seed=random_seed_count, city=city
        )

        # TODO: need to ensure each agent has different random seed for each population.
        # TODO: google this more...
        random_seed_count += 1

    chromosome = Chromosome(agents=agent_paths)
    chromosome.calculate_fitness()
    population.append(chromosome)


key = random.PRNGKey(12345)
# TODO: Selection, Cross over, mutation loop.
ga = GeneticAlgorithm(key=key)
avg_score = []

for evolution in range(100):
    # Step 1 - select the parents
    parents = ga.selection(population=population)
    parent_pairs = list(zip(parents[::2], parents[1::2]))

    # Step 2 - create the children
    children = []
    for pair in parent_pairs:
        parent_children = ga.crossover(parents=pair)
        children += parent_children

    # Step 3 - mutate 
    new_children = []
    for child in children:
        replacement_child, mutation_occured = ga.mutation(chromosome=child, city=city)
        if mutation_occured:
            new_children.append(replacement_child)
        
    # Step 4 - select the population going forward e.g. all children or based on fitness score.
    population += [child for child in new_children + children]
    # TODO: what do I want to track over the evolutions
    [chromosome.calculate_fitness() for chromosome in population]
    fitness = [chromosome.fitness for chromosome in population]
    
    
    pairs = list(zip(population, fitness))
    sorted_pairs = sorted(pairs, key=lambda x: x[1])
    population = [key for key, _ in sorted_pairs[:population_size]]
    
    avg_score.append(np.mean(fitness))
    print(f'Finished evolution # {evolution} with average fitness score: {np.mean(fitness)}\n')

plt.plot(range(len(avg_score)), avg_score)
plt.title('Average Fitness Score over each evolution')
plt.xlabel('Evolution')
plt.ylabel('Average Fitness Score')
plt.show()
