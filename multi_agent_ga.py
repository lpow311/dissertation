from utils.environment import Environment
from utils.agent import PopulationCreation
from utils.ga import GeneticAlgorithm
from utils.visualisation_heatmap import CityEvacuationHeatmap
import numpy as np
import matplotlib.pyplot as plt


def simulate_evacuation(
    city: Environment,
    num_agents: int,
    population_size: int,
    algorithm_seed: int,
    max_evolutions: int = 100,
):
    # 1 Generate the population
    creator = PopulationCreation(city=city, pop_size=population_size, num_agents=num_agents)
    population = creator.create_initial_population()

    # 2 Set up
    exit_criteria = {"max_evolutions": max_evolutions}
    ga = GeneticAlgorithm(
        exit_criteria=exit_criteria,
        pop_size=population_size,
        seed=algorithm_seed,
        num_agents=num_agents,
    )

    evolution = 1
    terminate = ga.extract_termination_criteria(evolution=evolution)

    avg_score = []

    while not terminate:
        # 3 Parent Selection
        parents = ga.parent_selection(population=population, num_parents=population_size)

        # 4 Crossover
        children = ga.chromosome_crossover(parents=parents)

        # 5 Mutation
        mutated_children = ga.agent_mutation(children=children)

        # 6 Survivor Selection
        population = ga.survivor_selection(
            children=mutated_children, old_population=population, keep_best=True
        )

        new_population_fitness = [c.fitness for c in population]
        avg = np.mean(new_population_fitness)
        best = np.max(new_population_fitness)
        worst = np.min(new_population_fitness)
        avg_score.append(avg)

        avg_path_length = np.mean([c.calculate_average_path() for c in population])

        print(
            f"Evolution {evolution:4d} | best: {best:.3f} | avg: {avg:.3f} | min: {worst:.3f} | avg path length={avg_path_length:.3f} | pop={len(population)}"
        )

        # 7 Termination
        evolution += 1
        terminate = ga.extract_termination_criteria(evolution=evolution)

    # TODO: might want to track best so far and use that at the end?
    final_solution = sorted(population, key=lambda c: c.fitness, reverse=True)[0]
    visualiser = CityEvacuationHeatmap(solution=final_solution, city=city)
    visualiser.animate_solution()

    plt.plot(range(len(avg_score)), avg_score)
    plt.title("Average Fitness Score over each evolution")
    plt.xlabel("Evolution")
    plt.ylabel("Average Fitness Score")
    plt.show()


if __name__ == "__main__":
    city = Environment(city_name="super_small_city_graph")

    simulate_evacuation(
        city=city, num_agents=10, population_size=50, algorithm_seed=29, max_evolutions=5
    )
