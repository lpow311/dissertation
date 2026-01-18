from utils.environment import Environment
from utils.agent import PopulationCreation
from utils.ga import GeneticAlgorithm
from utils.evaluation_metrics import Evaluation
from utils.visualisation_heatmap import CityEvacuationHeatmap
import numpy as np
import matplotlib.pyplot as plt


def print_log_line():
    print(
        "------------------------------------------------------------------------------------------------------"
    )


def simulate_evacuation(
    city: Environment,
    num_agents: int,
    population_size: int,
    algorithm_seed: int,
    simulation_params: dict,
    max_evolutions: int = 100,
):
    # 1 Generate the population
    creator = PopulationCreation(city=city, pop_size=population_size, num_agents=num_agents)
    population, human_traits = creator.create_initial_population(
        simulation_params=simulation_params
    )

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

    evaluation = Evaluation()

    avg_walking_speed = np.mean(human_traits["walking_speeds"])

    print_log_line()
    print(
        f" City: {city.city_name} | Population size: {len(population)} | Walking Speed: {avg_walking_speed:.2}"
    )
    print_log_line()

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

        evaluation.calculate_metrics(population, evolution)

        # 7 Termination
        evolution += 1
        terminate = ga.extract_termination_criteria(evolution=evolution)

    # TODO: might want to track best so far and use that at the end?
    final_solution = sorted(population, key=lambda c: c.fitness, reverse=True)[0]
    # TODO: think I broke this with congestion so ignoring it.
    # visualiser = CityEvacuationHeatmap(solution=final_solution, city=city)
    # visualiser.animate_solution()

    plt.plot(range(len(evaluation.avg_score)), evaluation.avg_score)
    plt.title("Average Fitness Score over each evolution")
    plt.xlabel("Evolution")
    plt.ylabel("Average Fitness Score")
    plt.show()


if __name__ == "__main__":
    city = Environment(city_name="super_small_city_graph")

    simulate_evacuation(
        city=city,
        num_agents=10,
        population_size=50,
        algorithm_seed=29,
        max_evolutions=20,
        simulation_params={"congestion": True, "human": False},
    )
