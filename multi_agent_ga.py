from utils.environment import Environment
from utils.agent import PopulationCreation
from utils.genetic_algorithm import GeneticAlgorithm
from utils.evaluation_metrics import Evaluation, FinalEvaluationMetrics
from utils.greedy import Greedy
import numpy as np
import matplotlib.pyplot as plt
from jax.random import PRNGKey, split


def print_log_line():
    print("-" * 150)


def greedy_algorithm(num_agents: int, simulation_params: dict, city: Environment):
    creator = PopulationCreation(city=city, pop_size=1, num_agents=num_agents)
    population, human_traits = creator.create_initial_population(
        simulation_params=simulation_params
    )

    greedy = Greedy(population=population)
    solution = greedy.solve()

    return solution


class KeyManager:
    def __init__(self, seed: int = 42) -> PRNGKey:
        self.key = PRNGKey(seed)

    def next_key(self):
        self.key, subkey = split(self.key)
        return subkey


def simulate_evacuation(
    city: Environment,
    num_agents: int,
    population_size: int,
    algorithm_seed: int,
    simulation_params: dict,
    max_evolutions: int = 100,
):
    """
    Main function to running the genetic algorithm.
    """
    # 1 Generate the population
    key_manager = KeyManager(seed=algorithm_seed)
    creator = PopulationCreation(
        city=city, pop_size=population_size, num_agents=num_agents, key_manager=key_manager
    )
    population, human_traits = creator.create_initial_population(
        simulation_params=simulation_params
    )

    # 2 Set up
    exit_criteria = {"max_evolutions": max_evolutions}
    ga = GeneticAlgorithm(
        exit_criteria=exit_criteria,
        pop_size=population_size,
        key_manager=key_manager,
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
        parents = ga.parent_selection(population=population)

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
    final_solution = sorted(population, key=lambda c: c.fitness)[0]
    # TODO: think I broke this with congestion so ignoring it.
    # visualiser = CityEvacuationHeatmap(solution=final_solution, city=city)
    # visualiser.animate_solution()
    final_eval = FinalEvaluationMetrics(solution=final_solution, params=simulation_params)
    final_eval.score()

    plt.plot(range(len(evaluation.avg_score)), evaluation.avg_score)
    plt.title("Average Fitness Score over each evolution")
    plt.xlabel("Evolution")
    plt.ylabel("Average Fitness Score")
    plt.show()

    return final_solution, evaluation


if __name__ == "__main__":
    city = Environment(city_name="super_small_city_graph")

    # greedy_algorithm(
    #     num_agents=10,
    #     simulation_params={"congestion": True, "walking": False, "panic": True},
    # )

    simulate_evacuation(
        city=city,
        num_agents=30,
        population_size=50,
        algorithm_seed=29,
        max_evolutions=21,
        simulation_params={"congestion": True, "walking": False, "fitness": "max"},
    )
