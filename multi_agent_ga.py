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


class KeyManager:
    def __init__(self, seed: int = 42) -> PRNGKey:
        self.key = PRNGKey(seed)

    def next_key(self):
        self.key, subkey = split(self.key)
        return subkey


def simulate_greedy_evacuation(
    num_agents: int, simulation_params: dict, city: Environment, algorithm_seed: int
) -> None:
    key_manager = KeyManager(seed=algorithm_seed)
    creator = PopulationCreation(
        city=city, pop_size=1, num_agents=num_agents, key_manager=key_manager
    )
    population, human_traits = creator.create_initial_population(
        simulation_params=simulation_params
    )

    greedy = Greedy(
        population=population, simulation_params=simulation_params, key_manager=key_manager
    )
    solution = greedy.solve()
    chromosome = greedy.turn_into_chromosome_for_evaluation()

    final_eval = FinalEvaluationMetrics(
        solution=chromosome,
        params=simulation_params,
    )
    final_eval.score()

    return solution, final_eval


def simulate_ga_evacuation(
    city: Environment,
    num_agents: int,
    population_size: int,
    algorithm_seed: int,
    simulation_params: dict,
    max_evolutions: int = 100,
    verbose: int = 0,
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

    if verbose:
        print_log_line()
        print(
            f" City: {city.city_name} | Population size: {len(population)} | Walking Speed: {avg_walking_speed:.2}"
        )
        print_log_line()

    while not terminate:
        parents = ga.parent_selection(population=population)
        children = ga.chromosome_crossover(parents=parents)
        mutated_children = ga.agent_mutation(children=children)
        population = ga.survivor_selection(
            children=mutated_children, old_population=population, keep_best=True
        )

        evaluation.calculate_metrics(population, evolution, verbose)

        # 7 Termination
        evolution += 1
        terminate = ga.extract_termination_criteria(evolution=evolution)

    final_solution = sorted(population, key=lambda c: c.fitness)[0]
    final_eval = FinalEvaluationMetrics(solution=final_solution, params=simulation_params)
    final_eval.score()
    final_eval.add_evolution_scores(evaluation.avg_score)

    return final_solution, final_eval


if __name__ == "__main__":
    city = Environment(city_name="super_small_city_graph")
    algorithm_seed = 29

    greedy_algorithm(
        num_agents=10,
        simulation_params={"congestion": True, "walking": False, "fitness": "max"},
        algorithm_seed=algorithm_seed,
        city=city,
    )

    simulate_evacuation(
        city=city,
        num_agents=30,
        population_size=50,
        algorithm_seed=algorithm_seed,
        max_evolutions=21,
        simulation_params={"congestion": True, "walking": False, "fitness": "max"},
        verbose=10,
    )
