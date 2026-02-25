import numpy as np
import matplotlib.pyplot as plt
from jax.random import PRNGKey, split
from tqdm import tqdm

from utils.environment import Environment
from utils.agent import PopulationCreation
from utils.genetic_algorithm import GeneticAlgorithm
from utils.evaluation_metrics import Evaluation, FinalEvaluationMetrics
from utils.greedy import Greedy
from utils.algorithm_evaluation import AlgorithmComparison


def print_log_line():
    print("-" * 150)


class KeyManager:
    def __init__(self, seed: int = 42) -> PRNGKey:
        self.key = PRNGKey(seed)

    def next_key(self):
        self.key, subkey = split(self.key)
        return subkey


def simulate_greedy_evacuation(
    num_agents: int,
    simulation_params: dict,
    city: Environment,
    algorithm_seed: int,
    verbose: bool = True,
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
    final_eval.score(verbose=verbose)

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
    final_eval.score(verbose=verbose)
    final_eval.add_evolution_scores(evaluation.avg_score)

    return final_solution, final_eval


if __name__ == "__main__":
    city_name = "bottleneck_severe_small"
    city = Environment(city_name=city_name)

    num_agents = 30
    simulation_params = {"congestion": True, "walking": False, "fitness": "max"}
    n_experiments = 5

    rng = np.random.default_rng(42)
    algorithm_seeds = rng.integers(0, 10**6, size=n_experiments)

    greedy_outputs = []
    ga_outputs = []
    for seed_idx in tqdm(range(n_experiments)):

        _, greedy_evaluation = simulate_greedy_evacuation(
            num_agents=num_agents,
            simulation_params=simulation_params,
            algorithm_seed=algorithm_seeds[seed_idx],
            city=city,
            verbose=False,
        )
        greedy_outputs.append(greedy_evaluation)

        _, ga_evaluation = simulate_ga_evacuation(
            city=city,
            num_agents=num_agents,
            population_size=20,
            algorithm_seed=algorithm_seeds[seed_idx],
            max_evolutions=20,
            simulation_params={"congestion": True, "walking": False, "fitness": "max"},
            verbose=0,
        )
        ga_outputs.append(ga_evaluation)

    comparison = AlgorithmComparison(greedy_outputs=greedy_outputs, ga_outputs=ga_outputs)
    results = comparison.statistical_tests(verbose=True)
    poa = comparison.price_of_anarchy(verbose=True, city=city)

    comparison.exit_utilisation(city_name=city_name)
    comparison.exit_time_by_start_location()
