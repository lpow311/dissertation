import numpy as np
import matplotlib.pyplot as plt
from jax.random import PRNGKey, split
from tqdm import tqdm

from utils.environment import Environment
from utils.agent import PopulationCreation, generate_agent_options
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
    seed: PRNGKey,
    verbose: bool = True,
) -> None:

    attributes = {
        "starts": generate_agent_options(city.starts, num_agents, seed),
        "walking": generate_agent_options([1, 2, 3], num_agents, seed),
    }

    key_manager = KeyManager(seed=seed)
    creator = PopulationCreation(
        city=city,
        pop_size=1,
        num_agents=num_agents,
        key_manager=key_manager,
        attributes=attributes,
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
        solution=chromosome, params=simulation_params, algorithm="GREEDY"
    )
    final_eval.score(verbose=verbose)

    return solution, final_eval


def simulate_ga_evacuation(
    city: Environment,
    num_agents: int,
    population_size: int,
    seed: PRNGKey,
    simulation_params: dict,
    hyperparams: dict,
    verbose: int = 0,
):
    """
    Main function to running the genetic algorithm.
    """
    # 1 Generate the population
    attributes = {
        "starts": generate_agent_options(city.starts, num_agents, seed),
        "walking": generate_agent_options([1, 2, 3], num_agents, seed),
    }

    key_manager = KeyManager(seed=seed)
    creator = PopulationCreation(
        city=city,
        pop_size=population_size,
        num_agents=num_agents,
        key_manager=key_manager,
        attributes=attributes,
    )
    population, human_traits = creator.create_initial_population(
        simulation_params=simulation_params
    )

    # 2 Set up
    exit_criteria = {"max_evolutions": hyperparams["max_evolutions"]}
    ga = GeneticAlgorithm(
        exit_criteria=exit_criteria,
        pop_size=population_size,
        key_manager=key_manager,
        num_agents=num_agents,
        hyperparams=hyperparams,
    )

    evolution = 1
    terminate = ga.extract_termination_criteria(evolution=evolution)

    evaluation = Evaluation()

    avg_walking_speed = np.mean(human_traits["walking"])

    if verbose:
        print_log_line()
        print(
            f" City: {city.city_name} | Population size: {len(population)} | Walking Speed: {avg_walking_speed:.2}"
        )
        print_log_line()

    parent_method = "roulette" if hyperparams["tournament_size"] is None else "tournament"

    while not terminate:
        # PARENT SELECTION
        parents = ga.parent_selection(population=population, method=parent_method)

        # CROSSOVER
        children = ga.chromosome_crossover(parents=parents)

        # MUTATION
        mutated_children = ga.agent_mutation(children=children)

        # SURVIVOR SELECTION
        old_population = [p for p in population]

        population = ga.survivor_selection(
            children=mutated_children,
            old_population=old_population,
            method=hyperparams["survivor_method"],
        )

        evaluation.calculate_metrics(population, evolution, verbose)

        # 7 Termination
        evolution += 1
        terminate = ga.extract_termination_criteria(evolution=evolution)

    final_solution = sorted(population, key=lambda c: c.fitness)[0]
    final_eval = FinalEvaluationMetrics(
        solution=final_solution, params=simulation_params, algorithm="GA"
    )
    final_eval.score(verbose=verbose)
    final_eval.add_evolution_scores(evaluation.avg_score)

    return final_solution, final_eval


if __name__ == "__main__":
    n_experiments = 5
    num_agents = 20
    params = {"congestion": False, "walking": False, "fitness": "max-median", "alpha": 0.75}

    grid_city = Environment("grid_city")
    grid_city.congestion_amount = 2

    rng = np.random.default_rng(123)
    algorithm_seeds = rng.integers(0, 10**6, size=n_experiments)

    hyperparams = {
        "crossover": 0.8,
        "mutation": 0.1,
        "epsilon": 0.8,
        "max_evolutions": 50,
        "survivor_method": "elite_percentage",
        "tournament_size": None,
    }

    moderate_ga_solution_small, moderate_ga_eval_small = simulate_ga_evacuation(
        city=grid_city,
        num_agents=num_agents,
        population_size=50,
        simulation_params=params,
        hyperparams=hyperparams,
        seed=algorithm_seeds[0],
        verbose=10,
    )
