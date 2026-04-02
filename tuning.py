from multiprocessing import Pool
import numpy as np

from utils.environment import Environment
from main import simulate_ga_evacuation, simulate_greedy_evacuation

from time import perf_counter
import pickle


def basic_function(extra_params):
    walking, delay, compliance, city_name = extra_params

    num_agents = 100

    n_experiments = 10

    original_params = {
        "congestion": True,
        "walking": walking,
        "delay_start": delay,
        "compliance": compliance,
        "fitness": "max-median",
        "alpha": 0.8,
    }

    city = Environment(city_name)

    rng = np.random.default_rng(123)
    algorithm_seeds = rng.integers(0, 10**6, size=n_experiments)

    original_hyperparams = {
        "crossover": 0.8,
        "mutation": 0.1,
        "epsilon": 1,
        "max_evolutions": 100,  # 50
        "survivor_method": "elite_percentage",
        "tournament_size": 3,
    }

    params = original_params
    hyperparams = original_hyperparams
    pop_size = 50

    solutions_ga, evals_ga = [], []
    solutions_greedy, evals_greedy = [], []
    for seed in algorithm_seeds:
        seed_solution, seed_eval = simulate_ga_evacuation(
            city=city,
            num_agents=num_agents,
            population_size=pop_size,  # 50
            simulation_params=params,
            hyperparams=hyperparams,
            seed=seed,
            verbose=0,
        )
        solutions_ga.append(seed_solution)
        evals_ga.append(seed_eval)

        greedy_solution_seed, greedy_evals_seed = simulate_greedy_evacuation(
            num_agents=num_agents, simulation_params=params, city=city, seed=seed, verbose=False
        )
        solutions_greedy.append(greedy_solution_seed)
        evals_greedy.append(greedy_evals_seed)

    compliance_str = str(compliance).replace(".", "")
    path = f"outputs/runs/{city_name}_{walking}walking_{delay}delay_{compliance_str}compliance.pkl"
    with open(path, "wb") as f:
        results = {
            "ga": {
                # results
                "solutions": solutions_ga,
                "evals": evals_ga,
                # metadata
                "num_agents": num_agents,
                "hyperparams": hyperparams,
                "params": params,
                "population_size": pop_size,
                "n_experiments": n_experiments,
                "population_creation_seed": algorithm_seeds,
            },
            "greedy": {
                # results
                "solutions": solutions_greedy,
                "evals": evals_greedy,
                # metadata
                "num_agents": num_agents,
                "hyperparams": hyperparams,
                "params": params,
                "population_size": 1,
                "n_experiments": n_experiments,
                "population_creation_seed": algorithm_seeds,
            },
        }
        pickle.dump(results, f)


if __name__ == "__main__":
    print("Starting propoer runs....")
    # phase 3 - agent tuning

    experiments = [
        # walking - delay - compliance - city name
        # Walking Speed (+ congestion)
        (True, False, 1, "grid_city"),
        (True, False, 1, "moderate_city"),
        (True, False, 1, "bottleneck_city"),
        # Delayed Start (+ congestion)
        (False, True, 1, "grid_city"),
        (False, True, 1, "moderate_city"),
        (False, True, 1, "bottleneck_city"),
        # Compliance (+ congestion) - grid city
        (False, False, 0, "grid_city"),
        (False, False, 0.25, "grid_city"),
        (False, False, 0.5, "grid_city"),
        (False, False, 0.75, "grid_city"),
        (False, False, 1, "grid_city"),
        # Compliance (+ congestion) - moderate city
        (False, False, 0, "moderate_city"),
        (False, False, 0.25, "moderate_city"),
        (False, False, 0.5, "moderate_city"),
        (False, False, 0.75, "moderate_city"),
        (False, False, 1, "moderate_city"),
        # Compliance (+ congestion) - bottleneck city
        (False, False, 0, "bottleneck_city"),
        (False, False, 0.25, "bottleneck_city"),
        (False, False, 0.5, "bottleneck_city"),
        (False, False, 0.75, "bottleneck_city"),
        (False, False, 1, "bottleneck_city"),
        # Combined - grid city
        (True, True, 0, "grid_city"),
        (True, True, 0.25, "grid_city"),
        (True, True, 0.5, "grid_city"),
        (True, True, 0.75, "grid_city"),
        (True, True, 1, "grid_city"),
        # Combined - moderate city
        (True, True, 0, "moderate_city"),
        (True, True, 0.25, "moderate_city"),
        (True, True, 0.5, "moderate_city"),
        (True, True, 0.75, "moderate_city"),
        (True, True, 1, "moderate_city"),
        # Combined - bottleneck city
        (True, True, 0, "bottleneck_city"),
        (True, True, 0.25, "bottleneck_city"),
        (True, True, 0.5, "bottleneck_city"),
        (True, True, 0.75, "bottleneck_city"),
        (True, True, 1, "bottleneck_city"),
    ]

    n_cores = 6
    with Pool(processes=n_cores) as pool:
        pool.map(basic_function, experiments)
    print("\nAll experiments complete for phase 3 size.")
