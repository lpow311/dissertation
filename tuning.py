from multiprocessing import Pool
import numpy as np

from utils.environment import Environment
from main import simulate_ga_evacuation

from time import perf_counter
import pickle


def basic_function(extra_params):
    n_experiments = 10
    num_agents = 100  # 100
    original_params = {
        "congestion": True,
        "walking": False,
        "delay_start": False,
        "compliance": 1,
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

    # params = {**original_params, **extra_params}
    params = original_params

    hyperparams = {**original_hyperparams, **extra_params}
    # hyperparams = original_hyperparams
    pop_size = 50

    start = perf_counter()

    solutions, evals = [], []
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
        solutions.append(seed_solution)
        evals.append(seed_eval)

    end = perf_counter()

    parent_method = "roulette" if hyperparams["tournament_size"] is None else "tournament"
    tournament_size = hyperparams["tournament_size"]
    survivor_method = hyperparams["survivor_method"]

    path = f"outputs/tuning/{city.city_name}_{n_experiments}_test_tournament3.pkl"
    with open(path, "wb") as f:
        results = {
            # results
            "solutions": solutions,
            "evals": evals,
            "timings": end - start,
            # metadata
            "num_agents": num_agents,
            "hyperparams": hyperparams,
            "params": params,
            "population_size": pop_size,
            "n_experiments": n_experiments,
            "population_creation_seed": algorithm_seeds,
        }
        pickle.dump(results, f)


if __name__ == "__main__":
    print("Starting phase 2 experiements...")
    # phase 2 - method tuning
    phase2 = [
        {"survivor_method": "elite_percentage", "tournament_size": None},
        {"survivor_method": "elite_percentage", "tournament_size": 3},
        {"survivor_method": "elite_percentage", "tournament_size": 5},
        {"survivor_method": "elite", "tournament_size": None},
        {"survivor_method": "elite", "tournament_size": 3},
        {"survivor_method": "elite", "tournament_size": 5},
        {"survivor_method": "children", "tournament_size": None},
        {"survivor_method": "children", "tournament_size": 3},
        {"survivor_method": "children", "tournament_size": 5},
    ]

    n_cores = 6
    with Pool(processes=n_cores) as pool:
        pool.map(basic_function, phase2)
    print("\nAll experiments complete for phase 2 size.")
