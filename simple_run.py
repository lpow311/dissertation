from utils.environment import Environment
from utils.notebook import ga_tuning_function

import pickle

for city_name in ["grid_city", "bottleneck_city", "moderate_city"]:
    city = Environment(city_name=city_name)
    n_experiments = 20
    num_agents = 10
    params = {"congestion": True, "walking": False, "fitness": "max"}
    population_size = 20

    hyperparams = {
        "crossover": 0.8,
        "mutation": 0.05,
        "epsilon": 0.2,
        "max_evolutions": 50,
        "survivor_method": "elite_percentage",
        "tournament_size": 3,
    }

    solutions, evolution_stats, timings = ga_tuning_function(
        city, n_experiments, num_agents, params, population_size, hyperparams
    )

    path = f"outputs/{city_name}_{n_experiments}_experiments_analysis_tournament.pkl"
    with open(path, "wb") as f:
        results = {
            # results
            "solutions": solutions,
            "evolution_stats": evolution_stats,
            "timings": timings,
            # metadata
            "num_agents": num_agents,
            "hyperparams": hyperparams,
            "population_size": population_size,
            "n_experiments": n_experiments,
        }
        pickle.dump(results, f)
