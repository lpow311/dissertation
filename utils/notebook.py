import numpy as np
from tqdm import tqdm
from time import perf_counter

from multi_agent_ga import KeyManager
from utils.agent import PopulationCreation
from utils.genetic_algorithm import GeneticAlgorithm


def ga_tuning_function(city, n_experiments, num_agents, params, population_size, hyperparams: dict):
    """
    I would like to say this is absolutely horrific code but I need to get all this info
    out so this feels the easiest to run it once although messy...
    """
    rng = np.random.default_rng(123)
    algorithm_seeds = rng.integers(0, 10**6, size=n_experiments)

    # Run it.
    all_solutions, all_stats, all_times = [], [], []

    for seed_idx in tqdm(range(n_experiments)):
        start_time = perf_counter()
        # This ensures the agents all start from the same place so can compare how the random route selection varies (hopefully)
        key_manager = KeyManager(seed=1234)
        creator = PopulationCreation(
            city=city, pop_size=population_size, num_agents=num_agents, key_manager=key_manager
        )
        population, human_traits = creator.create_initial_population(simulation_params=params)
        creation_time = perf_counter()

        seed = algorithm_seeds[seed_idx]

        ga_key_manager = KeyManager(seed=seed)
        ga = GeneticAlgorithm(
            exit_criteria={"max_evolutions": hyperparams["max_evolutions"]},
            pop_size=population_size,
            key_manager=ga_key_manager,
            num_agents=num_agents,
            hyperparams=hyperparams,
        )

        evolution = 1
        terminate = ga.extract_termination_criteria(evolution=evolution)

        all_generation_stats = {
            evolution: {} for evolution in range(1, hyperparams["max_evolutions"] + 1)
        }

        loop_start = perf_counter()
        steps = {step: [] for step in ["parent", "crossover", "mutation", "survivor"]}
        while not terminate:
            evolution_start = perf_counter()

            # PARENT SELECTION
            parents = ga.parent_selection(population=population)
            parent_time = perf_counter()

            # CROSSOVER
            children = ga.chromosome_crossover(parents=parents)
            crossover_time = perf_counter()

            # MUTATION
            mutated_children = ga.agent_mutation(children=children)
            mutation_time = perf_counter()

            # SURVIVOR SELECTION
            old_population = [p for p in population]

            survivor_start = perf_counter()
            population = ga.survivor_selection(
                children=mutated_children,
                old_population=population,
                method=hyperparams["survivor_method"],
            )
            survivor_time = perf_counter()
            survivor_analysis = ga.analyse_survivor_selection(
                old_population, mutated_children, population
            )

            steps["parent"].append(parent_time - evolution_start)
            steps["crossover"].append(crossover_time - parent_time)
            steps["mutation"].append(mutation_time - crossover_time)
            steps["survivor"].append(survivor_time - survivor_start)

            fitnesses = [c.fitness for c in population]
            inverted = [1 / f for f in fitnesses]

            all_generation_stats[evolution] = {
                "parent_selection": {
                    "cv": (np.std(fitnesses) / np.mean(fitnesses)) * 100,
                    "dominance": max(fitnesses) / (min(fitnesses) + 1e-9),
                    "best_share": (max(inverted) / sum(inverted)) * 100,
                },
                "survivor_selection": survivor_analysis,
                "fitnesses": fitnesses,
            }

            # 7 Termination
            evolution += 1
            terminate = ga.extract_termination_criteria(evolution=evolution)

        loop_end = perf_counter()
        final_solution = sorted(population, key=lambda c: c.fitness)[0]
        end_time = perf_counter()

        all_solutions.append(final_solution)
        all_stats.append(all_generation_stats)

        experiment_times = {
            "total": end_time - start_time,
            "population_creation": creation_time - start_time,
            "ga_components": steps,
            "evolution": loop_end - loop_start,
        }

        all_times.append(experiment_times)

    return all_solutions, all_stats, all_times
