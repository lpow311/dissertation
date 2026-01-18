import numpy as np


class Evaluation:

    def __init__(self) -> None:
        self.avg_score = []

    def calculate_metrics(self, population: list, evolution: int) -> None:
        avg, best, worst = self.fitness_metrics(population=population)

        # Path Length Metrics
        avg_path_length = np.mean([c.calculate_average_path() for c in population])

        # TODO: congestion index
        congestion_score = np.mean([c.congestion_score for c in population])

        # TODO: exit utilisation balance

        # TODO: altruism impact i.e. full time steps.

        print(
            f"Evolution {evolution:4d} | best: {best:.3f} | avg: {avg:.3f} | worst: {worst:.3f} | avg path length: {avg_path_length:.3f} | congestion: {congestion_score:.3f}"
        )

    def fitness_metrics(self, population: list) -> tuple:
        population_fitness = [c.fitness for c in population]

        avg = np.mean(population_fitness)
        best = np.max(population_fitness)
        worst = np.min(population_fitness)

        self.avg_score.append(avg)

        return avg, best, worst
