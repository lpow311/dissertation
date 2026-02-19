import numpy as np


class Evaluation:

    def __init__(self) -> None:
        self.avg_score = []
        self.scores = {
            "fitness": [],
            "avg_path_length": 0,
            "congestion": 0,
            "exit_utilisation": [],
            "altruism": 0,
        }

    def calculate_metrics(self, population: list, evolution: int) -> None:
        avg, best, worst = self.fitness_metrics(population=population)
        self.scores["fitness"] = [avg, best, worst]

        # Path Length Metrics
        avg_path_length = np.mean([c.calculate_average_path() for c in population])
        self.scores["avg_path_length"] = avg_path_length

        # Congestion index
        congestion_score = np.mean([c.congestion_score for c in population])
        self.scores["congestion"] = congestion_score

        # Exit utilisation balance
        exit_mean, exit_best, exit_worst = self.exit_utilisation_metric(population)
        self.scores["exit_utilisation"] = [exit_mean, exit_best, exit_worst]

        # TODO: altruism impact i.e. full time steps.

        string_components = [
            f"Evolution {evolution:4d}",
            f"best fit: {best:.3f} | avg fit: {avg:.3f} | worst fit: {worst:.3f}",
            f"avg path: {avg_path_length:.3f}",
            f"congestion: {congestion_score:.3f}",
            f"exit utilisation: {exit_mean:.3f}",
        ]
        full_string = " | ".join(string_components)
        print(full_string)

    def fitness_metrics(self, population: list) -> tuple:
        population_fitness = [c.fitness for c in population]

        avg = np.mean(population_fitness)
        best = np.max(population_fitness)
        worst = np.min(population_fitness)

        self.avg_score.append(avg)

        return avg, best, worst

    def exit_utilisation_metric(self, population: list):
        exit_utilisation = []

        for chromosome in population:
            chromosome_exits = {}
            for agent in chromosome.agents.values():
                exit_location = agent.path[-1]
                if exit_location not in chromosome_exits:
                    chromosome_exits[exit_location] = 1
                else:
                    chromosome_exits[exit_location] += 1

            utilisation_score = self.gini_coefficient(counts=list(chromosome_exits.values()))
            exit_utilisation.append(utilisation_score)

        best = min(exit_utilisation)
        worst = max(exit_utilisation)
        avg = np.mean(exit_utilisation)

        return avg, best, worst

    @staticmethod
    def gini_coefficient(counts):
        """
        counts: iterable of exit counts (list or dict values)
        returns: Gini coefficient between 0 and 1
        """
        x = np.array(counts, dtype=float)
        if np.sum(x) == 0:
            return 0.0

        x = np.sort(x)
        n = len(x)
        cumulative = np.cumsum(x)
        gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n

        return gini
