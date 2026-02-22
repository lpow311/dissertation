import numpy as np
from utils.agent import Agent
from networkx import shortest_path_length


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

    def calculate_metrics(self, population: list, evolution: int, verbose: int = 1) -> None:
        avg, best, worst = self.fitness_metrics(population=population)
        self.scores["fitness"] = [avg, best, worst]

        if not ((verbose > 0) and (evolution % verbose == 0 or evolution == 1)):
            return None

        # Path Length Metrics
        avg_path_length = np.mean([c.calculate_average_path() for c in population])
        self.scores["avg_path_length"] = avg_path_length

        # Congestion index
        congestion_score = np.mean([c.congestion_score for c in population])
        self.scores["congestion"] = congestion_score

        # Exit utilisation balance
        exit_mean, exit_best, exit_worst = self.exit_utilisation_metric(population)
        self.scores["exit_utilisation"] = [exit_mean, exit_best, exit_worst]

        # Population diversity
        pop_diversity = self.diversity_metric(population=population)

        string_components = [
            f"Evolution {evolution:4d}",
            f"best fit: {best:.3f} | avg fit: {avg:.3f} | worst fit: {worst:.3f}",
            f"avg path: {avg_path_length:.3f}",
            f"congestion: {congestion_score:.3f}",
            f"exit use: {exit_mean:.3f}",
            f"diveristy: {pop_diversity:.3f}",
        ]

        print(" | ".join(string_components))

    def fitness_metrics(self, population: list) -> tuple:
        population_fitness = [c.fitness for c in population]

        avg = np.mean(population_fitness)
        best = np.min(population_fitness)
        worst = np.max(population_fitness)

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

    def diversity_metric(self, population: list) -> float:
        similarities = []

        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                chromosome1, chromosome2 = population[i], population[j]

                agent_similarities = []
                for agent_num in range(chromosome1.num_agents):
                    path1 = set(chromosome1.agents[agent_num].path)
                    path2 = set(chromosome2.agents[agent_num].path)

                    intersection = len(path1.intersection(path2))
                    union = len(path1.union(path2))

                    jaccard = intersection / union if union > 0 else 1
                    agent_similarities.append(jaccard)

                similarities.append(np.mean(agent_similarities))

        return 1 - np.mean(similarities) if similarities else 0


class FinalEvaluationMetrics:

    def __init__(self, solution: dict, params: dict) -> None:
        self.solution = solution.agents
        self.chromosome = solution

        self.params = params
        self.metrics = {}
        self.evolution_scores = None

    def score(self) -> dict[str, float]:
        agent_time, path_lengths, congestion_impact, congestion_delayed = (
            self.calculate_agent_times()
        )

        exit_utilisation = self.exit_utilisation_metric()
        path_efficiency = self.path_efficiency_metric()

        string_components = [
            f"Avg path length: {np.mean(path_lengths):.2f}",
            f"Total time: {np.max(agent_time):.2f}",
            f"Average time: {np.mean(agent_time):.2f}",
            f"Congestion index: {np.mean(congestion_delayed)*100:.0f}%",
            f"Avg Congestion delay: {np.mean(congestion_impact):.2f}",
            f"Exit utilisation: {exit_utilisation:.2f}",
            f"Avg Path Efficiency: {path_efficiency:.2f}",
        ]
        self.print_final_results(results=string_components)

        self.metrics = {
            "Avg path length": np.mean(path_lengths),
            "Total time": np.max(agent_time),
            "Average time": np.mean(agent_time),
            "Congestion index": np.mean(congestion_delayed),
            "Avg Congestion delay": np.mean(congestion_impact),
            "Exit utilisation": exit_utilisation,
            "Avg Path Efficiency": path_efficiency,
        }
        return self.metrics

    def print_final_results(self, results: list) -> None:
        result_string = " | ".join(results)
        header = (
            "\n"
            + "=" * int(len(result_string) / 2)
            + " FINAL RESULTS "
            + "=" * int(len(result_string) / 2)
        )
        footer = "=" * int(len(header))

        for string in [header, result_string, footer]:
            print(string)

    def calculate_agent_times(self) -> tuple:
        path_lengths = self.chromosome.path_lengths
        agent_time = self.chromosome.path_times
        congestion_impact = self.chromosome.congestion_score
        congestion_delayed = [i > 0 for i in congestion_impact]

        return agent_time, path_lengths, congestion_impact, congestion_delayed

    def exit_utilisation_metric(self):
        chromosome_exits = {}

        for agent in self.solution.values():
            exit_location = agent.path[-1]
            if exit_location not in chromosome_exits:
                chromosome_exits[exit_location] = 1
            else:
                chromosome_exits[exit_location] += 1

        utilisation_score = Evaluation.gini_coefficient(counts=list(chromosome_exits.values()))

        return utilisation_score

    def path_efficiency_metric(self):
        path_efficiency = []

        for agent in self.solution.values():
            # Shortest path to the exit the agent actually took
            shortest = shortest_path_length(
                agent.city.graph, source=agent.start_point, target=agent.path[-1]
            )
            path_efficiency.append(len(agent.path) / shortest)

        return np.mean(path_efficiency)

    def add_evolution_scores(self, scores: list) -> None:
        self.evolution_scores = scores
