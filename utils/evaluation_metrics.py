import numpy as np
from collections import defaultdict
from utils.agent import Agent


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
        print(" | ".join(string_components))

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


class FinalEvaluationMetrics:

    def __init__(self, solution: dict, params: dict, chromosome: bool = True) -> None:
        if chromosome:
            self.solution = solution.agents
        else:
            self.solution = solution
        self.params = params

    def score(self) -> None:
        agent_time, congestion_delays, path_lengths = self.calculate_agent_times()

        exit_utilisation = self.exit_utilisation_metric()

        string_components = [
            f"Average path length: {np.mean(path_lengths):.3f}",
            f"Total path length: {np.max(path_lengths):.3f}",
            f"Average duration: {np.mean(agent_time):.3f}",
            f"Congestion delays: {np.mean(congestion_delays):.3f}",
            f"Exit utilisation: {exit_utilisation:.3f}",
        ]
        print(" | ".join(string_components))

    def calculate_agent_times(self) -> tuple:
        node_occupancy = self.calculate_node_congestion()
        agent_time = []
        congestion_delays = []
        path_lengths = []

        for agent in self.solution.values():
            path = agent.path
            congestion_delay = self.calculate_agent_congestion_delay(agent, node_occupancy)
            congestion_delays.append(congestion_delay)

            path_lengths.append(len(path))
            path_timesteps = len(path) * agent.speed if self.params["walking"] else len(path)
            path_fitness = path_timesteps
            if self.params["congestion"]:
                path_fitness += congestion_delay

            agent_time.append(path_fitness)

        return agent_time, congestion_delays, path_lengths

    @staticmethod
    def calculate_agent_congestion_delay(agent: Agent, occupancy: defaultdict) -> int:
        delay = 0
        for t, node in enumerate(agent.path):
            capacity = agent.city.congestion_amount
            if occupancy[t][node] > capacity:
                delay += occupancy[t][node] - capacity

        return delay

    def calculate_node_congestion(self) -> defaultdict:
        """
        Weakly time-dependent (non-causal) congestion - It is not fully dynamic, and delays
        do not propagate forward.
        i.e. So if an agent is delayed at time t, the model still assumes it arrives at t+1 next.
        """
        occupancy = defaultdict(lambda: defaultdict(int))

        for agent in self.solution.values():
            for t, node in enumerate(agent.path):
                occupancy[t][node] += 1

        return occupancy

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
