from scipy import stats
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from collections import defaultdict


plt.style.use("seaborn-v0_8")

from utils.agent import Chromosome


class AlgorithmComparison:

    def __init__(self, greedy_outputs: list[Chromosome], ga_outputs: list[Chromosome]) -> None:
        self.greedy = greedy_outputs
        self.ga = ga_outputs

    def statistical_tests(self, verbose: bool = False) -> dict[str, dict]:
        metrics = [
            "Total time",
            "Average time",
            "Avg Congestion delay",
            "Exit utilisation",
            "Avg Path Efficiency",
        ]

        results = {}

        for metric in metrics:
            greedy_results = [experiement.metrics[metric] for experiement in self.greedy]
            ga_results = [experiement.metrics[metric] for experiement in self.ga]

            _, p_value = stats.wilcoxon(ga_results, greedy_results)
            results[metric] = {"p_value": p_value, "significant": p_value < 0.05}

        if verbose:
            for name, values in results.items():
                print(
                    f"{name} p-value: {values['p_value']:.3f} (Significant: {values['significant']})"
                )

        return results

    def ga_convergence_plots(self) -> None:
        for experiment in self.ga:
            evolution_scores = experiment.evolution_scores
            plt.plot(range(len(evolution_scores)), evolution_scores)

        plt.title("Genetic Algorithm Experiment Best Fitness Evolution")
        plt.xlabel("Evolution")
        plt.ylabel("Best Fitness Score")
        plt.show()

    def exit_utilisation(self, city_name: str) -> None:
        greedy_x, ga_x = {}, {}

        for i in range(len(self.greedy)):
            greedy_exits = [agent.path[-1] for agent in self.greedy[i].solution.values()]
            greedy_counter = Counter(greedy_exits)
            greedy_x = self.add_exits_to_dictionary(counter=greedy_counter, exit_dict=greedy_x)

            ga_exits = [agent.path[-1] for agent in self.ga[i].solution.values()]
            ga_counter = Counter(ga_exits)
            ga_x = self.add_exits_to_dictionary(counter=ga_counter, exit_dict=ga_x)

        greedy_exit_use = {exit: float(np.mean(val)) for exit, val in greedy_x.items()}
        ga_exit_use = {exit: float(np.mean(val)) for exit, val in ga_x.items()}

        self.plot_exit_utilisation(ga_exit_use, greedy_exit_use, city_name)

    @staticmethod
    def add_exits_to_dictionary(counter: Counter, exit_dict: dict) -> dict:
        for exit, count in counter.items():
            if exit not in exit_dict:
                exit_dict[exit] = [count]
            else:
                exit_dict[exit].append(count)

        return exit_dict

    def plot_exit_utilisation(self, ga_exits: dict, greedy_exits: dict, city_name: str) -> None:
        exits = list(set(list(ga_exits.keys()) + list(greedy_exits.keys())))

        ga_values, greedy_values = [], []
        for exit in exits:
            ga_values.append(ga_exits[exit] if exit in ga_exits else 0)
            greedy_values.append(greedy_exits[exit] if exit in greedy_exits else 0)

        x = np.arange(len(exits))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))

        ga_bars = ax.bar(x - width / 2, ga_values, width, label="GA", color="steelblue")
        greedy_bars = ax.bar(x + width / 2, greedy_values, width, label="Greedy", color="coral")

        ax.set_xlabel("Exit")
        ax.set_ylabel("Average Number of Agents")
        ax.set_title(f"Exit Utilisation Distribution — {city_name.replace('_', ' ').capitalize()}")
        ax.set_xticks(x)
        ax.set_xticklabels(exits)
        ax.legend()

        # Add value labels on top of bars
        for bar in ga_bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{bar.get_height():.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        for bar in greedy_bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{bar.get_height():.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()
        plt.show()

    def price_of_anarchy(self, verbose: bool = True) -> float:
        poa_values = []

        for ga_experiment, greedy_experiment in zip(self.ga, self.greedy):
            greedy_time = greedy_experiment.metrics["Total time"]
            ga_time = ga_experiment.metrics["Total time"]

            poa = greedy_time / ga_time
            poa_values.append(poa)

        mean_poa = np.mean(poa_values)
        std_poa = np.std(poa_values)

        if verbose:
            print(f"Price of Anarchy: {mean_poa:.3f} ± {std_poa:.3f}")

        return mean_poa, std_poa

    def plot_exit_time_by_start_location(self) -> None:
        avg_ga_starts, avg_greedy_starts = self.calculate_start_location_exit_times()
        self.plot_avg_time_by_start(ga_times=avg_ga_starts, greedy_times=avg_greedy_starts)

    def calculate_start_location_exit_times(self) -> tuple:
        start_time_greedy, start_time_ga = defaultdict(list), defaultdict(list)
        for ga_exp, greedy_exp in zip(self.ga, self.greedy):
            greedy_times = greedy_exp.chromosome.path_times
            ga_times = ga_exp.chromosome.path_times

            starts = [agent.start_point for agent in ga_exp.chromosome.agents.values()]
            greedy_avg_times = self.calculate_average_times(starts=starts, times=greedy_times)
            ga_avg_times = self.calculate_average_times(starts=starts, times=ga_times)

            for start in starts:
                start_time_greedy[start].append(greedy_avg_times[start])
                start_time_ga[start].append(ga_avg_times[start])

        overall_avg_greedy = {start: np.mean(times) for start, times in start_time_greedy.items()}
        overall_avg_ga = {start: np.mean(times) for start, times in start_time_ga.items()}

        return overall_avg_ga, overall_avg_greedy

    def calculate_average_times(self, starts: list, times: list) -> dict:
        location_times = defaultdict(list)
        for location, time in zip(starts, times):
            location_times[location].append(time)

        avg_times = {location: np.mean(times) for location, times in location_times.items()}
        return avg_times

    def plot_avg_time_by_start(self, ga_times: dict, greedy_times: dict):
        locations = list(ga_times.keys())
        ga_values = [ga_times[loc] for loc in locations]
        greedy_values = [greedy_times[loc] for loc in locations]

        x = np.arange(len(locations))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))

        ga_bars = ax.bar(x - width / 2, ga_values, width, label="GA", color="steelblue")
        greedy_bars = ax.bar(x + width / 2, greedy_values, width, label="Greedy", color="coral")

        ax.set_xlabel("Start Location")
        ax.set_ylabel("Average Evacuation Time")
        ax.set_title("Average Evacuation Time by Start Location — GA vs Greedy")
        ax.set_xticks(x)
        ax.set_xticklabels(locations)
        ax.legend()

        for bar in ga_bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{bar.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        for bar in greedy_bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{bar.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()
        plt.savefig("avg_time_by_start.png", dpi=150)
        plt.show()
