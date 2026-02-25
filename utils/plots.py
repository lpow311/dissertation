import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import networkx as nx


class PlotRoomUsage:

    def __init__(self, G: nx.Graph, ga_agents: dict, greedy_agents: dict) -> None:
        self.G = G
        self.ga_agents = ga_agents
        self.greedy_agents = greedy_agents

    def plot(self) -> None:
        ga_room_usage = self.calculate_room_usage(self.ga_agents)
        greedy_room_usage = self.calculate_room_usage(self.greedy_agents)

        # Room usage bar chart
        self.plot_room_usage(
            ga_room_usage,
            greedy_room_usage,
        )

    def calculate_room_usage(self, agents: dict) -> dict:
        """
        For each agent, record which rooms they passed through.
        Returns the proportion of agents who passed through each room.
        """
        room_counts = defaultdict(int)

        for agent in agents.values():
            rooms_visited = set()
            for node in agent.path:
                room = self.G.nodes[node].get("room", None)
                if room:
                    rooms_visited.add(room)
            for room in rooms_visited:
                room_counts[room] += 1

        num_agents = len(agents)
        return {room: count / num_agents for room, count in room_counts.items()}

    def plot_room_usage(
        self,
        ga_usage: dict,
        greedy_usage: dict,
    ):
        """
        Bar chart comparing proportion of agents passing through each room
        under GA vs greedy routing.

        High transit room usage under GA = agents taking the longer open route.
        Low transit room usage under greedy = agents taking direct bottlenecked route.
        """
        # Combine all rooms from both results
        all_rooms = sorted(set(list(ga_usage.keys()) + list(greedy_usage.keys())))

        ga_values = [ga_usage.get(r, 0) * 100 for r in all_rooms]
        greedy_values = [greedy_usage.get(r, 0) * 100 for r in all_rooms]

        x = np.arange(len(all_rooms))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))

        ga_bars = ax.bar(x - width / 2, ga_values, width, label="GA", color="steelblue")
        greedy_bars = ax.bar(x + width / 2, greedy_values, width, label="Greedy", color="coral")

        # Value labels on bars
        for bar in ga_bars:
            if bar.get_height() > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{bar.get_height():.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        for bar in greedy_bars:
            if bar.get_height() > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{bar.get_height():.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax.set_xlabel("Room")
        ax.set_ylabel("% of Agents Passing Through")
        ax.set_title("Room Usage — GA vs Greedy")
        ax.set_xticks(x)
        ax.set_xticklabels(all_rooms, rotation=15, ha="right")
        ax.set_ylim(0, 110)
        ax.legend()
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

        plt.tight_layout()

        plt.show()
