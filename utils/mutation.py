from utils.agent import Agent, Chromosome
from jax import random


class Mutation:

    def __init__(self):
        self.mutation_prob = 0.1

        self.min_path_length = 3

    def partial_path_mutation(self, children: Chromosome) -> list:
        mutated_children = []
        for child_chromosome in children:
            for agent_name, agent in child_chromosome.agents.items():

                agent.key, mutation_sub_key = random.split(agent.key)
                agent_mutation_prob = random.uniform(
                    key=mutation_sub_key, shape=(), minval=0, maxval=1
                )

                if agent_mutation_prob <= self.mutation_prob:

                    point1, point2, key = self.extract_path_points(agent=agent)

                    agent_path = agent.path
                    sub_path, agent.key = self.find_new_subpath(
                        agent_path[point1], agent_path[point2], key, agent.city
                    )
                    new_path = agent_path[:point1] + sub_path[:-1] + agent_path[point2:]
                    child_chromosome.agents[agent_name].path = new_path

            mutated_children.append(child_chromosome)

        return mutated_children

    def extract_path_points(self, agent) -> tuple:
        # TODO: this needs a think again just needs to work...
        agent_path = agent.path

        key_point1, sub_key_point1 = random.split(agent.key)
        point1 = random.randint(key=sub_key_point1, shape=(), minval=1, maxval=len(agent_path) - 2)

        key_point2, sub_key_point2 = random.split(key_point1)
        max_point = min(len(agent_path) - 1, point1 + self.min_path_length)
        point2 = random.randint(
            key=sub_key_point2, shape=(), minval=max_point, maxval=len(agent_path)
        )

        return point1, point2, key_point2

    def find_new_subpath(self, point1, point2, key, city) -> tuple:
        current_location = point1
        new_path = [current_location]

        city = city.graph

        # TODO: maybe also add in option if reach an exit this is also fine.
        while current_location != point2:
            key, sub_key = random.split(key)
            neighbours = list(city.neighbors(current_location))

            next_location_idx = random.randint(
                key=sub_key, shape=(), minval=0, maxval=len(neighbours)
            )
            current_location = neighbours[next_location_idx]
            new_path.append(current_location)

        return new_path, key
