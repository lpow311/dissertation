from jax import random
import jax.numpy as jnp

from utils.agent import Chromosome


class CrossOver:

    def __init__(self, num_agents: int):
        self.crossover_prob = 0.8  # 0.6 and 0.9

        self.num_agents = num_agents

    def perform_crossover(self, parents: list, key: random.PRNGKey) -> tuple:
        parent_pairs, key = self.extract_parent_pairs(parents=parents, key=key)

        children = []

        for parent1, parent2 in parent_pairs:
            key, sub_key = random.split(key=key)

            prob_c = random.uniform(key=sub_key, shape=(), minval=0, maxval=1)

            if prob_c <= self.crossover_prob:
                key, sub_key = random.split(key=key)
                children += self.single_point_crossover(
                    parent1=parents[parent1], parent2=parents[parent2], key=sub_key
                )

        return children, key

    def single_point_crossover(self, parent1, parent2, key) -> list:
        # No point crossing over the first or last agent so +1 and -1 for start and end.
        key, sub_key = random.split(key=key)
        crossover_point = random.randint(
            key=sub_key, shape=(), minval=1, maxval=self.num_agents - 1
        )

        child1, child2 = {}, {}
        for agent_name in range(self.num_agents):
            if agent_name <= crossover_point:
                child1[agent_name] = parent1.agents[agent_name]
                child2[agent_name] = parent2.agents[agent_name]
            else:
                child1[agent_name] = parent2.agents[agent_name]
                child2[agent_name] = parent1.agents[agent_name]

        child1_key, child2_key = random.split(key=key)
        children = [
            # Parameters should be the same in both parents so shouldn't be an issue...
            Chromosome(agents=child1, seed=child1_key, params=parent1.params),
            Chromosome(agents=child2, seed=child2_key, params=parent1.params),
        ]

        return children

    def extract_parent_pairs(self, parents: list, key: random.PRNGKey) -> list:
        key, sub_key = random.split(key=key)
        shuffled = random.permutation(sub_key, jnp.array(range(len(parents))))
        pairs = list(zip(shuffled[0::2], shuffled[1::2]))
        return pairs, key
