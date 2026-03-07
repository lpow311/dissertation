import pickle


def read_pickle(name: str) -> tuple:
    with open(f"graphs/{name}.pkl", "rb") as f:
        city = pickle.load(f)

    G = city["graph"]
    exits = city["exits"]
    starts = city["starts"]
    metrics = city["metrics"]
    return G, exits, starts, metrics
