## City Tuning

### 1 City Set Up
* 2-1-2 room structure for starts, transition and exit rooms might realistically mirror realistic evacuation scenarios with multiple entry, coridoor and exit points.
* Selected 3 exit and start points to avoid nodes becoming bottlenecks themselves.

| City Name | Bottleneck Corridoors | Open Corridoors | Expected PoA |
| --------- | --------------------- | --------------- | ------------ |
| Connected | 2 | 3| Low |
| Moderate | 3 | 3| Medium |
| Severe | 1 | 3| High |
| Extreme | 1 | 1| Very High |

### 2 Tune the Genetic Algorithm Parameters
* Population size (you use 50)
* Max evolutions (you use 50)
* Mutation probability
* Selection method
* Crossover method

### 3 Tune the number of agents
* Does the GA advantage remain stable or change significantly?
* Does runtime scale acceptably?
* Is there a floor effect at low agent counts i.e. no congestion problems.