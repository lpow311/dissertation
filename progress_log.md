# Progress Log
This is so I can keep track of what I've done, what still needs doing and my most recent thoughts / maybe also a backlog of thoughts.

### so what's been done...
In the `multi_agent_ga.py` file and the `utils` folder I have done the following:
* Created the population - controlled by **population_size** and **num_agents** as input parameters.
* Set up genetic algorithm loop and exit criteria - exit criteria is currently **max_evolutions**.
* Tracking the average score of the evolutions.
* Selection - combination of elite and roulette selection. 
* Crossover - children are created by selecting a half way point across agent numbers and then swapping between the parents (single point crossover).
* Mutation - rerouting an agent between 2 randomly selected points on the path.
* Survivor Selection - got two options keeping only the best across old and new population or keeping the children and filling any gap with the old elite population.

Other additonal functionality includes:
* `city_creation.py` - creates different city environments and saves them so they can be read in.
* Plotting the results I've done but not 100% sure what I'm even trying to show yet.


### my recent thoughts are...
* Need to add in some stuff to make it more "real" e.g. congestion  and route familiarity now that the basic version is working.
* Need to make sure this is configurable so not overwritting basic version.


### pretty sure a long list of what needs doing....
##### General functionality:
* Exit criteria
* Walking speed - the fastest person needs to be they can move to a node in 1 time step and then increase.
* Agent characteristics e.g. walking speed
* Congestion
* What metric am I tracking - or multiple?
    * Side note need to get all the other metrics implemented at some point.

##### Selection 
* Currently allow for duplicates in parent selection might need to understand and justify this.

##### Crossover
* Crossover probability?


##### Mutation
* Mutation probability?


##### Survivor Selection 
* Why would I be losing children and having to fill with parents? Is it because of the cross over probability?



#### Human behaviours to add in:
* City familiarity
* Panic factor
* altruism - which likely means I'll need to add some form of characteristic around requiring support so people see it to stop?