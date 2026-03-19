## Another place to keep track of next steps..

## 1 Algorith Tuning
* Need to add tournament selection in and compare the results.
* Compare outputs of grid search - hope the fitness function starts converging.
* Look at the performance of the percentage elite.

## 2 Agent Tuning
* Run the different cities with differing number of agents to see what the best number should be.

## 3 Given above metrics compare algorithms
* Run the algorithms and see how they differ

## 4 Add agent compliance
* Add functionality to compare the performance with different levels of compliance.




# Research Summary
## Main Research Question
"To what extent does globally optimised multi-agent routing via Genetic Algorithms produce better evacuation outcomes than individually optimised greedy routing under varying human behavioural characteristics and dynamic environmental conditions?"
## Sub Questions & Approach
##### SQ1 — Baseline Performance Does GA outperform greedy routing across structurally different city networks?
Why: Establishes the fundamental comparison before adding complexity. Validates that your three cities produce meaningfully different results.
Approach: Run GA and greedy across Grid, Moderate and Bottleneck cities across N seeds. Compare congestion scores. ✅ Already done.

##### SQ2 — Dynamic Fire Environment How does a dynamically spreading fire affect the performance gap between GA and greedy?
Why: Makes the simulation more realistic. Fire punishes agents who are congested at the wrong place at the wrong time — directly amplifying greedy's weakness at bottlenecks.

**Approach:**
* Add timestep simulation (agent positions tracked at each step)
* Add node capacity limit N with random queue ordering
* Add agent speeds (variable movement rates)
* Fire starts at one node, spreads to all adjacent nodes every K timesteps
* Agents cannot see fire coming
* Agents on fire node receive large penalty
* Test across multiple fire spread rates (slow, medium, fast)

##### SQ3 — Compliance / Altruism At what compliance rate does collective GA routing begin to outperform pure greedy?
Why: Bridges the gap between purely individual and purely collective routing. Models realistic human behaviour — not everyone will follow an evacuation plan.
Approach:
At initialisation each agent rolls a probability p
If p < compliance rate → agent takes GA assigned path
If p ≥ compliance rate → agent takes greedy path
Test compliance rates [0.0, 0.25, 0.5, 0.75, 1.0]
Run across all three cities
Identify threshold compliance rate where GA begins to outperform greedy

##### SQ4 — Visibility How does limited visibility of fire affect evacuation outcomes under GA vs greedy routing?
Why: Agents following a pre-planned GA route may be routed into unseen fire — visibility limits could make compliance harmful rather than helpful. Creates a nuanced finding around when GA's global plan is actually beneficial.
Approach:
Visibility radius limits which nodes an agent can consider when replanning
Safe nodes = visible nodes not on fire
Test visibility radii [full, medium, minimal]
Combine with compliance rates to create a compliance × visibility experiment


Tomorrow's To Do List
Priority 1 — Timestep Simulation (foundation for SQ2, SQ3, SQ4)

 Add timestep_positions tracking to agent — where is each agent at each timestep
 Define agent speeds — how many timesteps to traverse one edge
 Add node capacity limit N
 Add random queue ordering when node is at capacity
 Build core simulation loop:

    for each timestep:
        spread fire
        for each agent:
            check fire penalty
            check capacity → queue if needed
            move if able
        record positions
Priority 2 — Fire Model

 Define fire start node per city
 Define spread rate K (timesteps between spread events)
 Add fire penalty to scoring
 Test across spread rates [slow, medium, fast]

Priority 3 — Update Scoring

 Move congestion scoring from static (post-hoc) to timestep-level (in-the-moment)
 Add fire penalty to fitness function
 Validate new scoring against existing results

Priority 4 — Compliance

 Add compliance rate parameter to agent initialisation
 At initialisation assign each agent GA or greedy path based on compliance roll
 Test compliance rates [0.0, 0.25, 0.5, 0.75, 1.0]

Priority 5 — Visibility (if time allows)

 Add visibility radius parameter to agent
 Filter available nodes to visibility radius at each step
 Remove fire nodes from visible options
 Test visibility radii [full, medium, minimal]