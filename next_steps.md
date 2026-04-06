# Analysis & Results Plan

## For each experiment, the comparison is always:
1. GA with condition vs Greedy with condition — does GA still outperform?
2. GA with condition vs GA baseline — how much does the condition hurt GA?
3. Greedy with condition vs Greedy baseline — how much does the condition hurt Greedy?

---

## Objective 1 — Baseline GA vs Greedy across cities

### For each city (Grid, Moderate, Bottleneck):
- [ ] Table: total time, avg time, congestion index, avg congestion delay, exit utilisation — GA vs Greedy at each agent count [10, 20, 50, 100, 200]
- [ ] Plot: greedy scaling plot (total time, avg time, congestion delay vs agent count) — already done
- [ ] Plot: cumulative exits over time — GA vs Greedy at 100 agents
- [ ] Plot: start node dispersal (x=3) — GA vs Greedy at 100 agents
- [ ] Plot: agent path density map — GA vs Greedy at 100 agents
- [ ] Statistical test: Mann-Whitney U test on total time GA vs Greedy across 10 seeds

### Cross-city summary:
- [ ] Table: GA vs Greedy at 100 agents across all three cities side by side
- [ ] Written conclusion: does city structure affect the GA advantage? Which city benefits most?

---

## Objective 2 — Agent Dispersal

- [ ] Plot: start node dispersal at x=1, x=2, x=3 — GA vs Greedy for each city at 100 agents
- [ ] Written conclusion: do GA agents clear the danger zone faster? Is this consistent across cities?

---

## Objective 3a — Walking Speed

### For each city at 100 agents:
- [ ] Table: baseline vs walking speed condition — GA and Greedy side by side
    - Columns: total time, avg time, congestion index, avg congestion delay, exit utilisation
    - Rows: GA baseline | GA + walking | Greedy baseline | Greedy + walking
- [ ] Plot: cumulative exits — GA vs Greedy with and without walking speed
- [ ] Plot: start node dispersal — GA vs Greedy with and without walking speed
- [ ] Statistical test: is the difference between walking and baseline significant for each algorithm?

### Cross-city summary:
- [ ] Table: delta from baseline for GA and Greedy across all three cities
    - i.e. how much does walking speed hurt each algorithm?
- [ ] Written conclusion: does walking speed hurt Greedy more than GA? Does GA route slow agents differently?

---

## Objective 3b — Delayed Starts

### For each city at 100 agents:
- [ ] Table: baseline vs delay condition — GA and Greedy side by side
    - Columns: total time, avg time, congestion index, avg congestion delay, exit utilisation
    - Rows: GA baseline | GA + delay | Greedy baseline | Greedy + delay
- [ ] Plot: cumulative exits — GA vs Greedy with and without delay
- [ ] Plot: start node dispersal — GA vs Greedy with and without delay
- [ ] Statistical test: is the difference between delay and baseline significant for each algorithm?

### Cross-city summary:
- [ ] Table: delta from baseline for GA and Greedy across all three cities
- [ ] Written conclusion: do delayed starts amplify greedy's congestion problem? Does GA handle waves of agents better?

---

## Objective 3c — Compliance / Anxiety

### For each city at 100 agents:
- [ ] Plot: compliance threshold curve — total time vs compliance rate [0.0, 0.25, 0.5, 0.75, 1.0] for GA and Greedy on same plot
- [ ] Plot: congestion index vs compliance rate
- [ ] Plot: exit utilisation vs compliance rate
- [ ] Table: all metrics at each compliance rate for each city
- [ ] Identify: at what compliance rate does GA begin to outperform Greedy?

### Cross-city summary:
- [ ] Table: compliance threshold per city
- [ ] Written conclusion: is there a consistent compliance threshold? Does city structure affect when GA advantage emerges?

---

## Objective 3d — Combined Condition (Walking + Delay + Compliance)

### For each city at 100 agents:
- [ ] Table: baseline | combined 0.25 | combined 0.5 | combined 0.75 | combined 1.0 — GA and Greedy
- [ ] Plot: cumulative exits — GA vs Greedy at each compliance rate under combined condition
- [ ] Plot: bar chart — total time for GA and Greedy across all conditions and cities

### Cross-city summary:
- [ ] Written conclusion: under fully realistic human behaviour does GA still add value? At what compliance rate?

---

## Final Summary (for Discussion chapter)

- [ ] Table: GA vs Greedy across all conditions and cities at 100 agents — one master summary table
- [ ] Written conclusion for each sub question:
    - SQ1: does city structure affect GA advantage?
    - SQ2: do GA agents disperse faster and reduce danger exposure?
    - SQ3: how does each human factor affect the performance gap?
- [ ] Limitations noted:
    - Simplified city structures
    - Simplified human behaviour model
    - Sequential tuning not full grid search
    - Walking speed not considered in fitness function
    - Compliance assumes binary compliant/non-compliant not partial
- [ ] Future work identified:
    - Exit familiarity / visitor vs local
    - Dynamic fire model
    - Speed-aware fitness function
    - Real city graph structures


## Analysis & Plots
- [ ] Cumulative exits plot — GA vs greedy (already built)
- [ ] Start node dispersal plot at x=1, 2, 3 (already built)
- [ ] Exit utilisation comparison plot
- [ ] Path visualisation on city graph
- [ ] Congestion index across agent counts for GA vs greedy
- [ ] Compliance rate threshold plot
- [ ] Familiarity degradation curve
