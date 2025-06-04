**NSGA-II Algorithm for the Algorithm Selection Problem using Decision Trees**

**Version:** v2

**How to run an experiment:** python3.11 .\main.py `{nog}` `{nogwi}` `{ps}` `{cr}` `{mr}` `{mtd}` `{r}` `{ilnp}`

**Script to run multiple experiments:** python3.11 .\run_experiments.py

**Evaluated Parameters**

| Parameters                       | Values                |
| --------------------------------- | ---------------------- |
| Number of generations `{nog}`                        | `(1000)`
| Number of generations without improvement `{nogwi}`                        | `(50)`
| Population size `{ps}`	 | `(10, 100)`               |
| Crossover rate `{cr}`                    | `(0.9, 0.8, 0.7)` |
| Mutation rate `{mr}`                    | `(0.2, 0.4)` |
| Maximum tree depth `{mtd}`                    | `(7, 10)` |
| Representativeness (%) `{r}`                    | `(0.05)` |
| Is leaf node probability `{ilnp}`	                    | `(0.5)` |

📈 **Summary Analysis of Result**

| Method                                | Value      |
|-------------------------------------|------------:|
| VBS (Virtual Best Solver)            | 22257.99  |
| SBA (Single Best Algorithm)          | 92942.48  |
| Default (InitialSolve)                | 118465.86 |
| VND* | 46209.56  |
| NSGA-II-v2                             | 38945.78  |

* [paper link](https://onlinelibrary.wiley.com/doi/abs/10.1111/itor.12724)

## Análise dos Resultados / Results Analysis

Our approach is the only one that considers conflicting objectives in the optimization problem. In this context, we present some interesting results below: even though one or another node is indicated as having low representativeness, the values are very close to the desired threshold, which reinforces the quality of the solutions found.

The threshold used for representativeness is 50.2, calculated as |I| = 1004 multiplied by r = 0.05.

### Examples of selected individuals

|  Individual |  Performance Degradation | Nodes with Low Representativeness |  Indicated Nodes                             |
|-----------------------|----------------------------------------------------|--------------------------------------------------------------------|-------------------------------------------------------------|
| 97                    | 34250.41                                           | 7.14                                                               | 117 71 66 52 47 58 44 49 113 75 57 43 52 40 43 77           |
| 70                    | 36757.69                                           | 4.03                                                               | 67 117 71 66 54 92 52 47 49 113 75 57 43 57 **44**           |
| 74                    | 36826.72                                           | 3.03                                                               | 67 117 54 92 50 63 49 113 55 77 43 63 51 56 **54**           |
| 39                    | 37422.61                                           | 3.00                                                               | 67 117 71 66 54 52 47 49 113 75 57 70 65 57 **44**           |
| 54                    | 35995.94                                           | 1.00                                                               | 66 56 96 52 76 77 70 53 65 48 61 108 97 79                    |




📌 **Release Notes - Version v2**

New features and fixes implemented:

- **New input parameter**: Number of generations without improvement (nogwi). The algorithm can now stop early when no progress is observed for a defined number of generations. Thus, the stopping criteria are {nog} and {nogwi}.

- **Avoid duplication in the new population**: In version v1, it was observed that the new population often contained duplicate individuals. Now, when selecting by Pareto front and crowding distance, if an individual already exists in the new population, new individuals are generated until the population is completed without duplicates.

- **Correction in penalty for nodes with low representativeness**: The previous metric only checked if the number of instances in the node was below a threshold, without considering how far below it was. Now the penalty is proportional: ```if n_instances < (r * |I|): penalty += max(1, (((r * |I|) - n_instances) ** 2) / (r * |I|))```

- **Fix in level selection probability during individual construction**: Since the number of nodes per level grows exponentially (level 0 has 1 node, level 1 has 2, level 10 has 1024), lower levels (deeper in the tree) had a much higher chance of being selected, leading to upper levels (closer to the root) being underrepresented. Now, all levels have the same chance of being chosen, promoting a fairer tree construction.

- **More careful initial choice for the root node**: The completely random selection of feature and cutoff point at the root node could negatively impact evolution since this node does not undergo crossover or mutation. Now, 5 combinations of feature and cutoff point are sampled, and one non-dominated choice is selected. If multiple non-dominated exist, one is randomly chosen.

- **Minimum representativeness in internal node splits**: When choosing feature and cutoff point at internal nodes, the algorithm avoids producing a child node with very few instances. If the proposed cutoff results in fewer than `min(10, (0.1 * |I|))`, a new cutoff is generated. If that fails, a new feature is tried. If still unsuccessful, the node becomes a leaf.
