**NSGA-II Algorithm for the Algorithm Selection Problem using Decision Trees**

**Version:** v1

**How to run an experiment:** python3.11 .\main.py `{nog}` `{ps}` `{cr}` `{mr}` `{mtd}` `{r}` `{ilnp}`

**Script to run multiple experiments:** python3.11 .\run_experiments.py

**Evaluated Parameters**

| Parameters                       | Values                |
| --------------------------------- | ---------------------- |
| Number of generations `{nog}`                        | `(10, 100)`
| Population size `{ps}`	 | `(10, 100)`               |
| Crossover rate `{cr}`                    | `(0.9, 0.8, 0.7)` |
| Mutation rate `{mr}`                    | `(0.1, 0.2, 0.4)` |
| Maximum tree depth `{mtd}`                    | `(3, 5, 7, 10)` |
| Representativeness (%) `{r}`                    | `(0.02, 0.05, 0.10)` |
| Is leaf node probability `{ilnp}`	                    | `(0.3, 0.5, 0.7)` |



