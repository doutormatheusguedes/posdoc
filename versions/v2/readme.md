**NSGA-II Algorithm for the Algorithm Selection Problem using Decision Trees**

**Version:** v2

### 🆕 Summary of Updates from v1 to v2
- **New input parameter:** `Number of generations without improvement` (`nogwi`). This allows the algorithm to stop early when no further progress is observed over a defined number of generations.

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

- Teste 1

- Teste 2

