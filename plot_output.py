import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import re
from itertools import cycle

# Função para verificar dominância (minimização)
def is_dominated(sol, others):
    return any((o[0] <= sol[0] and o[1] <= sol[1]) and (o[0] < sol[0] or o[1] < sol[1]) for o in others)

# Função para extrair os parâmetros
def parse_params(param_str):
    match = re.match(r"g(\d+)p(\d+)c([\d.]+)m([\d.]+)lvl(\d+)r([\d.]+)ln([\d.]+)\+id\d+", param_str)
    if match:
        g, p, c, m, lvl, r, ln = match.groups()
        return f"g={g}, p={p}, c={c}, m={m}, lvl={lvl}, r={r}, ln={ln}"
    return "unknown"

# Função para extrair objetivos das soluções
def extract_solutions(sol_str):
    solutions = []
    entries = sol_str.split(';')
    for entry in entries:
        match = re.match(r"\s*([\d.]+)\((\d+)\)", entry.strip())
        if match:
            f1 = float(match.group(1))  # desempenho
            f2 = int(match.group(2))    # número de nós ruins
            solutions.append((f1, f2))
    return solutions

# Lê o CSV
file_path = "resumo_saida1.txt"  # Substitua com o caminho correto
df = pd.read_csv(file_path, sep="|")

# Coletar todas as soluções com seus parâmetros
all_solutions = []

for idx, row in df.iterrows():
    param_str = row.iloc[0].strip()
    param_label = parse_params(param_str)
    sol_str = str(row.iloc[2])  # sol_pop_final
    solutions = extract_solutions(sol_str)
    for s in solutions:
        all_solutions.append({
            "f1": s[0],
            "f2": s[1],
            "params": param_label
        })

# Filtrar soluções não dominadas
pareto_solutions = []
for i, s in enumerate(all_solutions):
    if not is_dominated((s["f1"], s["f2"]), [(o["f1"], o["f2"]) for j, o in enumerate(all_solutions) if j != i]):
        pareto_solutions.append(s)

# Plot
plt.figure(figsize=(12, 8))
colors = cycle(plt.cm.tab20.colors)
param_colors = {}
opt_degradacao = 22257.99
opt_nos = 0
vnd_degradacao = 46209.56
vnd_nos = 0
default_degradacao = 118465.86
default_nos = 0

for sol in pareto_solutions:
    print(f"sol = {sol}")
    key = sol["params"]
    if key not in param_colors:
        param_colors[key] = next(colors)
    plt.scatter(sol["f2"], sol["f1"], label=key, color=param_colors[key], alpha=0.8, edgecolor='k')
plt.scatter(opt_nos, opt_degradacao, c='red', marker='X', s=50, label='Valor Ótimo')
plt.scatter(vnd_nos, vnd_degradacao, c='green', marker='X', s=50, label='Model+VND(Paper)')
#plt.scatter(default_nos, default_degradacao, c='yellow', marker='X', s=150, label='Default')
plt.scatter([], [], label='Default: 118465.86(?)', c='yellow', marker='X', s=50)
plt.scatter([], [], label='SBA: 92942.48(?)', c='blue', marker='X', s=50)
plt.scatter([], [], label='\n--Parameters--\ngenerations(10,100)\npop(10,100)\ncrossover(0.7,0.8,0.9)\nmutation(0.1,0.2,0.4)\nmax_level(3,5,7,10)\npercentage_min_instances(0.02,0.05,0.10)\nis_leaf_node(0.3,0.5,0.7)', c='black', marker='', s=50)


# Remover duplicadas na legenda
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

plt.ylabel("Degradação de performance (f1)")
plt.xlabel("Nós com baixa representatividade (f2)")
plt.title("Fronteira de Pareto - Soluções Não Dominadas")
plt.grid(True)
plt.tight_layout()
plt.show()
