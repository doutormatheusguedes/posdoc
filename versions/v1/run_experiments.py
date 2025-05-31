import subprocess
from itertools import product
import os

# Parâmetros e seus valores possíveis
generations = [10, 100]
pop_size = [10, 100]
crossover_prob = [0.9, 0.8, 0.7]
mutation_prob = [0.1, 0.2, 0.4]
max_level = [3, 5, 7, 10]
percentage_min_instances = [0.02, 0.05, 0.1]
node_is_leaf = [0.3, 0.5, 0.7]

# Diretório para salvar os logs
log_dir = "experiment_logs_valids"
os.makedirs(log_dir, exist_ok=True)

# Gerar todas as combinações
combinations = product(
    generations, pop_size, crossover_prob, mutation_prob,
    max_level, percentage_min_instances, node_is_leaf
)

for idx, combo in enumerate(combinations):
    gen, pop, cross_prob, mut_prob, max_lvl, perc_min, node_leaf = combo
    cmd = [
        "python",
        ".\\main.py",
        str(gen),
        str(pop),
        str(cross_prob),
        str(mut_prob),
        str(max_lvl),
        str(perc_min),
        str(node_leaf)
    ]

    # Nome do arquivo baseado na configuração
    log_filename = f"idx{idx}_gen{gen}_pop{pop}_cross{cross_prob}_mut{mut_prob}_maxlvl{max_lvl}_perc{perc_min}_leaf{node_leaf}.txt"
    log_path = os.path.join(log_dir, log_filename)

    with open(log_path, "w", encoding="utf-8") as f:
        # Cabeçalho com configuração e id
        f.write(f"Executando combinação ID: {idx}\n")
        f.write(f"Parâmetros:\n")
        f.write(f" generations={gen}\n")
        f.write(f" pop_size={pop}\n")
        f.write(f" crossover_prob={cross_prob}\n")
        f.write(f" mutation_prob={mut_prob}\n")
        f.write(f" max_level={max_lvl}\n")
        f.write(f" percentage_min_instances={perc_min}\n")
        f.write(f" node_is_leaf={node_leaf}\n")
        f.write("="*40 + "\n\n")

        print(f"Rodando: {' '.join(cmd)}  --> log: {log_path}")

        # Executar e capturar saída
        processo = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        # Escrever saída do processo no arquivo
        f.write(processo.stdout)
