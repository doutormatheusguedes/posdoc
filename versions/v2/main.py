import utils
import sys
from utils import ler_arquivo_entrada, algoritmo_menor_degradacao
import time


file_path = "../../bases/input_data_paper_2019.txt"
features, instances, algorithms, vbs = ler_arquivo_entrada(file_path)
print(f"\nVirtual Best Solver (VBS): {vbs:.2f}")
algoritmo_menor_degradacao(algorithms)
for alg in algorithms.values():
    if alg.getNome() == "initialSolve":
        print(f"default ({alg.getNome()}): {alg.performanceDegradationAllProblems():.2f}")

# set parameters
generations = int(sys.argv[1])
generations_no_improve = int(sys.argv[2])
pop_size = int(sys.argv[3])
crossover_prob = float(sys.argv[4])
mutation_prob = float(sys.argv[5])
max_level = int(sys.argv[6])
max_nodes = (2 ** max_level) - 1
percentage_min_instances = float(sys.argv[7])
min_instances_per_leafnode = percentage_min_instances * len(instances)
node_is_leaf = float(sys.argv[8])
start_time = time.time()
utils.nsga(generations, generations_no_improve, pop_size, max_nodes, features, algorithms, instances, crossover_prob, mutation_prob, min_instances_per_leafnode, node_is_leaf)
end_time = time.time()

execution_time = end_time - start_time

print(f"Tempo total de execução: {execution_time:.2f} segundos")