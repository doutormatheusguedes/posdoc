import os
import re
from collections import Counter

def parse_individuals(lines):
    individuals = []
    i = 0
    while i < len(lines):
        if lines[i].startswith(" ind ") or lines[i].startswith("ind "):
            match = re.search(r'perfdegrad = ([\d.]+) and nodeLowRepres = (\d+)', lines[i])
            if match:
                perf = float(match.group(1))
                low_rep = int(match.group(2))
                i += 1
                #instance_counts = list(map(int, lines[i].strip().split()))
                #individual = (perf, low_rep, tuple(instance_counts))
                individual = (perf, low_rep)
                individuals.append(individual)
        i += 1
    return individuals

def is_dominated(a, b):
    """Verifica se a é dominado por b (minimização)"""
    return (b[0] <= a[0] and b[1] <= a[1]) and (b[0] < a[0] or b[1] < a[1])

def get_pareto_front(individuals):
    pareto = []
    for ind in individuals:
        if not any(is_dominated(ind, other) for other in individuals if ind != other):
            pareto.append(ind)
    return pareto

def format_solution(sol):
    perf, low_rep = sol
    return f"{perf:.2f}({low_rep})"

# Função para extrair os indivíduos (performance, lowRepres, tupla de instâncias)
def extrair_individuos(linhas):
    individuos = []
    i = 0
    while i < len(linhas):
        linha = linhas[i].lstrip()
        if linha.startswith("ind ") and "has perfdegrad" in linha:
            match = re.search(r'perfdegrad\s*=\s*([0-9.]+)\s*and\s*nodeLowRepres\s*=\s*(\d+)', linha)
            if match:
                perf = float(match.group(1))
                lowrep = int(match.group(2))
                i += 1
                if i < len(linhas):
                    instancia_line = linhas[i].strip()
                    if instancia_line:
                        instancias = tuple(map(int, instancia_line.split()))
                        individuo = (perf, lowrep, instancias)
                        individuos.append(individuo)
        i += 1
    return individuos

def process_file(filepath):
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        #print(f"linhes = {lines}")
    
    id_match = next((re.search(r'ID: (\d+)', line) for line in lines if "ID:" in line), None)
    id_value = id_match.group(1) if id_match else "N/A"
    #print(f"id_value = {id_value}")
    sep_index = next((i for i, line in enumerate(lines) if '----' in line), None)
    initial_lines = lines[:sep_index]
    #print(f"initial  = {initial_lines}")
    final_lines = lines[sep_index+1:]
    #print(f"final = {final_lines}")

    iniciais = extrair_individuos(initial_lines)
    finais = extrair_individuos(final_lines)

    ini_counter = Counter(iniciais)
    ini_qtd_dif = len(ini_counter)
    ini_reps = " ".join(str(ini_counter[k]) for k in ini_counter)

    fin_counter = Counter(finais)
    fin_qtd_dif = len(fin_counter)
    fin_reps = " ".join(str(fin_counter[k]) for k in fin_counter)



    # Extrair população inicial e final
    init_individuals = parse_individuals(initial_lines)
    #print(f"init => {init_individuals}")
    final_individuals = parse_individuals(final_lines)
    #print(f"finais => {final_individuals}")

    # Pegar 1ª fronteira (não dominados)
    init_front = get_pareto_front(init_individuals)
    final_front = get_pareto_front(final_individuals)

    # Agrupar por solução
    init_counter = Counter(init_front)
    final_counter = Counter(final_front)
    nome_limpo = os.path.splitext(filepath)[0]
    match = re.search(r'gen(\d+)_pop(\d+)_cross([\d.]+)_mut([\d.]+)_maxlvl(\d+)_perc([\d.]+)_leaf([\d.]+)', nome_limpo)
    if match:
        g, p, c, m, lvl, r, ln = match.groups()
        short_id = f"g{g}p{p}c{c}m{m}lvl{lvl}r{r}ln{ln}"
    else:
        short_id = "invalid_filename"
    # Formatar saída
    init_str = '; '.join(format_solution(sol) for sol in init_counter)
    init_diff = len(init_counter)
    init_counts = ' | '.join(str(count) for count in init_counter.values())

    final_str = '; '.join(format_solution(sol) for sol in final_counter)
    final_diff = len(final_counter)
    final_counts = ' | '.join(str(count) for count in final_counter.values())

    #return f"{id_value} | {init_str} | {ini_qtd_dif} | {ini_reps} | {final_str} | {fin_qtd_dif} | {fin_reps}"
    return f"{short_id}+id{id_value} | {init_str} | {final_str}"

def main():
    pasta = './experiment_logs_valids'  # Caminho da pasta
    arquivos = [f for f in os.listdir(pasta) if f.endswith('.txt')]
    saida = []

    for arq in arquivos:
        path = os.path.join(pasta, arq)
        linha = process_file(path)
        saida.append(linha)

    with open('resumo_saida1.txt', 'w') as f:
        #f.write("id | sol_pop_inicial | qtd_dif_inds | rep_inds | sol_pop_final | qtd_dif_inds | rep_inds\n")
        f.write("param+id | sol_pop_inicial | sol_pop_final\n")
        
        for linha in saida:
            f.write(linha + '\n')

if __name__ == "__main__":
    main()
