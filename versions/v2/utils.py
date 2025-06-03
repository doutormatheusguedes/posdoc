import random
import copy
import sys
import os
import math


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(BASE_DIR)


from features import Feature
from instances import Instance
from algorithms import Algorithm

def make_hashable(value):
    if isinstance(value, list):
        return tuple(make_hashable(v) for v in value)
    elif isinstance(value, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in value.items()))
    else:
        return value


def nsga(generations, generations_no_improve, pop_size, max_nodes, features, algorithms, instances, crossover_prob, mutation_prob, min_instances_per_leafnode, node_is_leaf):
    # generate initial population
    population = generate_initial_population(0,pop_size, max_nodes, features, algorithms, instances, min_instances_per_leafnode, node_is_leaf)
    
    best_front = [] 
    no_improve_counter = 0
    gen = 0

    # loop nsga
    while no_improve_counter < generations_no_improve and gen < generations:
        fronts = non_dominated_sort(population)
        if gen == 0:
            for faux in range(len(fronts)):
                if faux < 2:
                    print(f"f{faux} = {fronts[faux]}")
        # 2. Atribuir rank (nível da frente) a cada indivíduo
        rank = [0] * len(population)
        for i, front in enumerate(fronts):
            for idx in front:
                rank[idx] = i

        # 3. Calcular crowding distances por indivíduo
        crowding_distance = [0.0] * len(population)
        for front in fronts:
            #print(f"front {front}:")
            dist = calculate_crowding_distance(front, population)
            for idx in front:
                crowding_distance[idx] = dist[idx]

        # 4. Seleção por torneio binário para gerar filhos
        offspring = []
        while len(offspring) < pop_size:
            a1 = random.randint(0, len(population) - 1)
            b1 = random.randint(0, len(population) - 1)
            winner_p1 = binary_tournament(a1, b1, rank, crowding_distance)
            a2 = random.randint(0, len(population) - 1)
            b2 = random.randint(0, len(population) - 1)
            winner_p2 = binary_tournament(a2, b2, rank, crowding_distance)
            #print(f"crossover_prob = {crossover_prob}")
            if random.random() < crossover_prob:
                # Realiza crossover
                child1, child2 = subtree_crossover_vector(population[winner_p1], population[winner_p2], max_nodes)
            else:
                # Clona os pais
                child1 = copy.deepcopy(population[winner_p1])
                child2 = copy.deepcopy(population[winner_p2])
            # MUTAÇÕES EM child1
            #print(f"mutation_prob = {mutation_prob}")
            if random.random() < mutation_prob:
                if random.random() < 0.5:
                    mutate_replace_subtree(instances, child1, features, max_nodes, node_is_leaf)
                else:
                    mutate_prune_or_grow(instances, child1, features, max_nodes, node_is_leaf)

            # MUTAÇÕES EM child2
            if random.random() < mutation_prob:
                if random.random() < 0.5:
                    mutate_replace_subtree(instances, child2, features, max_nodes, node_is_leaf)
                else:
                    mutate_prune_or_grow(instances, child2, features, max_nodes, node_is_leaf)

            # Avaliação
            evaluate_individual(child1, algorithms, instances, min_instances_per_leafnode, max_nodes)
            evaluate_individual(child2, algorithms, instances, min_instances_per_leafnode, max_nodes)

            offspring.append(child1)
            if len(offspring) < pop_size:
                offspring.append(child2)
           
        # 6. União de pais e filhos para nova avaliação
        population.extend(offspring)  # agora tem 2N indivíduos
        #print(f"len pop = {len(population)}")
        # Reavaliação: voltar ao início do laço com nova população (após seleção de N melhores)

        # Ordenar novamente
        fronts = non_dominated_sort(population)

        # início do código novo
        current_front = [population[i] for i in fronts[0]]

        def is_not_dominated_by_best(sol, best_solutions):
            for best in best_solutions:
                # Caso 1: best domina sol
                if (
                    best['value_performance_degradation'] <= sol['value_performance_degradation'] and
                    best['count_low_representation_nodes'] <= sol['count_low_representation_nodes'] and
                    (
                        best['value_performance_degradation'] < sol['value_performance_degradation'] or
                        best['count_low_representation_nodes'] < sol['count_low_representation_nodes']
                    )
                ):
                    return False  # sol é dominada por best

                # Caso 2: sol é idêntica a best
                if (
                    best['value_performance_degradation'] == sol['value_performance_degradation'] and
                    best['count_low_representation_nodes'] == sol['count_low_representation_nodes']
                ):
                    return False  # sol é idêntica a best → não considerar como nova
            
            # Caso 3: sol não é dominada nem idêntica a nenhuma solução de best_solutions
            return True


        if any(is_not_dominated_by_best(sol, best_front) for sol in current_front):
            
            #print("best_front = ")
            #for i in range(len(best_front)):
                #print(f" ind {i} has perfdegrad = {best_front[i]['value_performance_degradation']:.2f} and nodeLowRepres = {best_front[i]['count_low_representation_nodes']}")
        

            #print("current_front = ")
            #for i in range(len(current_front)):
                #print(f" ind {i} has perfdegrad = {current_front[i]['value_performance_degradation']:.2f} and nodeLowRepres = {current_front[i]['count_low_representation_nodes']}")
        
            best_front = current_front

            
           
            no_improve_counter = 0
        else:
            no_improve_counter += 1

        print(f"[Geração {gen}] Sem melhora consecutiva: {no_improve_counter}")
        gen += 1

        # fim do código novo

        # Construir nova população com no máximo pop_size
        # xxxx
        # Construir nova população com no máximo pop_size, evitando duplicatas
        new_population = []
        unique_individuals = {}  # chave = str(ind), valor = ind

        for front in fronts:
            front_unique = []
            for i in front:
                ind = population[i]
                #key = str(ind)
                #key = str(sorted(ind.items()))
                #ind_rounded = {k: round(v, 5) if isinstance(v, float) else v for k,v in ind.items()}
                #key = tuple(sorted(ind_rounded.items()))
                #ind_rounded = {k: round(v, 5) if isinstance(v, float) else v for k, v in ind.items()}
                #ind_hashable = {k: make_hashable(v) for k, v in ind_rounded.items()}
                #key = tuple(sorted(ind_hashable.items()))
                key = (
                    round(ind['value_performance_degradation'], 5),
                    round(ind['count_low_representation_nodes'], 5)
                )
                
                if key not in unique_individuals:
                    unique_individuals[key] = ind
                    front_unique.append((i, ind))  # Salva para crowding distance

            # Verifica se pode incluir todos os únicos desta frente
            if len(new_population) + len(front_unique) <= pop_size:
                #print(f"size de newpop antes = {len(new_population)}")
                new_population.extend([ind for _, ind in front_unique])
                #print("new population dentro do if: ")
                #for i in range(len(new_population)):
                    #print(f" ind {i} has perfdegrad = {new_population[i]['value_performance_degradation']:.2f} and nodeLowRepres = {new_population[i]['count_low_representation_nodes']}")
                #print(f"size de newpop depois = {len(new_population)}")
                
            else:
                if len(new_population) >= pop_size:
                    break
                #print(f"entrou aqui no else que calcula crowding. |np| = {len(new_population)}, |front| =  + {len(front_unique)}, f = {front}")
                # Precisamos selecionar apenas alguns indivíduos desta frente
                remaining = pop_size - len(new_population)
                indices = [i for i, _ in front_unique]
                individuals = [ind for _, ind in front_unique]

                # Calcula crowding distance para os indivíduos desta frente
                dist = calculate_crowding_distance(indices, population)
                sorted_by_distance = sorted(
                    zip(individuals, dist), key=lambda x: x[1], reverse=True
                )
                new_population.extend([ind for ind, _ in sorted_by_distance[:remaining]])
                #print("new population dentro do else: ")
                #for i in range(len(new_population)):
                    #print(f" ind {i} has perfdegrad = {new_population[i]['value_performance_degradation']:.2f} and nodeLowRepres = {new_population[i]['count_low_representation_nodes']}")
        
                break  # População preenchida

        # Agora, new_population contém até pop_size indivíduos,
        # sem duplicatas, priorizando frentes anteriores e maior diversidade.


        





        #print("new population: ")
        #for i in range(len(new_population)):
                #print(f" ind {i} has perfdegrad = {new_population[i]['value_performance_degradation']:.2f} and nodeLowRepres = {new_population[i]['count_low_representation_nodes']}")
        
        #print("completando ind = ")
        #for i, item in enumerate(sorted_by_distance[:remaining]):
        #    individuo = item[0][1]
        #    print(f" ind {i} has perfdegrad = {individuo['value_performance_degradation']:.2f} and nodeLowRepres = {individuo['count_low_representation_nodes']}")
        

        
        # Passo 2: Se ainda não atingiu pop_size, completa com novos individuos aleatórios
        if len(new_population) < pop_size:
            print(f"Tá completando populacao com {(pop_size - len(new_population))} indivíduos")
            remaining = pop_size - len(new_population)
            extra_individuals = generate_initial_population(flag=1,
                pop_size=remaining,
                max_nodes=max_nodes,
                features=features,
                algorithms=algorithms,
                instances=instances,
                min_instances_per_leafnode=min_instances_per_leafnode,
                node_is_leaf=node_is_leaf
            )
            new_population.extend(extra_individuals)

        population = new_population  # Atualizar para próxima geração








        #xxx
    for i in range(pop_size):
        print(f" ind {i} has perfdegrad = {population[i]['value_performance_degradation']:.2f} and nodeLowRepres = {population[i]['count_low_representation_nodes']}")
        for z in range(max_nodes):
            if population[i]["leafNode"][z] == 1 and len(population[i]["problemsInLeafNode"][z]) > 0:
                print(len(population[i]['problemsInLeafNode'][z]), sep=" ", end = " ")
        print("")
    if no_improve_counter == (generations_no_improve) or gen == generations:
        fronts = non_dominated_sort(population)
        for faux in range(len(fronts)):
            if faux < 2:
                print(f"f{faux} = {fronts[faux]}")
        print(f"generations = {gen}")

def ler_arquivo_entrada(caminho_arquivo):
    with open(caminho_arquivo, "r") as f:
        linhas = [linha.strip() for linha in f if linha.strip() != ""]

    dados = {}
    chave_atual = None
    conteudo_atual = []

    # Separar seções do arquivo baseado nas linhas chave
    for linha in linhas:
        if linha in {"Problems", "Features", "Algorithms",
                     "ValueProblemFeature", "ValueProblemAlgorithm"}:
            if chave_atual:
                dados[chave_atual] = conteudo_atual
            chave_atual = linha
            conteudo_atual = []
        else:
            conteudo_atual.append(linha)
    if chave_atual:
        dados[chave_atual] = conteudo_atual

    # Processar Problems
    problems = dados["Problems"][0].split()

    # Processar Features
    features_nomes = dados["Features"][0].split()

    # Processar Algorithms
    algorithms_nomes = dados["Algorithms"][0].split()

    # Processar ValueProblemFeature
    # Armazena em um dicionário: {problema: {feature: valor}}
    valores_features_por_problema = {p: {} for p in problems}
    for linha in dados["ValueProblemFeature"]:
        partes = linha.split()
        #print(partes)
        problema, feature, valor = partes[0], partes[1], float(partes[2])
        valores_features_por_problema[problema][feature] = valor

    # Criar matriz de features na ordem de problems x features_nomes
    matriz_features = []
    for p in problems:
        linha = [valores_features_por_problema[p][f] for f in features_nomes]
        matriz_features.append(linha)

    # Processar valueProblemAlgorithm
    # Armazena em um dicionário: {problema: {algoritmo: valor}}
    valores_algoritmos_por_problema = {p: {} for p in problems}
    for linha in dados["ValueProblemAlgorithm"]:
        partes = linha.split()
        problema, algoritmo, valor = partes[0], partes[1], float(partes[2])
        valores_algoritmos_por_problema[problema][algoritmo] = valor

    # Criar matriz de algoritmos na ordem de problems x algorithms_nomes
    matriz_algoritmos = []
    for p in problems:
        linha = [valores_algoritmos_por_problema[p][a] for a in algorithms_nomes]
        matriz_algoritmos.append(linha)

    # Criar instâncias
    instancias = []
    for i, p in enumerate(problems):
        feats = {features_nomes[j]: matriz_features[i][j] for j in range(len(features_nomes))}
        degradacoes = {algorithms_nomes[j]: matriz_algoritmos[i][j] for j in range(len(algorithms_nomes))}
        inst = Instance(p, feats, degradacoes)
        instancias.append(inst)

    # Criar features com valores de corte gerados automaticamente
    features = {}
    for j, fname in enumerate(features_nomes):
        valores = sorted(set(matriz_features[i][j] for i in range(len(problems))))
        f = Feature(fname)
        for v in valores:
            f.adicionar_valor(v)
        features[fname] = f

    # Criar algoritmos
    algorithms = {}
    for aname in algorithms_nomes:
        alg = Algorithm(aname)
        alg.atualizar(instancias)
        algorithms[aname] = alg

    # Calcular VBS
    vbs = sum(min(inst.degradacoes.values()) for inst in instancias)
    return features, instancias, algorithms, vbs

def evaluate_individual(ind, algorithms, instances, min_instances_per_leafnode, max_nodes):
    ind["problemsInLeafNode"] = {}
    ind["value_performance_degradation"] = 0.0
    ind["count_low_representation_nodes"] = 0

    for i in range(max_nodes):
        if ind["leafNode"][i] == 1:
            insts_ate_aqui = get_instances_for_leaf(
                i,
                ind["featureNode"],
                ind["cutoff_pointNode"],
                instances
            )

            ind["problemsInLeafNode"][i] = [inst.nome for inst in insts_ate_aqui]


             
                

            if insts_ate_aqui:
                min_total = float("inf")
                best_alg = None
                for alg_name in algorithms:
                    soma = sum(inst.degradacoes[alg_name] for inst in insts_ate_aqui)
                    if soma < min_total:
                        min_total = soma
                        best_alg = alg_name
                ind["algorithmLeafNode"][i] = best_alg
                if len(insts_ate_aqui) < min_instances_per_leafnode and len(insts_ate_aqui) > 0:
                    penalty = max(1, ((min_instances_per_leafnode - len(insts_ate_aqui))**2)/min_instances_per_leafnode)
                    ind["count_low_representation_nodes"] += penalty
                ind["value_performance_degradation"] += min_total
            else:
                ind["algorithmLeafNode"][i] = None

def get_subtree_indices(start_index, max_nodes):
    """Retorna os índices da subárvore a partir de um nó start_index (heap indexing)."""
    indices = []
    def dfs(i):
        if i >= max_nodes:
            return
        indices.append(i)
        dfs(2 * i + 1)
        dfs(2 * i + 2)
    dfs(start_index)
    return indices

def mutate_prune_or_grow(conjunto_instances, individual, features, max_nodes, node_is_leaf):
    tipo = random.choice(["poda", "crescimento"])
    
    if tipo == "poda":
        # Seleciona nó interno com filhos válidos
        candidatos = [i for i in range(max_nodes) if individual["leafNode"][i] == 0 and 2 * i + 1 < max_nodes]
        if not candidatos:
            return
        no = sortear_no_por_nivel_uniforme(individual, candidatos)
        # Poda
        individual["leafNode"][no] = 1
        individual["featureNode"][no] = None
        individual["cutoff_pointNode"][no] = None
        invalidate_subtree(2 * no + 1, max_nodes, [True]*max_nodes, individual)
        invalidate_subtree(2 * no + 2, max_nodes, [True]*max_nodes, individual)

    else:  # crescimento
        candidatos = [i for i in range(max_nodes) if individual["leafNode"][i] == 1 and 2 * i + 1 < max_nodes]
        if not candidatos:
            return
        no = sortear_no_por_nivel_uniforme(individual, candidatos)

        def recriar(i, primeiro_no=False):
            if i >= max_nodes:
                return
            is_last_level = (2 * i + 1 >= max_nodes)

            if primeiro_no:
                # Força nó interno no primeiro nó
                fname, feature = random.choice(list(features.items()))
                cutoff = random.choice(list(feature.pontos_de_corte))
                individual["featureNode"][i] = fname
                individual["cutoff_pointNode"][i] = cutoff
                individual["leafNode"][i] = 0
                recriar(2 * i + 1)
                recriar(2 * i + 2)

            else:
                if not is_last_level and random.random() >= node_is_leaf:
                    fname, feature = random.choice(list(features.items()))
                    cutoff = random.choice(list(feature.pontos_de_corte))

                    insts_no_no = get_instances_for_leaf(
                        i,
                        individual["featureNode"],
                        individual["cutoff_pointNode"],
                        conjunto_instances
                    )
                    subtrees = split_instances(insts_no_no, fname, cutoff)
                    #print(f"len insts_no_no = {len(insts_no_no)}")
                    qtd_total = len(conjunto_instances)
                    #print(f"qtd = {qtd_total}")
                    representativo = all(len(subset) >= min(10, 0.1 * qtd_total) for subset in subtrees)

                    if not representativo:
                        # TENTA UM NOVO PAR (FEATURE, CUTOFF)
                        fname2, feature2 = random.choice(list(features.items()))
                        cutoff2 = random.choice(list(feature2.pontos_de_corte))
                        subtrees2 = split_instances(insts_no_no, fname2, cutoff2)
                        representativo2 = all(len(subset) >= min(10, 0.1 * qtd_total) for subset in subtrees2)

                        if not representativo2:
                            # Falhou novamente → vira folha
                            individual["leafNode"][i] = 1
                            individual["featureNode"][i] = None
                            individual["cutoff_pointNode"][i] = None
                            return
                        else:
                            # Sucesso com nova feature e cutoff
                            individual["featureNode"][i] = fname2
                            individual["cutoff_pointNode"][i] = cutoff2
                            individual["leafNode"][i] = 0
                    else:
                        # Sucesso com primeiro par
                        individual["featureNode"][i] = fname
                        individual["cutoff_pointNode"][i] = cutoff
                        individual["leafNode"][i] = 0

                    # Continua recursão para os filhos
                    
                    recriar(2 * i + 1)
                    recriar(2 * i + 2)
                else:
                    individual["leafNode"][i] = 1
                    individual["featureNode"][i] = None
                    individual["cutoff_pointNode"][i] = None

        recriar(no, primeiro_no=True)

def sortear_indice_por_nivel_uniforme(indices_validos):
    """
    Sorteia um índice com chance uniforme por nível (nível calculado dinamicamente).
    """
    

    # Agrupar os índices por nível
    por_nivel = {}
    for i in indices_validos:
        nivel = int(math.floor(math.log2(i + 1)))
        por_nivel.setdefault(nivel, []).append(i)

    # Escolher nível com mesma chance
    nivel_escolhido = random.choice(list(por_nivel.keys()))

    # Escolher um índice dentro do nível
    return random.choice(por_nivel[nivel_escolhido])



def subtree_crossover_vector(p1, p2, max_nodes):
    """Executa crossover por subárvore entre dois indivíduos codificados vetorialmente."""
    # 1. Escolhe um nó interno aleatório como ponto de crossover (evita folhas)
    valid_indices = [i for i in range(max_nodes) if p1["leafNode"][i] == 0 and p2["leafNode"][i] == 0]
    if not valid_indices:
        # Não há crossover possível, retorna cópias
        return copy.deepcopy(p1), copy.deepcopy(p2)

    crossover_point = sortear_indice_por_nivel_uniforme(valid_indices)

    # 2. Pega os índices da subárvore
    #print(f"indice do no escolhido para crossover: {crossover_point}")
    indices_subtree = get_subtree_indices(crossover_point, max_nodes)

    # 3. Cria os filhos como cópias dos pais
    child1 = copy.deepcopy(p1)
    child2 = copy.deepcopy(p2)

    # 4. Troca as subárvores nos filhos
    for i in indices_subtree:
        for key in ["featureNode", "cutoff_pointNode", "leafNode", "algorithmLeafNode"]:
            child1[key][i] = copy.deepcopy(p2[key][i])
            child2[key][i] = copy.deepcopy(p1[key][i])

    # 5. Apaga o mapeamento de instâncias e recalcula depois da avaliação
    child1["problemsInLeafNode"] = {}
    child1["value_performance_degradation"] = 0.0
    child1["count_low_representation_nodes"] = 0

    child2["problemsInLeafNode"] = {}
    child2["value_performance_degradation"] = 0.0
    child2["count_low_representation_nodes"] = 0

    return child1, child2


def algoritmo_menor_degradacao(algorithms):
 
        # Se for dict, pegar valores
        if isinstance(algorithms, dict):
            algorithms = algorithms.values()
        
        # Filtrar apenas algoritmos que já foram atualizados (total_instancias > 0)
        atualizados = [alg for alg in algorithms if alg.total_instancias > 0]
        if not atualizados:
            print("Nenhum algoritmo atualizado com instâncias.")
            return None

        # Encontrar o algoritmo com menor degradacao_total
        melhor_alg = min(atualizados, key=lambda alg: alg.performanceDegradationAllProblems())
        
        print(f"{melhor_alg.getNome()} (SBA) = {melhor_alg.performanceDegradationAllProblems():.2f}")


def calculate_crowding_distance(front, population):
    """
    Calcula a crowding distance para cada indivíduo em uma frente.
    Retorna um dicionário {indiv_id: distance}.
    """
    distance = {ind: 0.0 for ind in front}

    # Objetivos usados (ajuste conforme seu modelo)
    objectives_names = ["value_performance_degradation", "count_low_representation_nodes"]

    for obj in objectives_names:
        # Ordena os indivíduos da frente com base neste objetivo
        front_sorted = sorted(front, key=lambda ind: population[ind][obj])
        
        # Valores mínimos e máximos do objetivo para normalização
        obj_min = population[front_sorted[0]][obj]
        obj_max = population[front_sorted[-1]][obj]
        range_obj = obj_max - obj_min if obj_max != obj_min else 1e-9
        #print(f"range_obj = {range_obj}")

        # Atribui distância infinita aos extremos (fronteiras)
        distance[front_sorted[0]] = 999999999
        distance[front_sorted[-1]] = 999999999
        

        # Calcula distância normalizada para os elementos internos
        for i in range(1, len(front_sorted) - 1):
            prev = population[front_sorted[i - 1]][obj]
            next_ = population[front_sorted[i + 1]][obj]
            dist = (next_ - prev) / range_obj
            
            distance[front_sorted[i]] += dist

    #print(distance)
    return distance  # Dict: {ind_id: distance}


def binary_tournament(a, b, rank, distance):
    if rank[a] < rank[b]:
        return a
    elif rank[a] > rank[b]:
        return b
    else:
        if distance[a] > distance[b]:
            return a
        elif distance[a] < distance[b]:
            return b
        else:
            print("Entrou aqui: 2 indivíduos são iguais em fronteira e crowding distance!")
            return random.choice([a, b])
        
def sortear_no_por_nivel_uniforme(individual, candidatos):
    """
    Sorteia um nó interno entre os candidatos com chance igual por nível.
    Calcula o nível dinamicamente com base na posição do nó.
    
    individual: dicionário que contém "leafNode"
    candidatos: lista de índices dos nós internos (leafNode[i] == 0)
    """
    # Agrupar os candidatos por nível, calculando o nível dinamicamente
    nos_por_nivel = {}
    for i in candidatos:
        nivel = int(math.floor(math.log2(i + 1)))
        nos_por_nivel.setdefault(nivel, []).append(i)

    # Sortear um nível entre os níveis presentes (chance igual por nível)
    nivel_escolhido = random.choice(list(nos_por_nivel.keys()))

    # Sortear um nó dentro do nível escolhido
    no_alvo = random.choice(nos_por_nivel[nivel_escolhido])
    return no_alvo


def mutate_replace_subtree(conjunto_instances, individual, features, max_nodes, node_is_leaf):
    # Seleciona um nó interno aleatório
    candidatos = [i for i in range(max_nodes) if individual["leafNode"][i] == 0]
    if not candidatos:
        return  # nada a fazer
    no_alvo = sortear_no_por_nivel_uniforme(individual, candidatos)

    # Invalida subárvore a partir do nó alvo
    invalidate_subtree(no_alvo, max_nodes, [True]*max_nodes, individual)
    
    # Recria a subárvore a partir do nó alvo
    def recriar(i):
        if i >= max_nodes:
            return
        is_last_level = (2 * i + 1 >= max_nodes)
        if not is_last_level and random.random() >= node_is_leaf:
            fname, feature = random.choice(list(features.items()))
            cutoff = random.choice(list(feature.pontos_de_corte))
            insts_no_no = get_instances_for_leaf(
                i,
                individual["featureNode"],
                individual["cutoff_pointNode"],
                conjunto_instances
            )
            subtrees = split_instances(insts_no_no, fname, cutoff)

            qtd_total = len(conjunto_instances)
            representativo = all(len(subset) >= min(10, 0.1 * qtd_total) for subset in subtrees)

            if not representativo:
                # TENTA UM NOVO PAR (FEATURE, CUTOFF)
                fname2, feature2 = random.choice(list(features.items()))
                cutoff2 = random.choice(list(feature2.pontos_de_corte))
                subtrees2 = split_instances(insts_no_no, fname2, cutoff2)
                representativo2 = all(len(subset) >= min(10, 0.1 * qtd_total) for subset in subtrees2)

                if not representativo2:
                    # Falhou novamente → vira folha
                    individual["leafNode"][i] = 1
                    individual["featureNode"][i] = None
                    individual["cutoff_pointNode"][i] = None
                    return
                else:
                    # Sucesso com nova feature e cutoff
                    individual["featureNode"][i] = fname2
                    individual["cutoff_pointNode"][i] = cutoff2
                    individual["leafNode"][i] = 0
            else:
                # Sucesso com primeiro par
                individual["featureNode"][i] = fname
                individual["cutoff_pointNode"][i] = cutoff
                individual["leafNode"][i] = 0

            recriar(2 * i + 1)
            recriar(2 * i + 2)


        else:
            individual["leafNode"][i] = 1
            individual["featureNode"][i] = None
            individual["cutoff_pointNode"][i] = None
            # filhos são automaticamente ignorados por sua função de avaliação

    recriar(no_alvo)





def get_instances_for_leaf(i, feature_nodes, cutoff_nodes, instances):
    """
    Retorna as instâncias que chegam até o nó folha i,
    considerando as decisões no caminho da raiz até esse nó.
    """
    selected_instances = instances.copy()

    # Reconstrói o caminho da raiz até o nó i
    path = []
    while i > 0:
        parent = (i - 1) // 2
        go_left = (2 * parent + 1 == i)
        path.append((parent, go_left))
        i = parent
    path.reverse()

    # Filtra as instâncias com base nas decisões dos nós no caminho
    for parent_idx, went_left in path:
        fname = feature_nodes[parent_idx]
        cutoff = cutoff_nodes[parent_idx]
        if fname is None or cutoff is None:
            continue

        if went_left:
            selected_instances = [
                inst for inst in selected_instances
                if inst.get_valor_feature(fname) <= cutoff
            ]
        else:
            selected_instances = [
                inst for inst in selected_instances
                if inst.get_valor_feature(fname) > cutoff
            ]

    return selected_instances


def invalidate_subtree(start_index, max_nodes, valid_nodes, individual):
    """Marca todos os descendentes de start_index como inválidos."""
    queue = [start_index]
    while queue:
        i = queue.pop(0)
        if i >= max_nodes:
            continue
        valid_nodes[i] = False
        individual["leafNode"][i] = -1
        queue.append(2 * i + 1)  # filho esquerdo
        queue.append(2 * i + 2)  # filho direito

def dominates(obj_a, obj_b):
    """Retorna True se obj_a domina obj_b"""
    return all(a <= b for a, b in zip(obj_a, obj_b)) and any(a < b for a, b in zip(obj_a, obj_b))

def non_dominated_sort(population):
    """
    Retorna uma lista de frentes, cada uma com os índices dos indivíduos da população.
    """
    num_individuals = len(population)
    S = [[] for _ in range(num_individuals)]  # S[i] = quem é dominado por i
    n = [0] * num_individuals                 # n[i] = quantos dominam i
    rank = [0] * num_individuals              # rank[i] = frente de i
    fronts = [[]]                             # lista de frentes

    # Extrair objetivos
    objectives = [
        (ind["value_performance_degradation"], ind["count_low_representation_nodes"])
        for ind in population
    ]

    for p in range(num_individuals):
        for q in range(num_individuals):
            if p == q:
                continue
            if dominates(objectives[p], objectives[q]):
                S[p].append(q)
            elif dominates(objectives[q], objectives[p]):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            fronts[0].append(p)

    i = 0
    while i < len(fronts):
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        i += 1
        if next_front:
            fronts.append(next_front)
        else:
            break
    return fronts  # Lista de frentes (com índices)

def split_instances(instances, feature_name, cutoff):
    esquerda = [inst for inst in instances if inst.features[feature_name] <= cutoff]
    direita = [inst for inst in instances if inst.features[feature_name] > cutoff]
    return [esquerda, direita]

def tentar_corrigir_divisao(conjunto_instances, instances, features, feature_atual, cutoff_atual, min_instances_per_leafnode):
    
    
    lim_min = min(10, (0.1 * len(conjunto_instances))) if len(conjunto_instances) > 0 else 0
    #print(f"lim_min = {lim_min}")
    # 1) Tenta mudar só o ponto de corte (1 tentativa)
    pontos_corte = list(features[feature_atual].pontos_de_corte)
    pontos_corte = [c for c in pontos_corte if c != cutoff_atual]
    random.shuffle(pontos_corte)
    if pontos_corte:
        novo_corte = pontos_corte[0]
        left_subset, right_subset = split_instances(instances, feature_atual, novo_corte)
        if len(left_subset) >= lim_min and len(right_subset) >= lim_min:
            return feature_atual, novo_corte

    # 2) Tenta mudar feature e ponto de corte juntos (1 tentativa)
    features_lista = list(features.items())
    random.shuffle(features_lista)
    for fname, feature in features_lista:
        if fname == feature_atual:
            continue
        pontos_corte = list(feature.pontos_de_corte)
        if pontos_corte:
            cutoff = random.choice(pontos_corte)
            left_subset, right_subset = split_instances(instances, fname, cutoff)
            if len(left_subset) >= lim_min and len(right_subset) >= lim_min:
                return fname, cutoff

    # 3) Não conseguiu, retorna None para virar folha
    return None, None

def generate_initial_population(flag, pop_size, max_nodes, features, algorithms, instances, min_instances_per_leafnode, node_is_leaf):
    population = []

    for _ in range(pop_size):
        individual = {
            "featureNode": [None] * max_nodes,
            "cutoff_pointNode": [None] * max_nodes,
            "leafNode": [0] * max_nodes,
            "algorithmLeafNode": [None] * max_nodes,
            "problemsInLeafNode": {},
            "value_performance_degradation": 0.0,
            "count_low_representation_nodes": 0
        }
        # Usar uma lista de nós válidos para evitar crescimento de subárvore sob folhas
        valid_nodes = [True] * max_nodes  # Indica se o nó pode ser processado
        # RANDOMIZE internal nodes (binary tree traversal)
        for i in range(max_nodes):
            if not valid_nodes[i]:
                continue  # pula nós que estão sob uma folha

            # Se o pai deste nó for uma folha, este nó deve ser ignorado
            if i != 0:  # exceto para a raiz
                pai = (i - 1) // 2
                if individual["leafNode"][pai] == 1:
                    continue
            
            is_last_level = (2 * i + 1 >= max_nodes)

            # ===== NÓ RAIZ COM AVALIAÇÃO DE 5 FEATURES E 5 PONTOS DE CORTE =====
            if i == 0:
                candidatos = []
                for _ in range(5):
                    fname, feature = random.choice(list(features.items()))
                    cutoff = random.choice(list(feature.pontos_de_corte))
                    
                    # Divide instâncias com o par (fname, cutoff)
                    subtrees = split_instances(instances, fname, cutoff)
                    
                    # Calcula os dois objetivos
                    total_degradacao = 0
                    total_penalidade = 0
                    
                    for subset in subtrees:
                        if subset:
                            # Melhor algoritmo para o subset (mínimo da soma das degradações)
                            best = min(
                                (sum(inst.degradacoes[alg] for inst in subset), alg)
                                for alg in algorithms
                            )
                            total_degradacao += best[0]
                            
                            # Penalidade para baixa representação
                            tamanho = len(subset)
                            if tamanho < min_instances_per_leafnode and tamanho > 0:
                                penalidade = max(1, ((min_instances_per_leafnode - tamanho)**2) / min_instances_per_leafnode)
                            else:
                                penalidade = 0
                            total_penalidade += penalidade
                    
                    candidatos.append({
                        "feature": fname,
                        "cutoff": cutoff,
                        "degradacao": total_degradacao,
                        "penalidade": total_penalidade
                    })
                
                # Função para verificar se a solução a domina b
                def domina(a, b):
                    return (a["degradacao"] <= b["degradacao"] and a["penalidade"] <= b["penalidade"]) and \
                        (a["degradacao"] < b["degradacao"] or a["penalidade"] < b["penalidade"])
                
                # Encontra os não dominados
                nao_dominados = []
                for c in candidatos:
                    if not any(domina(outro, c) for outro in candidatos if outro != c):
                        nao_dominados.append(c)
                
                # Escolhe aleatoriamente entre os não dominados, ou se vazio, aleatório entre todos
                if nao_dominados:
                    escolhido = random.choice(nao_dominados)
                else:
                    escolhido = random.choice(candidatos)
                
                #print(f"candidatos = {candidatos}")
                individual["featureNode"][i] = escolhido["feature"]
                individual["cutoff_pointNode"][i] = escolhido["cutoff"]
                #print(f"Escolhido f {escolhido['feature']} e pc {escolhido['cutoff']}")
                continue


            # Randomly decide if this node will be a leaf, except for the root node (node 0)
            if i != 0 and (random.random() < node_is_leaf or is_last_level):
                individual["leafNode"][i] = 1

                # Invalida toda a subárvore a partir dos filhos
                invalidate_subtree(2 * i + 1, max_nodes, valid_nodes, individual)
                invalidate_subtree(2 * i + 2, max_nodes, valid_nodes, individual)
                
                # Determina as instâncias que chegaram até este nó
                insts_ate_aqui = get_instances_for_leaf(
                    i,
                    individual["featureNode"],
                    individual["cutoff_pointNode"],
                    instances
                )

                individual["problemsInLeafNode"][i] = [inst.nome for inst in insts_ate_aqui]

                if insts_ate_aqui:
                    min_total = float("inf")
                    best_alg = None
                    for alg_name in algorithms:
                        soma = sum(inst.degradacoes[alg_name] for inst in insts_ate_aqui)
                        if soma < min_total:
                            min_total = soma
                            best_alg = alg_name
                    individual["algorithmLeafNode"][i] = best_alg
                    if len(insts_ate_aqui) < min_instances_per_leafnode and len(insts_ate_aqui) > 0:
                        penalty = max(1, ((min_instances_per_leafnode - len(insts_ate_aqui))**2)/min_instances_per_leafnode)
                        individual["count_low_representation_nodes"] += penalty
                    individual["value_performance_degradation"] += min_total
                else:
                    individual["algorithmLeafNode"][i] = None

                continue

            # Randomly select feature and cutoff
            # Determina as instâncias que chegaram até este nó
            insts_no_no = get_instances_for_leaf(
                i,
                individual["featureNode"],
                individual["cutoff_pointNode"],
                instances
            )
            fname, feature = random.choice(list(features.items()))
            cutoffs = list(feature.pontos_de_corte)
            cutoff = random.choice(cutoffs)
            left_subset, right_subset = split_instances(insts_no_no, fname, cutoff)
            #print(f"instances = {len(instances)}")
            lim_min = min(10, (0.1 * len(instances))) if len(insts_no_no) > 0 else 0
            #print(f"lim_min = {lim_min}")
            if len(left_subset) < lim_min or len(right_subset) < lim_min:
                fname_corrigido, cutoff_corrigido = tentar_corrigir_divisao(instances, insts_no_no, features, fname, cutoff, min_instances_per_leafnode)

                if fname_corrigido is None:
                    # vira folha
                    individual["leafNode"][i] = 1
                    insts_ate_aqui = insts_no_no
                    individual["problemsInLeafNode"][i] = [inst.nome for inst in insts_ate_aqui]

                    if insts_ate_aqui:
                        min_total = float("inf")
                        best_alg = None
                        for alg_name in algorithms:
                            soma = sum(inst.degradacoes[alg_name] for inst in insts_ate_aqui)
                            if soma < min_total:
                                min_total = soma
                                best_alg = alg_name
                        individual["algorithmLeafNode"][i] = best_alg
                        if len(insts_ate_aqui) < min_instances_per_leafnode and len(insts_ate_aqui) > 0:
                            penalty = max(1, ((min_instances_per_leafnode - len(insts_ate_aqui)) ** 2) / min_instances_per_leafnode)
                            individual["count_low_representation_nodes"] += penalty
                        individual["value_performance_degradation"] += min_total
                    else:
                        individual["algorithmLeafNode"][i] = None

                    invalidate_subtree(2 * i + 1, max_nodes, valid_nodes, individual)
                    invalidate_subtree(2 * i + 2, max_nodes, valid_nodes, individual)
                    continue
                else:
                    fname, cutoff = fname_corrigido, cutoff_corrigido

            individual["featureNode"][i] = fname
            individual["cutoff_pointNode"][i] = cutoff
        population.append(individual)
        
        
    if flag == 0:   
        print(":::::::::População inicial:::::::::")
        for i in range(pop_size):
            print(f" ind {i} has perfdegrad = {population[i]['value_performance_degradation']:.2f} and nodeLowRepres = {population[i]['count_low_representation_nodes']}")
            for z in range(max_nodes):
                if population[i]["leafNode"][z] == 1 and len(population[i]["problemsInLeafNode"][z]) > 0:
                    print(len(population[i]['problemsInLeafNode'][z]), sep=" ", end = " ")
            print("")
        print("------------------------------")
        """for i in range(pop_size):
            if population[i]["value_performance_degradation"] < 48000:
                print(f" ind {i} has pd = {population[i]['value_performance_degradation']} and nodeBr = {population[i]['count_low_representation_nodes']}")
                for z in range(max_nodes):
                    if population[i]["leafNode"][z] == 1 and len(population[i]["problemsInLeafNode"][z]) > 0:
                        print(f"Leaf node {z} has {len(population[i]['problemsInLeafNode'][z])}")
        """

    return population