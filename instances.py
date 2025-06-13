class Instance:
    def __init__(self, nome, features, degradacoes):
        self.nome = nome
        self.features = features  
        self.degradacoes = degradacoes 
        self.melhor_algoritmo = self._calcular_melhor_algoritmo()
        self.posicoes_algoritmos = self._calcular_posicoes_algoritmos()
        self.features_normalizados = {}



    def _calcular_melhor_algoritmo(self):
        # Retorna o algoritmo com a menor degradação
        return min(self.degradacoes, key=self.degradacoes.get)
    
    def _calcular_posicoes_algoritmos(self):
        # Ordena algoritmos por degradação (menor primeiro)
        algoritmos_ordenados = sorted(self.degradacoes.items(), key=lambda x: x[1])
        # Cria um dict {algoritmo: posição}, posição começa em 1
        return {alg: pos+1 for pos, (alg, _) in enumerate(algoritmos_ordenados)}


    def get_valor_feature(self, nome_feature):
        return self.features.get(nome_feature, None)
    

    def __repr__(self):
        return f"Instancia(nome={self.nome}, melhor_algoritmo={self.melhor_algoritmo})"
