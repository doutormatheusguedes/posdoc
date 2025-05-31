class Instance:
    def __init__(self, nome, features, degradacoes):
        self.nome = nome
        self.features = features  
        self.degradacoes = degradacoes 
        self.melhor_algoritmo = self._calcular_melhor_algoritmo()

    def _calcular_melhor_algoritmo(self):
        # Retorna o algoritmo com a menor degradação
        return min(self.degradacoes, key=self.degradacoes.get)

    def get_valor_feature(self, nome_feature):
        return self.features.get(nome_feature, None)

    def __repr__(self):
        return f"Instancia(nome={self.nome}, melhor_algoritmo={self.melhor_algoritmo})"
