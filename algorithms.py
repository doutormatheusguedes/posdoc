class Algorithm:
    def __init__(self, nome):
        self.nome = nome
        self.degradacao_total = 0.0
        self.num_melhor = 0
        self.num_pior = 0
        self.total_instancias = 0

    def getNome(self):
        return self.nome
    
    def performanceDegradationAllProblems(self):
        return self.degradacao_total

    def atualizar(self, instancias):
        """
        Atualiza as métricas do algoritmo com base numa lista de instâncias.
        
        :param instancias: lista de Instancia
        """
        self.degradacao_total = 0.0
        self.num_melhor = 0
        self.num_pior = 0
        self.total_instancias = len(instancias)

        for inst in instancias:
            degr = inst.degradacoes.get(self.nome, None)
            if degr is None:
                continue  # algoritmo não rodou nessa instância
            
            self.degradacao_total += degr

            # Verifica se é melhor ou pior para essa instância
            min_degr = min(inst.degradacoes.values())
            max_degr = max(inst.degradacoes.values())

            if degr == min_degr:
                self.num_melhor += 1
            if degr == max_degr:
                self.num_pior += 1

    def porcentagem_melhor(self):
        if self.total_instancias == 0:
            return 0.0
        return self.num_melhor / self.total_instancias * 100

    def porcentagem_pior(self):
        if self.total_instancias == 0:
            return 0.0
        return self.num_pior / self.total_instancias * 100

    def __repr__(self):
        return (f"Algorithm(nome={self.nome}, degradacao_total={self.degradacao_total:.3f}, "
                f"num_melhor={self.num_melhor}, pct_melhor={self.porcentagem_melhor():.1f}%, "
                f"num_pior={self.num_pior}, pct_pior={self.porcentagem_pior():.1f}%)")
    
    