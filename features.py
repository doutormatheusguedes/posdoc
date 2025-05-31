class Feature:
    def __init__(self, nome):
        """
        :param nome: str - nome da feature (ex: "f1")
        """
        self.nome = nome
        self.pontos_de_corte = set()

    def adicionar_valor(self, valor):
        """
        Adiciona um novo valor ao conjunto de pontos de corte possíveis.
        :param valor: valor da feature em uma instância
        """
        self.pontos_de_corte.add(valor)

    def get_pontos_de_corte(self):
        """
        Retorna os pontos de corte ordenados.
        """
        return sorted(self.pontos_de_corte)

    def __repr__(self):
        return f"Feature(nome={self.nome}, pontos_de_corte={self.get_pontos_de_corte()})"

