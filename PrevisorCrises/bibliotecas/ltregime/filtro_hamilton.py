"""
Filtro de Hamilton
Autor: Luiz Tiago Wilcke

Implementa o filtro de Hamilton para inferência de probabilidades de regime.
"""

import numpy as np


class FiltroHamilton:
    """Filtro de Hamilton para Markov Switching Models"""
    
    def __init__(self, modelo_ms):
        """
        Parâmetros:
        -----------
        modelo_ms : ModeloMarkovSwitching
            Modelo MS ajustado
        """
        self.modelo = modelo_ms
    
    def filtrar(self, dados):
        """Aplica filtro de Hamilton aos dados"""
        if self.modelo.parametros is None:
            raise ValueError("Modelo não ajustado")
        
        return self.modelo.probabilidades_filtradas
    
    def suavizar(self, dados):
        """Aplica suavizador (backward pass)"""
        return self.modelo.probabilidades_suavizadas
