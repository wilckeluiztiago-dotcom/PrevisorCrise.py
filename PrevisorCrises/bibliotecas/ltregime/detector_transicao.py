"""
Detector de Transições de Regime
Autor: Luiz Tiago Wilcke

Detecta mudanças abruptas de regime usando CUSUM bayesiano.
"""

import numpy as np


class DetectorTransicao:
    """Detecta transições abruptas entre regimes"""
    
    def __init__(self, threshold=5.0):
        self.threshold = threshold
    
    def detectar(self, serie, janela=20):
        """
        Detecta pontos de mudança de regime
        
        Retorna:
        --------
        pontos_mudanca : list
            Índices onde ocorreram transições
        """
        n = len(serie)
        cusum = np.zeros(n)
        
        media_global = np.mean(serie)
        
        for i in range(1, n):
            cusum[i] = max(0, cusum[i-1] + (serie[i] - media_global))
        
        pontos_mudanca = np.where(cusum > self.threshold)[0].tolist()
        
        return pontos_mudanca
