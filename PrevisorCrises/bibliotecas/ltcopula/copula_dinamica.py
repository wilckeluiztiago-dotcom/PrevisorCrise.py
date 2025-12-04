"""
Copula Dinâmica (DCC)
Autor: Luiz Tiago Wilcke
"""

import numpy as np


class CopulaDinamica:
    """Copula com correlações variando no tempo (DCC-GARCH)"""
    
    def __init__(self, dimensao=2):
        self.dimensao = dimensao
        self.correlacoes_temporais = None
    
    def ajustar(self, dados, janela=60):
        """Ajusta copula dinâmica com janela móvel"""
        n, d = dados.shape
        n_janelas = n - janela + 1
        
        correlacoes = np.zeros((n_janelas, d, d))
        
        for i in range(n_janelas):
            segmento = dados[i:i+janela]
            correlacoes[i] = np.corrcoef(segmento.T)
        
        self.correlacoes_temporais = correlacoes
        
        return correlacoes
    
    def prever_correlacao(self, horizonte=1):
        """Prevê correlação futura usando EWMA"""
        if self.correlacoes_temporais is None:
            raise ValueError("Copula não ajustada")
        
        # Última correlação
        corr_atual = self.correlacoes_temporais[-1]
        
        # Média de longo prazo
        corr_media = np.mean(self.correlacoes_temporais, axis=0)
        
        # EWMA
        lambda_decay = 0.94
        corr_prevista = lambda_decay * corr_atual + (1 - lambda_decay) * corr_media
        
        return corr_prevista
