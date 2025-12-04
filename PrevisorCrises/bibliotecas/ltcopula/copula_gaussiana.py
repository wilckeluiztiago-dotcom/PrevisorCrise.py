"""
Copula Gaussiana
Autor: Luiz Tiago Wilcke
"""

import numpy as np
from scipy.stats import norm, multivariate_normal


class CopulaGaussiana:
    """Copula Gaussiana Multivariada"""
    
    def __init__(self, dimensao=2):
        self.dimensao = dimensao
        self.correlacao = None
    
    def ajustar(self, dados):
        """Ajusta a copula aos dados"""
        # Transformar para uniformes via ranks
        n, d = dados.shape
        U = np.zeros_like(dados)
        
        for i in range(d):
            ranks = dados[:, i].argsort().argsort()
            U[:, i] = (ranks + 1) / (n + 1)
        
        # Transformar para normais
        Z = norm.ppf(U)
        
        # Estimar correlação
        self.correlacao = np.corrcoef(Z.T)
        
        return self.correlacao
    
    def simular(self, n_amostras=1000):
        """Simula da copula"""
        if self.correlacao is None:
            raise ValueError("Copula não ajustada")
        
        # Simular normais correlacionadas
        Z = multivariate_normal.rvs(mean=np.zeros(self.dimensao),
                                     cov=self.correlacao,
                                     size=n_amostras)
        
        # Transformar para uniformes
        U = norm.cdf(Z)
        
        return U
