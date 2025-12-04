"""
Copula t-Student
Autor: Luiz Tiago Wilcke
"""

import numpy as np
from scipy.stats import norm, t, multivariate_t


class CopulaT:
    """Copula t-Student (captura dependências de cauda)"""
    
    def __init__(self, dimensao=2, nu=5):
        self.dimensao = dimensao
        self.nu = nu  # graus de liberdade
        self.correlacao = None
    
    def ajustar(self, dados):
        """Ajusta copula t aos dados"""
        n, d = dados.shape
        U = np.zeros_like(dados)
        
        for i in range(d):
            ranks = dados[:, i].argsort().argsort()
            U[:, i] = (ranks + 1) / (n + 1)
        
        # Transformar para t-Student
        X = t.ppf(U, df=self.nu)
        
        # Estimar correlação
        self.correlacao = np.corrcoef(X.T)
        
        return self.correlacao
    
    def simular(self, n_amostras=1000):
        """Simula da copula t"""
        if self.correlacao is None:
            raise ValueError("Copula não ajustada")
        
        # Simular t multivariada
        X = multivariate_t.rvs(shape=self.correlacao,
                               df=self.nu,
                               size=n_amostras)
        
        # Transformar para uniformes
        U = t.cdf(X, df=self.nu)
        
        return U
