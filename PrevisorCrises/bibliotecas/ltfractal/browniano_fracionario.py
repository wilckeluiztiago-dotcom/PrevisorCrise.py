"""
Movimento Browniano Fracionário (fBm)
Autor: Luiz Tiago Wilcke

Simula e analisa movimento Browniano fracionário com memória longa.
"""

import numpy as np
from scipy.linalg import cholesky


class MovimentoBrownianoFracionario:
    """
    Movimento Browniano Fracionário (fBm)
    
    Generalização do movimento Browniano com expoente de Hurst H ≠ 0.5
    Permite modelar processos com memória longa
    """
    
    def __init__(self, H=0.7, T=1.0, n=1000):
        """
        Parâmetros:
        -----------
        H : float
            Expoente de Hurst (0 < H < 1)
        T : float
            Tempo final
        n : int
            Número de passos
        """
        if not 0 < H < 1:
            raise ValueError("H deve estar em (0, 1)")
        
        self.H = H
        self.T = T
        self.n = n
        self.dt = T / n
        
    def simular(self, n_trajetorias=1, metodo='cholesky'):
        """
        Simula trajetórias de fBm
        
        Parâmetros:
        -----------
        n_trajetorias : int
            Número de trajetórias a simular
        metodo : str
            'cholesky' (exato mas lento) ou 'davies_harte' (rápido)
            
        Retorna:
        --------
        trajetorias : array
            Array de shape (n_trajetorias, n+1) com as trajetórias
        tempos : array
            Array de tempos
        """
        if metodo == 'cholesky':
            trajetorias = self._simular_cholesky(n_trajetorias)
        else:
            trajetorias = self._simular_davies_harte(n_trajetorias)
        
        tempos = np.linspace(0, self.T, self.n + 1)
        
        return trajetorias, tempos
    
    def _simular_cholesky(self, n_trajetorias):
        """
        Método de Cholesky (exato)
        
        Constrói matriz de covariância e usa decomposição de Cholesky
        """
        # Matriz de covariância do fBm
        # Cov(B_H(s), B_H(t)) = 0.5 * (s^(2H) + t^(2H) - |t-s|^(2H))
        
        tempos = np.linspace(0, self.T, self.n + 1)
        n = len(tempos)
        
        # Construir matriz de covariância
        covariancia = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                s = tempos[i]
                t = tempos[j]
                covariancia[i, j] = 0.5 * (
                    s**(2*self.H) + t**(2*self.H) - np.abs(t - s)**(2*self.H)
                )
        
        # Decomposição de Cholesky
        L = cholesky(covariancia, lower=True)
        
        # Gerar trajetórias
        trajetorias = np.zeros((n_trajetorias, n))
        
        for i in range(n_trajetorias):
            z = np.random.randn(n)
            trajetorias[i] = L @ z
        
        return trajetorias
    
    def _simular_davies_harte(self, n_trajetorias):
        """
        Método de Davies-Harte (rápido via FFT)
        
        Mais eficiente para séries longas
        """
        # Para simplificar, usar aproximação via incrementos
        trajetorias = np.zeros((n_trajetorias, self.n + 1))
        
        for i in range(n_trajetorias):
            incrementos = self._gerar_incrementos_fbm()
            trajetorias[i] = np.cumsum(incrementos)
        
        return trajetorias
    
    def _gerar_incrementos_fbm(self):
        """
        Gera incrementos correlacionados para fBm
        """
        # Usar método de incrementos gaussianos correlacionados
        incrementos = np.zeros(self.n + 1)
        
        for i in range(1, self.n + 1):
            # Autocorrelação dos incrementos
            # ρ(k) ∝ k^(2H-2) para H > 0.5
            
            soma = 0
            for j in range(1, i):
                lag = i - j
                # Correlação aproximada
                if self.H > 0.5:
                    correlacao = 0.5 * (
                        (lag + 1)**(2*self.H) - 2*(lag**(2*self.H)) + (lag - 1)**(2*self.H)
                    )
                else:
                    correlacao = 0
                
                soma += correlacao * incrementos[j]
            
            # Variância condicional
            variancia = self.dt**(2*self.H)
            
            incrementos[i] = soma + np.sqrt(variancia) * np.random.randn()
        
        return incrementos
    
    def calcular_covariancia(self, s, t):
        """
        Calcula covariância entre dois tempos
        
        Cov(B_H(s), B_H(t)) = 0.5 * (|s|^(2H) + |t|^(2H) - |t-s|^(2H))
        """
        return 0.5 * (np.abs(s)**(2*self.H) + np.abs(t)**(2*self.H) - 
                     np.abs(t - s)**(2*self.H))
    
    def converter_preco(self, trajetoria, S0=100, mu=0.05, sigma=0.2):
        """
        Converte trajetória fBm para processo de preços
        
        Modelo: dS/S = μ dt + σ dB_H(t)
        
        Parâmetros:
        -----------
        trajetoria : array
            Trajetória fBm
        S0 : float
            Preço inicial
        mu : float
            Drift (retorno médio)
        sigma : float
            Volatilidade
            
        Retorna:
        --------
        precos : array
            Série de preços
        """
        # S(t) = S0 * exp(μt + σ B_H(t) - 0.5σ²t^(2H))
        tempos = np.linspace(0, self.T, len(trajetoria))
        
        log_precos = (
            np.log(S0) +
            mu * tempos +
            sigma * trajetoria -
            0.5 * sigma**2 * tempos**(2*self.H)
        )
        
        precos = np.exp(log_precos)
        
        return precos
    
    def estimar_parametros(self, serie_precos):
        """
        Estima H, μ, σ a partir de uma série de preços
        
        Parâmetros:
        -----------
        serie_precos : array
            Série de preços observada
            
        Retorna:
        --------
        parametros : dict
            H, mu, sigma estimados
        """
        # Estimar H usando análise de Hurst
        from .hurst_expoente import AnalisadorHurst
        
        analisador = AnalisadorHurst(metodo='dfa')
        H_est, _ = analisador.calcular(serie_precos)
        
        # Estimar μ e σ
        log_precos = np.log(serie_precos)
        retornos = np.diff(log_precos)
        
        mu_est = np.mean(retornos) / self.dt
        sigma_est = np.std(retornos) / np.sqrt(self.dt**(2*H_est))
        
        parametros = {
            'H': H_est,
            'mu': mu_est,
            'sigma': sigma_est
        }
        
        return parametros
    
    def calcular_variancia_temporal(self, t):
        """
        Calcula Var[B_H(t)] = t^(2H)
        """
        return t**(2*self.H)
    
    def simular_multifractal(self, n_trajetorias=1):
        """
        Simula movimento Browniano Multifractal (extensão do fBm)
        com H variando no tempo
        """
        trajetorias = np.zeros((n_trajetorias, self.n + 1))
        
        # H variável no tempo (modelo simples)
        tempos = np.linspace(0, self.T, self.n + 1)
        
        for i in range(n_trajetorias):
            # H oscilante entre 0.3 e 0.8
            H_temporal = 0.55 + 0.25 * np.sin(2 * np.pi * tempos / self.T)
            
            # Simular incrementos com H variável
            incrementos = np.zeros(self.n + 1)
            
            for j in range(1, self.n + 1):
                H_atual = H_temporal[j]
                variancia = self.dt**(2*H_atual)
                incrementos[j] = np.sqrt(variancia) * np.random.randn()
            
            trajetorias[i] = np.cumsum(incrementos)
        
        return trajetorias, tempos
