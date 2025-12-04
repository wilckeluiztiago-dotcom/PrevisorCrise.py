"""
Modelos Markov Switching
Autor: Luiz Tiago Wilcke

Implementa modelos com mudança de regime onde parâmetros dependem de uma cadeia de Markov oculta.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize


class ModeloMarkovSwitching:
    """
    Modelo de Mudança de Regime Markoviano
    
    y_t | S_t=j ~ N(μ_j, σ_j²)
    P(S_t=j | S_{t-1}=i) = p_{ij}
    """
    
    def __init__(self, n_regimes=2):
        """
        Parâmetros:
        -----------
        n_regimes : int
            Número de regimes (estados)
        """
        self.n_regimes = n_regimes
        self.parametros = None
        self.probabilidades_filtradas = None
        self.probabilidades_suavizadas = None
        
    def ajustar(self, dados, max_iter=100, tol=1e-4):
        """
        Ajusta o modelo usando algoritmo EM (Expectation-Maximization)
        
        Parâmetros:
        -----------
        dados : array
            Série temporal observada
        max_iter : int
            Máximo de iterações EM
        tol : float
            Tolerância para convergência
            
        Retorna:
        --------
        parametros : dict
            Parâmetros estimados
        """
        n = len(dados)
        k = self.n_regimes
        
        # Inicialização
        medias = np.percentile(dados, np.linspace(10, 90, k))
        variancias = np.ones(k) * np.var(dados)
        
        # Matriz de transição inicial (simétrica)
        P = np.ones((k, k)) / k
        
        # Probabilidades iniciais
        pi = np.ones(k) / k
        
        log_vero_anterior = -np.inf
        
        for iteracao in range(max_iter):
            # E-step: Forward-Backward algorithm
            # Forward pass
            alpha = np.zeros((n, k))
            
            # Inicialização
            for j in range(k):
                alpha[0, j] = pi[j] * norm.pdf(dados[0], medias[j], np.sqrt(variancias[j]))
            
            alpha[0] = alpha[0] / np.sum(alpha[0])
            
            # Recursão forward
            for t in range(1, n):
                for j in range(k):
                    soma = 0
                    for i in range(k):
                        soma += alpha[t-1, i] * P[i, j]
                    
                    alpha[t, j] = soma * norm.pdf(dados[t], medias[j], np.sqrt(variancias[j]))
                
                # Normalização
                alpha[t] = alpha[t] / (np.sum(alpha[t]) + 1e-10)
            
            # Backward pass
            beta = np.zeros((n, k))
            beta[n-1] = np.ones(k)
            
            for t in range(n-2, -1, -1):
                for i in range(k):
                    soma = 0
                    for j in range(k):
                        soma += P[i, j] * norm.pdf(dados[t+1], medias[j], np.sqrt(variancias[j])) * beta[t+1, j]
                    
                    beta[t, i] = soma
                
                # Normalização
                beta[t] = beta[t] / (np.sum(beta[t]) + 1e-10)
            
            # Probabilidades suavizadas
            gamma = alpha * beta
            gamma = gamma / (np.sum(gamma, axis=1, keepdims=True) + 1e-10)
            
            # Probabilidades de transição
            xi = np.zeros((n-1, k, k))
            
            for t in range(n-1):
                for i in range(k):
                    for j in range(k):
                        numerador = alpha[t, i] * P[i, j] * \
                                   norm.pdf(dados[t+1], medias[j], np.sqrt(variancias[j])) * beta[t+1, j]
                        
                        denominador = np.sum(alpha[t]) + 1e-10
                        
                        xi[t, i, j] = numerador / denominador
            
            # M-step: Atualizar parâmetros
            # Médias
            for j in range(k):
                medias[j] = np.sum(gamma[:, j] * dados) / np.sum(gamma[:, j])
            
            # Variâncias
            for j in range(k):
                variancias[j] = np.sum(gamma[:, j] * (dados - medias[j])**2) / np.sum(gamma[:, j])
                variancias[j] = max(variancias[j], 1e-6)
            
            # Matriz de transição
            for i in range(k):
                for j in range(k):
                    P[i, j] = np.sum(xi[:, i, j]) / (np.sum(gamma[:-1, i]) + 1e-10)
            
            # Normalizar linhas de P
            P = P / (np.sum(P, axis=1, keepdims=True) + 1e-10)
            
            # Probabilidades iniciais
            pi = gamma[0]
            
            # Calcular log-verossimilhança
            log_vero = 0
            for t in range(n):
                prob_t = 0
                for j in range(k):
                    prob_t += gamma[t, j] * norm.pdf(dados[t], medias[j], np.sqrt(variancias[j]))
                
                log_vero += np.log(prob_t + 1e-10)
            
            # Verificar convergência
            if abs(log_vero - log_vero_anterior) < tol:
                break
            
            log_vero_anterior = log_vero
        
        # Armazenar resultados
        self.parametros = {
            'medias': medias,
            'variancias': variancias,
            'matriz_transicao': P,
            'prob_inicial': pi,
            'log_verossimilhanca': log_vero,
            'n_iteracoes': iteracao + 1
        }
        
        self.probabilidades_filtradas = alpha
        self.probabilidades_suavizadas = gamma
        
        return self.parametros
    
    def prever_regime(self, dados=None, horizonte=1):
        """
        Prevê o regime futuro
        
        Parâmetros:
        -----------
        dados : array, opcional
            Novos dados para atualizar probabilidades
        horizonte : int
            Passos à frente
            
        Retorna:
        --------
        prob_regimes : array
            Probabilidades de cada regime no horizonte
        """
        if self.parametros is None:
            raise ValueError("Modelo não ajustado")
        
        P = self.parametros['matriz_transicao']
        
        # Estado atual (última probabilidade filtrada)
        if dados is not None:
            # Atualizar com novos dados
            prob_atual = self.probabilidades_filtradas[-1]
        else:
            prob_atual = self.probabilidades_filtradas[-1]
        
        # Projetar horizonte passos à frente
        prob_futuro = prob_atual
        for _ in range(horizonte):
            prob_futuro = prob_futuro @ P
        
        return prob_futuro
    
    def classificar_regimes(self):
        """
        Classifica cada regime automaticamente
        
        Retorna:
        --------
        classificacao : dict
            Nome de cada regime
        """
        if self.parametros is None:
            raise ValueError("Modelo não ajustado")
        
        medias = self.parametros['medias']
        variancias = self.parametros['variancias']
        
        # Ordenar por média
        ordem = np.argsort(medias)
        
        classificacao = {}
        
        if self.n_regimes == 2:
            if medias[ordem[0]] < 0:
                classificacao[ordem[0]] = 'CRISE'
                classificacao[ordem[1]] = 'NORMAL'
            else:
                classificacao[ordem[0]] = 'BAIXA_VOLATILIDADE'
                classificacao[ordem[1]] = 'ALTA_VOLATILIDADE'
        
        elif self.n_regimes == 3:
            classificacao[ordem[0]] = 'CRISE'
            classificacao[ordem[1]] = 'NORMAL'
            classificacao[ordem[2]] = 'BOLHA'
        
        else:
            for i, idx in enumerate(ordem):
                classificacao[idx] = f'REGIME_{i+1}'
        
        return classificacao
    
    def duracao_esperada(self):
        """
        Calcula duração média de cada regime
        
        Retorna:
        --------
        duracoes : array
            Duração esperada em cada regime
        """
        if self.parametros is None:
            raise ValueError("Modelo não ajustado")
        
        P = self.parametros['matriz_transicao']
        
        # Duração esperada = 1 / (1 - p_ii)
        duracoes = 1 / (1 - np.diag(P) + 1e-10)
        
        return duracoes
