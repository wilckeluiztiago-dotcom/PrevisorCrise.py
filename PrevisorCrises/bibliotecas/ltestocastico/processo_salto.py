"""
Processos de Salto (Jump Processes)
Autor: Luiz Tiago Wilcke

Implementa processos de Poisson composto para modelar eventos extremos (crashes).
"""

import numpy as np
from scipy.stats import poisson, gamma, norm, expon


class ProcessoSalto:
    """
    Processos de Salto para modelar eventos extremos
    
    Combina difusão contínua com saltos discretos:
    dS = μS dt + σS dW + S dJ
    
    Onde dJ é um processo de Poisson composto
    """
    
    def __init__(self, S0, lambda_salto, distribuicao_salto='normal'):
        """
        Parâmetros:
        -----------
        S0 : float
            Valor inicial
        lambda_salto : float
            Intensidade dos saltos (saltos por unidade de tempo)
        distribuicao_salto : str
            'normal', 'exponencial', ou 'gamma'
        """
        self.S0 = S0
        self.lambda_salto = lambda_salto
        self.distribuicao_salto = distribuicao_salto
        
    def simular_merton_jump_diffusion(self, mu, sigma, mu_J, sigma_J, T=1, n_passos=1000, n_trajetorias=1):
        """
        Modelo de Merton Jump-Diffusion
        
        dS/S = μ dt + σ dW + dJ
        
        Onde J tem saltos normalmente distribuídos
        """
        dt = T / n_passos
        tempos = np.linspace(0, T, n_passos + 1)
        
        trajetorias = np.zeros((n_trajetorias, n_passos + 1))
        
        for i in range(n_trajetorias):
            S = self.S0
            trajetorias[i, 0] = S
            
            for j in range(1, n_passos + 1):
                # Parte difusão
                dW = np.random.randn() * np.sqrt(dt)
                difusao = mu * dt + sigma * dW
                
                # Parte salto (Poisson composto)
                n_saltos = poisson.rvs(self.lambda_salto * dt)
                
                if n_saltos > 0:
                    # Tamanho dos saltos
                    saltos = np.random.normal(mu_J, sigma_J, n_saltos)
                    salto_total = np.sum(saltos)
                else:
                    salto_total = 0
                
                # Atualização
                S = S * np.exp(difusao + salto_total)
                trajetorias[i, j] = S
        
        return trajetorias, tempos
    
    def simular_kou_jump_diffusion(self, mu, sigma, p, eta1, eta2, T=1, n_passos=1000, n_trajetorias=1):
        """
        Modelo de Kou (assimetria nos saltos)
        
        Saltos positivos e negativos com distribuições exponenciais diferentes
        """
        dt = T / n_passos
        tempos = np.linspace(0, T, n_passos + 1)
        
        trajetorias = np.zeros((n_trajetorias, n_passos + 1))
        
        for i in range(n_trajetorias):
            S = self.S0
            trajetorias[i, 0] = S
            
            for j in range(1, n_passos + 1):
                # Difusão
                dW = np.random.randn() * np.sqrt(dt)
                difusao = mu * dt + sigma * dW
                
                # Saltos
                n_saltos = poisson.rvs(self.lambda_salto * dt)
                
                salto_total = 0
                for _ in range(n_saltos):
                    # Salto positivo ou negativo
                    if np.random.rand() < p:
                        # Salto positivo (exponencial com taxa eta1)
                        salto = expon.rvs(scale=1/eta1)
                    else:
                        # Salto negativo (exponencial com taxa eta2)
                        salto = -expon.rvs(scale=1/eta2)
                    
                    salto_total += salto
                
                # Atualização
                S = S * np.exp(difusao + salto_total)
                trajetorias[i, j] = S
        
        return trajetorias, tempos
    
    def detectar_saltos(self, serie_precos, threshold=3.0):
        """
        Detecta saltos em uma série de preços observada
        
        Parâmetros:
        -----------
        serie_precos : array
            Série de preços
        threshold : float
            Threshold em desvios padrão para considerar um salto
            
        Retorna:
        --------
        indices_saltos : array
            Índices onde ocorreram saltos
        tamanhos_saltos : array
            Tamanho dos saltos detectados
        """
        # Retornos logarítmicos
        log_retornos = np.diff(np.log(serie_precos))
        
        # Estatísticas robustas
        mediana = np.median(log_retornos)
        mad = np.median(np.abs(log_retornos - mediana))
        
        # MAD robusto
        desvio_robusto = 1.4826 * mad
        
        # Detectar outliers
        z_scores = np.abs(log_retornos - mediana) / desvio_robusto
        
        indices_saltos = np.where(z_scores > threshold)[0]
        tamanhos_saltos = log_retornos[indices_saltos]
        
        return indices_saltos, tamanhos_saltos
    
    def estimar_intensidade(self, serie_precos, threshold=3.0):
        """
        Estima a intensidade λ dos saltos a partir de dados
        """
        indices_saltos, _ = self.detectar_saltos(serie_precos, threshold)
        
        n_saltos = len(indices_saltos)
        n_periodos = len(serie_precos) - 1
        
        lambda_estimado = n_saltos / n_periodos
        
        return lambda_estimado
    
    def calcular_variancia_total(self, sigma, mu_J, sigma_J, T):
        """
        Calcula variância total (difusão + saltos)
        
        Var[log(S_T)] = σ²T + λT(μ_J² + σ_J²)
        """
        var_difusao = sigma**2 * T
        var_saltos = self.lambda_salto * T * (mu_J**2 + sigma_J**2)
        
        return var_difusao + var_saltos
    
    def simular_processo_hawkes(self, alpha, beta, T=1, n_simulacoes=1):
        """
        Processo de Hawkes (auto-excitante)
        
        λ(t) = λ_0 + Σ α exp(-β(t - t_i))
        
        Saltos geram mais saltos (clustering de volatilidade)
        """
        trajetorias = []
        
        for _ in range(n_simulacoes):
            tempos_saltos = []
            t = 0
            lambda_t = self.lambda_salto
            
            while t < T:
                # Próximo evento
                u = np.random.rand()
                t = t - np.log(u) / lambda_t
                
                if t >= T:
                    break
                
                # Aceitar/rejeitar
                lambda_atual = self.lambda_salto + sum(
                    alpha * np.exp(-beta * (t - s)) for s in tempos_saltos if s < t
                )
                
                if np.random.rand() * lambda_t <= lambda_atual:
                    tempos_saltos.append(t)
                
                lambda_t = lambda_atual
            
            trajetorias.append(np.array(tempos_saltos))
        
        return trajetorias
