"""
Solver para Equações Diferenciais Estocásticas (EDE)
Autor: Luiz Tiago Wilcke

Resolve EDEs usando métodos de Euler-Maruyama, Milstein, e Runge-Kutta estocástico.
"""

import numpy as np


class SolverEDE:
    """
    Resolve EDEs da forma:
    dX(t) = a(t, X) dt + b(t, X) dW(t)
    
    Onde:
    - a(t, X): coeficiente de drift
    - b(t, X): coeficiente de difusão
    - dW(t): incremento de Wiener
    """
    
    def __init__(self, drift, difusao, x0, t0=0, T=1, dt=0.01):
        """
        Parâmetros:
        -----------
        drift : callable
            Função a(t, x) do drift
        difusao : callable
            Função b(t, x) da difusão
        x0 : float ou array
            Condição inicial
        t0 : float
            Tempo inicial
        T : float
            Tempo final
        dt : float
            Passo de tempo
        """
        self.drift = drift
        self.difusao = difusao
        self.x0 = np.array(x0) if np.isscalar(x0) else x0
        self.t0 = t0
        self.T = T
        self.dt = dt
        self.n_passos = int((T - t0) / dt)
        
    def resolver_euler_maruyama(self, n_trajetorias=1):
        """
        Método de Euler-Maruyama (ordem fraca 1, ordem forte 0.5)
        
        X_{n+1} = X_n + a(t_n, X_n) Δt + b(t_n, X_n) ΔW_n
        """
        dim = self.x0.shape[0] if self.x0.ndim > 0 else 1
        
        trajetorias = np.zeros((n_trajetorias, self.n_passos + 1, dim))
        tempos = np.linspace(self.t0, self.T, self.n_passos + 1)
        
        for i in range(n_trajetorias):
            X = self.x0.copy()
            trajetorias[i, 0] = X
            
            for j in range(self.n_passos):
                t = tempos[j]
                
                # Incremento de Wiener
                dW = np.random.randn(dim) * np.sqrt(self.dt)
                
                # Atualização de Euler-Maruyama
                drift_term = self.drift(t, X) * self.dt
                difusao_term = self.difusao(t, X) * dW
                
                X = X + drift_term + difusao_term
                trajetorias[i, j+1] = X
        
        return trajetorias.squeeze(), tempos
    
    def resolver_milstein(self, n_trajetorias=1):
        """
        Método de Milstein (ordem forte 1)
        
        Mais preciso que Euler-Maruyama para difusão não-linear
        """
        dim = self.x0.shape[0] if self.x0.ndim > 0 else 1
        
        trajetorias = np.zeros((n_trajetorias, self.n_passos + 1, dim))
        tempos = np.linspace(self.t0, self.T, self.n_passos + 1)
        
        epsilon = 1e-6  # Para derivadas numéricas
        
        for i in range(n_trajetorias):
            X = self.x0.copy()
            trajetorias[i, 0] = X
            
            for j in range(self.n_passos):
                t = tempos[j]
                dW = np.random.randn(dim) * np.sqrt(self.dt)
                
                # Termos do drift e difusão
                a = self.drift(t, X)
                b = self.difusao(t, X)
                
                # Derivada parcial de b em relação a x (numérica)
                b_plus = self.difusao(t, X + epsilon)
                db_dx = (b_plus - b) / epsilon
                
                # Atualização de Milstein
                X = X + a * self.dt + b * dW + 0.5 * b * db_dx * (dW**2 - self.dt)
                
                trajetorias[i, j+1] = X
        
        return trajetorias.squeeze(), tempos
    
    def resolver_rk_estocastico(self, n_trajetorias=1):
        """
        Método de Runge-Kutta estocástico (ordem 1.5)
        """
        dim = self.x0.shape[0] if self.x0.ndim > 0 else 1
        
        trajetorias = np.zeros((n_trajetorias, self.n_passos + 1, dim))
        tempos = np.linspace(self.t0, self.T, self.n_passos + 1)
        
        for i in range(n_trajetorias):
            X = self.x0.copy()
            trajetorias[i, 0] = X
            
            for j in range(self.n_passos):
                t = tempos[j]
                dW = np.random.randn(dim) * np.sqrt(self.dt)
                
                # Valores auxiliares
                a_n = self.drift(t, X)
                b_n = self.difusao(t, X)
                
                X_til = X + a_n * self.dt + b_n * np.sqrt(self.dt)
                b_til = self.difusao(t + self.dt, X_til)
                
                # Atualização RK
                X = X + a_n * self.dt + 0.5 * (b_n + b_til) * dW + \
                    0.5 * (b_til - b_n) * ((dW**2 - self.dt) / np.sqrt(self.dt))
                
                trajetorias[i, j+1] = X
        
        return trajetorias.squeeze(), tempos
    
    def simular_ornstein_uhlenbeck(self, theta, mu, sigma, n_trajetorias=1):
        """
        Resolve o processo de Ornstein-Uhlenbeck:
        dX = θ(μ - X) dt + σ dW
        
        Tem solução analítica exata
        """
        tempos = np.linspace(self.t0, self.T, self.n_passos + 1)
        trajetorias = np.zeros((n_trajetorias, len(tempos)))
        
        for i in range(n_trajetorias):
            X = self.x0
            trajetorias[i, 0] = X
            
            for j in range(1, len(tempos)):
                dt = tempos[j] - tempos[j-1]
                
                # Solução exata
                media = X * np.exp(-theta * dt) + mu * (1 - np.exp(-theta * dt))
                variancia = (sigma**2 / (2 * theta)) * (1 - np.exp(-2 * theta * dt))
                
                X = np.random.normal(media, np.sqrt(variancia))
                trajetorias[i, j] = X
        
        return trajetorias, tempos
    
    def simular_cev(self, mu, sigma, beta, n_trajetorias=1):
        """
        Processo CEV (Constant Elasticity of Variance):
        dS = μS dt + σS^β dW
        
        Usado para modelar volatilidade dependente do nível de preços
        """
        drift_cev = lambda t, S: mu * S
        difusao_cev = lambda t, S: sigma * np.abs(S)**beta
        
        solver_temp = SolverEDE(drift_cev, difusao_cev, self.x0, 
                                self.t0, self.T, self.dt)
        
        return solver_temp.resolver_milstein(n_trajetorias)
    
    def simular_cir(self, kappa, theta, sigma, n_trajetorias=1):
        """
        Processo CIR (Cox-Ingersoll-Ross):
        dr = κ(θ - r) dt + σ√r dW
        
        Usado para taxas de juros (sempre positivo)
        """
        drift_cir = lambda t, r: kappa * (theta - r)
        difusao_cir = lambda t, r: sigma * np.sqrt(np.maximum(r, 0))
        
        solver_temp = SolverEDE(drift_cir, difusao_cir, self.x0, 
                                self.t0, self.T, self.dt)
        
        return solver_temp.resolver_milstein(n_trajetorias)
