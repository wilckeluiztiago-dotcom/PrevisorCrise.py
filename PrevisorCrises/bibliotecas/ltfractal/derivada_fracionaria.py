"""
Derivadas e Integrais Fracionárias
Autor: Luiz Tiago Wilcke

Implementa operadores de Caputo, Riemann-Liouville e Grünwald-Letnikov
para cálculo fracionário.
"""

import numpy as np
from scipy.special import gamma
from scipy.integrate import quad


class DerivadaFracionaria:
    """
    Calcula derivadas e integrais de ordem fracionária
    
    Suporta diferentes definições:
    - Caputo (útil para condições iniciais)
    - Riemann-Liouville (padrão clássico)
    - Grünwald-Letnikov (implementação numérica direta)
    """
    
    def __init__(self, tipo='caputo'):
        """
        Parâmetros:
        -----------
        tipo : str
            Tipo de derivada: 'caputo', 'riemann', ou 'grunwald'
        """
        if tipo not in ['caputo', 'riemann', 'grunwald']:
            raise ValueError("Tipo deve ser 'caputo', 'riemann' ou 'grunwald'")
        
        self.tipo = tipo
    
    def derivada(self, funcao, ordem, pontos, h=0.01):
        """
        Calcula a derivada fracionária de uma função
        
        Parâmetros:
        -----------
        funcao : callable ou array
            Função f(x) ou array de valores
        ordem : float
            Ordem da derivada (pode ser fracionária, ex: 0.5)
        pontos : array
            Pontos onde calcular a derivada
        h : float
            Passo para diferenciação numérica
            
        Retorna:
        --------
        derivada : array
            Derivada fracionária nos pontos especificados
        """
        if callable(funcao):
            valores = funcao(pontos)
        else:
            valores = funcao
        
        if self.tipo == 'grunwald':
            return self._grunwald_letnikov(valores, ordem, h)
        elif self.tipo == 'caputo':
            return self._caputo(valores, ordem, pontos)
        else:  # riemann
            return self._riemann_liouville(valores, ordem, pontos)
    
    def _grunwald_letnikov(self, valores, ordem, h):
        """
        Implementação de Grünwald-Letnikov
        
        D^α f(x) ≈ h^{-α} Σ_{k=0}^{n} (-1)^k (α choose k) f(x - kh)
        """
        n = len(valores)
        resultado = np.zeros(n)
        
        # Calcular coeficientes binomiais generalizados
        max_k = min(100, n)  # Limitar para eficiência
        coeficientes = self._coeficientes_binomiais_generalizados(ordem, max_k)
        
        for i in range(n):
            soma = 0
            for k in range(min(i + 1, max_k)):
                soma += coeficientes[k] * valores[i - k]
            
            resultado[i] = soma / (h ** ordem)
        
        return resultado
    
    def _caputo(self, valores, ordem, pontos):
        """
        Derivada de Caputo
        
        Mais adequada para problemas com condições iniciais
        """
        n = len(valores)
        resultado = np.zeros(n)
        
        # Ordem inteira superior
        m = int(np.ceil(ordem))
        
        # Derivada de ordem m (usando diferenças finitas)
        derivada_m = valores.copy()
        for _ in range(m):
            derivada_m = np.gradient(derivada_m, pontos)
        
        # Integração fracionária de ordem (m - α)
        beta = m - ordem
        
        for i in range(1, n):
            soma = 0
            for j in range(i):
                dt = pontos[i] - pontos[j]
                peso = (dt ** beta) / gamma(beta + 1)
                soma += peso * derivada_m[j]
            
            resultado[i] = soma
        
        return resultado
    
    def _riemann_liouville(self, valores, ordem, pontos):
        """
        Derivada de Riemann-Liouville
        
        Definição clássica do cálculo fracionário
        """
        n = len(valores)
        resultado = np.zeros(n)
        
        # Ordem inteira superior
        m = int(np.ceil(ordem))
        alpha = ordem
        
        # Integral fracionária de ordem (m - α), depois derivada de ordem m
        beta = m - alpha
        
        # Primeiro: integral fracionária
        integral_frac = np.zeros(n)
        for i in range(1, n):
            soma = 0
            for j in range(i):
                dt = pontos[i] - pontos[j]
                if dt > 0:
                    peso = (dt ** (beta - 1)) / gamma(beta)
                    soma += peso * valores[j] * (pontos[1] - pontos[0])
            
            integral_frac[i] = soma
        
        # Depois: derivada de ordem m
        resultado = integral_frac.copy()
        for _ in range(m):
            resultado = np.gradient(resultado, pontos)
        
        return resultado
    
    def integral_fracionaria(self, valores, ordem, pontos):
        """
        Calcula a integral fracionária (derivada de ordem negativa)
        
        Parâmetros:
        -----------
        valores : array
            Valores da função
        ordem : float
            Ordem da integral (positiva)
        pontos : array
            Pontos de avaliação
            
        Retorna:
        --------
        integral : array
            Integral fracionária
        """
        n = len(valores)
        resultado = np.zeros(n)
        
        for i in range(1, n):
            soma = 0
            for j in range(i):
                dt = pontos[i] - pontos[j]
                if dt > 0:
                    peso = (dt ** (ordem - 1)) / gamma(ordem)
                    soma += peso * valores[j] * (pontos[1] - pontos[0])
            
            resultado[i] = soma
        
        return resultado
    
    def _coeficientes_binomiais_generalizados(self, alpha, n_max):
        """
        Calcula coeficientes binomiais generalizados
        
        c_k = (-1)^k (alpha choose k) = Γ(k-alpha) / (Γ(-alpha) Γ(k+1))
        
        Ou recursivamente: c_0 = 1, c_k = c_{k-1} * (k - 1 - alpha) / k
        """
        coeficientes = np.zeros(n_max)
        coeficientes[0] = 1
        
        for k in range(1, n_max):
            coeficientes[k] = coeficientes[k-1] * (k - 1 - alpha) / k
        
        return coeficientes
    
    def transformada_mittag_leffler(self, z, alpha, beta=1):
        """
        Calcula a função de Mittag-Leffler
        
        E_{α,β}(z) = Σ_{k=0}^∞ z^k / Γ(αk + β)
        
        Fundamental para soluções de EDFs (Equações Diferenciais Fracionárias)
        """
        soma = 0
        k = 0
        termo = 1
        
        while abs(termo) > 1e-10 and k < 100:
            termo = (z ** k) / gamma(alpha * k + beta)
            soma += termo
            k += 1
        
        return soma
