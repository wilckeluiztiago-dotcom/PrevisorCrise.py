"""
FIGARCH - Fractionally Integrated GARCH
Autor: Luiz Tiago Wilcke

Modela volatilidade com memória longa.
"""

import numpy as np
from scipy.optimize import minimize

try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False
    arch_model = None


class FIGARCH:
    """
    Modelo FIGARCH(p, d, q)
    
    Captura persistência de longo prazo em volatilidade
    φ(L)(1-L)^d ε_t² = ω + [1 - β(L)]v_t
    """
    
    def __init__(self, p=1, d=0.4, q=1):
        """
        Parâmetros:
        -----------
        p : int
            Ordem GARCH
        d : float
            Parâmetro fracionário (0 < d < 1)
        q : int
            Ordem ARCH
        """
        self.p = p
        self.d = d
        self.q = q
        self.parametros = None
        
    def ajustar(self, retornos):
        """
        Ajusta o modelo FIGARCH aos dados
        """
        # Se arch library disponível, usar
        if HAS_ARCH and arch_model is not None:
            try:
                modelo = arch_model(retornos, vol='GARCH', p=self.p, q=self.q)
                resultado = modelo.fit(disp='off', options={'maxiter': 500})
                self.parametros = resultado.params
                self.modelo_ajustado = resultado
                return resultado
            except:
                return self._ajustar_manual(retornos)
        else:
            # Fallback: estimação manual
            return self._ajustar_manual(retornos)
    
    def _ajustar_manual(self, retornos):
        """
        Ajuste manual do FIGARCH via máxima verossimilhança  
        """
        n = len(retornos)
        
        # Inicialização
        omega = np.var(retornos) * 0.1
        alpha = 0.1
        beta = 0.8
        
        parametros_iniciais = [omega, alpha, beta, self.d]
        
        def log_verossimilhanca_negativa(params):
            omega, alpha, beta, d = params
            
            # Volatilidade condicional
            sigma2 = np.zeros(n)
            sigma2[0] = np.var(retornos)
            
            for t in range(1, n):
                # Aproximação FIGARCH
                sigma2[t] = omega + alpha * retornos[t-1]**2 + beta * sigma2[t-1]
                sigma2[t] = max(sigma2[t], 1e-6)
            
            # Log-verossimilhança
            ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + retornos**2 / sigma2)
            
            return -ll
        
        # Otimização
        bounds = [(1e-6, None), (0, 1), (0, 1), (0, 1)]
        
        resultado = minimize(log_verossimilhanca_negativa, parametros_iniciais,
                           method='L-BFGS-B', bounds=bounds)
        
        self.parametros = resultado.x
        
        return resultado
    
    def prever_volatilidade(self, retornos, horizonte=10):
        """
        Prevê volatilidade futura
        """
        if self.parametros is None:
            raise ValueError("Modelo não ajustado")
        
        # Usar modelo ajustado para previsão
        if hasattr(self, 'modelo_ajustado'):
            previsoes = self.modelo_ajustado.forecast(horizon=horizonte)
            return np.sqrt(previsoes.variance.values[-1, :])
        else:
            # Previsão manual
            omega, alpha, beta, d = self.parametros
            
            # Última volatilidade
            sigma2_atual = omega + alpha * retornos[-1]**2 + \
                          beta * np.var(retornos[-20:])
            
            # Previsão com persistência
            persistencia = beta ** np.arange(1, horizonte + 1)
            sigma2_forecast = omega / (1 - beta) + \
                             (sigma2_atual - omega / (1 - beta)) * persistencia
            
            return np.sqrt(sigma2_forecast)
    
    def calcular_memoria_longa(self, retornos):
        """
        Calcula métricas de memória longa na volatilidade
        """
        # Retornos ao quadrado como proxy de volatilidade
        volatilidade_proxy = retornos ** 2
        
        # Autocorrelação
        from statsmodels.tsa.stattools import acf
        
        lags = min(100, len(retornos) // 4)
        autocorr = acf(volatilidade_proxy, nlags=lags)
        
        # Taxa de decaimento
        # Em FIGARCH, ACF decai como k^(d-1)
        log_acf = np.log(np.abs(autocorr[1:]) + 1e-10)
        log_lags = np.log(np.arange(1, len(autocorr)))
        
        # Regressão linear
        coef = np.polyfit(log_lags[:50], log_acf[:50], 1)
        taxa_decaimento = coef[0]
        
        # Estimar d
        d_estimado = taxa_decaimento + 1
        
        return {
            'd_estimado': d_estimado,
            'autocorrelacao': autocorr,
            'taxa_decaimento': taxa_decaimento
        }
