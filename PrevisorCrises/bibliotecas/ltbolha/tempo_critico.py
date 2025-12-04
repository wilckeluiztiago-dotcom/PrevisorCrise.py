"""
Estimador de Tempo Crítico
Autor: Luiz Tiago Wilcke

Estima o tempo crítico t_c quando uma crise/crash é esperado ocorrer.
Usa múltiplas abordagens para robustez.
"""

import numpy as np
from scipy.optimize import minimize_scalar, brentq
from scipy.stats import norm
import warnings


class EstimadorTempoCritico:
    """
    Estima o tempo crítico de uma crise iminente
    
    Combina múltiplas técnicas:
    1. Extrapolação LPPL
    2. Análise de singularidade
    3. Modelo de Ornstein-Uhlenbeck invertido
    4. Detecção de ponto de inflexão
    """
    
    def __init__(self, horizonte_maximo=252):
        """
        Parâmetros:
        -----------
        horizonte_maximo : int
            Horizonte máximo de previsão (em dias)
        """
        self.horizonte_maximo = horizonte_maximo
        self.tc_estimado = None
        self.intervalo_confianca = None
        
    def estimar_multiplos_metodos(self, tempos, precos, volumes=None):
        """
        Estima t_c usando múltiplos métodos e combina os resultados
        
        Parâmetros:
        -----------
        tempos : array
            Tempos
        precos : array
            Preços
        volumes : array, opcional
            Volumes de negociação
            
        Retorna:
        --------
        tc_medio : float
            Tempo crítico estimado (consenso)
        tc_por_metodo : dict
            Estimativas individuais por método
        probabilidade_crise : float
            Probabilidade de crise no horizonte [0, 1]
        """
        estimativas = {}
        pesos = {}
        
        # Método 1: LPPL
        try:
            tc_lppl, conf_lppl = self._estimar_tc_lppl(tempos, precos)
            estimativas['lppl'] = tc_lppl
            pesos['lppl'] = conf_lppl
        except:
            warnings.warn("Método LPPL falhou")
        
        # Método 2: Singularidade
        try:
            tc_sing, conf_sing = self._estimar_tc_singularidade(tempos, precos)
            estimativas['singularidade'] = tc_sing
            pesos['singularidade'] = conf_sing
        except:
            warnings.warn("Método de singularidade falhou")
        
        # Método 3: Ornstein-Uhlenbeck invertido
        try:
            tc_ou, conf_ou = self._estimar_tc_ou(tempos, precos)
            estimativas['ornstein_uhlenbeck'] = tc_ou
            pesos['ornstein_uhlenbeck'] = conf_ou
        except:
            warnings.warn("Método OU falhou")
        
        # Método 4: Volatilidade crescente
        try:
            tc_vol, conf_vol = self._estimar_tc_volatilidade(tempos, precos)
            estimativas['volatilidade'] = tc_vol
            pesos['volatilidade'] = conf_vol
        except:
            warnings.warn("Método de volatilidade falhou")
        
        if len(estimativas) == 0:
            raise ValueError("Todos os métodos de estimação falharam")
        
        # Combinar estimativas com média ponderada
        pesos_array = np.array(list(pesos.values()))
        pesos_norm = pesos_array / np.sum(pesos_array)
        estimativas_array = np.array(list(estimativas.values()))
        
        tc_medio = np.sum(estimativas_array * pesos_norm)
        
        # Intervalo de confiança (baseado na dispersão)
        tc_std = np.sqrt(np.sum(pesos_norm * (estimativas_array - tc_medio) ** 2))
        intervalo_confianca = (tc_medio - 2*tc_std, tc_medio + 2*tc_std)
        
        # Calcular probabilidade de crise
        dias_ate_tc = tc_medio - tempos[-1]
        probabilidade_crise = self._calcular_probabilidade_crise(
            dias_ate_tc, tc_std
        )
        
        self.tc_estimado = tc_medio
        self.intervalo_confianca = intervalo_confianca
        
        return tc_medio, estimativas, probabilidade_crise
    
    def dias_ate_crise(self, tempo_atual):
        """
        Calcula quantos dias faltam até o tempo crítico estimado
        
        Parâmetros:
        -----------
        tempo_atual : float
            Tempo atual
            
        Retorna:
        --------
        dias : float
            Dias até a crise (pode ser negativo se já passou)
        """
        if self.tc_estimado is None:
            raise ValueError("Execute estimar_multiplos_metodos() primeiro")
        
        return self.tc_estimado - tempo_atual
    
    def nivel_urgencia(self, tempo_atual):
        """
        Determina o nível de urgência baseado na proximidade de t_c
        
        Parâmetros:
        -----------
        tempo_atual : float
            Tempo atual
            
        Retorna:
        --------
        nivel : str
            'IMEDIATO', 'URGENTE', 'MODERADO', 'BAIXO'
        dias : float
            Dias até a crise
        """
        dias = self.dias_ate_crise(tempo_atual)
        
        if dias < 0:
            return 'EXPIRADO', dias
        elif dias < 5:
            return 'IMEDIATO', dias
        elif dias < 20:
            return 'URGENTE', dias
        elif dias < 60:
            return 'MODERADO', dias
        else:
            return 'BAIXO', dias
    
    # Métodos de estimação individuais
    def _estimar_tc_lppl(self, tempos, precos):
        """Estima t_c usando modelo LPPL"""
        from .log_periodicidade import AnalisadorLogPeriodico
        
        analisador = AnalisadorLogPeriodico()
        params = analisador.ajustar_lppl(tempos, precos)
        confianca, _ = analisador.calcular_confianca_lppl(tempos, precos)
        
        return params['tc'], confianca
    
    def _estimar_tc_singularidade(self, tempos, precos):
        """
        Estima t_c detectando singularidade na derivada segunda
        """
        # Calcular retornos e aceleração
        log_precos = np.log(precos)
        primeira_derivada = np.gradient(log_precos, tempos)
        segunda_derivada = np.gradient(primeira_derivada, tempos)
        
        # Encontrar onde a segunda derivada diverge
        # Ajustar modelo: d²p/dt² ~ 1/(t_c - t)^α
        
        # Usar últimos 60 pontos
        n = min(60, len(tempos))
        t_fit = tempos[-n:]
        deriv2_fit = np.abs(segunda_derivada[-n:])
        
        # Evitar zeros e valores muito pequenos
        deriv2_fit = np.maximum(deriv2_fit, 1e-10)
        
        def modelo_singularidade(tc, t, deriv2):
            dt = tc - t
            dt = np.maximum(dt, 0.01)
            alpha = 1.5
            predicao = 1.0 / (dt ** alpha)
            return np.sum((deriv2 - predicao) ** 2)
        
        # Otimizar
        resultado = minimize_scalar(
            lambda tc: modelo_singularidade(tc, t_fit, deriv2_fit),
            bounds=(tempos[-1] + 1, tempos[-1] + self.horizonte_maximo),
            method='bounded'
        )
        
        tc_sing = resultado.x
        
        # Confiança baseada na qualidade do ajuste
        confianca = np.exp(-resultado.fun / len(t_fit))
        
        return tc_sing, confianca
    
    def _estimar_tc_ou(self, tempos, precos):
        """
        Estima t_c usando processo de Ornstein-Uhlenbeck invertido
        (força de reversão à média se tornando negativa indica instabilidade)
        """
        log_precos = np.log(precos)
        
        # Estimar parâmetros OU: dX = θ(μ - X)dt + σdW
        # Usar janela móvel para detectar quando θ se torna negativo
        
        janela = 30
        thetas = []
        
        for i in range(janela, len(log_precos)):
            x = log_precos[i-janela:i]
            t = tempos[i-janela:i]
            
            # Estimativa de θ via regressão
            dx = np.diff(x)
            x_lag = x[:-1]
            mu = np.mean(x)
            
            # θ = cov(dx, x_lag - mu) / var(x_lag - mu)
            theta = np.cov(dx, x_lag - mu)[0, 1] / np.var(x_lag - mu)
            thetas.append((tempos[i], theta))
        
        thetas = np.array(thetas)
        
        # Encontrar quando θ cruza zero (torna-se negativo)
        if np.any(thetas[:, 1] < 0):
            idx_negativo = np.where(thetas[:, 1] < 0)[0][0]
            
            # Extrapolar para quando θ = -infinito (singularidade)
            # Ajustar modelo exponencial
            t_theta = thetas[max(0, idx_negativo-10):, 0]
            theta_vals = thetas[max(0, idx_negativo-10):, 1]
            
            # Modelo: θ(t) ~ -1/(t_c - t)
            if len(t_theta) > 5 and np.all(theta_vals < 0):
                try:
                    # Regressão não-linear simplificada
                    tc_ou = t_theta[-1] + np.abs(1.0 / theta_vals[-1])
                    confianca = 0.6
                except:
                    tc_ou = tempos[-1] + 30
                    confianca = 0.3
            else:
                tc_ou = tempos[-1] + 30
                confianca = 0.3
        else:
            # Nenhuma instabilidade detectada ainda
            tc_ou = tempos[-1] + self.horizonte_maximo
            confianca = 0.2
        
        return tc_ou, confianca
    
    def _estimar_tc_volatilidade(self, tempos, precos):
        """
        Estima t_c baseado em volatilidade crescente explosivamente
        """
        retornos = np.diff(np.log(precos))
        
        # Volatilidade móvel
        janela = 20
        volatilidades = []
        
        for i in range(janela, len(retornos)):
            vol = np.std(retornos[i-janela:i]) * np.sqrt(252)
            volatilidades.append((tempos[i], vol))
        
        volatilidades = np.array(volatilidades)
        
        # Ajustar modelo exponencial: σ(t) ~ exp(a + b*t)
        t_vol = volatilidades[:, 0]
        sigma_vol = volatilidades[:, 1]
        
        # Regressão linear em log(σ)
        log_sigma = np.log(sigma_vol + 1e-10)
        coef = np.polyfit(t_vol, log_sigma, 1)
        b = coef[0]  # Taxa de crescimento
        
        if b > 0:
            # Volatilidade está crescendo exponencialmente
            # Estimar quando σ atinge nível crítico (ex: 100%)
            sigma_critico = 1.0
            
            # σ_0 * exp(b*t) = σ_critico
            sigma_atual = sigma_vol[-1]
            t_atual = t_vol[-1]
            
            if sigma_atual > 0:
                tc_vol = t_atual + np.log(sigma_critico / sigma_atual) / b
                confianca = min(b * 10, 0.8)  # Maior b = maior confiança
            else:
                tc_vol = tempos[-1] + self.horizonte_maximo / 2
                confianca = 0.3
        else:
            # Volatilidade não está crescendo
            tc_vol = tempos[-1] + self.horizonte_maximo
            confiança = 0.1
        
        return tc_vol, confianca
    
    def _calcular_probabilidade_crise(self, dias_ate_tc, incerteza):
        """
        Calcula probabilidade de crise ocorrer dentro do horizonte
        
        Usa distribuição normal centrada em t_c com desvio padrão = incerteza
        """
        if dias_ate_tc <= 0:
            return 1.0
        
        # Probabilidade de crise nos próximos 30 dias
        horizonte = 30
        
        # P(crise em [agora, agora + horizonte]) = P(T_c < agora + horizonte)
        z_score = (horizonte - dias_ate_tc) / (incerteza + 1e-10)
        probabilidade = norm.cdf(z_score)
        
        return np.clip(probabilidade, 0, 1)
