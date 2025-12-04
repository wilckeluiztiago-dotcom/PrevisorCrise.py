"""
MÓDULO 15-17 - Previsão e Alertas
Autor: Luiz Tiago Wilcke

Módulos para previsão probabilística, cálculo de risco sistêmico e sistema de alertas.
"""

import numpy as np
from scipy import stats
from bibliotecas.ltbolha import IndicadorBolhaWilcke, AnalisadorLogPeriodico


class SistemaPrevisaoAlertas:
    """Sistema completo de previsão e alertas de crises"""
    
    def __init__(self, dados):
        self.dados = dados
        self.previsoes = {}
        self.alertas = []
    
    # MÓDULO 15: Previsão Probabilística
    def gerar_previsao_probabilistica(self, horizonte=30):
        """
        Gera previsões probabilísticas de crise
        
        Retorna:
        --------
        previsoes : dict
            Previsões com intervalos de confiança
        """
        precos = self.dados['preco'].values
        volumes = self.dados['volume'].values
        
        # Calcular IBW (Indicador de Bolha de Wilcke)
        ibw_calc = IndicadorBolhaWilcke()
        ibw, componentes = ibw_calc.calcular_indice_composto(precos, volumes)
        
        # Analisar log-periodicidade
        analisador_lp = AnalisadorLogPeriodico()
        tempos = np.arange(len(precos))
        
        try:
            params_lppl = analisador_lp.ajustar_lppl(tempos, precos)
            confianca_lppl, metricas = analisador_lp.calcular_confianca_lppl(tempos, precos)
        except:
            params_lppl = None
            confianca_lppl = 0
        
        # Probabilidade de crise baseada em IBW
        ibw_atual = ibw[-1] if len(ibw) > 0 else 50
        prob_crise_ibw = min(ibw_atual / 100, 1.0)
        
        # Combinar evidências
        if params_lppl is not None:
            prob_crise_total = 0.6 * prob_crise_ibw + 0.4 * confianca_lppl
        else:
            prob_crise_total = prob_crise_ibw
        
        # Previsão de preço com intervalos
        retornos = np.diff(np.log(precos))
        mu = np.mean(retornos)
        sigma = np.std(retornos)
        
        # Simular trajetórias futuras
        n_sim = 1000
        previsoes_futuras = np.zeros((n_sim, horizonte))
        preco_atual = precos[-1]
        
        for i in range(n_sim):
            for h in range(horizonte):
                retorno = np.random.normal(mu, sigma)
                if h == 0:
                    previsoes_futuras[i, h] = preco_atual * np.exp(retorno)
                else:
                    previsoes_futuras[i, h] = previsoes_futuras[i, h-1] * np.exp(retorno)
        
        # Percentis
        p5 = np.percentile(previsoes_futuras, 5, axis=0)
        p50 = np.percentile(previsoes_futuras, 50, axis=0)
        p95 = np.percentile(previsoes_futuras, 95, axis=0)
        
        self.previsoes = {
            'horizonte': horizonte,
            'probabilidade_crise': prob_crise_total,
            'ibw_atual': ibw_atual,
            'nivel_ibw': ibw_calc.nivel_alerta(ibw_atual),
            'preco_atual': preco_atual,
            'previsao_mediana': p50,
            'intervalo_5': p5,
            'intervalo_95': p95,
            'parametros_lppl': params_lppl,
            'confianca_lppl': confianca_lppl
        }
        
        return self.previsoes
    
    # MÓDULO 16: Risco Sistêmico
    def calcular_risco_sistemico(self):
        """
        Calcula métricas de risco sistêmico (VaR, CVaR, etc.)
        """
        retornos = np.diff(np.log(self.dados['preco'].values))
        
        # VaR (Value at Risk) - 95%
        var_95 = np.percentile(retornos, 5) * 100
        
        # CVaR (Conditional VaR) - Expected Shortfall
        cvar_95 = np.mean(retornos[retornos <= np.percentile(retornos, 5)]) * 100
        
        # Maximum Drawdown
        precos_acum = self.dados['preco'].values
        running_max = np.maximum.accumulate(precos_acum)
        drawdown = (precos_acum - running_max) / running_max * 100
        max_drawdown = np.min(drawdown)
        
        # SRISK simplificado (capital em risco)
        volatilidade = np.std(retornos) * np.sqrt(252)
        valor_mercado = precos_acum[-1]
        srisk = max(0, valor_mercado * (0.08 - (1 - volatilidade * 2)))
        
        risco = {
            'var_95': var_95,
            'cvar_95': cvar_95,
            'max_drawdown': max_drawdown,
            'drawdown_atual': drawdown[-1],
            'srisk': srisk,
            'volatilidade_anual': volatilidade * 100
        }
        
        return risco
    
    # MÓDULO 17: Sistema de Alertas
    def gerar_alertas(self):
        """
        Gera alertas de crise em múltiplos níveis
        """
        if not self.previsoes:
            self.gerar_previsao_probabilistica()
        
        risco = self.calcular_risco_sistemico()
        
        self.alertas = []
        
        # Alerta baseado em IBW
        ibw_atual = self.previsoes['ibw_atual']
        if ibw_atual > 70:
            self.alertas.append({
                'nivel': 'CRÍTICO',
                'tipo': 'INDICADOR_BOLHA',
                'mensagem': f'IBW em nível CRÍTICO: {ibw_atual:.1f}/100',
                'prioridade': 1
            })
        elif ibw_atual > 50:
            self.alertas.append({
                'nivel': 'ALTO',
                'tipo': 'INDICADOR_BOLHA',
                'mensagem': f'IBW em nível ALTO: {ibw_atual:.1f}/100',
                'prioridade': 2
            })
        
        # Alerta baseado em probabilidade de crise
        prob_crise = self.previsoes['probabilidade_crise']
        if prob_crise > 0.7:
            self.alertas.append({
                'nivel': 'CRÍTICO',
                'tipo': 'PROBABILIDADE_CRISE',
                'mensagem': f'Probabilidade de crise: {prob_crise*100:.1f}%',
                'prioridade': 1
            })
        elif prob_crise > 0.5:
            self.alertas.append({
                'nivel': 'ALTO',
                'tipo': 'PROBABILIDADE_CRISE',
                'mensagem': f'Probabilidade de crise: {prob_crise*100:.1f}%',
                'prioridade': 2
            })
        
        # Alerta baseado em risco
        if risco['max_drawdown'] < -30:
            self.alertas.append({
                'nivel': 'CRÍTICO',
                'tipo': 'DRAWDOWN',
                'mensagem': f'Máximo drawdown: {risco["max_drawdown"]:.1f}%',
                'prioridade': 1
            })
        
        # Ordenar por prioridade
        self.alertas.sort(key=lambda x: x['prioridade'])
        
        return self.alertas
    
    def gerar_relatorio_completo(self):
        """Gera relatório completo de previsão e risco"""
        self.gerar_previsao_probabilistica()
        risco = self.calcular_risco_sistemico()
        alertas = self.gerar_alertas()
        
        relatorio = {
            'previsoes': self.previsoes,
            'risco_sistemico': risco,
            'alertas': alertas,
            'resumo': {
                'nivel_alerta_geral': 'CRÍTICO' if any(a['nivel'] == 'CRÍTICO' for a in alertas) else 'MODERADO',
                'n_alertas_criticos': sum(1 for a in alertas if a['nivel'] == 'CRÍTICO'),
                'probabilidade_crise': self.previsoes['probabilidade_crise']
            }
        }
        
        return relatorio
