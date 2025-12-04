"""
MÓDULO 9-14 - Modelagem Avançada
Autor: Luiz Tiago Wilcke

Módulos para análise de liquidez, crédito, regime, EDEs, redes e singularidades.
"""

import numpy as np
import pandas as pd
from bibliotecas.ltregime import ModeloMarkovSwitching
from bibliotecas.ltestocastico import SolverEDE
from bibliotecas.ltbolha import EstimadorTempoCritico


class ModeladorAvancado:
    """Modelagem avançada para detecção de crises"""
    
    def __init__(self, dados):
        self.dados = dados
        self.modelos = {}
    
    # MÓDULO 11: Mudança de Regime
    def detectar_mudanca_regime(self):
        """Detecta regimes de mercado (bull/bear/crise)"""
        retornos = np.diff(np.log(self.dados['preco'].values))
        
        # Ajustar modelo Markov Switching de 3 regimes
        modelo_ms = ModeloMarkovSwitching(n_regimes=3)
        params = modelo_ms.ajustar(retornos, max_iter=20)  # Reduzido de 50 para 20
        
        # Classificar regimes
        classificacao = modelo_ms.classificar_regimes()
        
        # Probabilidades suavizadas
        prob_regimes = modelo_ms.probabilidades_suavizadas
        
        # Regime mais provável em cada período
        regimes = np.argmax(prob_regimes, axis=1)
        
        self.modelos['regime'] = {
            'modelo': modelo_ms,
            'regimes': regimes,
            'probabilidades': prob_regimes,
            'classificacao': classificacao,
            'parametros': params
        }
        
        return self.modelos['regime']
    
    # MÓDULO 12: Simulação com EDEs
    def simular_dinamica_precos(self, n_trajetorias=100):
        """Simula dinâmica futura de preços com EDE"""
        preco_atual = self.dados['preco'].values[-1]
        retornos = np.diff(np.log(self.dados['preco'].values))
        
        # Estimar parâmetros
        mu = np.mean(retornos) * 252
        sigma = np.std(retornos) * np.sqrt(252)
        
        # Definir EDE: dS = μS dt + σS dW
        def drift(t, S):
            return mu * S
        
        def difusao(t, S):
            return sigma * S
        
        solver = SolverEDE(drift, difusao, x0=preco_atual, T=1, dt=1/252)
        trajetorias, tempos = solver.resolver_euler_maruyama(n_trajetorias)
        
        self.modelos['ede'] = {
            'trajetorias': trajetorias,
            'tempos': tempos,
            'mu': mu,
            'sigma': sigma
        }
        
        return self.modelos['ede']
    
    # MÓDULO 14: Análise de Singularidade
    def detectar_singularidade(self):
        """Detecta pontos de singularidade (potenciais crashes)"""
        precos = self.dados['preco'].values
        tempos = np.arange(len(precos))
        
        estimador = EstimadorTempoCritico()
        
        try:
            tc, estimativas, prob_crise = estimador.estimar_multiplos_metodos(
                tempos[-200:], precos[-200:]
            )
            
            dias_ate_crise = tc - tempos[-1]
            
            self.modelos['singularidade'] = {
                'tempo_critico': tc,
                'dias_ate_crise': dias_ate_crise,
                'probabilidade_crise': prob_crise,
                'estimativas_por_metodo': estimativas
            }
        except:
            self.modelos['singularidade'] = {
                'tempo_critico': None,
                'dias_ate_crise': None,
                'probabilidade_crise': 0.0,
                'estimativas_por_metodo': {}
            }
        
        return self.modelos['singularidade']
    
    # MÓDULO 13: Rede de Correlação (simplificado)
    def construir_rede_correlacao(self):
        """Constrói rede de correlação entre variáveis"""
        # Usar múltiplas variáveis
        variaveis = ['preco', 'volume', 'taxa_juros', 'inflacao', 'credito']
        matriz_corr = self.dados[variaveis].corr().values
        
        # Centralidade (soma das correlações absolutas)
        centralidade = np.sum(np.abs(matriz_corr), axis=1)
        
        self.modelos['rede'] = {
            'matriz_correlacao': matriz_corr,
            'centralidade': centralidade,
            'variaveis': variaveis
        }
        
        return self.modelos['rede']
    
    def processar_completo(self):
        """Processa todas as modelagens"""
        self.detectar_mudanca_regime()
        self.simular_dinamica_precos(n_trajetorias=50)
        self.detectar_singularidade()
        self.construir_rede_correlacao()
        
        return self.modelos
