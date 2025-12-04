"""
MÓDULO 1 - Carregador de Dados
Autor: Luiz Tiago Wilcke

Carrega dados de múltiplas fontes para análise de crises econômicas.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class CarregadorDados:
    """Carrega e gerencia dados financeiros"""
    
    def __init__(self):
        self.dados = None
        self.metadados = {}
    
    def gerar_dados_sinteticos(self, n_dias=1000, com_bolha=True, com_crise=True):
        """
        Gera dados sintéticos para demonstração
        
        Parâmetros:
        -----------
        n_dias : int
            Número de dias a simular
        com_bolha : bool
            Se True, inclui período de bolha
        com_crise : bool
            Se True, inclui crash
            
        Retorna:
        --------
        dados : DataFrame
            Dados sintéticos com preços, volumes, etc.
        """
        np.random.seed(42)
        
        # Datas
        data_inicial = datetime(2020, 1, 1)
        datas = [data_inicial + timedelta(days=i) for i in range(n_dias)]
        
        # Preços (processo estocástico com bolha e crise)
        precos = np.zeros(n_dias)
        precos[0] = 100
        
        for i in range(1, n_dias):
            # Drift e volatilidade base
            mu = 0.0005
            sigma = 0.02
            
            # Adicionar bolha (aceleração superexponencial)
            if com_bolha and 600 < i < 800:
                # Log-periodicidade
                t_c = 820  # Tempo crítico
                omega = 7
                A = 0.003
                bolha_termo = A * np.cos(omega * np.log(max(t_c - i, 1))) / max((t_c - i)**0.3, 0.01)
                mu += bolha_termo
                sigma *= 1.5  # Volatilidade aumenta
            
            # Adicionar crise (crash)
            if com_crise and i == 800:
                mu = -0.15  # Crash súbito
                sigma *= 3
            
            # Equação estocástica
            retorno = mu + sigma * np.random.randn()
            precos[i] = precos[i-1] * np.exp(retorno)
        
        # Volumes (correlacionados com volatilidade)
        retornos = np.diff(np.log(precos))
        volatilidade = np.abs(retornos)
        volumes = 1000000 * (1 + 2 * volatilidade)
        volumes = np.insert(volumes, 0, 1000000)  # Primeiro dia
        volumes = volumes + np.random.randn(n_dias) * 100000
        volumes = np.maximum(volumes, 0)
        
        # Indicadores macro (correlacionados)
        taxa_juros = 5 + np.cumsum(np.random.randn(n_dias) * 0.01)
        inflacao = 3 + np.cumsum(np.random.randn(n_dias) * 0.005)
        credito = 100 + np.cumsum(np.random.randn(n_dias) * 0.5)
        
        # Criar DataFrame
        self.dados = pd.DataFrame({
            'data': datas,
            'preco': precos,
            'volume': volumes,
            'taxa_juros': taxa_juros,
            'inflacao': inflacao,
            'credito': credito
        })
        
        self.metadados = {
            'fonte': 'Dados Sintéticos',
            'n_observacoes': n_dias,
            'data_inicial': datas[0],
            'data_final': datas[-1],
            'com_bolha': com_bolha,
            'com_crise': com_crise
        }
        
        return self.dados
    
    def calcular_estatisticas_basicas(self):
        """Calcula estatísticas básicas dos dados"""
        if self.dados is None:
            raise ValueError("Carregue dados primeiro")
        
        estatisticas = {
            'n_observacoes': len(self.dados),
            'preco_medio': self.dados['preco'].mean(),
            'preco_min': self.dados['preco'].min(),
            'preco_max': self.dados['preco'].max(),
            'volatilidade_anual': self.dados['preco'].pct_change().std() * np.sqrt(252) * 100,
            'volume_medio': self.dados['volume'].mean()
        }
        
        return estatisticas
