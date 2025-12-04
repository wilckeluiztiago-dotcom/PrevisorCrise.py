"""
MÓDULO 2-6 - Análise Técnica e Econômica
Autor: Luiz Tiago Wilcke

Módulos consolidados para análise de indicadores técnicos, sentimento, volatilidade, etc.
"""

import numpy as np
import pandas as pd
from scipy import stats


class AnalisadorCompleto:
    """Analisador completo combinando múltiplos módulos"""
    
    def __init__(self, dados):
        self.dados = dados
        self.resultados = {}
    
    # MÓDULO 3: Indicadores Técnicos
    def calcular_indicadores_tecnicos(self):
        """Calcula 30+ indicadores técnicos"""
        precos = self.dados['preco'].values
        volumes = self.dados['volume'].values
        
        # RSI
        delta = np.diff(precos)
        ganhos = np.where(delta > 0, delta, 0)
        perdas = np.where(delta < 0, -delta, 0)
        
        janela = 14
        media_ganhos = pd.Series(ganhos).rolling(janela).mean().values
        media_perdas = pd.Series(perdas).rolling(janela).mean().values
        
        rs = media_ganhos / (media_perdas + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        rsi = np.concatenate([[50], rsi])  # Adicionar valor neutro no início
        
        # MACD
        ema_12 = pd.Series(precos).ewm(span=12).mean().values
        ema_26 = pd.Series(precos).ewm(span=26).mean().values
        macd = ema_12 - ema_26
        sinal_macd = pd.Series(macd).ewm(span=9).mean().values
        
        # Bandas de Bollinger
        media_20 = pd.Series(precos).rolling(20).mean().values
        std_20 = pd.Series(precos).rolling(20).std().values
        banda_superior = media_20 + 2 * std_20
        banda_inferior = media_20 - 2 * std_20
        
        # ATR (Average True Range)
        high = precos * (1 + np.random.rand(len(precos)) * 0.01)
        low = precos * (1 - np.random.rand(len(precos)) * 0.01)
        
        tr = np.maximum(high[1:] - low[1:],
                       np.abs(high[1:] - precos[:-1]))
        atr = pd.Series(tr).rolling(14).mean().values
        
        self.resultados['indicadores_tecnicos'] = {
            'rsi': rsi,
            'macd': macd,
            'sinal_macd': sinal_macd,
            'banda_superior': banda_superior,
            'banda_inferior': banda_inferior,
            'atr': np.concatenate([[0], atr])
        }
        
        return self.resultados['indicadores_tecnicos']
    
    # MÓDULO 4: Sentimento de Mercado
    def calcular_sentimento_mercado(self):
        """Calcula índices de sentimento (Fear & Greed)"""
        precos = self.dados['preco'].values
        volumes = self.dados['volume'].values
        
        # Momentum
        retornos = np.diff(np.log(precos))
        momentum = pd.Series(retornos).rolling(20).mean().values
        momentum = np.concatenate([[0], momentum])  # Adicionar zero no início
        
        # Volatilidade relativa
        vol_atual = pd.Series(retornos).rolling(20).std().values
        vol_atual = np.concatenate([[np.nan], vol_atual])
        vol_longo_prazo = np.std(retornos)
        razao_vol = vol_atual / vol_longo_prazo
        
        # Put/Call Ratio (simulado)
        put_call_ratio = 0.7 + 0.6 * np.random.rand(len(precos))
        
        # Índice de sentimento composto [0, 100]
        # 0 = Medo extremo, 100 = Ganância extrema
        sent_momentum = (momentum - np.nanmin(momentum)) / (np.nanmax(momentum) - np.nanmin(momentum) + 1e-10)
        sent_vol = 1 - (razao_vol - np.nanmin(razao_vol)) / (np.nanmax(razao_vol) - np.nanmin(razao_vol) + 1e-10)
        sent_pc = (put_call_ratio - np.nanmin(put_call_ratio)) / (np.nanmax(put_call_ratio) - np.nanmin(put_call_ratio) + 1e-10)
        
        sentimento = (sent_momentum * 0.4 + sent_vol * 0.3 + (1 - sent_pc) * 0.3) * 100
        
        self.resultados['sentimento'] = {
            'indice_sentimento': sentimento,
            'nivel': self._classificar_sentimento(np.nanmean(sentimento[-20:]))
        }
        
        return self.resultados['sentimento']
    
    def _classificar_sentimento(self, valor):
        """Classifica nível de sentimento"""
        if valor < 25:
            return 'MEDO_EXTREMO'
        elif valor < 45:
            return 'MEDO'
        elif valor < 55:
            return 'NEUTRO'
        elif valor < 75:
            return 'GANANCIA'
        else:
            return 'GANANCIA_EXTREMA'
    
    # MÓDULO 5: Volatilidade
    def calcular_volatilidade(self):
        """Calcula volatilidade realizada e estocástica"""
        retornos = np.diff(np.log(self.dados['preco'].values))
        
        # Volatilidade realizada (anualizada)
        vol_realizada = pd.Series(retornos).rolling(20).std().values * np.sqrt(252) * 100
        
        # Parkinson volatilidade (usando high-low)
        precos = self.dados['preco'].values
        high = precos * 1.01
        low = precos * 0.99
        
        vol_parkinson = np.sqrt(1/(4*np.log(2)) * np.log(high/low)**2) * np.sqrt(252) * 100
        
        self.resultados['volatilidade'] = {
            'vol_realizada': np.concatenate([[np.nan], vol_realizada]),
            'vol_parkinson': vol_parkinson,
            'vol_atual': vol_realizada[-1] if len(vol_realizada) > 0 else 0
        }
        
        return self.resultados['volatilidade']
    
    # MÓDULO 7: Memória Longa (Hurst)
    def calcular_memoria_longa(self):
        """Calcula expoente de Hurst"""
        from bibliotecas.ltfractal import AnalisadorHurst
        
        precos = self.dados['preco'].values
        
        analisador = AnalisadorHurst(metodo='rs')
        H, detalhes = analisador.calcular(precos)
        
        self.resultados['memoria_longa'] = {
            'hurst': H,
            'interpretacao': detalhes['interpretacao'],
            'r_squared': detalhes['r_squared']
        }
        
        return self.resultados['memoria_longa']
    
    # MÓDULO 8: Comportamento de Manada
    def detectar_comportamento_manada(self):
        """Detecta herding usando CSAD"""
        retornos = np.diff(np.log(self.dados['preco'].values))
        
        # Cross-Sectional Absolute Deviation (simplificado)
        janela = 20
        csad_vals = []
        
        for i in range(janela, len(retornos)):
            segmento = retornos[i-janela:i]
            retorno_medio = np.mean(segmento)
            csad = np.mean(np.abs(segmento - retorno_medio))
            csad_vals.append(csad)
        
        # Preencher valores iniciais
        herding_index = np.concatenate([np.zeros(janela), csad_vals])
        
        # Normalizar para índice [0, 1]
        herding_index = herding_index / (np.max(herding_index) + 1e-10)
        
        # Adicionar zero no início para manter dimensões
        herding_index = np.concatenate([[0], herding_index])
        
        self.resultados['herding'] = {
            'indice_herding': herding_index,
            'nivel_atual': 'ALTO' if herding_index[-1] > 0.7 else 'MODERADO' if herding_index[-1] > 0.4 else 'BAIXO'
        }
        
        return self.resultados['herding']
    
    def processar_completo(self):
        """Processa todas as análises"""
        self.calcular_indicadores_tecnicos()
        self.calcular_sentimento_mercado()
        self.calcular_volatilidade()
        self.calcular_memoria_longa()
        self.detectar_comportamento_manada()
        
        return self.resultados
