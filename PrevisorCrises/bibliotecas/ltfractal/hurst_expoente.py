"""
Análise do Expoente de Hurst
Autor: Luiz Tiago Wilcke

Calcula o expoente de Hurst para caracterizar memória longa em séries temporais.
Usa múltiplos métodos: R/S analysis, DFA, e variância temporal.
"""

import numpy as np
from scipy.stats import linregress


class AnalisadorHurst:
    """
    Calcula o expoente de Hurst H para séries temporais
    
    H > 0.5: Persistência (memória longa positiva)
    H = 0.5: Movimento Browniano (sem memória)
    H < 0.5: Anti-persistência (reversão à média)
    """
    
    def __init__(self, metodo='rs'):
        """
        Parâmetros:
        -----------
        metodo : str
            'rs' (Rescaled Range), 'dfa' (Detrended Fluctuation Analysis),
            ou 'variance' (Análise de variância temporal)
        """
        if metodo not in ['rs', 'dfa', 'variance']:
            raise ValueError("Método deve ser 'rs', 'dfa' ou 'variance'")
        
        self.metodo = metodo
    
    def calcular(self, serie):
        """
        Calcula o expoente de Hurst
        
        Parâmetros:
        -----------
        serie : array
            Série temporal
            
        Retorna:
        --------
        H : float
            Expoente de Hurst
        detalhes : dict
            Informações detalhadas do cálculo
        """
        if self.metodo == 'rs':
            return self._hurst_rs(serie)
        elif self.metodo == 'dfa':
            return self._hurst_dfa(serie)
        else:
            return self._hurst_variance(serie)
    
    def _hurst_rs(self, serie):
        """
        Método R/S (Rescaled Range)
        
        Analisa a relação E[R(n)/S(n)] ~ n^H
        onde R é o range e S é o desvio padrão
        """
        n = len(serie)
        
        # Testar diferentes tamanhos de janela
        tamanhos_janela = []
        rs_valores = []
        
        # Usar potências de 2 para eficiência
        min_janela = 8
        max_janela = n // 4
        
        janela = min_janela
        while janela <= max_janela:
            tamanhos_janela.append(janela)
            
            # Calcular R/S para esta janela
            rs_media = self._calcular_rs_janela(serie, janela)
            rs_valores.append(rs_media)
            
            janela *= 2
        
        # Regressão log-log: log(R/S) = H*log(n) + constante
        log_tamanhos = np.log(tamanhos_janela)
        log_rs = np.log(rs_valores)
        
        slope, intercept, r_value, p_value, std_err = linregress(log_tamanhos, log_rs)
        
        H = slope
        
        detalhes = {
            'hurst': H,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'std_error': std_err,
            'tamanhos_janela': tamanhos_janela,
            'rs_valores': rs_valores,
            'interpretacao': self._interpretar_hurst(H)
        }
        
        return H, detalhes
    
    def _calcular_rs_janela(self, serie, janela):
        """
        Calcula R/S médio para um determinado tamanho de janela
        """
        n = len(serie)
        num_janelas = n // janela
        
        rs_list = []
        
        for i in range(num_janelas):
            inicio = i * janela
            fim = inicio + janela
            segmento = serie[inicio:fim]
            
            # Média do segmento
            media = np.mean(segmento)
            
            # Desvios acumulados
            desvios = segmento - media
            desvios_acum = np.cumsum(desvios)
            
            # Range
            R = np.max(desvios_acum) - np.min(desvios_acum)
            
            # Desvio padrão
            S = np.std(segmento, ddof=1)
            
            if S > 0:
                rs_list.append(R / S)
        
        return np.mean(rs_list) if rs_list else 1.0
    
    def _hurst_dfa(self, serie):
        """
        Detrended Fluctuation Analysis (DFA)
        
        Mais robusto para séries não-estacionárias
        """
        n = len(serie)
        
        # Série acumulada
        serie_acum = np.cumsum(serie - np.mean(serie))
        
        # Escalas a testar
        escalas = []
        flutuacoes = []
        
        min_escala = 4
        max_escala = n // 4
        
        escala = min_escala
        while escala <= max_escala:
            escalas.append(escala)
            
            # Dividir série em segmentos
            num_segmentos = n // escala
            flutuacao_segmentos = []
            
            for i in range(num_segmentos):
                inicio = i * escala
                fim = inicio + escala
                segmento = serie_acum[inicio:fim]
                
                # Ajustar polinômio de grau 1 (tendência linear)
                x = np.arange(escala)
                coef = np.polyfit(x, segmento, 1)
                tendencia = np.polyval(coef, x)
                
                # Flutuação
                flutuacao = np.sqrt(np.mean((segmento - tendencia) ** 2))
                flutuacao_segmentos.append(flutuacao)
            
            # Flutuação média para esta escala
            F = np.mean(flutuacao_segmentos)
            flutuacoes.append(F)
            
            escala = int(escala * 1.5)
        
        # Regressão log-log: log(F) = H*log(escala) + constante
        log_escalas = np.log(escalas)
        log_flutuacoes = np.log(flutuacoes)
        
        slope, intercept, r_value, p_value, std_err = linregress(log_escalas, log_flutuacoes)
        
        H = slope
        
        detalhes = {
            'hurst': H,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'std_error': std_err,
            'escalas': escalas,
            'flutuacoes': flutuacoes,
            'interpretacao': self._interpretar_hurst(H)
        }
        
        return H, detalhes
    
    def _hurst_variance(self, serie):
        """
        Método baseado em análise de variância temporal
        
        Var(X(t+τ) - X(t)) ~ τ^(2H)
        """
        n = len(serie)
        
        # Lags a testar
        lags = []
        variancias = []
        
        max_lag = n // 10
        
        for lag in range(1, max_lag, max(1, max_lag // 20)):
            lags.append(lag)
            
            # Diferenças com lag
            diferencas = serie[lag:] - serie[:-lag]
            variancia = np.var(diferencas)
            
            variancias.append(variancia)
        
        # Regressão log-log: log(Var) = 2H*log(lag) + constante
        log_lags = np.log(lags)
        log_variancias = np.log(variancias)
        
        slope, intercept, r_value, p_value, std_err = linregress(log_lags, log_variancias)
        
        H = slope / 2  # Dividir por 2 porque Var ~ τ^(2H)
        
        detalhes = {
            'hurst': H,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'std_error': std_err,
            'lags': lags,
            'variancias': variancias,
            'interpretacao': self._interpretar_hurst(H)
        }
        
        return H, detalhes
    
    def _interpretar_hurst(self, H):
        """
        Interpreta o valor do expoente de Hurst
        """
        if H > 0.65:
            return "FORTE PERSISTÊNCIA - Tendências de longo prazo dominantes"
        elif H > 0.55:
            return "PERSISTÊNCIA MODERADA - Memória longa presente"
        elif H > 0.45:
            return "MOVIMENTO BROWNIANO - Processo sem memória"
        elif H > 0.35:
            return "ANTI-PERSISTÊNCIA MODERADA - Reversão à média"
        else:
            return "FORTE ANTI-PERSISTÊNCIA - Reversão rápida à média"
    
    def calcular_horizonte_previsibilidade(self, H, frequencia_dados='diaria'):
        """
        Estima o horizonte de previsibilidade baseado em H
        
        Parâmetros:
        -----------
        H : float
            Expoente de Hurst
        frequencia_dados : str
            'horaria', 'diaria', 'semanal'
            
        Retorna:
        --------
        horizonte : int
            Horizonte em unidades de tempo
        """
        # Quanto maior H, maior o horizonte de previsibilidade
        
        if frequencia_dados == 'horaria':
            base = 24  # 1 dia
        elif frequencia_dados == 'diaria':
            base = 30  # 1 mês
        else:  # semanal
            base = 12  # 3 meses
        
        # Fator de escala baseado em H
        # H=0.5 → fator=1, H=0.8 → fator≈4
        fator = np.exp(3 * (H - 0.5))
        
        horizonte = int(base * fator)
        
        return horizonte
    
    def detectar_mudanca_regime_hurst(self, serie, janela=100):
        """
        Detecta mudanças no expoente de Hurst ao longo do tempo
        (indica mudanças de regime de mercado)
        
        Parâmetros:
        -----------
        serie : array
            Série temporal
        janela : int
            Tamanho da janela móvel
            
        Retorna:
        --------
        hurst_temporal : array
            Evolução temporal do expoente H
        mudancas_regime : list
            Índices onde ocorreram mudanças significativas
        """
        n = len(serie)
        hurst_temporal = np.zeros(n - janela)
        
        for i in range(n - janela):
            segmento = serie[i:i+janela]
            H, _ = self.calcular(segmento)
            hurst_temporal[i] = H
        
        # Detectar mudanças abruptas (diferença > 0.15)
        diferencas = np.abs(np.diff(hurst_temporal))
        threshold = 0.15
        
        mudancas_regime = np.where(diferencas > threshold)[0].tolist()
        
        return hurst_temporal, mudancas_regime
