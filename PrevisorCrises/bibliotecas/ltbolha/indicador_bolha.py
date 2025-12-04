"""
Indicador de Bolha de Wilcke (IBW)
Autor: Luiz Tiago Wilcke

Implementa um indicador multidimensional que combina:
- Desvio da tendência fundamental
- Volatilidade excessiva
- Volume anormal
- Momentum extremo
- Sentimento de mercado
"""

import numpy as np
from scipy.signal import detrend
from scipy.stats import zscore


class IndicadorBolhaWilcke:
    """
    Indicador de Bolha de Wilcke (IBW)
    
    Calcula um índice composto [0, 100] que indica a probabilidade
    de uma bolha especulativa estar se formando.
    """
    
    def __init__(self, janela_tendencia=252, janela_volatilidade=60):
        """
        Parâmetros:
        -----------
        janela_tendencia : int
            Janela para cálculo de tendência fundamental (dias)
        janela_volatilidade : int
            Janela para cálculo de volatilidade (dias)
        """
        self.janela_tendencia = janela_tendencia
        self.janela_volatilidade = janela_volatilidade
        
    def calcular_desvio_tendencia(self, precos):
        """
        Calcula o desvio dos preços em relação à tendência fundamental
        
        Parâmetros:
        -----------
        precos : array
            Série temporal de preços
            
        Retorna:
        --------
        desvio_normalizado : array
            Desvio normalizado [0, 1]
        """
        # Removendo tendência linear
        precos_sem_tendencia = detrend(precos, type='linear')
        
        # Calculando média móvel como proxy da tendência fundamental
        tendencia_fundamental = np.convolve(
            precos, 
            np.ones(self.janela_tendencia) / self.janela_tendencia, 
            mode='same'
        )
        
        # Desvio percentual
        desvio_percentual = (precos - tendencia_fundamental) / tendencia_fundamental
        
        # Normalização usando tangente hiperbólica
        desvio_normalizado = (np.tanh(desvio_percentual * 5) + 1) / 2
        
        return desvio_normalizado
    
    def calcular_volatilidade_excessiva(self, retornos):
        """
        Detecta volatilidade excessiva anormal
        
        Parâmetros:
        -----------
        retornos : array
            Retornos logarítmicos
            
        Retorna:
        --------
        indice_volatilidade : array
            Índice de volatilidade excessiva [0, 1]
        """
        # Volatilidade realizada
        volatilidade_realizada = self._volatilidade_movel(
            retornos, self.janela_volatilidade
        )
        
        # Volatilidade de longo prazo (benchmark)
        volatilidade_longo_prazo = np.std(retornos) * np.sqrt(252)
        
        # Razão de volatilidade
        razao_volatilidade = volatilidade_realizada / volatilidade_longo_prazo
        
        # Normalização
        indice_volatilidade = (np.tanh((razao_volatilidade - 1) * 3) + 1) / 2
        
        return indice_volatilidade
    
    def calcular_volume_anormal(self, volumes):
        """
        Detecta volume de negociação anormalmente alto
        
        Parâmetros:
        -----------
        volumes : array
            Volume de negociação
            
        Retorna:
        --------
        indice_volume : array
            Índice de volume anormal [0, 1]
        """
        # Z-score do volume
        z_volume = zscore(volumes)
        
        # Normalização usando função logística
        indice_volume = 1 / (1 + np.exp(-z_volume))
        
        return indice_volume
    
    def calcular_momentum_extremo(self, precos, janela=20):
        """
        Detecta momentum extremo (possível sobrecompra)
        
        Parâmetros:
        -----------
        precos : array
            Série temporal de preços
        janela : int
            Janela para cálculo de momentum
            
        Retorna:
        --------
        indice_momentum : array
            Índice de momentum extremo [0, 1]
        """
        # RSI (Relative Strength Index)
        rsi = self._calcular_rsi(precos, janela)
        
        # Normalizar RSI para [0, 1] com ênfase em valores extremos
        indice_momentum = np.where(
            rsi > 50,
            (rsi - 50) / 50,  # Mapeando [50, 100] para [0, 1]
            0
        )
        
        # Aplicar função de ênfase para valores extremos
        indice_momentum = indice_momentum ** 2
        
        return indice_momentum
    
    def calcular_indice_composto(self, precos, volumes, pesos=None):
        """
        Calcula o Indicador de Bolha de Wilcke (IBW) composto
        
        Parâmetros:
        -----------
        precos : array
            Série temporal de preços
        volumes : array
            Volume de negociação
        pesos : array, opcional
            Pesos para cada componente [desvio, volatilidade, volume, momentum]
            
        Retorna:
        --------
        ibw : array
            Indicador de Bolha de Wilcke [0, 100]
        componentes : dict
            Componentes individuais do indicador
        """
        # Pesos padrão (podem ser otimizados)
        if pesos is None:
            pesos = np.array([0.35, 0.30, 0.20, 0.15])
        
        # Normalizar pesos
        pesos = pesos / np.sum(pesos)
        
        # Calcular retornos logarítmicos
        retornos = np.diff(np.log(precos))
        retornos = np.insert(retornos, 0, 0)  # Adicionar zero no início
        
        # Calcular componentes
        comp_desvio = self.calcular_desvio_tendencia(precos)
        comp_volatilidade = self.calcular_volatilidade_excessiva(retornos)
        comp_volume = self.calcular_volume_anormal(volumes)
        comp_momentum = self.calcular_momentum_extremo(precos)
        
        # Índice composto ponderado
        ibw = (
            pesos[0] * comp_desvio +
            pesos[1] * comp_volatilidade +
            pesos[2] * comp_volume +
            pesos[3] * comp_momentum
        ) * 100
        
        componentes = {
            'desvio_tendencia': comp_desvio,
            'volatilidade_excessiva': comp_volatilidade,
            'volume_anormal': comp_volume,
            'momentum_extremo': comp_momentum
        }
        
        return ibw, componentes
    
    def nivel_alerta(self, ibw):
        """
        Determina o nível de alerta baseado no IBW
        
        Parâmetros:
        -----------
        ibw : float
            Valor do Indicador de Bolha de Wilcke
            
        Retorna:
        --------
        nivel : str
            Nível de alerta ('BAIXO', 'MODERADO', 'ALTO', 'CRÍTICO')
        """
        if ibw < 30:
            return 'BAIXO'
        elif ibw < 50:
            return 'MODERADO'
        elif ibw < 70:
            return 'ALTO'
        else:
            return 'CRÍTICO'
    
    # Métodos auxiliares
    def _volatilidade_movel(self, retornos, janela):
        """Calcula volatilidade móvel (anualizada)"""
        volatilidade = np.zeros(len(retornos))
        
        for i in range(janela, len(retornos)):
            volatilidade[i] = np.std(retornos[i-janela:i]) * np.sqrt(252)
        
        # Preencher valores iniciais com a primeira volatilidade calculada
        volatilidade[:janela] = volatilidade[janela]
        
        return volatilidade
    
    def _calcular_rsi(self, precos, janela):
        """Calcula o Relative Strength Index (RSI)"""
        # Calcular variações de preço
        deltas = np.diff(precos)
        deltas = np.insert(deltas, 0, 0)
        
        # Separar ganhos e perdas
        ganhos = np.where(deltas > 0, deltas, 0)
        perdas = np.where(deltas < 0, -deltas, 0)
        
        # Médias móveis exponenciais
        media_ganhos = self._ema(ganhos, janela)
        media_perdas = self._ema(perdas, janela)
        
        # Calcular RS e RSI
        rs = np.divide(media_ganhos, media_perdas, 
                      out=np.ones_like(media_ganhos), 
                      where=media_perdas != 0)
        
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _ema(self, dados, janela):
        """Calcula média móvel exponencial"""
        alpha = 2 / (janela + 1)
        ema = np.zeros(len(dados))
        ema[0] = dados[0]
        
        for i in range(1, len(dados)):
            ema[i] = alpha * dados[i] + (1 - alpha) * ema[i-1]
        
        return ema
