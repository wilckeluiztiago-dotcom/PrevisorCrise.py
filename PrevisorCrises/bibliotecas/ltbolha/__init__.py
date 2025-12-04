"""
Biblioteca LTBolha - Detecção de Bolhas Especulativas
Autor: Luiz Tiago Wilcke
Data: 2025-12-04

Esta biblioteca implementa algoritmos avançados para detecção de bolhas
especulativas em mercados financeiros, incluindo:
- Indicador de Bolha de Wilcke (IBW)
- Análise de log-periodicidade
- Estimação de tempo crítico
"""

from .indicador_bolha import IndicadorBolhaWilcke
from .log_periodicidade import AnalisadorLogPeriodico
from .tempo_critico import EstimadorTempoCritico

__version__ = "1.0.0"
__author__ = "Luiz Tiago Wilcke"

__all__ = [
    'IndicadorBolhaWilcke',
    'AnalisadorLogPeriodico',
    'EstimadorTempoCritico'
]
