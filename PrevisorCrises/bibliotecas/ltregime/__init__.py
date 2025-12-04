"""
Biblioteca LTRegime - Modelos de Mudança de Regime
Autor: Luiz Tiago Wilcke
Data: 2025-12-04

Implementa modelos Markov Switching para detectar transições de regime em mercados.
"""

from .markov_switching import ModeloMarkovSwitching
from .filtro_hamilton import FiltroHamilton
from .detector_transicao import DetectorTransicao

__version__ = "1.0.0"
__author__ = "Luiz Tiago Wilcke"

__all__ = [
    'ModeloMarkovSwitching',
    'FiltroHamilton',
    'DetectorTransicao'
]
