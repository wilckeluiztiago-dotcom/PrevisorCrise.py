"""
Biblioteca LTFractal - Cálculo Fracionário e Memória Longa
Autor: Luiz Tiago Wilcke
Data: 2025-12-04

Implementa operadores de cálculo fracionário, análise de memória longa,
e processos com dependência de longo alcance.
"""

from .derivada_fracionaria import DerivadaFracionaria
from .hurst_expoente import AnalisadorHurst
from .browniano_fracionario import MovimentoBrownianoFracionario

__version__ = "1.0.0"
__author__ = "Luiz Tiago Wilcke"

__all__ = [
    'DerivadaFracionaria',
    'AnalisadorHurst',
    'MovimentoBrownianoFracionario'
]
