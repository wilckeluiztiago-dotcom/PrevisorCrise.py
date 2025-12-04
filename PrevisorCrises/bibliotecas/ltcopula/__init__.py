"""
Biblioteca LTCopula - Modelagem de Dependências Não-lineares
Autor: Luiz Tiago Wilcke
Data: 2025-12-04

Implementa copulas para modelar dependências complexas entre variáveis.
"""

from .copula_gaussiana import CopulaGaussiana
from .copula_t import CopulaT
from .copula_dinamica import CopulaDinamica

__version__ = "1.0.0"
__author__ = "Luiz Tiago Wilcke"

__all__ = [
    'CopulaGaussiana',
    'CopulaT',
    'CopulaDinamica'
]
