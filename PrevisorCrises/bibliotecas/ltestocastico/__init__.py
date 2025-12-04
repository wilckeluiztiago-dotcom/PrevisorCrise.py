"""
Biblioteca LTEstocastico - Processos Estocásticos Avançados
Autor: Luiz Tiago Wilcke
Data: 2025-12-04

Implementa solvers para EDEs, processos de salto, e modelos GARCH fracion

ários.
"""

from .ede_solver import SolverEDE
from .processo_salto import ProcessoSalto
from .garch_fracionario import FIGARCH

__version__ = "1.0.0"
__author__ = "Luiz Tiago Wilcke"

__all__ = [
    'SolverEDE',
    'ProcessoSalto',
    'FIGARCH'
]
