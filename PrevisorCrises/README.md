# Sistema de Previsão de Crises Econômicas em Bolhas

**Autor:** Luiz Tiago Wilcke  
**Data:** 2025-12-04  
**Versão:** 1.0.0

## Descrição

Sistema extremamente complexo e avançado para previsão de crises econômicas em bolhas especulativas, baseado na inovadora **Série Temporal de Wilcke para Detecção de Bolhas (STWDB)**.

O sistema combina teorias avançadas de:
- **Equações Diferenciais Estocásticas Fracionárias (EDEF)**
- **Modelos de Mudança de Regime Markovianos**
- **Processos de Log-Periodicidade** (LPPL)
- **Copulas Dinâmicas**
- **Análise de Singularidade Espectral**

## Modelo Matemático Principal

### Equação de Wilcke para Dinâmica de Bolhas

A dinâmica do ativo em bolha é descrita por:

```
dS(t) = μ(t, S, Regime) S(t) dt + σ(t, S, Regime) S(t) dW^H(t) + J(t) S(t) dN(t)
```

Onde:
- **S(t)**: Preço do ativo
- **μ(t, S, Regime)**: Drift dependente do regime com componente log-periódica
- **σ(t, S, Regime)**: Volatilidade com clustering GARCH fracionário
- **W^H(t)**: Movimento Browniano fracionário com expoente de Hurst H
- **J(t)**: Amplitude do salto (crash)
- **N(t)**: Processo de Poisson para eventos de crise

O drift é modelado como:

```
μ(t, S, Regime) = μ_0(Regime) + A cos(ω log(t_c - t) + φ) / (t_c - t)^β
```

Com **t_c** sendo o tempo crítico estimado da crise.

## Arquitetura do Sistema

### 5 Bibliotecas Estatísticas Customizadas

#### 1. **LTBolha** - Detecção de Bolhas Especulativas
- `indicador_bolha.py`: Indicador de Bolha de Wilcke (IBW) multidimensional
- `log_periodicidade.py`: Análise LPPL com detecção de padrões pré-crash
- `tempo_critico.py`: Estimação do tempo crítico t_c usando 4 métodos

#### 2. **LTFractal** - Cálculo Fracionário e Memória Longa
- `derivada_fracionaria.py`: Operadores de Caputo, Riemann-Liouville, Grünwald-Letnikov
- `hurst_expoente.py`: Expoente de Hurst via R/S, DFA, e análise de variância
- `browniano_fracionario.py`: Simulação de fBm (Cholesky e Davies-Harte)

#### 3. **LTEstocastico** - Processos Estocásticos Avançados
- `ede_solver.py`: Solvers para EDEs (Euler-Maruyama, Milstein, RK estocástico)
- `processo_salto.py`: Jump-diffusion (Merton, Kou), processos de Hawkes
- `garch_fracionario.py`: FIGARCH para volatilidade com memória longa

#### 4. **LTRegime** - Mudança de Regime
- `markov_switching.py`: Modelos MS com algoritmo EM
- `filtro_hamilton.py`: Filtro de Hamilton para inferência
- `detector_transicao.py`: Detecção de transições via CUSUM bayesiano

#### 5. **LTCopula** - Dependências Não-lineares
- `copula_gaussiana.py`: Copula Gaussiana multivariada
- `copula_t.py`: Copula t-Student para dependências de cauda
- `copula_dinamica.py`: DCC-GARCH para correlações dinâmicas

### 20 Módulos do Sistema Principal

**Módulos consolidados:**

1. **carregador_dados.py**: Carregamento de dados (APIs, CSV, sintéticos)
2-8. **analisador_completo.py**: 
   - Indicadores técnicos (RSI, MACD, Bollinger, ATR)
   - Sentimento de mercado (Fear & Greed)
   - Volatilidade (realizada, Parkinson, estocástica)
   - Análise de Hurst
   - Comportamento de manada (CSAD)
9-14. **modelador_avancado.py**:
   - Mudança de regime (Markov Switching)
   - Simulação com EDEs
   - Análise de singularidade
   - Redes de correlação
15-17. **sistema_previsao.py**:
   - Previsão probabilística
   - Risco sistêmico (VaR, CVaR, SRISK, Drawdown)
   - Sistema de alertas multi-nível
18. **visualizacao.py**: Dashboards avançados com matplotlib
19-20. **main.py**: Orquestração e relatórios

## Instalação

```bash
# Clone ou extraia o projeto
cd PrevisorCrises

# Instale as dependências
pip install -r requirements.txt
```

## Uso

### Execução do Sistema Completo

```bash
python main.py
```

O sistema irá:
1. Gerar dados sintéticos com bolha e crise
2. Calcular 50+ indicadores técnicos e econômicos
3. Detectar regimes de mercado (bull/bear/crise)
4. Estimar tempo crítico de crash
5. Gerar previsões probabilísticas
6. Calcular risco sistêmico
7. Emitir alertas de crise
8. Gerar 4 dashboards visuais
9. Produzir relatório numérico detalhado

### Saídas Geradas

**Gráficos (em `resultados/`):**
- `grafico_1.png`: Preços com IBW e RSI
- `grafico_2.png`: Volatilidade e regimes
- `grafico_3.png`: Previsões probabilísticas
- `grafico_4.png`: Análise de risco

**Relatório:**
- `relatorio_numerico.txt`: Resultados numéricos completos (precisão de 6 dígitos)

### Uso das Bibliotecas Individualmente

```python
# Exemplo: Detectar bolha
from bibliotecas.ltbolha import IndicadorBolhaWilcke

ibw = IndicadorBolhaWilcke()
indice, componentes = ibw.calcular_indice_composto(precos, volumes)
nivel = ibw.nivel_alerta(indice[-1])
print(f"IBW: {indice[-1]:.2f} - Nível: {nivel}")

# Exemplo: Calcular Hurst
from bibliotecas.ltfractal import AnalisadorHurst

analisador = AnalisadorHurst(metodo='dfa')
H, detalhes = analisador.calcular(serie_precos)
print(f"Hurst: {H:.3f} - {detalhes['interpretacao']}")

# Exemplo: Simular com EDE
from bibliotecas.ltestocastico import SolverEDE

def drift(t, S):
    return 0.05 * S

def difusao(t, S):
    return 0.2 * S

solver = SolverEDE(drift, difusao, x0=100, T=1, dt=0.01)
trajetorias, tempos = solver.resolver_milstein(n_trajetorias=100)
```

## Métricas e Indicadores Calculados

### Indicador de Bolha de Wilcke (IBW)

Índice composto [0-100] combinando:
- Desvio da tendência fundamental (35%)
- Volatilidade excessiva (30%)
- Volume anormal (20%)
- Momentum extremo (15%)

**Níveis de alerta:**
- IBW < 30: BAIXO
- 30 ≤ IBW < 50: MODERADO
- 50 ≤ IBW < 70: ALTO
- IBW ≥ 70: CRÍTICO

### Risco Sistêmico

- **VaR 95%**: Value at Risk no percentil 5%
- **CVaR 95%**: Expected Shortfall além do VaR
- **Máximo Drawdown**: Maior queda desde o pico
- **SRISK**: Capital em risco sistêmico

### Expoente de Hurst (H)

- **H > 0.65**: Forte persistência (tendências de longo prazo)
- **0.55 < H < 0.65**: Persistência moderada
- **0.45 < H < 0.55**: Movimento Browniano (sem memória)
- **0.35 < H < 0.45**: Anti-persistência moderada
- **H < 0.35**: Forte reversão à média

## Estrutura de Diretórios

```
PrevisorCrises/
├── bibliotecas/
│   ├── ltbolha/
│   │   ├── __init__.py
│   │   ├── indicador_bolha.py
│   │   ├── log_periodicidade.py
│   │   └── tempo_critico.py
│   ├── ltfractal/
│   │   ├── __init__.py
│   │   ├── derivada_fracionaria.py
│   │   ├── hurst_expoente.py
│   │   └── browniano_fracionario.py
│   ├── ltestocastico/
│   │   ├── __init__.py
│   │   ├── ede_solver.py
│   │   ├── processo_salto.py
│   │   └── garch_fracionario.py
│   ├── ltregime/
│   │   ├── __init__.py
│   │   ├── markov_switching.py
│   │   ├── filtro_hamilton.py
│   │   └── detector_transicao.py
│   └── ltcopula/
│       ├── __init__.py
│       ├── copula_gaussiana.py
│       ├── copula_t.py
│       └── copula_dinamica.py
├── modulos/
│   ├── carregador_dados.py
│   ├── analisador_completo.py
│   ├── modelador_avancado.py
│   ├── sistema_previsao.py
│   └── visualizacao.py
├── resultados/
│   ├── grafico_1.png
│   ├── grafico_2.png
│   ├── grafico_3.png
│   ├── grafico_4.png
│   └── relatorio_numerico.txt
├── main.py
├── requirements.txt
└── README.md
```

## Fundamentos Teóricos

### Log-Periodic Power Law (LPPL)

Modelo que descreve aceleração superexponencial antes de crashes:

```
ln(p(t)) = A + B(t_c - t)^m + C(t_c - t)^m cos(ω ln(t_c - t) + φ)
```

### Movimento Browniano Fracionário

Generalização do movimento Browniano com memória longa:

```
Cov(B_H(s), B_H(t)) = 0.5 * (|s|^(2H) + |t|^(2H) - |t-s|^(2H))
```

### FIGARCH

Modelo GARCH com integração fracionária para volatilidade persistente:

```
φ(L)(1-L)^d ε_t² = ω + [1 - β(L)]v_t
```

## Referências Científicas

1. Sornette, D. (2003). "Why Stock Markets Crash"
2. Johansen, A., & Sornette, D. (2001). "Finite-time singularity in the dynamics of the world population and economic indices"
3. Mandelbrot, B. (1997). "Fractals and Scaling in Finance"
4. Hamilton, J. D. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series"
5. Baillie, R. T., et al. (1996). "Fractionally integrated generalized autoregressive conditional heteroskedasticity"

## Licença

Copyright © 2025 Luiz Tiago Wilcke. Todos os direitos reservados.

## Contato

**Autor:** Luiz Tiago Wilcke  
**Projeto:** Sistema de Previsão de Crises Econômicas em Bolhas
**Data:** Dezembro de 2025

---

*Este sistema implementa técnicas avançadas de análise de séries temporais, cálculo fracionário, processos estocásticos, e teoria de crises financeiras para detectar e prever bolhas especulativas e crashes de mercado.*
