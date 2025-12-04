"""
SISTEMA DE PREVISÃO DE CRISES ECONÔMICAS EM BOLHAS
Autor: Luiz Tiago Wilcke
Data: 2025-12-04

Sistema completo para detectar e prever crises econômicas usando
a Série Temporal de Wilcke para Detecção de Bolhas (STWDB).
"""

import sys
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Adicionar diretório ao path
sys.path.insert(0, '/home/luiztiagowilcke188/Área de trabalho/Projetos/PrevisorCrises')

from modulos.carregador_dados import CarregadorDados
from modulos.analisador_completo import AnalisadorCompleto
from modulos.modelador_avancado import ModeladorAvancado
from modulos.sistema_previsao import SistemaPrevisaoAlertas
from modulos.visualizacao import VisualizadorCrises


def imprimir_cabecalho():
    """Imprime cabeçalho do sistema"""
    print("=" * 80)
    print("SISTEMA DE PREVISÃO DE CRISES ECONÔMICAS EM BOLHAS".center(80))
    print("Série Temporal de Wilcke para Detecção de Bolhas (STWDB)".center(80))
    print("Autor: Luiz Tiago Wilcke".center(80))
    print("Data: 2025-12-04".center(80))
    print("=" * 80)
    print()


def imprimir_secao(titulo):
    """Imprime separador de seção"""
    print("\n" + "─" * 80)
    print(f"█ {titulo}")
    print("─" * 80)


def main():
    """Função principal do sistema"""
    
    imprimir_cabecalho()
    
    # PASSO 1: Carregar dados
    imprimir_secao("PASSO 1/6: CARREGAMENTO DE DADOS")
    print("Gerando dados sintéticos com bolha e crise...")
    
    carregador = CarregadorDados()
    dados = carregador.gerar_dados_sinteticos(n_dias=1000, com_bolha=True, com_crise=True)
    
    stats = carregador.calcular_estatisticas_basicas()
    print(f"✓ Dados carregados: {stats['n_observacoes']} observações")
    print(f"  • Preço médio: ${stats['preco_medio']:.2f}")
    print(f"  • Volatilidade anual: {stats['volatilidade_anual']:.2f}%")
    print(f"  • Preço mín/máx: ${stats['preco_min']:.2f} / ${stats['preco_max']:.2f}")
    
    # PASSO 2: Análise técnica e econômica
    imprimir_secao("PASSO 2/6: ANÁLISE TÉCNICA E ECONÔMICA")
    print("Calculando indicadores técnicos, sentimento, volatilidade...")
    
    analisador = AnalisadorCompleto(dados)
    resultados_analise = analisador.processar_completo()
    
    print(f"✓ Indicadores técnicos calculados")
    print(f"  • RSI atual: {resultados_analise['indicadores_tecnicos']['rsi'][-1]:.1f}")
    print(f"  • Sentimento: {resultados_analise['sentimento']['nivel']}")
    print(f"  • Volatilidade atual: {resultados_analise['volatilidade']['vol_atual']:.2f}%")
    print(f"  • Expoente de Hurst: {resultados_analise['memoria_longa']['hurst']:.3f}")
    print(f"  • Interpretação: {resultados_analise['memoria_longa']['interpretacao']}")
    
    # PASSO 3: Modelagem avançada
    imprimir_secao("PASSO 3/6: MODELAGEM AVANÇADA")
    print("Detectando regimes, simulando EDEs, analisando singularidades...")
    
    modelador = ModeladorAvancado(dados)
    resultados_modelo = modelador.processar_completo()
    
    regime_atual = resultados_modelo['regime']['regimes'][-1]
    classificacao = resultados_modelo['regime']['classificacao']
    print(f"✓ Modelagem concluída")
    print(f"  • Regime atual: {classificacao.get(regime_atual, 'DESCONHECIDO')}")
    print(f"  • Número de regimes detectados: {len(classificacao)}")
    
    if resultados_modelo['singularidade']['tempo_critico'] is not None:
        dias = resultados_modelo['singularidade']['dias_ate_crise']
        prob = resultados_modelo['singularidade']['probabilidade_crise']
        print(f"  • Tempo crítico estimado: ~{dias:.0f} dias")
        print(f"  • Probabilidade de crise: {prob*100:.1f}%")
    else:
        print(f"  • Nenhuma singularidade detectada no horizonte")
    
    # PASSO 4: Previsões e alertas
    imprimir_secao("PASSO 4/6: PREVISÕES E SISTEMA DE ALERTAS")
    print("Gerando previsões probabilísticas e calculando risco sistêmico...")
    
    sistema_prev = SistemaPrevisaoAlertas(dados)
    relatorio = sistema_prev.gerar_relatorio_completo()
    
    print(f"✓ Previsões geradas (horizonte: {relatorio['previsoes']['horizonte']} dias)")
    print(f"  • IBW (Indicador de Bolha de Wilcke): {relatorio['previsoes']['ibw_atual']:.1f}/100")
    print(f"  • Nível IBW: {relatorio['previsoes']['nivel_ibw']}")
    print(f"  • Probabilidade de crise: {relatorio['previsoes']['probabilidade_crise']*100:.1f}%")
    print(f"  • VaR 95%: {relatorio['risco_sistemico']['var_95']:.2f}%")
    print(f"  • CVaR 95%: {relatorio['risco_sistemico']['cvar_95']:.2f}%")
    print(f"  • Máximo Drawdown: {relatorio['risco_sistemico']['max_drawdown']:.2f}%")
    
    print(f"\n  ALERTAS ATIVOS: {len(relatorio['alertas'])}")
    for alerta in relatorio['alertas'][:5]:  # Mostrar top 5
        emoji = "🔴" if alerta['nivel'] == 'CRÍTICO' else "🟠" if alerta['nivel'] == 'ALTO' else "🟡"
        print(f"    {emoji} [{alerta['nivel']}] {alerta['mensagem']}")
    
    # PASSO 5: Visualização
    imprimir_secao("PASSO 5/6: GERAÇÃO DE GRÁFICOS")
    print("Criando visualizações avançadas...")
    
    # Criar diretório de resultados
    os.makedirs('/home/luiztiagowilcke188/Área de trabalho/Projetos/PrevisorCrises/resultados', exist_ok=True)
    
    # Combinar resultados
    resultados_completos = {
        'dados': dados,
        'analisador': resultados_analise,
        'modelador': resultados_modelo,
        'previsoes': relatorio['previsoes'],
        'alertas': relatorio['alertas']
    }
    
    visualizador = VisualizadorCrises(dados, resultados_completos)
    figuras = visualizador.gerar_dashboard_completo(salvar=True)
    
    print(f"✓ {len(figuras)} gráficos gerados e salvos")
    print(f"  Localização: /home/luiztiagowilcke188/Área de trabalho/Projetos/PrevisorCrises/resultados/")
    
    # PASSO 6: Relatório numérico
    imprimir_secao("PASSO 6/6: RELATÓRIO NUMÉRICO DETALHADO")
    
    gerar_relatorio_numerico(dados, resultados_completos, relatorio)
    
    # Resumo final
    imprimir_secao("RESUMO EXECUTIVO")
    print(f"NÍVEL DE ALERTA GERAL: {relatorio['resumo']['nivel_alerta_geral']}")
    print(f"Alertas críticos: {relatorio['resumo']['n_alertas_criticos']}")
    print(f"Probabilidade de crise: {relatorio['resumo']['probabilidade_crise']*100:.1f}%")
    
    if relatorio['resumo']['nivel_alerta_geral'] == 'CRÍTICO':
        print("\n⚠️  ATENÇÃO: Sinais de bolha especulativa detectados!")
        print("   Recomenda-se cautela e redução de exposição ao risco.")
    else:
        print("\n✓ Mercado em condições normais de operação.")
    
    print("\n" + "=" * 80)
    print("ANÁLISE CONCLUÍDA".center(80))
    print("=" * 80 + "\n")


def gerar_relatorio_numerico(dados, resultados, relatorio):
    """Gera relatório numérico detalhado"""
    
    arquivo = '/home/luiztiagowilcke188/Área de trabalho/Projetos/PrevisorCrises/resultados/relatorio_numerico.txt'
    
    with open(arquivo, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("RELATÓRIO NUMÉRICO DETALHADO - SISTEMA DE PREVISÃO DE CRISES\n")
        f.write("Série Temporal de Wilcke para Detecção de Bolhas (STWDB)\n")
        f.write("Autor: Luiz Tiago Wilcke\n")
        f.write("=" * 100 + "\n\n")
        
        # Seção 1: Dados
        f.write("1. ESTATÍSTICAS DOS DADOS\n")
        f.write("-" * 100 + "\n")
        f.write(f"Número de observações: {len(dados)}\n")
        f.write(f"Preço inicial: ${dados['preco'].iloc[0]:.6f}\n")
        f.write(f"Preço final: ${dados['preco'].iloc[-1]:.6f}\n")
        f.write(f"Retorno total: {(dados['preco'].iloc[-1]/dados['preco'].iloc[0] - 1)*100:.6f}%\n")
        f.write(f"Volatilidade anualizada: {np.std(np.diff(np.log(dados['preco'].values))) * np.sqrt(252) * 100:.6f}%\n\n")
        
        # Seção 2: Indicadores
        f.write("2. INDICADORES TÉCNICOS E ECONÔMICOS\n")
        f.write("-" * 100 + "\n")
        if 'analisador' in resultados:
            f.write(f"RSI atual: {resultados['analisador']['indicadores_tecnicos']['rsi'][-1]:.6f}\n")
            f.write(f"Sentimento de mercado: {resultados['analisador']['sentimento']['nivel']}\n")
            f.write(f"Índice de sentimento: {resultados['analisador']['sentimento']['indice_sentimento'][-1]:.6f}\n")
            f.write(f"Volatilidade atual: {resultados['analisador']['volatilidade']['vol_atual']:.6f}%\n")
            f.write(f"Expoente de Hurst: {resultados['analisador']['memoria_longa']['hurst']:.6f}\n")
            f.write(f"R² (Hurst): {resultados['analisador']['memoria_longa']['r_squared']:.6f}\n\n")
        
        # Seção 3: Previsões
        f.write("3. PREVISÕES E RISCO SISTÊMICO\n")
        f.write("-" * 100 + "\n")
        f.write(f"IBW (Indicador de Bolha de Wilcke): {relatorio['previsoes']['ibw_atual']:.6f} / 100\n")
        f.write(f"Nível de alerta IBW: {relatorio['previsoes']['nivel_ibw']}\n")
        f.write(f"Probabilidade de crise: {relatorio['previsoes']['probabilidade_crise']:.6f}\n")
        f.write(f"VaR 95%: {relatorio['risco_sistemico']['var_95']:.6f}%\n")
        f.write(f"CVaR 95%: {relatorio['risco_sistemico']['cvar_95']:.6f}%\n")
        f.write(f"Máximo Drawdown: {relatorio['risco_sistemico']['max_drawdown']:.6f}%\n")
        f.write(f"Drawdown atual: {relatorio['risco_sistemico']['drawdown_atual']:.6f}%\n")
        f.write(f"SRISK: ${relatorio['risco_sistemico']['srisk']:.6f}\n\n")
        
        # Seção 4: Alertas
        f.write("4. ALERTAS DETECTADOS\n")
        f.write("-" * 100 + "\n")
        for i, alerta in enumerate(relatorio['alertas'], 1):
            f.write(f"{i}. [{alerta['nivel']}] {alerta['tipo']}: {alerta['mensagem']}\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write("FIM DO RELATÓRIO\n")
        f.write("=" * 100 + "\n")
    
    print(f"✓ Relatório numérico salvo: {arquivo}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERRO: {str(e)}")
        import traceback
        traceback.print_exc()
