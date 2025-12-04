"""
MÓDULO 18 - Visualização Avançada
Autor: Luiz Tiago Wilcke

Gera gráficos avançados para análise de crises.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class VisualizadorCrises:
    """Visualizador avançado para análise de crises"""
    
    def __init__(self, dados, resultados):
        self.dados = dados
        self.resultados = resultados
        self.figuras = []
    
    def gerar_dashboard_completo(self, salvar=True):
        """Gera dashboard completo com todos os gráficos"""
        
        # Figura 1: Análise de Preços e Indicadores
        self._grafico_precos_indicadores()
        
        # Figura 2: Análise de Volatilidade e Regime
        self._grafico_volatilidade_regime()
        
        # Figura 3: Previsões e Alertas
        self._grafico_previsoes()
        
        # Figura 4: Análise de Risco
        self._grafico_risco()
        
        if salvar:
            for i, fig in enumerate(self.figuras):
                fig.savefig(f'/home/luiztiagowilcke188/Área de trabalho/Projetos/PrevisorCrises/resultados/grafico_{i+1}.png',
                           dpi=150, bbox_inches='tight')
        
        return self.figuras
    
    def _grafico_precos_indicadores(self):
        """Gráfico 1: Preços com IBW e RSI"""
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)
        
        # Subplot 1: Preços e bandas
        ax1 = fig.add_subplot(gs[0])
        precos = self.dados['preco'].values
        datas = range(len(precos))
        
        ax1.plot(datas, precos, 'b-', label='Preço', linewidth=2)
        
        # IBW se disponível
        if 'previsoes' in self.resultados and 'ibw_atual' in self.resultados['previsoes']:
            ibw = self.resultados['previsoes'].get('ibw_atual', 50)
            cor_ibw = 'red' if ibw > 70 else 'orange' if ibw > 50 else 'green'
            ax1.axhspan(precos.min(), precos.max(), alpha=0.1, color=cor_ibw)
            ax1.text(0.02, 0.95, f'IBW: {ibw:.1f}/100', transform=ax1.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor=cor_ibw, alpha=0.3))
        
        ax1.set_ylabel('Preço', fontsize=12)
        ax1.set_title('Análise de Preços e Detecção de Bolhas - Sistema de Wilcke', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: RSI
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        if 'analisador' in self.resultados and 'indicadores_tecnicos' in self.resultados['analisador']:
            rsi = self.resultados['analisador']['indicadores_tecnicos'].get('rsi', np.zeros(len(datas)))
            ax2.plot(datas, rsi, 'purple', linewidth=1.5)
            ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Sobrecomprado')
            ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Sobrevendido')
            ax2.fill_between(datas, 30, 70, alpha=0.1, color='gray')
        
        ax2.set_ylabel('RSI', fontsize=11)
        ax2.set_ylim(0, 100)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # Subplot 3: Volume
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        volumes = self.dados['volume'].values
        ax3.bar(datas, volumes, alpha=0.6, color='skyblue')
        ax3.set_ylabel('Volume', fontsize=11)
        ax3.set_xlabel('Tempo (dias)', fontsize=11)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.figuras.append(fig)
    
    def _grafico_volatilidade_regime(self):
        """Gráfico 2: Volatilidade e Regimes"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        
        datas = range(len(self.dados))
        
        # Volatilidade
        if 'analisador' in self.resultados and 'volatilidade' in self.resultados['analisador']:
            vol = self.resultados['analisador']['volatilidade'].get('vol_realizada', np.zeros(len(datas)))
            ax1.plot(datas, vol, 'red', linewidth=2, label='Volatilidade Realizada')
            ax1.fill_between(datas, 0, vol, alpha=0.3, color='red')
        
        ax1.set_ylabel('Volatilidade Anualizada (%)', fontsize=12)
        ax1.set_title('Análise de Volatilidade e Mudança de Regime', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Regimes
        if 'modelador' in self.resultados and 'regime' in self.resultados['modelador']:
            regimes = self.resultados['modelador']['regime'].get('regimes', np.zeros(len(datas)-1))
            cores_regime = ['green', 'orange', 'red']
            
            for i in range(len(regimes) - 1):
                ax2.axvspan(i, i+1, alpha=0.3, color=cores_regime[regimes[i]])
            
            ax2.plot(datas[:-1], regimes, 'ko', markersize=2)
        
        ax2.set_ylabel('Regime', fontsize=12)
        ax2.set_xlabel('Tempo (dias)', fontsize=12)
        ax2.set_yticks([0, 1, 2])
        ax2.set_yticklabels(['Normal', 'Volátil', 'Crise'])
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.figuras.append(fig)
    
    def _grafico_previsoes(self):
        """Gráfico 3: Previsões Probabilísticas"""
        fig, ax = plt.subplots(figsize=(16, 8))
        
        precos = self.dados['preco'].values
        datas_hist = range(len(precos))
        
        # Histórico
        ax.plot(datas_hist, precos, 'b-', linewidth=2, label='Histórico')
        
        # Previsões
        if 'previsoes' in self.resultados:
            prev = self.resultados['previsoes']
            horizonte = prev.get('horizonte', 30)
            datas_fut = range(len(precos), len(precos) + horizonte)
            
            mediana = prev.get('previsao_mediana', np.zeros(horizonte))
            p5 = prev.get('intervalo_5', np.zeros(horizonte))
            p95 = prev.get('intervalo_95', np.zeros(horizonte))
            
            ax.plot(datas_fut, mediana, 'g--', linewidth=2, label='Previsão (Mediana)')
            ax.fill_between(datas_fut, p5, p95, alpha=0.3, color='green', label='IC 90%')
            
            prob_crise = prev.get('probabilidade_crise', 0)
            ax.text(0.02, 0.95, f'Prob. Crise: {prob_crise*100:.1f}%', 
                   transform=ax.transAxes, fontsize=14,
                   verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='red' if prob_crise > 0.5 else 'yellow', alpha=0.5))
        
        ax.set_ylabel('Preço', fontsize=12)
        ax.set_xlabel('Tempo (dias)', fontsize=12)
        ax.set_title('Previsões Probabilísticas de Crise - Modelo de Wilcke', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.figuras.append(fig)
    
    def _grafico_risco(self):
        """Gráfico 4: Análise de Risco"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
        
        # Drawdown
        precos = self.dados['preco'].values
        running_max = np.maximum.accumulate(precos)
        drawdown = (precos - running_max) / running_max * 100
        
        ax1.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.5, color='red')
        ax1.plot(drawdown, 'darkred', linewidth=1.5)
        ax1.set_ylabel('Drawdown (%)', fontsize=11)
        ax1.set_title('Drawdown', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Distribuição de retornos
        retornos = np.diff(np.log(precos)) * 100
        ax2.hist(retornos, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax2.axvline(x=np.percentile(retornos, 5), color='r', linestyle='--', linewidth=2, label='VaR 95%')
        ax2.set_xlabel('Retorno (%)', fontsize=11)
        ax2.set_ylabel('Frequência', fontsize=11)
        ax2.set_title('Distribuição de Retornos', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Expoente de Hurst
        if 'analisador' in self.resultados and 'memoria_longa' in self.resultados['analisador']:
            H = self.resultados['analisador']['memoria_longa'].get('hurst', 0.5)
            
            ax3.bar(['Hurst'], [H], color='purple', alpha=0.7)
            ax3.axhline(y=0.5, color='black', linestyle='--', label='Browniano')
            ax3.set_ylim(0, 1)
            ax3.set_ylabel('Expoente de Hurst', fontsize=11)
            ax3.set_title('Memória Longa', fontsize=12, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Resumo de Alertas
        if 'alertas' in self.resultados:
            alertas = self.resultados['alertas']
            n_critico = sum(1 for a in alertas if a['nivel'] == 'CRÍTICO')
            n_alto = sum(1 for a in alertas if a['nivel'] == 'ALTO')
            n_moderado = len(alertas) - n_critico - n_alto
            
            niveis = ['Crítico', 'Alto', 'Moderado']
            valores = [n_critico, n_alto, n_moderado]
            cores = ['red', 'orange', 'yellow']
            
            ax4.bar(niveis, valores, color=cores, alpha=0.7, edgecolor='black')
            ax4.set_ylabel('Número de Alertas', fontsize=11)
            ax4.set_title('Distribuição de Alertas', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        self.figuras.append(fig)
    
    def mostrar_graficos(self):
        """Exibe todos os gráficos"""
        plt.show()
