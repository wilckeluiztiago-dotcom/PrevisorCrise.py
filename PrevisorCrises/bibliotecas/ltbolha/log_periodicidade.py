"""
Análise de Log-Periodicidade
Autor: Luiz Tiago Wilcke

Detecta padrões log-periódicos que frequentemente precedem crashes de mercado.
Baseado no modelo LPPL (Log-Periodic Power Law).
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks


class AnalisadorLogPeriodico:
    """
    Detecta oscilações log-periódicas em séries temporais financeiras
    
    O modelo LPPL é dado por:
    ln(p(t)) = A + B(t_c - t)^m + C(t_c - t)^m cos(ω ln(t_c - t) + φ)
    
    Onde:
    - t_c: tempo crítico (momento esperado do crash)
    - m: expoente de potência (tipicamente 0 < m < 1)
    - ω: frequência log-periódica
    - φ: fase
    """
    
    def __init__(self, restricao_tc=None):
        """
        Parâmetros:
        -----------
        restricao_tc : tuple, opcional
            (tc_min, tc_max) - Restrições para o tempo crítico
        """
        self.restricao_tc = restricao_tc
        self.parametros_otimos = None
        
    def ajustar_lppl(self, tempos, precos, metodo='evolution'):
        """
        Ajusta o modelo LPPL aos dados
        
        Parâmetros:
        -----------
        tempos : array
            Array de tempos (indices ou timestamps)
        precos : array
            Preços correspondentes
        metodo : str
            'evolution' (robusto) ou 'gradient' (rápido)
            
        Retorna:
        --------
        parametros : dict
            Parâmetros ajustados do modelo LPPL
        """
        # Normalizar tempos para [0, 1]
        t_norm = (tempos - tempos[0]) / (tempos[-1] - tempos[0])
        log_precos = np.log(precos)
        
        # Definir bounds para os parâmetros
        # [A, B, tc, m, omega, phi]
        if self.restricao_tc is None:
            tc_min, tc_max = 1.01, 1.5  # tc deve estar no futuro
        else:
            tc_min, tc_max = self.restricao_tc
        
        bounds = [
            (log_precos.min(), log_precos.max()),  # A
            (-10, 10),  # B
            (tc_min, tc_max),  # tc
            (0.01, 0.99),  # m
            (1, 20),  # omega
            (-np.pi, np.pi)  # phi
        ]
        
        # Função objetivo
        def objetivo(params):
            A, B, tc, m, omega, phi = params
            predicao = self._modelo_lppl(t_norm, A, B, tc, m, omega, phi)
            return np.sum((log_precos - predicao) ** 2)
        
        # Otimização
        if metodo == 'evolution':
            resultado = differential_evolution(
                objetivo, 
                bounds, 
                seed=42, 
                maxiter=1000,
                popsize=20
            )
        else:
            # Chute inicial
            x0 = [
                np.mean(log_precos),  # A
                -1,  # B
                1.2,  # tc
                0.5,  # m
                6,  # omega
                0  # phi
            ]
            resultado = minimize(objetivo, x0, bounds=bounds, method='L-BFGS-B')
        
        # Extrair parâmetros
        A, B, tc, m, omega, phi = resultado.x
        
        # Desnormalizar tc
        tc_real = tempos[0] + tc * (tempos[-1] - tempos[0])
        
        self.parametros_otimos = {
            'A': A,
            'B': B,
            'tc': tc_real,
            't_c_norm': tc,
            'm': m,
            'omega': omega,
            'phi': phi,
            'erro_ajuste': resultado.fun,
            'sucesso': resultado.success
        }
        
        return self.parametros_otimos
    
    def prever(self, tempos_futuros):
        """
        Faz previsões usando o modelo LPPL ajustado
        
        Parâmetros:
        -----------
        tempos_futuros : array
            Tempos para os quais fazer previsões
            
        Retorna:
        --------
        precos_previstos : array
            Preços previstos (escala original)
        """
        if self.parametros_otimos is None:
            raise ValueError("Modelo ainda não foi ajustado. Execute ajustar_lppl() primeiro.")
        
        params = self.parametros_otimos
        predicao_log = self._modelo_lppl(
            tempos_futuros, 
            params['A'], 
            params['B'], 
            params['t_c_norm'], 
            params['m'], 
            params['omega'], 
            params['phi']
        )
        
        return np.exp(predicao_log)
    
    def calcular_confianca_lppl(self, tempos, precos):
        """
        Calcula um índice de confiança [0, 1] para o padrão LPPL detectado
        
        Parâmetros:
        -----------
        tempos : array
            Tempos
        precos : array
            Preços
            
        Retorna:
        --------
        confianca : float
            Índice de confiança [0, 1]
        metricas : dict
            Métricas detalhadas
        """
        if self.parametros_otimos is None:
            self.ajustar_lppl(tempos, precos)
        
        params = self.parametros_otimos
        
        # Critérios de qualidade
        # 1. Qualidade do ajuste (R²)
        log_precos = np.log(precos)
        t_norm = (tempos - tempos[0]) / (tempos[-1] - tempos[0])
        predicao = self._modelo_lppl(
            t_norm, params['A'], params['B'], params['t_c_norm'], 
            params['m'], params['omega'], params['phi']
        )
        ss_res = np.sum((log_precos - predicao) ** 2)
        ss_tot = np.sum((log_precos - np.mean(log_precos)) ** 2)
        r_quadrado = 1 - (ss_res / ss_tot)
        
        # 2. Parâmetros dentro de faixas típicas
        m_valido = 0.1 < params['m'] < 0.9
        omega_valido = 3 < params['omega'] < 15
        tc_valido = params['t_c_norm'] > 1.0  # tc no futuro
        
        parametros_validos = m_valido and omega_valido and tc_valido
        
        # 3. Significância da componente log-periódica
        C = params['B'] * np.cos(params['phi'])
        amplitude_relativa = abs(C) / abs(params['A'])
        componente_significativa = amplitude_relativa > 0.05
        
        # Combinar critérios
        confianca_base = r_quadrado
        
        if not parametros_validos:
            confianca_base *= 0.5
        
        if not componente_significativa:
            confianca_base *= 0.7
        
        confianca = np.clip(confianca_base, 0, 1)
        
        metricas = {
            'r_quadrado': r_quadrado,
            'm_valido': m_valido,
            'omega_valido': omega_valido,
            'tc_valido': tc_valido,
            'amplitude_relativa': amplitude_relativa,
            'confianca_final': confianca
        }
        
        return confianca, metricas
    
    def detectar_aceleracao(self, tempos, precos, janela=30):
        """
        Detecta aceleração superexponencial nos preços
        
        Parâmetros:
        -----------
        tempos : array
            Tempos
        precos : array
            Preços
        janela : int
            Tamanho da janela móvel
            
        Retorna:
        --------
        aceleracao : array
            Índice de aceleração ao longo do tempo
        """
        log_precos = np.log(precos)
        aceleracao = np.zeros(len(precos))
        
        for i in range(janela, len(precos)):
            # Ajustar modelo exponencial simples na janela
            t_janela = tempos[i-janela:i] - tempos[i-janela]
            p_janela = log_precos[i-janela:i]
            
            # Regressão linear em log-preços
            coef = np.polyfit(t_janela, p_janela, 1)
            taxa_crescimento = coef[0]
            
            # Detectar se a taxa está acelerando
            if i >= 2 * janela:
                t_anterior = tempos[i-2*janela:i-janela] - tempos[i-2*janela]
                p_anterior = log_precos[i-2*janela:i-janela]
                coef_anterior = np.polyfit(t_anterior, p_anterior, 1)
                taxa_anterior = coef_anterior[0]
                
                # Aceleração relativa
                if taxa_anterior != 0:
                    aceleracao[i] = (taxa_crescimento - taxa_anterior) / abs(taxa_anterior)
        
        return aceleracao
    
    def _modelo_lppl(self, t, A, B, tc, m, omega, phi):
        """
        Modelo LPPL (Log-Periodic Power Law)
        """
        # Evitar divisão por zero e valores inválidos
        dt = tc - t
        dt = np.maximum(dt, 1e-10)  # Evitar valores muito próximos de zero
        
        componente_potencia = B * (dt ** m)
        componente_periodica = B * (dt ** m) * np.cos(omega * np.log(dt) + phi)
        
        return A + componente_potencia + componente_periodica
    
    def analise_espectral(self, precos):
        """
        Análise espectral para detectar frequências log-periódicas dominantes
        
        Parâmetros:
        -----------
        precos : array
            Série de preços
            
        Retorna:
        --------
        frequencias_dominantes : array
            Frequências com maior amplitude
        espectro : array
            Espectro de potência completo
        """
        # Transformar para log-preços e detrend
        log_precos = np.log(precos)
        log_precos_detrend = log_precos - np.mean(log_precos)
        
        # FFT
        espectro = np.abs(fft(log_precos_detrend))
        frequencias = fftfreq(len(precos))
        
        # Considerar apenas frequências positivas
        mask = frequencias > 0
        espectro = espectro[mask]
        frequencias = frequencias[mask]
        
        # Encontrar picos
        picos, propriedades = find_peaks(espectro, height=np.max(espectro) * 0.2)
        
        if len(picos) > 0:
            frequencias_dominantes = frequencias[picos]
        else:
            frequencias_dominantes = np.array([])
        
        return frequencias_dominantes, espectro
