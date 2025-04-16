"""
Análise de Probabilidade de Adoção P(m): Implementação baseada no gpt_model_tests.py
Calcula a probabilidade de adoção para diferentes valores de opinião coletiva (m)
"""

import random
import time
from collections import Counter
import re
from datetime import datetime
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Importar a configuração centralizada e utilitários
from config import ConformityConfig, format_opinions, parse_llm_response, generate_unique_random_strings, handle_api_error

# Obter a configuração centralizada
config = ConformityConfig()
# Atualizar configuração para este script específico
config.num_agents = 10
config.num_simulations = 20  # Aumentar número de simulações para coletar mais dados de P(m)

# Configurar o modelo a ser testado
config.model = "gpt-4o"  # Modelo a ser utilizado

client = config.get_client()
MODEL = config.model

# --- Parâmetros da Simulação ---
NUM_AGENTS = config.num_agents
NUM_SIMULATIONS = config.num_simulations
TEMPERATURE = config.temperature
OPINION_NAMES_INITIAL = config.opinions_initial
OPINION_VALUES = config.opinion_values
MAX_STEPS = config.max_steps
CONVERGENCE_THRESHOLD = config.convergence_threshold

# --- Métricas de Simulação ---
class SimulationMetrics:
    def __init__(self):
        self.opinion_history = []
        self.convergence_times = []
        self.opinion_changes = 0
        self.data_points = {}  # Para armazenar (m, decisão) para cálculo de P(m)
        
    def record_state(self, opinions):
        """Registra o estado atual das opiniões."""
        self.opinion_history.append(list(opinions.values()))
        
    def record_opinion_change(self):
        """Registra uma mudança de opinião."""
        self.opinion_changes += 1
        
    def record_m_decision(self, m_value, decision):
        """
        Registra o valor de m e a decisão tomada.
        Args:
            m_value: Valor de m (opinião coletiva média)
            decision: Decisão tomada pelo agente (1 para A, -1 para B)
        """
        m_rounded = round(m_value, 2)  # Arredondar para 2 casas decimais para binning
        
        if m_rounded not in self.data_points:
            self.data_points[m_rounded] = {'count_A': 0, 'count_total': 0}
        
        self.data_points[m_rounded]['count_total'] += 1
        if decision == 1:  # Se escolheu A
            self.data_points[m_rounded]['count_A'] += 1
        
    def get_convergence_rate(self):
        """Retorna a taxa de convergência."""
        return len(self.convergence_times) / len(self.opinion_history) if self.opinion_history else 0
    
    def calculate_p_m(self):
        """
        Calcula P(m) para todos os valores de m registrados.
        Returns:
            Tupla de listas (m_values, p_m_values) ordenadas por m
        """
        m_values = []
        p_m_values = []
        
        for m, counts in sorted(self.data_points.items()):
            if counts['count_total'] > 0:
                p_m = counts['count_A'] / counts['count_total']
                m_values.append(m)
                p_m_values.append(p_m)
                
        return m_values, p_m_values

# --- Funções Auxiliares ---
def calculate_m(opinions_list):
    """Calcula a opinião coletiva média m."""
    return sum(opinions_list) / len(opinions_list)

def check_consensus(opinions_list):
    """Verifica se o consenso foi atingido (100% de uma opinião)."""
    from collections import Counter
    counter = Counter(opinions_list)
    most_common_opinion, count = counter.most_common(1)[0]
    return count / len(opinions_list) >= CONVERGENCE_THRESHOLD

def get_llm_opinion(agent_id, all_opinions, opinion_names):
    """Consulta o LLM para obter a nova opinião de um agente."""
    # Formatar as opiniões usando a função de utilidade
    formatted_opinions = format_opinions(all_opinions, opinion_names, agent_id)
    
    # Obter mensagens formatadas da configuração
    messages = config.get_messages(formatted_opinions, opinion_names)
    
    # Fazer a chamada à API com tratamento de erros de limite de taxa
    max_retries = 5
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Fazer a chamada à API
            response = client.chat.completions.create(
                model=MODEL, 
                messages=messages, 
                max_tokens=config.max_tokens, 
                temperature=TEMPERATURE
            )
            
            # Adicionar tempo de espera após a chamada à API para prevenir rate limits
            time.sleep(config.api_call_sleep_time)
            
            # Parsear a resposta usando a função de utilidade
            return parse_llm_response(response.choices[0].message.content, opinion_names)
            
        except Exception as e:
            retry_count += 1
            error_message = str(e)
            print(f"\n!!! ERRO NA CHAMADA À API !!!")
            print(f"Detalhes do erro: {error_message}")
            print(f"Tentativa {retry_count} de {max_retries}")
            
            # Verificar se é erro de limite de taxa
            is_rate_limit = "rate_limit" in error_message.lower() or "rate limit" in error_message.lower()
            if is_rate_limit:
                print(f"Tipo de erro: Limite de taxa atingido")
            else:
                print(f"Tipo de erro: Outro erro de API")
            
            wait_time, should_retry = handle_api_error(e, retry_count, max_retries)
            
            if should_retry:
                print(f"Aguardando {wait_time:.2f} segundos antes de tentar novamente...")
                time.sleep(wait_time)
                print(f"Retomando após espera...")
            else:
                print(f"Erro após {max_retries} tentativas. Desistindo.")
                raise

# --- Função Sigmóide para Ajuste ---
def sigmoid(x, L, x0, k, b):
    """
    Função sigmóide generalizada para ajuste de curva.
    L: valor máximo da curva
    x0: ponto médio da curva
    k: inclinação da curva
    b: deslocamento vertical
    """
    return L / (1 + np.exp(-k * (x - x0))) + b

# --- Simulação Principal ---
def run_simulation():
    """Executa uma simulação completa com registro de P(m)."""
    # Inicializa as opiniões aleatoriamente
    opinions = {i: random.choice([-1, 1]) for i in range(NUM_AGENTS)}
    metrics = SimulationMetrics()
    opinion_names = {k: v for k, v in OPINION_NAMES_INITIAL.items()}
    output_lines = []
    
    output_lines.extend([
        f"\nIniciando simulação com temperatura {TEMPERATURE}",
        f"Número de agentes: {NUM_AGENTS}",
        f"Modelo: {MODEL}",
        f"Opiniões: {opinion_names[1]} (+1), {opinion_names[-1]} (-1)",
        ""
    ])
    
    current_m = calculate_m(list(opinions.values()))
    output_lines.extend([
        f"Estado inicial (m={current_m:.2f}):",
        str({k: opinion_names[v] for k, v in opinions.items()}),
        "-" * 30,
        ""
    ])
    
    start_time = time.time()
    output_lines.append(f"Tempo de início: {datetime.now().strftime('%H:%M:%S')}")
    
    # Imprimir estado inicial
    print(f"\n--- Estado Inicial da Simulação ---")
    print(f"Opinião coletiva inicial (m): {current_m:.2f}")
    counts = Counter(opinions.values())
    print(f"Distribuição: +1: {counts[1]} agentes, -1: {counts[-1]} agentes")
    
    for step in range(MAX_STEPS):
        step_start_time = time.time()
        
        # A cada passo de tempo (dt = 1/N), selecionamos um único agente aleatoriamente
        agent_id = random.choice(list(opinions.keys()))
        
        # O agente selecionado consulta o LLM para atualizar sua opinião
        old_opinion = opinions[agent_id]
        
        # Calcular m ANTES da consulta ao LLM
        current_m = calculate_m(list(opinions.values()))
        
        # --- INÍCIO: Novo Bloco de Embaralhamento k/z ---
        # Com 50% de chance, troca os nomes das opiniões PARA ESTE PASSO
        if random.random() <= 0.5:
            # Troca os valores associados às chaves 1 e -1
            temp_label = opinion_names[1]
            opinion_names[1] = opinion_names[-1]
            opinion_names[-1] = temp_label
            # Imprimir troca de nomes
            if step % 10 == 0 or agent_id == 0:
                print(f"  Nomes das opiniões trocados PARA ESTE PASSO: +1={opinion_names[1]}, -1={opinion_names[-1]}")
        elif step % 10 == 0 or agent_id == 0:
            # Imprimir se não houve troca (apenas para debug)
            print(f"  Mapeamento mantido PARA ESTE PASSO: +1={opinion_names[1]}, -1={opinion_names[-1]}")
        # --- FIM: Novo Bloco de Embaralhamento k/z ---
        
        # Imprimir progresso a cada 10 passos ou quando o agente 0 for selecionado
        if step % 10 == 0 or agent_id == 0:
            print(f"Passo {step}: Agente {agent_id} sendo consultado... (opinião atual: {opinion_names[old_opinion]})")
            
        new_opinion = get_llm_opinion(agent_id, opinions, opinion_names)
        
        # Registrar m e a decisão para cálculo de P(m)
        if new_opinion is not None:
            metrics.record_m_decision(current_m, new_opinion)
            
        # Atualiza a opinião se necessário
        if new_opinion is not None and new_opinion != old_opinion:
            opinions[agent_id] = new_opinion
            metrics.record_opinion_change()
            
            # Imprimir mudança de opinião
            if agent_id == 0 or step % 10 == 0:
                print(f"  Mudança de opinião: Agente {agent_id} alterou de {opinion_names[old_opinion]} para {opinion_names[new_opinion]}")
        
        # Registra o estado atual
        metrics.record_state(opinions)
        
        # Verifica convergência
        if check_consensus(list(opinions.values())):
            metrics.convergence_times.append(step)
            output_lines.append(f"\nConvergência atingida após {step+1} etapas!")
            print(f"\n>>> Convergência atingida após {step+1} etapas! <<<")
            break
            
        step_end_time = time.time()
        step_duration = step_end_time - step_start_time
            
        # Imprimir estado atual a cada 10 passos
        if step % 10 == 0:
            counts = Counter(opinions.values())
            print(f"Passo {step} - Opinião coletiva (m): {current_m:.2f}, +1: {counts[1]} agentes, -1: {counts[-1]} agentes")
            print(f"Tempo de execução do passo {step}: {step_duration:.2f} segundos")
            print(f"{'-'*30}")
        # Imprimir tempo de execução do passo para agente 0
        elif agent_id == 0:
            print(f"Tempo de execução do passo {step} (agente 0): {step_duration:.2f} segundos")
            
    else:
        output_lines.append(f"\nSimulação interrompida após {MAX_STEPS} etapas sem convergência.")
        print(f"\n>>> Simulação interrompida após {MAX_STEPS} etapas sem convergência <<<")
    
    end_time = time.time()
    final_m = calculate_m(list(opinions.values()))
    final_counts = Counter(opinion_names[op] for op in opinions.values())
    
    simulation_duration = end_time - start_time
    
    output_lines.extend([
        "-" * 30,
        "--- Resultado Final ---",
        f"Opiniões Finais: {final_counts}",
        f"Opinião Coletiva Final (m): {final_m:.2f}",
        f"Estado Final: {[opinion_names[opinions[i]] for i in range(NUM_AGENTS)]}",
        f"Tempo de término: {datetime.now().strftime('%H:%M:%S')}",
        f"Tempo Total de Simulação: {simulation_duration:.2f} segundos",
        f"Taxa de Mudança: {metrics.opinion_changes / (step + 1):.2f}",
        f"Taxa de Convergência: {metrics.get_convergence_rate():.2%}",
        "=" * 80,
        ""
    ])
    
    return metrics, opinions, output_lines

def plot_p_m(m_values, p_m_values, output_file=None):
    """
    Plota a curva P(m) e salva em um arquivo.
    
    Args:
        m_values: Lista com valores de m
        p_m_values: Lista com valores de P(m)
        output_file: Caminho para salvar o gráfico (se None, apenas exibe)
    """
    plt.figure(figsize=(10, 6))
    
    # Plotar pontos
    plt.scatter(m_values, p_m_values, color='blue', alpha=0.7)
    
    # Tentar ajustar uma curva sigmóide
    try:
        # Parâmetros iniciais (L, x0, k, b)
        p0 = [1.0, 0.0, 4.0, 0.0]
        
        # Ajustar a curva
        popt, _ = curve_fit(sigmoid, m_values, p_m_values, p0=p0)
        
        # Gerar pontos para a curva
        x_fit = np.linspace(min(m_values), max(m_values), 100)
        y_fit = sigmoid(x_fit, *popt)
        
        # Plotar a curva ajustada
        plt.plot(x_fit, y_fit, 'r-', label=f'Ajuste: L={popt[0]:.2f}, x0={popt[1]:.2f}, k={popt[2]:.2f}, b={popt[3]:.2f}')
        plt.legend()
    except Exception as e:
        print(f"Erro ao ajustar curva: {str(e)}")
    
    # Configurar o gráfico
    plt.xlabel(r'Opinião coletiva $(m)$', fontsize=14)
    plt.ylabel(r'Probabilidade de adoção $P(m)$', fontsize=14)
    plt.title(f'Probabilidade de adoção $P(m)$ - {MODEL}', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # Salvar ou exibir
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo em: {output_file}")
    plt.show()

def save_p_m_data(m_values, p_m_values, output_file):
    """
    Salva os dados de P(m) em um arquivo CSV.
    
    Args:
        m_values: Lista com valores de m
        p_m_values: Lista com valores de P(m)
        output_file: Caminho para salvar os dados
    """
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['m', 'P(m)'])
        for m, p in zip(m_values, p_m_values):
            writer.writerow([m, p])
    print(f"Dados salvos em: {output_file}")

# --- Código Principal ---
if __name__ == "__main__":
    # Criar diretórios para resultados
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = "results/p_m_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Arquivos de saída
    model_name = MODEL.replace('-', '_')
    output_file = f"{output_dir}/{model_name}_p_m_analysis_{timestamp}.txt"
    output_csv = f"{output_dir}/{model_name}_p_m_data_{timestamp}.csv"
    output_plot = f"{output_dir}/{model_name}_p_m_plot_{timestamp}.png"
    
    # Variáveis para acumular dados de todas as simulações
    all_output_lines = []
    all_m_values = []
    all_p_m_values = []
    combined_data_points = {}
    
    print(f"\n{'='*50}")
    print(f"INICIANDO ANÁLISE DE P(m) COM {NUM_SIMULATIONS} SIMULAÇÕES")
    print(f"Modelo: {MODEL}")
    print(f"{'='*50}")
    
    total_start_time = time.time()
    
    # Executar simulações
    for i in range(NUM_SIMULATIONS):
        sim_start_time = time.time()
        print(f"\nIniciando simulação {i+1}/{NUM_SIMULATIONS} em {datetime.now().strftime('%H:%M:%S')}")
        
        # Executar a simulação
        metrics, final_opinions, sim_output_lines = run_simulation()
        
        # Acumular dados para P(m)
        for m, counts in metrics.data_points.items():
            if m not in combined_data_points:
                combined_data_points[m] = {'count_A': 0, 'count_total': 0}
            combined_data_points[m]['count_A'] += counts['count_A']
            combined_data_points[m]['count_total'] += counts['count_total']
        
        # Registrar saída
        sim_end_time = time.time()
        sim_duration = sim_end_time - sim_start_time
        
        print(f"Simulação {i+1} concluída em {datetime.now().strftime('%H:%M:%S')}")
        print(f"Duração: {sim_duration:.2f} segundos")
        print(f"{'-'*30}")
        
        all_output_lines.append(f"Simulação {i+1}")
        all_output_lines.extend(sim_output_lines)
    
    # Calcular P(m) combinado de todas as simulações
    m_values = []
    p_m_values = []
    
    for m, counts in sorted(combined_data_points.items()):
        if counts['count_total'] > 0:
            p_m = counts['count_A'] / counts['count_total']
            m_values.append(m)
            p_m_values.append(p_m)
    
    # Salvar dados de P(m)
    save_p_m_data(m_values, p_m_values, output_csv)
    
    # Plotar resultados
    plot_p_m(m_values, p_m_values, output_plot)
    
    # Salvar saída de texto
    all_output_lines.append("\n" + "="*50)
    all_output_lines.append("RESULTADOS DE P(m)")
    all_output_lines.append("="*50)
    
    for m, p in zip(m_values, p_m_values):
        all_output_lines.append(f"m = {m:.2f}, P(m) = {p:.4f}")
    
    # Calcular tempo total
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    all_output_lines.append("\n" + "="*50)
    all_output_lines.append(f"ANÁLISE CONCLUÍDA")
    all_output_lines.append(f"Tempo total de execução: {total_duration:.2f} segundos")
    all_output_lines.append(f"Tempo médio por simulação: {total_duration/NUM_SIMULATIONS:.2f} segundos")
    all_output_lines.append(f"{'='*50}")
    
    # Salvar saída completa
    with open(output_file, 'w') as f:
        f.write("\n".join(all_output_lines))
    
    print(f"\n{'='*50}")
    print(f"ANÁLISE CONCLUÍDA")
    print(f"Tempo total de execução: {total_duration:.2f} segundos")
    print(f"Tempo médio por simulação: {total_duration/NUM_SIMULATIONS:.2f} segundos")
    print(f"{'='*50}")
    print(f"\nResultados salvos em:")
    print(f"- Texto: {output_file}")
    print(f"- Dados: {output_csv}")
    print(f"- Gráfico: {output_plot}")
