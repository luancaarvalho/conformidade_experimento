import random
import time
from collections import Counter
import re
from datetime import datetime
import os
import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import ConformityConfig, format_opinions, parse_llm_response, generate_unique_random_strings, handle_api_error

print("="*80)
print("INICIANDO SCRIPT DE ANÁLISE P(m) - VERSÃO DEBUG")
print("="*80)

# Obter a configuração centralizada
print("Carregando configuração centralizada...")
config = ConformityConfig()

# Atualizar configuração para este script específico
config.num_agents = 5        
config.num_simulations = 50   
config.max_steps = 50       


config.model = "gpt-4o" 

print(f"Configuração carregada:")
print(f"- Modelo: {config.model}")
print(f"- Número de agentes: {config.num_agents}")
print(f"- Número de simulações: {config.num_simulations}")
print(f"- Passos máximos: {config.max_steps}")
print(f"- Temperatura: {config.temperature}")
print()

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

print(f"Nomes das opiniões iniciais: {OPINION_NAMES_INITIAL}")
print("-"*50)


class SimulationMetrics:
    def __init__(self):
        self.opinion_history = []
        self.convergence_times = []
        self.opinion_changes = 0
        self.data_points = {}  # Para armazenar (m, decisão) para cálculo de P(m)
        
    def record_state(self, opinions):
        self.opinion_history.append(list(opinions.values()))
        
    def record_opinion_change(self):
        self.opinion_changes += 1
        
    def record_m_decision(self, m_value, decision):
        """
        Registra o valor de m e a decisão tomada.
        Args:
            m_value: Valor de m (opinião coletiva média)
            decision: Decisão tomada pelo agente (1 para A, -1 para B)
        """
        m_rounded = round(m_value, 2)  
        
        # Mostrar o que está sendo registrado
        decision_name = "A (+1)" if decision == 1 else "B (-1)"
        print(f"    >>> Registrando: m={m_rounded:.2f}, decisão={decision_name}")
        
        # Inicializar o contador se este valor de m ainda não foi visto
        if m_rounded not in self.data_points:
            self.data_points[m_rounded] = {'count_A': 0, 'count_total': 0}
            print(f"        Primeiro registro para m={m_rounded:.2f}")
        
        # Incrementar contadores
        self.data_points[m_rounded]['count_total'] += 1
        if decision == 1:  # Se escolheu A (+1)
            self.data_points[m_rounded]['count_A'] += 1
            
        # Mostrar contadores atualizados
        counts = self.data_points[m_rounded]
        print(f"        Agora: A={counts['count_A']}, Total={counts['count_total']}, " + 
              f"P(m)={counts['count_A']/counts['count_total']:.2f}")
        
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
        
        print("\nCalculando P(m) para todos os valores de m registrados:")
        for m, counts in sorted(self.data_points.items()):
            if counts['count_total'] > 0:
                p_m = counts['count_A'] / counts['count_total']
                m_values.append(m)
                p_m_values.append(p_m)
                print(f"  m = {m:.2f}: {counts['count_A']} escolhas A em {counts['count_total']} total → P(m) = {p_m:.4f}")
                
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
    print(f"\n  > Consultando LLM para agente {agent_id}...")
    
    # Formatar as opiniões usando a função de utilidade
    formatted_opinions = format_opinions(all_opinions, opinion_names, agent_id)
    print(f"  > Opiniões formatadas: {formatted_opinions}")
    
    # Obter mensagens formatadas da configuração
    messages = config.get_messages(formatted_opinions, opinion_names)
    
    # Fazer a chamada à API com tratamento de erros de limite de taxa
    max_retries = 5
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            print(f"  > Fazendo chamada à API (tentativa {retry_count+1}/{max_retries})...")
            # Fazer a chamada à API
            response = client.chat.completions.create(
                model=MODEL, 
                messages=messages, 
                max_tokens=config.max_tokens, 
                temperature=TEMPERATURE
            )
            
            # Adicionar tempo de espera após a chamada à API para prevenir rate limits
            wait_time = config.api_call_sleep_time
            print(f"  > Chamada bem-sucedida. Aguardando {wait_time}s antes da próxima...")
            time.sleep(wait_time)
            
            # Parsear a resposta usando a função de utilidade
            response_content = response.choices[0].message.content
            print(f"  > Resposta do LLM: {response_content}")
            
            new_opinion = parse_llm_response(response_content, opinion_names)
            print(f"  > Opinião extraída: {new_opinion} ({opinion_names[new_opinion] if new_opinion is not None else 'Nenhuma'})")
            
            return new_opinion
            
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
    print("\n" + "="*50)
    print("INICIANDO NOVA SIMULAÇÃO")
    print("="*50)
    
    # Inicializa as opiniões aleatoriamente
    opinions = {i: random.choice([-1, 1]) for i in range(NUM_AGENTS)}
    print(f"Opiniões iniciais: {opinions}")
    
    metrics = SimulationMetrics()
    opinion_names = {k: v for k, v in OPINION_NAMES_INITIAL.items()}
    print(f"Nomes das opiniões: {opinion_names}")
    
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
        print(f"\n--- Passo {step+1}/{MAX_STEPS} ---")
        step_start_time = time.time()
        
        # A cada passo de tempo (dt = 1/N), selecionamos um único agente aleatoriamente
        agent_id = random.choice(list(opinions.keys()))
        print(f"Agente selecionado: {agent_id}")
        
        # O agente selecionado consulta o LLM para atualizar sua opinião
        old_opinion = opinions[agent_id]
        print(f"Opinião atual do agente {agent_id}: {old_opinion} ({opinion_names[old_opinion]})")
        
        # Calcular m ANTES da consulta ao LLM
        current_m = calculate_m(list(opinions.values()))
        print(f"Opinião coletiva (m) atual: {current_m:.4f}")
        
        # --- INÍCIO: Novo Bloco de Embaralhamento k/z ---
        # Com 50% de chance, troca os nomes das opiniões PARA ESTE PASSO
        if random.random() <= 0.5:
            # Troca os valores associados às chaves 1 e -1
            temp_label = opinion_names[1]
            opinion_names[1] = opinion_names[-1]
            opinion_names[-1] = temp_label
            print(f"  >>> NOMES TROCADOS: +1={opinion_names[1]}, -1={opinion_names[-1]}")
        else:
            print(f"  >>> NOMES MANTIDOS: +1={opinion_names[1]}, -1={opinion_names[-1]}")
        # --- FIM: Novo Bloco de Embaralhamento k/z ---
            
        new_opinion = get_llm_opinion(agent_id, opinions, opinion_names)
        
        # Registrar m e a decisão para cálculo de P(m)
        if new_opinion is not None:
            metrics.record_m_decision(current_m, new_opinion)
            
        # Atualiza a opinião se necessário
        if new_opinion is not None and new_opinion != old_opinion:
            print(f"  >>> MUDANÇA: Agente {agent_id} alterou de {opinion_names[old_opinion]} para {opinion_names[new_opinion]}")
            opinions[agent_id] = new_opinion
            metrics.record_opinion_change()
        else:
            print(f"  >>> SEM MUDANÇA: Agente {agent_id} manteve opinião {opinion_names[old_opinion]}")
        
        # Registra o estado atual
        metrics.record_state(opinions)
        
        # Recalcular m após a possível mudança
        new_m = calculate_m(list(opinions.values()))
        print(f"  Novo valor de m após decisão: {new_m:.4f}")
        
        # Mostrar todas as opiniões atuais
        print(f"  Estado atual: {opinions}")
        counts = Counter(opinions.values())
        print(f"  Distribuição: +1: {counts.get(1, 0)} agentes, -1: {counts.get(-1, 0)} agentes")
        
        # Verifica convergência
        if check_consensus(list(opinions.values())):
            metrics.convergence_times.append(step)
            output_lines.append(f"\nConvergência atingida após {step+1} etapas!")
            print(f"\n>>> CONVERGÊNCIA ATINGIDA após {step+1} etapas! <<<")
            break
            
        step_end_time = time.time()
        step_duration = step_end_time - step_start_time
        print(f"  Tempo do passo: {step_duration:.2f} segundos")
        print("-"*40)
            
    else:
        output_lines.append(f"\nSimulação interrompida após {MAX_STEPS} etapas sem convergência.")
        print(f"\n>>> Simulação interrompida após {MAX_STEPS} etapas sem convergência <<<")
    
    end_time = time.time()
    final_m = calculate_m(list(opinions.values()))
    final_counts = Counter(opinion_names[op] for op in opinions.values())
    
    simulation_duration = end_time - start_time
    
    print("\n--- RESULTADOS FINAIS ---")
    print(f"Opiniões Finais: {final_counts}")
    print(f"Opinião Coletiva Final (m): {final_m:.4f}")
    print(f"Tempo Total de Simulação: {simulation_duration:.2f} segundos")
    
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
    print("\n--- GERANDO GRÁFICO DE P(m) ---")
    print(f"Pontos de dados: {len(m_values)}")
    for i, (m, p) in enumerate(zip(m_values, p_m_values)):
        print(f"  Ponto {i+1}: m={m:.2f}, P(m)={p:.4f}")
    
    plt.figure(figsize=(10, 6))
    
    # Plotar pontos
    plt.scatter(m_values, p_m_values, color='blue', alpha=0.7, s=100, label='Dados observados')
    
    # Tentar ajustar uma curva sigmóide
    try:
        # Parâmetros iniciais (L, x0, k, b)
        p0 = [1.0, 0.0, 4.0, 0.0]
        
        print("Tentando ajustar curva sigmóide...")
        # Ajustar a curva
        popt, _ = curve_fit(sigmoid, m_values, p_m_values, p0=p0)
        print(f"Parâmetros ajustados: L={popt[0]:.4f}, x0={popt[1]:.4f}, k={popt[2]:.4f}, b={popt[3]:.4f}")
        
        # Gerar pontos para a curva
        x_fit = np.linspace(min(m_values), max(m_values), 100)
        y_fit = sigmoid(x_fit, *popt)
        
        # Plotar a curva ajustada
        plt.plot(x_fit, y_fit, 'r-', linewidth=2, 
                 label=f'Ajuste sigmóide\nL={popt[0]:.2f}, x0={popt[1]:.2f}, k={popt[2]:.2f}, b={popt[3]:.2f}')
        plt.legend(fontsize=10)
    except Exception as e:
        print(f"Erro ao ajustar curva: {str(e)}")
        print("Continuando sem a curva ajustada.")
    
    # Configurar o gráfico
    plt.xlabel(r'Opinião coletiva $(m)$', fontsize=14)
    plt.ylabel(r'Probabilidade de adoção $P(m)$', fontsize=14)
    plt.title(f'Probabilidade de adoção $P(m)$ - {MODEL}', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='P(m)=0.5')
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='m=0')
    
    # Texto explicativo
    explanation = """
    P(m) = Probabilidade de adoção da opinião A
    m = Opinião coletiva média
    m = -1: Todos têm opinião B
    m = 0: Metade A, metade B 
    m = +1: Todos têm opinião A
    """
    plt.figtext(0.02, 0.02, explanation, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Salvar ou exibir
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo em: {output_file}")
    
    print("Exibindo gráfico...")
    plt.show()

def save_p_m_data(m_values, p_m_values, output_file):
    """
    Salva os dados de P(m) em um arquivo CSV.
    
    Args:
        m_values: Lista com valores de m
        p_m_values: Lista com valores de P(m)
        output_file: Caminho para salvar os dados
    """
    print(f"\n--- SALVANDO DADOS DE P(m) em {output_file} ---")
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['m', 'P(m)'])
        for m, p in zip(m_values, p_m_values):
            writer.writerow([m, p])
            print(f"  Gravado: m={m:.2f}, P(m)={p:.4f}")
    print(f"Dados salvos com sucesso em: {output_file}")

# --- Código Principal ---
if __name__ == "__main__":
    print("\n" + "="*80)
    print("INICIANDO ANÁLISE P(m) - EXECUÇÃO PRINCIPAL")
    print("="*80)
    
    # Criar diretórios para resultados
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = "results/p_m_analysis_debug"
    os.makedirs(output_dir, exist_ok=True)
    
    # Arquivos de saída
    model_name = MODEL.replace('-', '_')
    output_file = f"{output_dir}/{model_name}_p_m_analysis_{timestamp}.txt"
    output_csv = f"{output_dir}/{model_name}_p_m_data_{timestamp}.csv"
    output_plot = f"{output_dir}/{model_name}_p_m_plot_{timestamp}.png"
    
    print(f"Arquivos de saída:")
    print(f"- Texto: {output_file}")
    print(f"- CSV: {output_csv}")
    print(f"- Gráfico: {output_plot}")
    
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
        print(f"\nINICIANDO SIMULAÇÃO {i+1}/{NUM_SIMULATIONS} em {datetime.now().strftime('%H:%M:%S')}")
        
        # Executar a simulação
        metrics, final_opinions, sim_output_lines = run_simulation()
        
        # Mostrar os dados coletados nesta simulação
        print("\n--- DADOS DE P(m) DESTA SIMULAÇÃO ---")
        for m, counts in sorted(metrics.data_points.items()):
            p_m = counts['count_A'] / counts['count_total'] if counts['count_total'] > 0 else 0
            print(f"  m={m:.2f}: {counts['count_A']} escolhas A em {counts['count_total']} total -> P(m)={p_m:.4f}")
        
        # Acumular dados para P(m)
        print("\nAcumulando dados para cálculo global de P(m)...")
        for m, counts in metrics.data_points.items():
            if m not in combined_data_points:
                combined_data_points[m] = {'count_A': 0, 'count_total': 0}
                print(f"  Novo valor de m={m:.2f} adicionado ao conjunto global")
            
            combined_data_points[m]['count_A'] += counts['count_A']
            combined_data_points[m]['count_total'] += counts['count_total']
            
            # Mostrar dados acumulados para este m
            cur_counts = combined_data_points[m]
            print(f"  m={m:.2f} agora tem: A={cur_counts['count_A']}, Total={cur_counts['count_total']}")
        
        # Registrar saída
        sim_end_time = time.time()
        sim_duration = sim_end_time - sim_start_time
        
        print(f"\nSimulação {i+1} concluída em {datetime.now().strftime('%H:%M:%S')}")
        print(f"Duração: {sim_duration:.2f} segundos")
        print(f"{'-'*30}")
        
        all_output_lines.append(f"Simulação {i+1}")
        all_output_lines.extend(sim_output_lines)
    
    # Calcular P(m) combinado de todas as simulações
    print("\n" + "="*50)
    print("CALCULANDO P(m) GLOBAL DE TODAS AS SIMULAÇÕES")
    print("="*50)
    
    m_values = []
    p_m_values = []
    
    for m, counts in sorted(combined_data_points.items()):
        if counts['count_total'] > 0:
            p_m = counts['count_A'] / counts['count_total']
            m_values.append(m)
            p_m_values.append(p_m)
            print(f"m = {m:.2f}: {counts['count_A']} escolhas A em {counts['count_total']} total → P(m) = {p_m:.4f}")
    
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
    
    print("\n" + "="*80)
    print("FIM DA EXECUÇÃO")
    print("="*80)
