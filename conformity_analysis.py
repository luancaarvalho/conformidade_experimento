"""
Análise de Conformidade: Relação entre a fração de opiniões opostas e a taxa de conformidade
Compara o comportamento entre diferentes modelos de LLM (GPT-4o e GPT-3.5)
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
import pandas as pd

# Importar a configuração centralizada e utilitários
from config import ConformityConfig, format_opinions, parse_llm_response, generate_unique_random_strings, handle_api_error

# --- Métricas de Convergência ---
class SimulationMetrics:
    def __init__(self):
        self.opinion_history = []
        self.convergence_times = []
        self.opinion_changes = 0
        self.conformity_events = 0  # Contador de eventos de conformidade
        self.non_conformity_events = 0  # Contador de eventos que não resultaram em conformidade
        
    def record_state(self, opinions):
        """Registra o estado atual das opiniões."""
        self.opinion_history.append(list(opinions.values()))
        
    def record_opinion_change(self, old_opinion, new_opinion, majority_opinion):
        """
        Registra uma mudança de opinião, identificando se foi conformidade.
        
        Args:
            old_opinion: Opinião anterior do agente
            new_opinion: Nova opinião do agente
            majority_opinion: Opinião da maioria no momento da mudança
        """
        self.opinion_changes += 1
        
        # Se o agente mudou para a opinião da maioria, é um evento de conformidade
        if new_opinion == majority_opinion and old_opinion != majority_opinion:
            self.conformity_events += 1
        else:
            self.non_conformity_events += 1
        
    def get_convergence_rate(self):
        """Retorna a taxa de convergência."""
        return len(self.convergence_times) / len(self.opinion_history) if self.opinion_history else 0
    
    def get_conformity_rate(self):
        """Retorna a taxa de conformidade (proporção de mudanças que seguiram a maioria)"""
        total_events = self.conformity_events + self.non_conformity_events
        return self.conformity_events / total_events if total_events > 0 else 0


# --- Funções Auxiliares ---
def calculate_m(opinions_list):
    """Calcula a opinião coletiva média m."""
    return sum(opinions_list) / len(opinions_list)

def get_majority_opinion(opinions_list):
    """Determina a opinião da maioria no grupo."""
    counts = Counter(opinions_list)
    return counts.most_common(1)[0][0] if counts else None

def check_consensus(opinions_list, threshold=1.0):
    """Verifica se o consenso foi atingido."""
    from collections import Counter
    counter = Counter(opinions_list)
    most_common_opinion, count = counter.most_common(1)[0]
    return count / len(opinions_list) >= threshold

def initialize_opinions_with_magnetization(num_agents, magnetization):
    """
    Inicializa opiniões com base na magnetização inicial.
    
    Args:
        num_agents: Número total de agentes
        magnetization: Valor entre -1 e 1 que representa a magnetização inicial
                     -1: todos os agentes têm opinião -1
                      0: metade dos agentes têm cada opinião
                     +1: todos os agentes têm opinião +1
        
    Returns:
        Dicionário de opiniões inicializadas
    """
    # Garantir que a magnetização está entre -1 e 1
    magnetization = max(-1, min(1, magnetization))
    
    # Calcular quantos agentes terão a opinião -1
    # Converter magnetização para fração: (1 - magnetization) / 2
    # Quando magnetização = -1 → fração = 1 (todos com opinião -1)
    # Quando magnetização = 0 → fração = 0.5 (metade com cada opinião)
    # Quando magnetização = 1 → fração = 0 (ninguém com opinião -1)
    fraction_negative = (1 - magnetization) / 2
    count_negative = int(num_agents * fraction_negative)
    count_positive = num_agents - count_negative
    
    # Criar o dicionário de opiniões
    opinions = {}
    
    # Atribuir opinião -1
    for i in range(count_negative):
        opinions[i] = -1
        
    # Atribuir opinião 1
    for i in range(count_negative, num_agents):
        opinions[i] = 1
        
    return opinions

def get_llm_opinion(agent_id, all_opinions, opinion_names, client, model, temperature, max_tokens, api_call_sleep_time, config):
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
                model=model, 
                messages=messages, 
                max_tokens=max_tokens, 
                temperature=temperature
            )
            
            # Adicionar tempo de espera após a chamada à API para prevenir rate limits
            time.sleep(api_call_sleep_time)
            
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

# --- Função para o formato S (Sigmóide) ---
def sigmoid(x, L, x0, k, b):
    """
    Função sigmóide generalizada.
    L: valor máximo da curva
    x0: ponto médio da curva
    k: inclinação da curva
    b: deslocamento vertical
    """
    return L / (1 + np.exp(-k * (x - x0))) + b

# --- Simulação Principal ---
def run_simulation(config, model, magnetization=0.0, num_steps=None):
    """
    Executa uma simulação completa.
    
    Args:
        config: Configuração de simulação
        model: Nome do modelo a ser usado
        magnetization: Magnetização inicial (-1 a 1)
                    -1: todos opinião -1
                     0: metade-metade
                    +1: todos opinião +1
        num_steps: Número de passos de simulação (se None, usa config.max_steps)
        
    Returns:
        Métricas, opiniões finais, linhas de saída, magnetização usada, taxa de conformidade
    """
    # Configurações
    client = config.get_client()
    num_agents = config.num_agents
    temperature = config.temperature
    max_tokens = config.max_tokens
    api_call_sleep_time = config.api_call_sleep_time
    opinion_names_initial = config.opinions_initial
    max_steps = num_steps if num_steps is not None else config.max_steps
    convergence_threshold = config.convergence_threshold
    
    # Inicializa as opiniões com a magnetização especificada
    opinions = initialize_opinions_with_magnetization(num_agents, magnetization)
    metrics = SimulationMetrics()
    opinion_names = {k: v for k, v in opinion_names_initial.items()}
    output_lines = []
    
    # Calcular as proporções a partir da magnetização
    fraction_negative = (1 - magnetization) / 2
    fraction_positive = (1 + magnetization) / 2
    
    output_lines.extend([
        f"\nIniciando simulação com temperatura {temperature}",
        f"Número de agentes: {num_agents}",
        f"Modelo: {model}",
        f"Magnetização inicial: {magnetization:.2f}",
        f"Proporção inicial: {fraction_positive*100:.1f}% {opinion_names[1]} (+1), {fraction_negative*100:.1f}% {opinion_names[-1]} (-1)",
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
    # Calcular magnetização inicial real (para verificação)
    initial_m = calculate_m(list(opinions.values()))
    print(f"Distribuição: +1: {counts[1]} agentes ({(counts[1]/num_agents)*100:.1f}%), -1: {counts[-1]} agentes ({(counts[-1]/num_agents)*100:.1f}%), Magnetização: {initial_m:.2f}")
    
    for step in range(max_steps):
        step_start_time = time.time()
        
        # A cada passo de tempo (dt = 1/N), selecionamos um único agente aleatoriamente
        agent_id = random.choice(list(opinions.keys()))
        
        # O agente selecionado consulta o LLM para atualizar sua opinião
        old_opinion = opinions[agent_id]
        
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
            
        new_opinion = get_llm_opinion(
            agent_id, 
            opinions, 
            opinion_names, 
            client, 
            model, 
            temperature, 
            max_tokens, 
            api_call_sleep_time, 
            config
        )
        
        # Determinar a opinião majoritária antes da mudança
        majority_opinion = get_majority_opinion(list(opinions.values()))
        
        # Atualiza a opinião se necessário
        if new_opinion is not None and new_opinion != old_opinion:
            opinions[agent_id] = new_opinion
            metrics.record_opinion_change(old_opinion, new_opinion, majority_opinion)
            
            # Imprimir mudança de opinião
            if agent_id == 0 or step % 10 == 0:
                conformity_txt = "CONFORMIDADE" if new_opinion == majority_opinion else "NÃO-CONFORMIDADE"
                print(f"  Mudança de opinião: Agente {agent_id} alterou de {opinion_names[old_opinion]} para {opinion_names[new_opinion]} ({conformity_txt})")
        
        # Registra o estado atual
        metrics.record_state(opinions)
        
        # Calcula a opinião coletiva
        current_m = calculate_m(list(opinions.values()))
        
        # Verifica convergência
        if check_consensus(list(opinions.values()), threshold=convergence_threshold):
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
        output_lines.append(f"\nSimulação interrompida após {max_steps} etapas sem convergência.")
        print(f"\n>>> Simulação interrompida após {max_steps} etapas sem convergência <<<")
    
    end_time = time.time()
    final_m = calculate_m(list(opinions.values()))
    final_counts = Counter(opinion_names[op] for op in opinions.values())
    
    simulation_duration = end_time - start_time
    
    # Calcular a taxa de conformidade
    conformity_rate = metrics.get_conformity_rate()
    
    output_lines.extend([
        "-" * 30,
        "--- Resultado Final ---",
        f"Opiniões Finais: {final_counts}",
        f"Opinião Coletiva Final (m): {final_m:.2f}",
        f"Estado Final: {[opinion_names[opinions[i]] for i in range(num_agents)]}",
        f"Tempo de término: {datetime.now().strftime('%H:%M:%S')}",
        f"Tempo Total de Simulação: {simulation_duration:.2f} segundos",
        f"Taxa de Mudança: {metrics.opinion_changes / (step + 1):.2f}",
        f"Taxa de Conformidade: {conformity_rate:.2%}",
        f"Taxa de Convergência: {metrics.get_convergence_rate():.2%}",
        "=" * 80,
        ""
    ])
    
    return metrics, opinions, output_lines, magnetization, conformity_rate

def run_conformity_analysis(config, model, output_dir=None):
    """
    Executa uma análise da relação entre magnetização inicial e conformidade.
    Testa vários valores de magnetização inicial e mede a taxa de conformidade para cada um.
    
    Args:
        config: Configuração de simulação
        model: Nome do modelo a usar
        output_dir: Diretório para salvar resultados (se None, usa o padrão)
    
    Returns:
        magnetization_values: Lista de valores de magnetização testados
        conformity_rates: Lista de taxas de conformidade correspondentes
        results_file: Caminho para o arquivo de resultados
        csv_results: Lista de tuplas (magnetização, taxa) para exportação CSV
    """
    print(f"\n{'='*50}")
    print(f"ANÁLISE DE CONFORMIDADE VS. MAGNETIZAÇÃO INICIAL")
    print(f"Modelo: {model}")
    print(f"{'='*50}")
    
    # Lista de magnetizações a testar (de -1 a 1 em 21 pontos)
    magnetization_values = np.linspace(-1.0, 1.0, 21)
    conformity_rates = []
    csv_results = []  # Lista de tuplas (magnetização, taxa) para exportá-las posteriormente
    
    # Criar diretório para os resultados da análise
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if output_dir is None:
        output_dir = "results/conformity_analysis"
    
    # Garantir que o diretório existe e criar subdiretório com timestamp para organizar melhor
    output_dir = f"{output_dir}/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Resultados serão salvos em: {output_dir}")
    
    model_short_name = model.replace("-", "_")
    results_file = f"{output_dir}/conformity_analysis_{model_short_name}_{timestamp}.txt"
    
    # Cabeçalho do arquivo de resultados
    with open(results_file, 'w') as f:
        f.write("# Análise de Conformidade vs. Magnetização Inicial\n")
        f.write("# Modelo: {}\n".format(model))
        f.write("# Temperatura: {}\n".format(config.temperature))
        f.write("# Número de Agentes: {}\n".format(config.num_agents))
        f.write("# Data: {}\n\n".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        f.write("magnetizacao_inicial,taxa_conformidade\n")
    
    for magnetization in magnetization_values:
        print(f"\nTestando magnetização inicial: {magnetization:.2f}")
        
        # Limitar o número de passos para tornar a análise mais eficiente
        reduced_steps = min(100, config.max_steps)
        
        # Executar simulação com esta magnetização inicial
        metrics, opinions, output_lines, magnetization_used, conformity_rate = run_simulation(
            config, 
            model, 
            magnetization=magnetization,
            num_steps=reduced_steps
        )
        
        # Armazenar resultado
        conformity_rates.append(conformity_rate)
        csv_results.append((magnetization, conformity_rate))
        
        # Adicionar ao arquivo de resultados
        with open(results_file, 'a') as f:
            f.write(f"{magnetization:.2f},{conformity_rate:.4f}\n")
        
        print(f"Magnetização inicial: {magnetization:.2f}, Taxa de conformidade: {conformity_rate:.2%}")
    
    return magnetization_values, conformity_rates, results_file, csv_results

def plot_conformity_comparison(models_data, output_dir="results/conformity_analysis"):
    """
    Plota um gráfico comparativo das taxas de conformidade para diferentes modelos.
    
    Args:
        models_data: Dicionário com nomes de modelos como chaves e tuplas 
                    (minority_fractions, conformity_rates) como valores
        output_dir: Diretório para salvar o gráfico
    """
    plt.figure(figsize=(12, 8))
    
    for model_name, (fractions, rates) in models_data.items():
        # Plotar os pontos de dados
        plt.plot(fractions, rates, 'o', label=f'Dados {model_name}')
        
        # Tentar ajustar uma curva sigmóide aos dados
        try:
            # Valores iniciais para os parâmetros da sigmóide
            # L (altura), x0 (ponto médio), k (inclinação), b (base)
            p0 = [max(rates) - min(rates), 0.25, 10, min(rates)]
            
            # Ajustar a curva
            popt, pcov = curve_fit(sigmoid, fractions, rates, p0=p0, maxfev=5000)
            
            # Gerar pontos suaves para a curva
            x_fit = np.linspace(0, 0.5, 100)
            y_fit = sigmoid(x_fit, *popt)
            
            # Plotar a curva ajustada
            plt.plot(x_fit, y_fit, '-', label=f'Modelo Sigmóide {model_name}')
            
            # Imprimir parâmetros do ajuste
            print(f"\nParâmetros da sigmóide para {model_name}:")
            print(f"L (altura): {popt[0]:.4f}")
            print(f"x0 (ponto médio): {popt[1]:.4f}")
            print(f"k (inclinação): {popt[2]:.4f}")
            print(f"b (base): {popt[3]:.4f}")
            
        except Exception as e:
            print(f"Erro ao ajustar curva sigmóide para {model_name}: {e}")
            # Se falhar, plotar apenas uma linha conectando os pontos
            plt.plot(fractions, rates, '-', label=f'Linha {model_name}')
    
    plt.xlabel('Magnetização Inicial (m)', fontsize=14)
    plt.ylabel('Taxa de Conformidade', fontsize=14)
    plt.title('Comparação da Relação entre Magnetização Inicial e Conformidade', fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Adicionar anotações explicativas
    plt.figtext(0.02, 0.02, 
                "A curva sigmóide (em S) demonstra como a taxa de conformidade varia com a magnetização inicial.\n"
                "Magnetização = -1: todos os agentes começam com opinião -1\n"
                "Magnetização = 0: 50% dos agentes com cada opinião\n"
                "Magnetização = 1: todos os agentes começam com opinião 1\n"
                "Uma inclinação mais acentuada indica uma transição mais rápida entre regimes de baixa e alta conformidade.",
                fontsize=10, wrap=True)
    
    # Salvar o gráfico em múltiplos formatos
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(output_dir, exist_ok=True)
    
    # Salvar em PNG (alta resolução)
    plot_file_png = f"{output_dir}/conformity_comparison_plot_{timestamp}.png"
    plt.savefig(plot_file_png, dpi=300, bbox_inches='tight')
    print(f"\nGráfico PNG salvo em: {plot_file_png}")
    
    # Salvar em PDF (vetorial, adequado para publicações)
    plot_file_pdf = f"{output_dir}/conformity_comparison_plot_{timestamp}.pdf"
    plt.savefig(plot_file_pdf, format='pdf', bbox_inches='tight')
    print(f"Gráfico PDF salvo em: {plot_file_pdf}")
    
    # Mostrar gráfico na tela se estiver em ambiente interativo
    plt.show()
    
    plot_file = plot_file_png  # Retorna o arquivo PNG para compatibilidade
    
    return plot_file

# --- Função Principal ---
def main():
    """Função principal que executa a análise de conformidade para diferentes modelos."""
    print("\n" + "="*80)
    print("INICIANDO ANÁLISE DE CONFORMIDADE SOCIAL EM MODELOS DE LINGUAGEM")
    print("="*80)
    
    # Obter a configuração centralizada
    config = ConformityConfig()
    
    # Atualizar configuração para este script específico
    config.num_agents = 10
    config.num_simulations = 1
    config.max_steps = 100  # Reduzir para análise mais rápida (cada ponto da curva executa até 100 passos)
    
    # Definir modelos a testar
    models = ["gpt-4o", "gpt-3.5-turbo"]
    
    # Diretório para resultados
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"results/conformity_analysis/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nTodos os resultados serão salvos em: {output_dir}")
    print(f"Modelos a serem analisados: {', '.join(models)}")
    print(f"Número de agentes por simulação: {config.num_agents}")
    print(f"Máximo de passos por simulação: {config.max_steps}")
    print(f"Temperatura: {config.temperature}")
    
    # Arquivo CSV consolidado para todos os resultados
    csv_file = f"{output_dir}/conformity_analysis_results.csv"
    
    # Criar arquivo CSV com cabeçalho
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'magnetization', 'conformity_rate', 'timestamp', 'num_agents', 'temperature'])
    
    # Armazenar resultados de cada modelo
    models_results = {}
    
    for model in models:
        print(f"\n{'='*80}")
        print(f"EXECUTANDO ANÁLISE DE CONFORMIDADE PARA O MODELO: {model}")
        print(f"{'='*80}")
        
        # Executar análise para este modelo
        fractions, rates, results_file, csv_results = run_conformity_analysis(config, model, output_dir)
        
        # Armazenar resultados
        models_results[model] = (fractions, rates)
        
        # Adicionar resultados ao CSV consolidado
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for magnetization, rate in csv_results:
                writer.writerow([model, magnetization, rate, timestamp, config.num_agents, config.temperature])
        
        print(f"\nAnálise para {model} concluída.")
        print(f"Resultados salvos em: {results_file}")
    
    # Plotar comparação entre modelos
    comparison_plot = plot_conformity_comparison(models_results, output_dir)
    
    # Converter CSV para DataFrame e salvar em formato Excel para fácil visualização
    try:
        df = pd.read_csv(csv_file)
        excel_file = f"{output_dir}/conformity_analysis_results.xlsx"
        df.to_excel(excel_file, index=False)
        print(f"\nResultados consolidados salvos em Excel: {excel_file}")
    except Exception as e:
        print(f"\nAviso: Não foi possível criar arquivo Excel: {e}")
    
    print(f"\n{'='*80}")
    print(f"ANÁLISE DE CONFORMIDADE CONCLUÍDA PARA TODOS OS MODELOS")
    print(f"Diretório de resultados: {output_dir}")
    print(f"Arquivo CSV consolidado: {csv_file}")
    print(f"Gráfico de comparação: {comparison_plot}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
