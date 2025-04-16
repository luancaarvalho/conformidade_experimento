import random
import time
from collections import Counter
import re
from datetime import datetime
import os

# Importar a configuração centralizada e utilitários
from config import ConformityConfig, format_opinions, parse_llm_response, generate_unique_random_strings, handle_api_error

# Obter a configuração centralizada
config = ConformityConfig()
# Atualizar configuração para este script específico
config.num_agents = 10
config.num_simulations = 1  # Definir número de simulações para 1

# Atualizar o modelo - anteriormente era gpt-3.5-turbo
# Histórico de modelos testados:
# - 2025-04-04: Testado com gpt-3.5-turbo
# config.model = "gpt-4-turbo-preview"  # Atualizado para GPT-4 Turbo
config.model = "gpt-4o"  # Atualizado para GPT-4o

client = config.get_client()
MODEL = config.model

# --- Parâmetros da Simulação ---
NUM_AGENTS = config.num_agents
NUM_SIMULATIONS = config.num_simulations  # Usar o parâmetro de configuração
TEMPERATURE = config.temperature
OPINION_NAMES_INITIAL = config.opinions_initial
OPINION_VALUES = config.opinion_values
MAX_STEPS = config.max_steps
CONVERGENCE_THRESHOLD = config.convergence_threshold

# --- Métricas de Convergência ---
class SimulationMetrics:
    def __init__(self):
        self.opinion_history = []
        self.convergence_times = []
        self.opinion_changes = 0
        
    def record_state(self, opinions):
        self.opinion_history.append(list(opinions.values()))
        
    def record_opinion_change(self):
        self.opinion_changes += 1
        
    def get_convergence_rate(self):
        return len(self.convergence_times) / len(self.opinion_history) if self.opinion_history else 0

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

# --- Simulação Principal ---
def run_simulation():
    """Executa uma simulação completa."""
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
        
        # Atualiza a opinião se necessário
        if new_opinion is not None and new_opinion != old_opinion:
            opinions[agent_id] = new_opinion
            metrics.record_opinion_change()
            
            # Imprimir mudança de opinião
            if agent_id == 0 or step % 10 == 0:
                print(f"  Mudança de opinião: Agente {agent_id} alterou de {opinion_names[old_opinion]} para {opinion_names[new_opinion]}")
        
        # Registra o estado atual
        metrics.record_state(opinions)
        
        # Calcula a opinião coletiva
        current_m = calculate_m(list(opinions.values()))
        
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

# Executar simulações
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# Atualizar o diretório de saída para refletir o novo nome de arquivo/propósito
output_dir = "results/gpt_model_tests"
os.makedirs(output_dir, exist_ok=True)  # Garantir que o diretório existe
# Destacar o nome do modelo no início do nome do arquivo
output_file = f"{output_dir}/{MODEL.replace('-', '_')}_simulacao_{timestamp}.txt"
all_output_lines = []

print(f"\n{'='*50}")
print(f"INICIANDO CONJUNTO DE {NUM_SIMULATIONS} SIMULAÇÕES")
print(f"{'='*50}")
total_start_time = time.time()

for i in range(NUM_SIMULATIONS):  # Executar simulações configuradas
    sim_start_time = time.time()
    print(f"\nIniciando simulação {i+1}/{NUM_SIMULATIONS} em {datetime.now().strftime('%H:%M:%S')}")
    
    metrics, final_opinions, sim_output_lines = run_simulation()
    
    sim_end_time = time.time()
    sim_duration = sim_end_time - sim_start_time
    convergence_rate = metrics.get_convergence_rate()
    
    print(f"Simulação {i+1} concluída em {datetime.now().strftime('%H:%M:%S')}")
    print(f"Duração: {sim_duration:.2f} segundos")
    print(f"Taxa de convergência: {convergence_rate:.2%}")
    print(f"{'-'*30}")
    
    all_output_lines.append(f"Simulação {i+1}, Taxa de Convergência: {convergence_rate}")
    all_output_lines.extend(sim_output_lines)

total_end_time = time.time()
total_duration = total_end_time - total_start_time
    
print(f"\n{'='*50}")
print(f"TODAS AS SIMULAÇÕES CONCLUÍDAS")
print(f"Tempo total de execução: {total_duration:.2f} segundos")
print(f"Tempo médio por simulação: {total_duration/NUM_SIMULATIONS:.2f} segundos")
print(f"{'='*50}")
    
with open(output_file, 'w') as f:
    f.write("\n".join(all_output_lines))

print(f"\nResultados salvos em: {output_file}")