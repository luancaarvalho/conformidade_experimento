import random
import time
from collections import Counter
import re
from datetime import datetime
import os

# Importar a configuração centralizada e utilitários
from config import ConformityConfig, format_opinions, parse_llm_response, handle_api_error

# Obter a configuração centralizada
config = ConformityConfig()
# Atualizar para 50 agentes específico para este script
config.num_agents = 50
client = config.get_client()
MODEL = config.model

# --- Parâmetros da Simulação ---
NUM_AGENTS = config.num_agents
TEMPERATURE = config.temperature
OPINION_NAMES_INITIAL = config.opinions_initial
OPINION_VALUES = config.opinion_values
MAX_STEPS = config.max_steps  # Ajustado automaticamente para NUM_AGENTS * 50
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
def generate_unique_random_strings(length, num_strings):
    """Gera uma lista de strings aleatórias únicas."""
    import string
    generated_strings = set()
    
    # Combina letras ASCII e dígitos para formar o conjunto de caracteres
    chars = string.ascii_letters + string.digits
    
    # Continua gerando até ter o número desejado de strings únicas
    while len(generated_strings) < num_strings:
        random_str = ''.join(random.choice(chars) for _ in range(length))
        generated_strings.add(random_str)

    return list(generated_strings)

def format_opinions_for_prompt(opinions_dict, opinion_names, agent_id_to_exclude):
    """Função legada para compatibilidade - usa a função format_opinions da config."""
    return format_opinions(opinions_dict, opinion_names, agent_id_to_exclude)

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
            wait_time, should_retry = handle_api_error(e, retry_count, max_retries)
            
            if should_retry:
                time.sleep(wait_time)
            else:
                print(f"Erro após {max_retries} tentativas: {e}")
                raise
                
def calculate_m(opinions_list):
    return sum(opinions_list) / len(opinions_list)


def check_consensus(opinions_list):
    counter = Counter(opinions_list)
    most_common_opinion, count = counter.most_common(1)[0]
    return count / len(opinions_list) >= CONVERGENCE_THRESHOLD

# --- Simulação Principal ---
def run_simulation():
    opinions = {i: random.choice([-1, 1]) for i in range(NUM_AGENTS)}
    metrics = SimulationMetrics()
    opinion_names = {k: v for k, v in OPINION_NAMES_INITIAL.items()}
    
    for step in range(MAX_STEPS):
        # A cada passo de tempo (dt = 1/N), selecionamos um único agente aleatoriamente
        agent_id = random.choice(list(opinions.keys()))
        
        # --- INÍCIO: Novo Bloco de Embaralhamento k/z ---
        # Com 50% de chance, troca os nomes das opiniões PARA ESTE PASSO
        if random.random() <= 0.5:
            # Troca os valores associados às chaves 1 e -1
            temp_label = opinion_names[1]
            opinion_names[1] = opinion_names[-1]
            opinion_names[-1] = temp_label
        # --- FIM: Novo Bloco de Embaralhamento k/z ---
        
        # O agente selecionado consulta o LLM para atualizar sua opinião
        new_opinion = get_llm_opinion(agent_id, opinions, opinion_names)
        
        # Atualiza a opinião se necessário
        if new_opinion is not None and new_opinion != opinions[agent_id]:
            opinions[agent_id] = new_opinion
            metrics.record_opinion_change()
        
        # Registra o estado atual
        metrics.record_state(opinions)
        
        # Verifica convergência
        if check_consensus(list(opinions.values())):
            metrics.convergence_times.append(step)
            break
            
    return metrics


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = config.simulation_50_agents_dir
    os.makedirs(output_dir, exist_ok=True)  # Garantir que o diretório existe
    output_file = f"{output_dir}/simulation_{NUM_AGENTS}_agents_output_{timestamp}.txt"
    all_output_lines = []
    
    print(f"Iniciando simulação com {NUM_AGENTS} agentes e temperatura {TEMPERATURE}...")
    print(f"Configuração centralizada: modelo={MODEL}, threshold={CONVERGENCE_THRESHOLD}")
    
    for i in range(20):  # Executar 20 simulações
        print(f"Executando simulação {i+1}/20...")
        metrics = run_simulation()
        convergence_rate = metrics.get_convergence_rate()
        all_output_lines.append(f"Simulação {i+1}, Taxa de Convergência: {convergence_rate}")
    
    print(f"Simulações concluídas. Salvando resultados em {output_file}...")
    with open(output_file, 'w') as f:
        f.write("\n".join(all_output_lines))
    
    print(f"Resultados salvos com sucesso em: {output_file}")

if __name__ == "__main__":
    main()
