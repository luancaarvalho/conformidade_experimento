import random
import string
import json
from openai import OpenAI
import time
from datetime import datetime
from collections import Counter
import re
import os
from dotenv import load_dotenv
from config import ConformityConfig, format_opinions, parse_llm_response, handle_api_error

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Obter a configuração centralizada
config = ConformityConfig()
client = config.get_client()

# Parâmetros do experimento
EXPERIMENT_CONFIG = {
    "model": config.model,
    "temperature": config.temperature,
    "max_tokens": config.max_tokens,
    "opinions_initial": config.opinions_initial,
    "opinion_values": config.opinion_values,
    "agent_id": 0,
    "other_agents_opinions": {
        1: 1,   # Agente 1: opinião k
        2: -1,  # Agente 2: opinião z
        3: 1,   # Agente 3: opinião k
    }
}

def generate_unique_random_strings(length, num_strings):
    """Gera uma lista de strings aleatórias únicas."""
    generated_strings = set()
    
    # Combina letras ASCII e dígitos para formar o conjunto de caracteres
    chars = string.ascii_letters + string.digits
    
    # Continua gerando até ter o número desejado de strings únicas
    while len(generated_strings) < num_strings:
        random_str = ''.join(random.choice(chars) for _ in range(length))
        generated_strings.add(random_str)

    return list(generated_strings)

def calculate_m(opinions_list):
    """Calcula a opinião coletiva média m."""
    return sum(opinions_list) / len(opinions_list)

def run_single_experiment(experiment_num=None):
    """Executa um único experimento com logs detalhados"""
    
    # Lista para armazenar as linhas de output
    output_lines = []
    
    # Cabeçalho inicial
    separator = "=" * 80
    output_lines.extend([
        separator,
        f"EXPERIMENTO ÚNICO DE CONFORMIDADE SOCIAL {'' if experiment_num is None else f'#{experiment_num}'}",
        separator,
        "",
        "1. CONFIGURAÇÃO DO EXPERIMENTO:",
        json.dumps(EXPERIMENT_CONFIG, indent=4),
        ""
    ])
    
    # 1. Configurar nomes de opiniões (podem ser trocados com 50% de chance)
    opinion_names = {k: v for k, v in EXPERIMENT_CONFIG["opinions_initial"].items()}
    
    print("\n=== CONFIGURAÇÃO INICIAL DE OPINIÕES ===")
    print(f"Mapeamento padrão: +1={opinion_names[1]}, -1={opinion_names[-1]}")
    
    # Com 50% de chance, troca os nomes das opiniões PARA ESTE EXPERIMENTO
    is_shuffled = random.random() <= 0.5
    if is_shuffled:
        # Troca os valores associados às chaves 1 e -1
        temp_label = opinion_names[1]
        opinion_names[1] = opinion_names[-1]
        opinion_names[-1] = temp_label
        output_lines.append(f"Nomes das opiniões foram trocados: +1={opinion_names[1]}, -1={opinion_names[-1]}")
        print(f"[EMBARALHAMENTO ATIVADO] Nomes das opiniões trocados!")
        print(f"Novo mapeamento: +1={opinion_names[1]}, -1={opinion_names[-1]}")
    else:
        output_lines.append(f"Mapeamento mantido: +1={opinion_names[1]}, -1={opinion_names[-1]}")
        print(f"[EMBARALHAMENTO NÃO ATIVADO] Mapeamento mantido")
        print(f"Mapeamento mantido: +1={opinion_names[1]}, -1={opinion_names[-1]}")
    
    # Estado inicial
    opinions_list = list(EXPERIMENT_CONFIG["other_agents_opinions"].values())
    current_m = calculate_m(opinions_list)
    counts = Counter(opinion_names[op] for op in opinions_list)
    
    output_lines.extend([
        f"Estado inicial (m={current_m:.2f}):",
        f"Opiniões: {opinion_names[1]} (+1), {opinion_names[-1]} (-1)",
        f"Distribuição inicial: {counts}",
        ""
    ])
    
    # 2. Preparar o prompt usando a configuração centralizada
    formatted_opinions = format_opinions(
        EXPERIMENT_CONFIG["other_agents_opinions"], 
        opinion_names
    )
    
    # Obter mensagens formatadas da configuração
    messages = config.get_messages(formatted_opinions, opinion_names)
    user_prompt = messages[1]["content"]
    system_prompt = messages[0]["content"]
    
    print("\n=== GERAÇÃO DO PROMPT ===")
    print(f"Usando mapeamento atual: +1={opinion_names[1]}, -1={opinion_names[-1]}")
    print(f"Este mapeamento será usado tanto para gerar o prompt quanto para interpretar a resposta")
    
    output_lines.extend([
        "2. PROMPTS UTILIZADOS:",
        f"System Prompt:\n{system_prompt}",
        f"\nUser Prompt:\n{user_prompt}",
        ""
    ])
    
    # 3. Executar a chamada e capturar a resposta
    start_time = time.time()
    output_lines.append("3. CHAMADA À API:")
    
    try:
        # Implementação com retentativas em caso de limite de taxa
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response = client.chat.completions.create(
                    model=EXPERIMENT_CONFIG["model"],
                    messages=messages,
                    max_tokens=EXPERIMENT_CONFIG["max_tokens"],
                    temperature=EXPERIMENT_CONFIG["temperature"]
                )
                # Adicionar tempo de espera após a chamada à API para prevenir rate limits
                time.sleep(config.api_call_sleep_time)
                break  # Se bem-sucedido, sai do loop
                
            except Exception as e:
                retry_count += 1
                output_lines.append(f"\nAVISO: Erro da API: {str(e)} (tentativa {retry_count}/{max_retries})")
                
                wait_time, should_retry = handle_api_error(e, retry_count, max_retries)
                
                if should_retry:
                    output_lines.append(f"Aguardando {wait_time:.2f}s antes de tentar novamente...")
                    time.sleep(wait_time)
                else:
                    output_lines.append(f"Excedido número máximo de tentativas.")
                    raise
        
        # 4. Processar a resposta
        response_content = response.choices[0].message.content.strip()
        end_time = time.time()
        
        # 5. Formatar a resposta da API
        output_lines.append("\n4. RESPOSTA DA API:")
        response_data = {
            "completion_id": response.id,
            "model": response.model,
            "created_timestamp": datetime.fromtimestamp(response.created).strftime("%Y-%m-%d %H:%M:%S"),
            "response_ms": int((end_time - start_time) * 1000),
            "choices": [
                {
                    "text": response_content,
                    "finish_reason": response.choices[0].finish_reason,
                    "index": i
                } for i, choice in enumerate(response.choices)
            ],
            "usage": {
                "completion_tokens": response.usage.completion_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
        output_lines.append(json.dumps(response_data, indent=4))
        
        # 6. Extrair a opinião usando a função de utilidade
        output_lines.append("\n5. ANÁLISE DA RESPOSTA:")
        output_lines.append(f"Resposta bruta: {response_content}")
        
        print("\n=== INTERPRETAÇÃO DA RESPOSTA ===")
        print(f"Resposta bruta do LLM: {response_content}")
        print(f"Mapeamento usado para interpretação: +1={opinion_names[1]}, -1={opinion_names[-1]}")
        
        # Usar a função de utilidade para parsear a resposta com o mapeamento atual
        opinion_value = parse_llm_response(response_content, opinion_names)
        
        if opinion_value is not None:
            # Determinar o nome da opinião usando o mapeamento atual
            extracted_opinion = opinion_names[opinion_value]
            output_lines.append(f"Opinião extraída: {extracted_opinion} (valor: {opinion_value})")
            print(f"[OK] Opinião extraída: '{extracted_opinion}' → interpretada como valor numérico: {opinion_value}")
            
            # 7. Análise de conformidade
            majority_opinion = sum(EXPERIMENT_CONFIG["other_agents_opinions"].values()) > 0
            majority_name = opinion_names[1] if majority_opinion else opinion_names[-1]
            conforms_to_majority = (opinion_value == 1 and majority_opinion) or (opinion_value == -1 and not majority_opinion)
            
            print("\n=== ANÁLISE DE CONFORMIDADE ===")
            print(f"Opinião majoritária na rede: '{majority_name}' ({'positiva (+1)' if majority_opinion else 'negativa (-1)'})")
            print(f"O agente {'SEGUIU' if conforms_to_majority else 'DIVERGIU DA'} maioria")
            print(f"Resultado: {'Conformidade social detectada! [CONFORMIDADE]' if conforms_to_majority else 'Independência detectada! [DIVERGÊNCIA]'}")
            
            output_lines.extend([
                "",
                "6. ANÁLISE DE CONFORMIDADE:",
                f"Opinião majoritária: {majority_name} ({'positiva' if majority_opinion else 'negativa'})",
                f"Conformidade com a maioria: {conforms_to_majority}",
                f"Explicação: O agente escolheu {'concordar' if conforms_to_majority else 'discordar'} com a opinião majoritária"
            ])
            
            # 8. Resultado final
            final_m = (sum(EXPERIMENT_CONFIG["other_agents_opinions"].values()) + opinion_value) / (len(EXPERIMENT_CONFIG["other_agents_opinions"]) + 1)
            output_lines.extend([
                "",
                "7. RESULTADO FINAL:",
                f"Opinião coletiva final (m): {final_m:.2f}",
                f"Tempo total de processamento: {end_time - start_time:.2f} segundos",
                separator
            ])
        else:
            output_lines.append("Não foi possível extrair a opinião claramente.")
        
    except Exception as e:
        output_lines.append(f"\nERRO: {str(e)}")
        
    # Salvar o resultado em arquivo apenas se não estivermos coletando vários experimentos
    if experiment_num is None:
        # Criar arquivo de output com timestamp no diretório apropriado
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = config.single_experiment_dir
        os.makedirs(output_dir, exist_ok=True)  # Garantir que o diretório existe
        output_file = f"{output_dir}/experiment_output_{timestamp}.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
            
        print(f"Experimento concluído. Resultados salvos em: {output_file}")
        return output_file
    else:
        # Apenas retornar as linhas de output para combinação posterior
        return output_lines

if __name__ == "__main__":
    print("\n=== EXECUTANDO DOIS EXPERIMENTOS CONSECUTIVOS ===\n")
    
    # Timestamp único para ambos os experimentos
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = config.single_experiment_dir
    os.makedirs(output_dir, exist_ok=True)  # Garantir que o diretório existe
    combined_output_file = f"{output_dir}/combined_experiments_{timestamp}.txt"
    
    combined_output_lines = []
    
    # Primeiro experimento
    print("Experimento 1:")
    output_lines1 = run_single_experiment(experiment_num=1)
    combined_output_lines.extend(output_lines1)
    
    # Separador entre experimentos
    separator = "-" * 80
    combined_output_lines.extend([
        "",
        separator,
        "",
        "COMBINANDO MÚLTIPLOS EXPERIMENTOS - RESULTADOS AGREGADOS",
        "",
        separator,
        ""
    ])
    
    # Segundo experimento
    print("\nExperimento 2:")
    output_lines2 = run_single_experiment(experiment_num=2)
    combined_output_lines.extend(output_lines2)
    
    # Salvar todos os resultados em um único arquivo
    with open(combined_output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(combined_output_lines))
    
    print(f"\n=== AMBOS OS EXPERIMENTOS FORAM CONCLUÍDOS ===")
    print(f"Resultados combinados salvos em: {combined_output_file}")
