"""
Utilitários compartilhados para experimentos de conformidade.
Contém funções comuns usadas pelos diferentes scripts de simulação.
"""

import re
import string
import random
from typing import Dict, List, Optional, Union


def generate_unique_random_strings(length: int, num_strings: int) -> List[str]:
    """
    Gera uma lista de strings aleatórias únicas.
    
    Args:
        length: Comprimento de cada string gerada
        num_strings: Número de strings únicas a gerar
        
    Returns:
        Lista de strings aleatórias únicas
    """
    generated_strings = set()
    
    # Combina letras ASCII e dígitos para formar o conjunto de caracteres
    chars = string.ascii_letters + string.digits
    
    # Continua gerando até ter o número desejado de strings únicas
    while len(generated_strings) < num_strings:
        random_str = ''.join(random.choice(chars) for _ in range(length))
        generated_strings.add(random_str)

    return list(generated_strings)


def format_opinions(
    opinions_dict: Dict[int, int], 
    opinion_names: Dict[int, str], 
    agent_id_to_exclude: Optional[int] = None
) -> str:
    """
    Formata as opiniões para o prompt usando nomes aleatórios.
    
    Args:
        opinions_dict: Dicionário mapeando IDs de agentes para valores de opinião (1 ou -1)
        opinion_names: Dicionário mapeando valores de opinião para nomes ('k' ou 'z')
        agent_id_to_exclude: ID do agente a ser excluído da lista, se especificado
        
    Returns:
        String formatada com nomes de agentes e suas opiniões
    """
    # Filtrar o agente a ser excluído, se especificado
    agent_ids = [aid for aid in opinions_dict.keys() if aid != agent_id_to_exclude]
    random_names = generate_unique_random_strings(3, len(agent_ids))
    
    items = []
    for i, agent_id in enumerate(agent_ids):
        items.append(f"{random_names[i]} {opinion_names[opinions_dict[agent_id]]}")
    
    formatted_opinions = "\n".join(items)
    return formatted_opinions


def parse_llm_response(response_text: str, opinion_names: Dict[int, str]) -> Optional[int]:
    """
    Extrai a opinião '[k]' ou '[z]' da resposta do LLM.
    
    Args:
        response_text: Texto de resposta do modelo
        opinion_names: Dicionário mapeando valores de opinião (1, -1) para nomes (k, z)
        
    Returns:
        Valor numérico da opinião (1 ou -1) ou None se não encontrado
    """
    # Tentar extrair opinião entre colchetes primeiro
    match = re.search(r'\[(k|z)\]', response_text)
    
    extracted_label = None
    if match:
        extracted_label = match.group(1)
    else:
        # Fallback para encontrar k ou z sem colchetes
        if 'k' in response_text.lower():
            extracted_label = 'k'
        elif 'z' in response_text.lower():
            extracted_label = 'z'
    
    # Se encontrou uma opinião, converte para o valor numérico correspondente
    if extracted_label:
        # Compara o rótulo extraído com os VALORES do dicionário opinion_names atual
        if opinion_names.get(1) == extracted_label:
            return 1  # Retorna a CHAVE 1 se o VALOR for o rótulo extraído
        elif opinion_names.get(-1) == extracted_label:
            return -1  # Retorna a CHAVE -1 se o VALOR for o rótulo extraído
        else:
            # O rótulo extraído não corresponde a nenhum valor no mapeamento atual
            print(f"WARN: Rótulo extraído '{extracted_label}' não encontrado nos valores do mapeamento atual: {opinion_names}")
            return None
    
    # Não foi possível extrair a opinião
    print(f"WARN: Não foi possível extrair a opinião da resposta: '{response_text}'")
    return None


def wait_for_rate_limit(retry_after=None):
    """
    Função de espera para lidar com erros de limite de taxa da API.
    
    Args:
        retry_after: Tempo sugerido para esperar em segundos.
                    Se None, usa um tempo padrão de 10 segundos.
    """
    import time
    import random
    
    # Se não tiver um tempo sugerido, espera pelo menos 10 segundos
    # e adiciona um jitter aleatório para evitar sincronização de múltiplas chamadas
    wait_time = retry_after if retry_after is not None else 10
    wait_time += random.uniform(0.5, 2.0)  # Adiciona jitter para evitar thundering herd
    
    print(f"Limite de taxa da API atingido. Esperando {wait_time:.2f} segundos...")
    time.sleep(wait_time)
    print("Retomando operação...")


def handle_api_error(error, retry_count, max_retries):
    """
    Função para lidar com erros da API OpenAI de forma genérica.
    
    Args:
        error: A exceção capturada
        retry_count: Número da tentativa atual
        max_retries: Número máximo de tentativas
        
    Returns:
        float: Tempo recomendado para esperar antes da próxima tentativa
        bool: True se deve continuar tentando, False se deve desistir
    """
    import re
    import random
    
    # Verificar se é um erro de limite de taxa
    error_message = str(error)
    is_rate_limit = "rate_limit" in error_message.lower() or "rate limit" in error_message.lower()
    
    # Tentar extrair o tempo de espera sugerido para erros de limite de taxa
    retry_after = None
    if is_rate_limit:
        match = re.search(r'Please try again in (\d+\.\d+)s', error_message)
        if match:
            retry_after = float(match.group(1))
    
    # Determinar se deve continuar tentando
    should_retry = retry_count < max_retries
    
    # Calcular tempo de espera baseado no tipo de erro
    if is_rate_limit:
        # Para erros de limite de taxa, usar o tempo sugerido ou um padrão
        wait_time = retry_after if retry_after is not None else 10
        # Adicionar jitter para evitar sincronização
        wait_time += random.uniform(0.5, 2.0)
        print(f"Limite de taxa da API atingido. Esperando {wait_time:.2f} segundos...")
    else:
        # Para outros erros, usar backoff exponencial
        wait_time = 2 ** retry_count + random.uniform(0, 1)
        print(f"Erro de API: {error_message}. Tentativa {retry_count}/{max_retries}. Esperando {wait_time:.2f}s...")
    
    return wait_time, should_retry
