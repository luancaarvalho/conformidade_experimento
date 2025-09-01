import os
import time
import re
import random
import argparse
import csv
import datetime
import yaml
from typing import Dict, List, Optional, Any
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo para threads
import matplotlib.pyplot as plt
from openai import OpenAI
from prompt_strategies import get_prompt_strategy, PromptStrategy
import pandas as pd
import fcntl
import psutil

# Variável global para configuração de prompts
PROMPT_VARIANT = "v20_lista_completa_meio_raciocinio_primeiro"  # Variante padrão de prompt

# Lista de todas as variantes disponíveis para teste
VARIANTES_TESTE = [
    'v5_original', 'v6_lista_indices', 'v7_offsets',
    'v8_visual', 'v9_lista_completa_meio', 'v10_lista_indice_especifico',
    'v11_lista_completa_meio_sem_current',
    'v12_python', 'v13_incidence', 'v14_json',
    'v15_compact_symbol', 'v16_cartesian',
    'v17_graph_of_thought', 'v18_rule', 'v19_lista_completa_meio_com_raciocinio',
    'v20_lista_completa_meio_raciocinio_primeiro'
]
# Importa a configuração se disponível, caso contrário usa configurações locais
try:
    from config import ConformityConfig
    config = ConformityConfig()
    client = config.get_client()
except ImportError:
    # local_client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")
    local_client = OpenAI(base_url="http://172.18.254.16:1234/v1", api_key="lm-studio")
    # local_client = OpenAI(base_url="http://192.168.68.109:1234/v1", api_key="lm-studio")
    # local_client = OpenAI(base_url="http://192.168.68.109:1234/v1", api_key="lm-studio")
    client = local_client

# Constantes
OPINION_MAP = {'k': 0, 'z': 1}  # Mapeia letras para valores numéricos
REVERSE_OPINION_MAP = {0: 'k', 1: 'z'}  # Mapeia valores numéricos para letras
# TEMPERATURES = [0.0, 0.2, 0.5, 0.8, 1.0]  # Diferentes temperaturas para teste
# TEMPERATURES = [0.0, 1.0]  # Diferentes temperaturas para teste
TEMPERATURES = [0.0]  # Diferentes temperaturas para teste
# TEMPERATURES = [round(x * 0.5, 2) for x in range(0, 7)]  # [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

# Modelos disponíveis no LabelStudio
AVAILABLE_MODELS = [
    "google/gemma-3-12b",
    "google/gemma-3-12b:2", 
    "google/gemma-3-12b:3",
    "google/gemma-3-12b:4",
    "google/gemma-3-12b:5"
]

MODEL = "gemma-12b"  # Modelo padrão (usado apenas para compatibilidade)
NUM_ITERATIONS = 1  # Número de iterações por configuração
NUM_NEIGHBORS = 9   # Número TOTAL de vizinhos no experimento (sempre ímpar)
TOTAL_PARTICIPANTS = 10  # Número total de participantes no experimento


# Configuração dos arquivos de log
LOG_FILE = None  
PROMPT_LOG_FILE = None

# ===============================================================================
# SISTEMA DE ORQUESTRAÇÃO DE EXPERIMENTOS
# ===============================================================================

def process_exists(pid_str):
    """Verifica se um processo ainda existe localmente"""
    # Lida com valores NaN do pandas
    if pd.isna(pid_str) or not pid_str:
        return False
    
    # Converte para string se for float/int
    if isinstance(pid_str, (int, float)):
        pid_str = str(int(pid_str))
    
    if not pid_str or pid_str.strip() == '':
        return False
    
    try:
        pid = int(pid_str)
        return psutil.pid_exists(pid)
    except (ValueError, TypeError):
        return False

def experiment_timed_out(start_time_str, timeout_minutes):
    """Verifica se um experimento passou do timeout"""
    # Lida com valores NaN do pandas
    if pd.isna(start_time_str) or pd.isna(timeout_minutes):
        return False
    
    if not start_time_str or not timeout_minutes:
        return False
    
    try:
        start_time = datetime.datetime.strptime(str(start_time_str), '%Y-%m-%d %H:%M:%S')
        current_time = datetime.datetime.now()
        elapsed_minutes = (current_time - start_time).total_seconds() / 60
        return elapsed_minutes > float(timeout_minutes)
    except (ValueError, TypeError):
        return False

def clear_experiment_fields(df, idx):
    """Limpa campos de um experimento de forma type-safe"""
    df.loc[idx, 'status'] = 'pendente'
    df.loc[idx, 'process_id'] = pd.NA
    df.loc[idx, 'hora_inicio'] = pd.NA
    df.loc[idx, 'modelo_executado'] = pd.NA
    df.loc[idx, 'caminho_csv_saida'] = pd.NA
    df.loc[idx, 'caminho_log_saida'] = pd.NA
    df.loc[idx, 'caminho_log_prompt_saida'] = pd.NA

def delete_experiment_files(row, verbose=False):
    """Deleta arquivos de um experimento órfão"""
    files_to_delete = [
        row.get('caminho_csv_saida', ''),
        row.get('caminho_log_saida', ''),
        row.get('caminho_log_prompt_saida', '')
    ]
    
    deleted_files = []
    for file_path in files_to_delete:
        # Converte para string e remove valores NaN
        if pd.isna(file_path) or not file_path:
            continue
            
        file_path = str(file_path).strip()
        if not file_path:
            continue
            
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                deleted_files.append(os.path.basename(file_path))
                if verbose:
                    print(f"  🗑️  Deletado: {os.path.basename(file_path)}")
            except Exception as e:
                if verbose:
                    print(f"  ⚠️  Erro ao deletar {os.path.basename(file_path)}: {e}")
    
    if verbose and deleted_files:
        print(f"  📁 {len(deleted_files)} arquivo(s) deletado(s) do experimento {row['id_experimento']}")
    
    return len(deleted_files)

class ExperimentManager:
    """Gerencia a planilha mestre de experimentos com suporte a paralelização"""
    
    def __init__(self, csv_path='experimentos/experimentos_master.csv'):
        self.csv_path = csv_path
        self.process_id = str(os.getpid())
        
        # Garante que o diretório existe
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        # Cria CSV vazio se não existir
        if not os.path.exists(csv_path):
            self._create_empty_csv()
    
    def _create_empty_csv(self):
        """Cria um CSV vazio com o cabeçalho correto"""
        headers = [
            'id_experimento', 'conjunto_experimento', 'status', 'process_id', 
            'timeout_minutes', 'hora_inicio', 'hora_fim', 'modelo', 'modelo_executado',
            'num_vizinhos', 'num_iteracoes', 'temperaturas', 'variante_prompt', 
            'notas', 'caminho_csv_saida', 'caminho_log_saida', 'caminho_log_prompt_saida'
        ]
        
        empty_df = pd.DataFrame(columns=headers)
        empty_df.to_csv(self.csv_path, index=False)
        print(f"Criado arquivo CSV vazio: {self.csv_path}")
    
    def _atomic_csv_operation(self, operation_func):
        """Executa operação no CSV com lock atômico"""
        max_retries = 5
        retry_delay = 0.1
        
        for attempt in range(max_retries):
            try:
                with open(self.csv_path, 'r+', encoding='utf-8') as f:
                    # Tenta obter lock exclusivo
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    
                    # Lê o CSV
                    f.seek(0)
                    df = pd.read_csv(f)
                    
                    # Executa a operação
                    result = operation_func(df)
                    
                    # Salva as mudanças se necessário
                    if result.get('save_changes', False):
                        f.seek(0)
                        f.truncate()
                        result['df'].to_csv(f, index=False)
                    
                    # Retorna o resultado
                    return result.get('return_value')
                    
            except (IOError, OSError) as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))  # Backoff exponencial
                    continue
                else:
                    print(f"❌ Falha ao obter lock do CSV após {max_retries} tentativas: {e}")
                    return None
    
    def recover_orphaned_experiments(self):
        """Recupera experimentos órfãos que ficaram em status 'executando'"""
        def operation(df):
            executando = df[df['status'] == 'executando']
            recovered_count = 0
            
            for idx, row in executando.iterrows():
                should_recover = False
                reason = ""
                
                # Verifica se processo ainda existe
                if not process_exists(row['process_id']):
                    should_recover = True
                    reason = f"Processo {row['process_id']} não existe mais"
                
                # Verifica timeout
                elif experiment_timed_out(row['hora_inicio'], row['timeout_minutes']):
                    should_recover = True
                    reason = f"Timeout de {row['timeout_minutes']} minutos excedido"
                
                if should_recover:
                    print(f"🔄 Recuperando experimento {row['id_experimento']}: {reason}")
                    
                    # Deleta arquivos existentes do experimento órfão
                    delete_experiment_files(row, verbose=True)
                    
                    # Reseta status e limpa caminhos de forma type-safe
                    clear_experiment_fields(df, idx)
                    recovered_count += 1
            
            if recovered_count > 0:
                print(f"✅ Recuperados {recovered_count} experimento(s) órfão(s)")
            
            return {
                'save_changes': recovered_count > 0,
                'df': df,
                'return_value': recovered_count
            }
        
        return self._atomic_csv_operation(operation)
    
    def get_next_experiment(self, preferred_model=None, start_id=None, end_id=None):
        """Obtém o próximo experimento disponível de forma atômica
        
        Args:
            preferred_model: Modelo específico para reservar (opcional)
            start_id: ID mínimo para considerar (opcional)
            end_id: ID máximo para considerar (opcional)
        """
        def operation(df):
            # Primeiro recupera experimentos órfãos
            executando = df[df['status'] == 'executando']
            for idx, row in executando.iterrows():
                if (not process_exists(row['process_id']) or 
                    experiment_timed_out(row['hora_inicio'], row['timeout_minutes'])):
                    
                    # Deleta arquivos existentes do experimento órfão
                    delete_experiment_files(row, verbose=False)
                    
                    df.loc[idx, 'status'] = 'pendente'
                    df.loc[idx, 'process_id'] = pd.NA
                    df.loc[idx, 'hora_inicio'] = ''
                    df.loc[idx, 'modelo_executado'] = ''
                    df.loc[idx, 'caminho_csv_saida'] = ''
                    df.loc[idx, 'caminho_log_saida'] = ''
                    df.loc[idx, 'caminho_log_prompt_saida'] = ''
            
            # Procura próximo experimento pendente
            pendentes = df[df['status'] == 'pendente']
            
            # Aplica filtros de range de ID se especificados
            if start_id is not None:
                pendentes = pendentes[pendentes['id_experimento'] >= start_id]
            if end_id is not None:
                pendentes = pendentes[pendentes['id_experimento'] <= end_id]
            
            if len(pendentes) == 0:
                return {'save_changes': False, 'return_value': None}
            
            # Pega o primeiro experimento pendente
            next_idx = pendentes.index[0]
            experiment_row = df.loc[next_idx].to_dict()
            
            # Determina qual modelo usar
            if preferred_model:
                model_to_use = preferred_model
            else:
                # Encontra um modelo disponível (não sendo usado por outro experimento)
                models_in_use = set()
                for _, running_exp in df[df['status'] == 'executando'].iterrows():
                    if pd.notna(running_exp['modelo_executado']) and running_exp['modelo_executado']:
                        models_in_use.add(running_exp['modelo_executado'])
                
                # Encontra primeiro modelo disponível
                available_models = [m for m in AVAILABLE_MODELS if m not in models_in_use]
                if not available_models:
                    # Todos os modelos estão ocupados
                    return {'save_changes': False, 'return_value': None}
                
                model_to_use = available_models[0]
            
            # Verifica se o modelo escolhido está disponível
            if preferred_model is None:  # Só verifica disponibilidade se não foi especificado
                models_in_use = set()
                for _, running_exp in df[df['status'] == 'executando'].iterrows():
                    if pd.notna(running_exp['modelo_executado']) and running_exp['modelo_executado']:
                        models_in_use.add(running_exp['modelo_executado'])
                
                if model_to_use in models_in_use:
                    return {'save_changes': False, 'return_value': None}
            
            # Marca como executando
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            df.loc[next_idx, 'status'] = 'executando'
            df.loc[next_idx, 'process_id'] = self.process_id
            df.loc[next_idx, 'hora_inicio'] = timestamp
            df.loc[next_idx, 'modelo_executado'] = model_to_use
            
            # Adiciona o modelo executado ao resultado
            experiment_row['modelo_executado'] = model_to_use
            
            return {
                'save_changes': True,
                'df': df,
                'return_value': experiment_row
            }
        
        return self._atomic_csv_operation(operation)
    
    def get_experiment_by_id(self, experiment_id, preferred_model=None):
        """Obtém um experimento específico por ID se estiver pendente
        
        Args:
            experiment_id: ID do experimento específico
            preferred_model: Modelo específico para usar (opcional)
        """
        def operation(df):
            # Primeiro recupera experimentos órfãos
            executando = df[df['status'] == 'executando']
            for idx, row in executando.iterrows():
                if (not process_exists(row['process_id']) or 
                    experiment_timed_out(row['hora_inicio'], row['timeout_minutes'])):
                    
                    # Deleta arquivos existentes do experimento órfão
                    delete_experiment_files(row, verbose=False)
                    
                    df.loc[idx, 'status'] = 'pendente'
                    df.loc[idx, 'process_id'] = pd.NA
                    df.loc[idx, 'hora_inicio'] = ''
                    df.loc[idx, 'modelo_executado'] = ''
                    df.loc[idx, 'caminho_csv_saida'] = ''
                    df.loc[idx, 'caminho_log_saida'] = ''
                    df.loc[idx, 'caminho_log_prompt_saida'] = ''
            
            # Procura experimento específico
            experiment_mask = (df['id_experimento'] == experiment_id) & (df['status'] == 'pendente')
            matching_experiments = df[experiment_mask]
            
            if len(matching_experiments) == 0:
                return {'save_changes': False, 'return_value': None}
            
            # Pega o experimento específico
            next_idx = matching_experiments.index[0]
            experiment_row = df.loc[next_idx].to_dict()
            
            # Determina qual modelo usar
            if preferred_model:
                model_to_use = preferred_model
            else:
                # Encontra um modelo disponível (não sendo usado por outro experimento)
                models_in_use = set()
                for _, running_exp in df[df['status'] == 'executando'].iterrows():
                    if pd.notna(running_exp['modelo_executado']) and running_exp['modelo_executado']:
                        models_in_use.add(running_exp['modelo_executado'])
                
                # Encontra primeiro modelo disponível
                available_models = [m for m in AVAILABLE_MODELS if m not in models_in_use]
                if not available_models:
                    # Todos os modelos estão ocupados
                    return {'save_changes': False, 'return_value': None}
                
                model_to_use = available_models[0]
            
            # Verifica se o modelo escolhido está disponível (só se não foi especificado)
            if preferred_model is None:
                models_in_use = set()
                for _, running_exp in df[df['status'] == 'executando'].iterrows():
                    if pd.notna(running_exp['modelo_executado']) and running_exp['modelo_executado']:
                        models_in_use.add(running_exp['modelo_executado'])
                
                if model_to_use in models_in_use:
                    return {'save_changes': False, 'return_value': None}
            
            # Marca como executando
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            df.loc[next_idx, 'status'] = 'executando'
            df.loc[next_idx, 'process_id'] = self.process_id
            df.loc[next_idx, 'hora_inicio'] = timestamp
            df.loc[next_idx, 'modelo_executado'] = model_to_use
            
            # Adiciona o modelo executado ao resultado
            experiment_row['modelo_executado'] = model_to_use
            
            return {
                'save_changes': True,
                'df': df,
                'return_value': experiment_row
            }
        
        return self._atomic_csv_operation(operation)
    
    def update_experiment_status(self, experiment_id, status, **kwargs):
        """Atualiza o status de um experimento"""
        def operation(df):
            mask = df['id_experimento'] == experiment_id
            if not mask.any():
                return {'save_changes': False, 'return_value': False}
            
            idx = df[mask].index[0]
            df.loc[idx, 'status'] = status
            
            # Atualiza outros campos se fornecidos
            for key, value in kwargs.items():
                if key in df.columns:
                    df.loc[idx, key] = value
            
            # Se completou ou deu erro, remove informações de processo
            if status in ['concluido', 'erro']:
                df.loc[idx, 'hora_fim'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                df.loc[idx, 'process_id'] = pd.NA
            
            return {
                'save_changes': True,
                'df': df,
                'return_value': True
            }
        
        return self._atomic_csv_operation(operation)
    
    def get_experiments_status(self):
        """Retorna o status atual de todos os experimentos"""
        def operation(df):
            status_summary = df['status'].value_counts().to_dict()
            return {
                'save_changes': False,
                'return_value': {
                    'total': len(df),
                    'status_counts': status_summary,
                    'experiments': df.to_dict('records')
                }
            }
        
        return self._atomic_csv_operation(operation)

# Variáveis globais para tracking de performance
TOKEN_STATS = {
    'total_tokens': 0,
    'total_time': 0.0,
    'request_count': 0,
    'running_avg_tps': 0.0
}  

def setup_log_files(experiment_id=None):
    """Inicializa os arquivos de log com timestamp no nome do arquivo"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Cria diretórios na estrutura correta
    logs_dir = os.path.join(os.path.dirname(__file__), 'resultados', 'logs')
    prompt_logs_dir = os.path.join(os.path.dirname(__file__), 'resultados', 'prompt_logs')
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(prompt_logs_dir, exist_ok=True)
    
    # Se temos ID do experimento, usa ele no nome para facilitar identificação
    if experiment_id:
        exp_suffix = f"_exp{experiment_id}"
    else:
        exp_suffix = ""
    
    # Arquivo de log regular com variante e número de vizinhos no nome
    log_path = os.path.join(logs_dir, f"respostas_log_{PROMPT_VARIANT}_n{NUM_NEIGHBORS}{exp_suffix}_{timestamp}_{MODEL.replace('-', '_')}.txt")
    
    # Arquivo de log de prompts com variante e número de vizinhos no nome
    prompt_log_path = os.path.join(prompt_logs_dir, f"prompts_log_{PROMPT_VARIANT}_n{NUM_NEIGHBORS}{exp_suffix}_{timestamp}_{MODEL.replace('-', '_')}.txt")
    
    # Cria arquivo de log regular com cabeçalho (substitui se existir)
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"=== LOG DE RESPOSTAS DO EXPERIMENTO DE VIZINHANÇA - {timestamp} ===\n")
        if experiment_id:
            f.write(f"Experimento ID: {experiment_id}\n")
        f.write(f"Modelo: {MODEL}\n")
        f.write(f"Número de vizinhos: {NUM_NEIGHBORS}\n")
        f.write(f"Data e hora: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # Cria arquivo de log de prompts com cabeçalho (substitui se existir)
    with open(prompt_log_path, 'w', encoding='utf-8') as f:
        f.write(f"=== LOG DE PROMPTS DO EXPERIMENTO DE VIZINHANÇA - {timestamp} ===\n")
        if experiment_id:
            f.write(f"Experimento ID: {experiment_id}\n")
        f.write(f"Modelo: {MODEL}\n")
        f.write(f"Número de vizinhos: {NUM_NEIGHBORS}\n")
        f.write(f"Data e hora: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    print(f"Arquivos de log criados em: {log_path} e {prompt_log_path}")
    return log_path, prompt_log_path

def log_response(neighbors: str, temperature: float, response_text: str, predicted_choice: str, attempt_number: int = 1, is_retry: bool = False):
    """
    Registra a resposta em arquivo
    
    Args:
        neighbors: Representação em string da configuração dos vizinhos
        temperature: Configuração de temperatura usada
        response_text: Resposta do LLM
        predicted_choice: Escolha detectada (k ou z)
        attempt_number: Número da tentativa atual
        is_retry: Indica se é uma nova tentativa devido a erro
    """
    if LOG_FILE is None:
        return
    
    retry_str = " [RETRY]" if is_retry else ""
    
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{retry_str}\n")
        f.write(f"Temperatura: {temperature}\n")
        f.write(f"Vizinhos: {neighbors}\n")
        f.write(f"Tentativa: {attempt_number}\n")
        f.write(f"Resposta completa: {response_text}\n")
        f.write(f"Escolha detectada: {predicted_choice}\n")
        f.write("-" * 50 + "\n")

def log_prompt(neighbors: str, temperature: float, full_prompt: str, response_text: str = None, predicted_choice: str = None, binary_config: str = None, attempt_number: int = 1, is_retry: bool = False):
    """
    Registra prompt em arquivo separado para registro completo
    
    Args:
        neighbors: Representação em string da configuração dos vizinhos
        temperature: Configuração de temperatura usada
        full_prompt: Prompt completo enviado ao LLM
        response_text: Texto de resposta do LLM (opcional)
        predicted_choice: Escolha detectada - k ou z (opcional)
        binary_config: Configuração binária completa (opcional)
        attempt_number: Número da tentativa atual
        is_retry: Indica se é uma nova tentativa devido a erro
    """
    if PROMPT_LOG_FILE is None:
        return
        
    retry_str = " [RETRY]" if is_retry else ""
    
    with open(PROMPT_LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{retry_str}\n")
        f.write(f"Temperatura: {temperature}\n")
        
        # Informações detalhadas sobre a vizinhança
        if binary_config:
            letter_config = ''.join([REVERSE_OPINION_MAP[int(bit)] for bit in binary_config])
            f.write(f"Configuração completa: {binary_config} ({letter_config})\n")
            
        f.write(f"Vizinhos: {neighbors}\n")
        f.write(f"Tentativa: {attempt_number}\n")
        
        # Registra resposta se disponível
        if response_text is not None:
            f.write(f"=== RESPOSTA DO LLM ===\n{response_text}\n")
            if predicted_choice is not None:
                f.write(f"Escolha detectada: {predicted_choice}\n")
            f.write("\n")
        
        f.write(f"=== PROMPT COMPLETO ===\n{full_prompt}\n")
        f.write("-" * 80 + "\n")



def log_token_performance(input_tokens: int, output_tokens: int, tempo_exec: float, neighbors_str: str):
    """
    Registra estatísticas de performance de tokens no formato do teste_chamada.py.
    
    Args:
        input_tokens: Número de tokens de entrada (real ou estimado)
        output_tokens: Número de tokens de saída (real ou estimado)
        tempo_exec: Tempo de execução em segundos
        neighbors_str: String de identificação da configuração
    """
    global TOKEN_STATS
    
    total_tokens = input_tokens + output_tokens
    tps = total_tokens / tempo_exec if tempo_exec > 0 else 0
    
    # Atualiza estatísticas globais
    TOKEN_STATS['total_tokens'] += total_tokens
    TOKEN_STATS['total_time'] += tempo_exec
    TOKEN_STATS['request_count'] += 1
    
    # Calcula média móvel
    if TOKEN_STATS['total_time'] > 0:
        TOKEN_STATS['running_avg_tps'] = TOKEN_STATS['total_tokens'] / TOKEN_STATS['total_time']
    
    # Log detalhado no arquivo no formato simplificado
    if LOG_FILE is not None:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(f"[PERF] Input tokens: {input_tokens} | Output tokens: {output_tokens} | Tempo: {tempo_exec:.2f}s | TPS: {tps:.1f}\n")
            f.write(f"STATS: Total tokens: {TOKEN_STATS['total_tokens']}, Avg TPS: {TOKEN_STATS['running_avg_tps']:.1f}, Requests: {TOKEN_STATS['request_count']}\n")

def parse_llm_response(response_text: str) -> Optional[str]:
    """
    Extrai a opinião '[k]' ou '[z]' da resposta do LLM.
    Somente aceita respostas exatamente no formato [k] ou [z].
    
    Args:
        response_text: Texto de resposta do modelo
        
    Returns:
        String com a opinião ('k' ou 'z') ou None se não encontrado
    """
    # Verifica se a resposta é exatamente [k] ou [z]
    response_clean = response_text.lower().strip()
    
    if response_clean == '[k]':
        return 'k'
    elif response_clean == '[z]':
        return 'z'
    
    # Como fallback, tenta encontrar [k] ou [z] em qualquer lugar da resposta
    match = re.search(r'\[(k|z)\]', response_clean)
    
    if match:
        return match.group(1)
    
    # Não foi possível extrair a opinião no formato correto
    return None

def query_agent(strategy: PromptStrategy, neighbor_data_kwargs: Dict, temperature: float, current_iteration: int = 1, config_info: Dict = None, model_name: str = None) -> Dict:
    """Consulta um agente LLM usando a estratégia de prompt fornecida.
    
    Args:
        strategy: Instância da estratégia de prompt a ser usada
        neighbor_data_kwargs: Dados dos vizinhos no formato esperado pela estratégia
        temperature: Temperatura para geração do LLM
        current_iteration: Iteração atual para identificação clara nos logs
        config_info: Informações da configuração atual (opcional, para melhor logging)
    """
    # A construção do prompt agora é delegada para a estratégia
    system_prompt, user_prompt = strategy.build_prompt(**neighbor_data_kwargs)
    
    # Combina system e user prompt em um único prompt
    full_prompt = f"SYSTEM: {system_prompt}\n\nUSER: {user_prompt}"
    
    # Cria uma representação em string dos vizinhos para registro
    if 'left' in neighbor_data_kwargs and 'right' in neighbor_data_kwargs:
        neighbors_str = f"Left: {neighbor_data_kwargs['left']}, Right: {neighbor_data_kwargs['right']}"
    else:
        neighbors_str = f"Strategy: {strategy.__class__.__name__}, Data: {neighbor_data_kwargs}"
    
    # Registra o prompt que vamos enviar APENAS UMA VEZ por iteração
    log_prompt(neighbors_str, temperature, full_prompt, attempt_number=current_iteration)
    
    # Track retries for errors
    max_retries = 3
    retry_count = 0
    
    # For reexecution to get a valid response format
    max_response_attempts = 5
    response_attempt = 0
    
    while retry_count < max_retries:
        try:
            valid_response = False
            response_attempt = 0
            
            while not valid_response and response_attempt < max_response_attempts:
                # Print retry message if needed
                if response_attempt > 0:
                    print(f"⚠️ Format retry #{response_attempt} for neighbors: {neighbors_str}")
                
                # Mede tempo de inferência e calcula TPS
                start_time = time.time()
                
                response = client.chat.completions.create(
                    model=model_name if model_name else "google/gemma-3-12b",
                    messages=[
                        {"role": "user", "content": full_prompt},
                    ],
                    # max_tokens=4,
                    temperature=temperature,
                    seed=42,  # Seed fixo para reprodutibilidade
                )
                
                end_time = time.time()
                
                response_text = response.choices[0].message.content.strip().lower()
                
                # Calcula métricas de performance usando dados reais da API
                tempo_exec = end_time - start_time
                
                # Usar dados reais de tokens da resposta da API
                if hasattr(response, 'usage') and response.usage:
                    input_tokens = response.usage.prompt_tokens
                    output_tokens = response.usage.completion_tokens
                    total_tokens = response.usage.total_tokens
                else:
                    # Se não houver dados de usage, definir como 0
                    input_tokens = 0
                    output_tokens = 0
                    total_tokens = 0

                
                tps = total_tokens / tempo_exec if tempo_exec > 0 else 0
                
                # Log performance no formato simples
                log_token_performance(input_tokens, output_tokens, tempo_exec, neighbors_str)
                
                # Print simplificado de performance igual ao teste_chamada.py
                # Exibe informações da configuração junto com as métricas
                if config_info:
                    config_display = f"{config_info['binary_config']} ({''.join(config_info['letter_config'])})"
                    print(f"    ✅ [PERF] {config_display} | In:{input_tokens} Out:{output_tokens} Tot:{total_tokens} | {tempo_exec:.2f}s | {tps:.1f} TPS")
                else:
                    print(f"    ✅ [PERF] In:{input_tokens} Out:{output_tokens} Tot:{total_tokens} | {tempo_exec:.2f}s | {tps:.1f} TPS")
                
                predicted_choice = parse_llm_response(response_text)
                
                # Log all responses, including invalid ones
                log_response(neighbors_str, temperature, response_text, predicted_choice, 
                            attempt_number=current_iteration)
                
                # Log apenas a resposta no arquivo de prompt (sem repetir o prompt)
                if PROMPT_LOG_FILE is not None:
                    with open(PROMPT_LOG_FILE, 'a', encoding='utf-8') as f:
                        f.write(f"=== RESPOSTA DO LLM (Tentativa {response_attempt+1}) ===\n{response_text}\n")
                        if predicted_choice is not None:
                            f.write(f"Escolha detectada: {predicted_choice}\n")
                        f.write("\n")
                
                # Check if we have a valid response in correct format
                if predicted_choice in ['k', 'z']:
                    valid_response = True
                    # Registro especial para respostas válidas
                    with open(LOG_FILE, 'a', encoding='utf-8') as f:
                        f.write(f"RESPOSTA VÁLIDA após {response_attempt+1} tentativas\n")
                        f.write("-" * 50 + "\n")
                else:
                    # Registra como tentativa inválida
                    with open(LOG_FILE, 'a', encoding='utf-8') as f:
                        f.write(f"FORMATO INVÁLIDO (tentativa {response_attempt+1}): Faltando [k] ou [z]\n")
                        f.write("-" * 50 + "\n")
                    response_attempt += 1
            
            # Se esgotamos todas as tentativas de resposta e ainda não temos uma resposta válida
            if not valid_response:
                print(f"❌ Falhou ao obter resposta válida após {max_response_attempts} tentativas para vizinhos: {neighbors_str}")
                with open(LOG_FILE, 'a', encoding='utf-8') as f:
                    f.write(f"AVISO: Falha ao obter resposta válida [k] ou [z] após {max_response_attempts} tentativas\n")
                    f.write("-" * 50 + "\n")
            
            return {
                "neighbors": neighbors_str,
                "response": response_text,
                "predicted_choice": predicted_choice,
                "binary_choice": OPINION_MAP.get(predicted_choice, None) if predicted_choice else None,
                "full_prompt": full_prompt,
                "full_response": response_text
            }
            
        except Exception as e:
            retry_count += 1
            wait_time = 2 ** retry_count
            print(f"🔄 API ERROR RETRY #{retry_count}/{max_retries} for neighbors: {neighbors_str}")
            print(f"   Error: {str(e)}. Retrying in {wait_time:.2f}s...")
            
            # Log prompt and error for this retry attempt
            if PROMPT_LOG_FILE is not None:
                with open(PROMPT_LOG_FILE, 'a', encoding='utf-8') as f:
                    f.write(f"=== ERRO [RETRY] ===\n{str(e)}\n")
                    f.write(f"Tentativa: {current_iteration}\n\n")
            log_response(neighbors_str, temperature, f"ERROR: {str(e)}", None, 
                        attempt_number=current_iteration, is_retry=True)
            
            time.sleep(wait_time)
    
    # If all retries fail
    print(f"❌❌ TODAS AS TENTATIVAS FALHARAM para vizinhos: {neighbors_str}")
    return {
        "neighbors": neighbors_str,
        "response": "ERRO",
        "predicted_choice": None,
        "binary_choice": None,
        "full_prompt": "",
        "full_response": "ERRO"
    }

def binary_to_letters(binary_config: str) -> List[str]:
    """Converte configuração binária (ex: '101') para opiniões em letras (ex: ['k', 'z', 'k'])"""
    return [REVERSE_OPINION_MAP[int(bit)] for bit in binary_config]

def count_ones(binary_string: str) -> int:
    """Conta o número de '1's em uma string binária"""
    return binary_string.count('1')

def generate_neighbor_configs(total_neighbors: int):
    """
    Gera todas as possíveis configurações de vizinhos para o número total de vizinhos.
    
    Args:
        total_neighbors: Número total de vizinhos (serão divididos entre 'before' e 'after')
        
    Returns:
        Lista de configurações binárias
    """
    return [format(i, f'0{total_neighbors}b') for i in range(2**total_neighbors)]

def run_experiment(temperature: float, model_name: str = None, num_neighbors: int = None):
    """
    Executa o experimento para uma temperatura específica.
    
    Usa as constantes globais:
    - NUM_NEIGHBORS: Número TOTAL de vizinhos no experimento (sempre ímpar)
    - TOTAL_PARTICIPANTS: Número total de participantes no experimento completo
    
    A configuração usa uma abordagem de anel onde o agente LLM é o centro da vizinhança,
    com vizinhos distribuídos igualmente antes e depois dele.
    
    IMPORTANTE: O agente central (LLM) não conhece sua própria opinião e decide apenas
    com base nas opiniões dos vizinhos ao seu redor.
    
    Returns:
        tuple: (averages, raw_data) onde:
            - averages: Um dicionário com a média de escolhas para cada número de 1's
            - raw_data: Uma lista de dicionários com os dados brutos de cada iteração
    """
    # Usa parâmetro ou fallback para global
    effective_neighbors = num_neighbors if num_neighbors is not None else NUM_NEIGHBORS
    
    print(f"\nRunning experiment with temp: {temperature}, prompt variant: {PROMPT_VARIANT}")
    print(f"Using {effective_neighbors} neighbors for this experiment")

    # Pega a estratégia UMA VEZ no início do experimento
    try:
        strategy = get_prompt_strategy(PROMPT_VARIANT)
    except (ValueError, NotImplementedError) as e:
        print(f"❌ Erro ao carregar estratégia de prompt: {e}")
        return {}, []  # Retorna vazio para não quebrar a execução
    
    # Calculamos o número de vizinhos para cada lado (floor division para garantir inteiros)
    neighbors_per_side = effective_neighbors // 2
    
    # Geramos configurações para o total de vizinhos
    binary_configs = generate_neighbor_configs(effective_neighbors)
    max_ones = effective_neighbors  # Número máximo de uns possível
    
    # Resultados serão organizados por número de 1's na configuração
    results_by_ones = {i: [] for i in range(max_ones + 1)}  # 0 até max_ones
    
    # Armazenar a configuração específica usada para cada escolha
    # A chave é (num_ones, índice da escolha), o valor é a configuração binária
    config_mapping = {}
    
    # Armazenar resultados por configuração para calcular maioria
    # A chave é binary_config, o valor é uma lista de escolhas para essa configuração
    results_by_config = {}
    
    # Armazenar dados completos das respostas (incluindo prompts)
    # A chave é binary_config, o valor é uma lista de dados completos das respostas
    config_responses = {}
    
    # Variáveis para estimativa de tempo
    experiment_start_time = time.time()
    total_configs = len(binary_configs)
    total_calls = total_configs * NUM_ITERATIONS
    
    # Pré-processa todas as configurações para evitar recálculos
    config_data = []
    for config_index, binary_config in enumerate(binary_configs):
        letter_config = binary_to_letters(binary_config)
        num_ones = count_ones(binary_config)
        
        # Lógica "Agente no Centro": o LLM está no meio da fila de vizinhos.
        # O índice do elemento central é ignorado ao construir o prompt.
        middle_index = effective_neighbors // 2

        # Os vizinhos 'left' são todos os elementos ANTES do índice do meio.
        left_neighbors = letter_config[:middle_index]

        # Os vizinhos 'right' são todos os elementos DEPOIS do índice do meio.
        right_neighbors = letter_config[middle_index + 1:]
        
        # Determina a opinião ATUAL REAL do agente a partir da configuração
        agent_opinion_real = letter_config[middle_index]

        # A contagem de vizinhos é atualizada para refletir a nova divisão.
        left_count = len(left_neighbors)
        right_count = len(right_neighbors)
        
        # Armazena os dados da configuração
        config_data.append({
            'config_index': config_index,
            'binary_config': binary_config,
            'letter_config': letter_config,
            'num_ones': num_ones,
            'left_neighbors': left_neighbors,
            'right_neighbors': right_neighbors,
            'agent_opinion_real': agent_opinion_real,
            'left_count': left_count,
            'right_count': right_count,
            'middle_index': middle_index
        })
    
    print(f"\n� PREPARAÇÃO DO EXPERIMENTO:")
    print(f"   🎯 Configurações geradas: {total_configs}")
    print(f"   🔄 Iterações por config: {NUM_ITERATIONS}")
    print(f"   📞 Total de calls ao LLM: {total_calls}")
    print(f"   🧠 Estratégia anti-cache: Por iteração completa")
    
    print(f"\n🔄 EXECUTANDO {NUM_ITERATIONS} ITERAÇÕES PARA EVITAR CACHE")
    
    # Executa por iteração para evitar cache (mudança principal)
    call_count = 0
    for iteration in range(1, NUM_ITERATIONS + 1):
        print(f"\n=== ITERAÇÃO {iteration}/{NUM_ITERATIONS} ===")
        iteration_start_time = time.time()
        
        for config_info in config_data:
            call_count += 1
            
            # Extrai os dados pré-processados
            config_index = config_info['config_index']
            binary_config = config_info['binary_config']
            letter_config = config_info['letter_config']
            num_ones = config_info['num_ones']
            left_neighbors = config_info['left_neighbors']
            right_neighbors = config_info['right_neighbors']
            agent_opinion_real = config_info['agent_opinion_real']
            left_count = config_info['left_count']
            right_count = config_info['right_count']
            middle_index = config_info['middle_index']
            
            # Mostra qual configuração está sendo processada
            print(f"  🔧 [Iter {iteration}] Config {config_index+1}/{total_configs}: {binary_config} ({''.join(letter_config)}) | Ones: {num_ones} | Left: {''.join(left_neighbors)} | Right: {''.join(right_neighbors)}")
            
            # A cada 10 chamadas dentro da iteração, mostra estimativa de tempo restante total
            calls_in_current_iteration = config_index + 1
            if calls_in_current_iteration % 10 == 0:
                elapsed_so_far = time.time() - experiment_start_time
                avg_time_per_call = elapsed_so_far / call_count if call_count > 0 else 0
                
                # Calcula quantas chamadas restam no experimento total
                remaining_calls_total = total_calls - call_count
                estimated_remaining_total = remaining_calls_total * avg_time_per_call
                
                print(f"    ⏰ [Checkpoint] {calls_in_current_iteration}/{total_configs} configs nesta iteração | {call_count}/{total_calls} calls totais")
                print(f"       Tempo restante estimado: {estimated_remaining_total/60:.1f}min | Velocidade: {avg_time_per_call:.2f}s/call")
            
            # Progresso geral a cada 50 calls (mantido para iterações longas)
            elif call_count % 50 == 0:
                elapsed_so_far = time.time() - experiment_start_time
                avg_time_per_call = elapsed_so_far / call_count if call_count > 0 else 0
                estimated_total_time = avg_time_per_call * total_calls
                estimated_remaining = estimated_total_time - elapsed_so_far
                
                print(f"  📈 Progresso Geral: {call_count}/{total_calls} calls ({call_count/total_calls*100:.1f}%) | Tempo médio: {avg_time_per_call:.2f}s/call | Restante: {estimated_remaining/60:.1f}min")
            
            # --- Ponto chave: Preparar os dados para a estratégia atual ---
            # A lógica de como montar os dados agora fica aqui,
            # mantendo a estratégia agnóstica.
            
            data_for_strategy = {}
            if PROMPT_VARIANT == 'v5_original':
                data_for_strategy = {
                    "left": left_neighbors,
                    "right": right_neighbors,
                    "current_opinion": agent_opinion_real
                }
            elif PROMPT_VARIANT == 'v6_lista_indices':
                # Para v6, precisamos criar uma lista completa de vizinhos incluindo o agente no meio
                # Usa a opinião real do agente a partir da configuração
                
                # Constrói a lista completa com o agente no meio
                full_neighborhood = left_neighbors.copy()
                full_neighborhood.append(agent_opinion_real)  # Insere o agente no meio
                full_neighborhood.extend(right_neighbors)
                
                data_for_strategy = {
                    "neighborhood": full_neighborhood,
                    "position": middle_index
                }
            elif PROMPT_VARIANT == 'v7_offsets':
                # Para v7, usamos o mesmo formato que v5 (left/right)
                data_for_strategy = {
                    "left": left_neighbors,
                    "right": right_neighbors,
                    "current_opinion": agent_opinion_real
                }
            elif PROMPT_VARIANT == 'v8_visual':
                # Para v8, usamos o mesmo formato que v5 (left/right)
                data_for_strategy = {
                    "left": left_neighbors,
                    "right": right_neighbors,
                    "current_opinion": agent_opinion_real
                }
            elif PROMPT_VARIANT == 'v9_lista_completa_meio':
                # Para v9, usamos left/right e passamos current_opinion real
                data_for_strategy = {
                    "left": left_neighbors,
                    "right": right_neighbors,
                    "current_opinion": agent_opinion_real
                }
            elif PROMPT_VARIANT == 'v10_lista_indice_especifico':
                # Para v10, usamos left/right e passamos current_opinion real
                data_for_strategy = {
                    "left": left_neighbors,
                    "right": right_neighbors,
                    "current_opinion": agent_opinion_real
                }
            elif PROMPT_VARIANT == 'v12_python':
                # Para v12_python, usamos o mesmo formato que v6_lista_indices 
                # (lista completa com agente no meio + posição)
                
                # Constrói a lista completa com o agente no meio
                full_neighborhood = left_neighbors.copy()
                full_neighborhood.append(agent_opinion_real)  # Insere o agente no meio
                full_neighborhood.extend(right_neighbors)
                
                data_for_strategy = {
                    "neighborhood": full_neighborhood,
                    "position": middle_index
                }
            elif PROMPT_VARIANT == 'v20_lista_completa_meio_raciocinio_primeiro':
                # Para v20, usamos o mesmo formato que v19 (left/right e current_opinion)
                data_for_strategy = {
                    "left": left_neighbors,
                    "right": right_neighbors,
                    "current_opinion": agent_opinion_real
                }
            else:
                # Fallback para formato v5_original
                data_for_strategy = {
                    "left": left_neighbors,
                    "right": right_neighbors,
                    "current_opinion": agent_opinion_real
                }
            
            # Registra a configuração binária completa para referência no prompt log (só na primeira iteração)
            if iteration == 1 and PROMPT_LOG_FILE is not None:
                with open(PROMPT_LOG_FILE, 'a', encoding='utf-8') as f:
                    f.write(f"\n=== NOVA CONFIGURAÇÃO ===\n")
                    f.write(f"Configuração binária completa: {binary_config} ({''.join(letter_config)})\n")
                    f.write(f"Posição central (agente LLM): {letter_config[middle_index]} na posição {middle_index}\n")
                    f.write(f"Vizinhos à esquerda: {''.join(left_neighbors)}, Vizinhos à direita: {''.join(right_neighbors)}\n")
                    f.write("-" * 80 + "\n")
            
            result = query_agent(strategy, data_for_strategy, temperature, iteration, config_info, model_name)
            
            # Store the binary choice (0 for 'k', 1 for 'z')
            if result["binary_choice"] is not None:
                # Inicializa a lista se não existir
                if binary_config not in results_by_config:
                    results_by_config[binary_config] = []
                
                # Adiciona à lista de escolhas desta configuração específica
                results_by_config[binary_config].append(result["binary_choice"])
                
                # Armazenar o índice desta escolha (mantido para compatibilidade com dados brutos)
                if num_ones not in config_mapping:
                    config_mapping[num_ones] = []
                config_mapping[num_ones].append(binary_config)
            
            # Armazenar dados completos da resposta (incluindo prompts)
            if binary_config not in config_responses:
                config_responses[binary_config] = []
            config_responses[binary_config].append(result)
            
            if result["binary_choice"] is None:
                # Log quando uma escolha não for válida
                with open(LOG_FILE, 'a', encoding='utf-8') as f:
                    f.write(f"WARNING: Iteration {iteration} config {binary_config} produced invalid choice\n")
                    f.write("-" * 50 + "\n")
        
        # Relatório de progresso da iteração
        iteration_elapsed = time.time() - iteration_start_time
        iteration_avg_per_call = iteration_elapsed / total_configs
        calls_this_iteration = total_configs
        
        print(f"✅ Iteração {iteration} completada:")
        print(f"   ⏱️  Tempo: {iteration_elapsed:.2f}s ({iteration_elapsed/60:.1f}min)")
        print(f"   Calls: {calls_this_iteration} configs")
        print(f"   🚀 Velocidade: {iteration_avg_per_call:.2f}s/config")
        
        # Estatísticas globais atualizadas
        total_elapsed_so_far = time.time() - experiment_start_time
        global_avg_per_call = total_elapsed_so_far / call_count if call_count > 0 else 0
        
        # Estimativa de tempo restante baseada na performance real
        if iteration < NUM_ITERATIONS:
            remaining_iterations = NUM_ITERATIONS - iteration
            remaining_calls = remaining_iterations * total_configs
            estimated_remaining = remaining_calls * global_avg_per_call
            
            print(f"   📈 Global: {call_count}/{total_calls} calls totais ({call_count/total_calls*100:.1f}%)")
            print(f"   ⏳ Estimativa restante: {estimated_remaining/60:.1f}min ({remaining_iterations} iterações)")
        else:
            print(f"   🎉 EXPERIMENTO CONCLUÍDO! Total: {call_count} calls em {total_elapsed_so_far/60:.1f}min")
    
    # Após todas as iterações, processa os resultados
    print(f"\nPROCESSANDO RESULTADOS...")
    
    for binary_config, config_choices in results_by_config.items():
        num_ones = count_ones(binary_config)
        
        if config_choices:
            # Calcula a escolha majoritária para esta configuração específica
            majority_choice = 1 if sum(config_choices) > len(config_choices) / 2 else 0
            
            # Adiciona a escolha majoritária aos resultados por número de 1's
            results_by_ones[num_ones].append(majority_choice)
            
            print(f"    Config {binary_config}: choices={config_choices}, majority={majority_choice}")
        else:
            print(f"    Config {binary_config}: No valid choices")
    
    # Log final com tempo total
    total_elapsed = time.time() - experiment_start_time
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"\n=== EXPERIMENTO CONCLUÍDO ===\n")
        f.write(f"Total de configurações: {total_configs}\n")
        f.write(f"Total de iterações: {NUM_ITERATIONS}\n") 
        f.write(f"Total de chamadas: {total_calls}\n")
        f.write(f"Tempo total: {total_elapsed:.2f} segundos ({total_elapsed/60:.1f} minutos)\n")
        f.write(f"Tempo médio por chamada: {total_elapsed/total_calls:.2f} segundos\n")
        f.write("-" * 50 + "\n")
    
    # Preparar dados brutos para retornar para a função main
    raw_data = []
    extended_raw_data = []  # Dados estendidos com prompts e respostas
    
    # Organiza os dados brutos em formato tabular usando as escolhas individuais por configuração
    for binary_config, individual_choices in results_by_config.items():
        num_ones = count_ones(binary_config)
        config_letras = ''.join([REVERSE_OPINION_MAP[int(bit)] for bit in binary_config])
        
        # Adiciona cada escolha individual (mantém o formato original dos dados brutos)
        for choice in individual_choices:
            raw_data.append({
                'temperatura': temperature,
                'configuracao_binaria': binary_config,
                'configuracao_letras': config_letras,
                'num_ones': num_ones,
                'escolha': choice
            })
    
    # Organiza os dados estendidos com prompts e respostas
    for binary_config, config_data in config_responses.items():
        num_ones = count_ones(binary_config)
        config_letras = ''.join([REVERSE_OPINION_MAP[int(bit)] for bit in binary_config])
        
        for response_data in config_data:
            extended_raw_data.append({
                'temperatura': temperature,
                'configuracao_binaria': binary_config,
                'configuracao_letras': config_letras,
                'num_ones': num_ones,
                'escolha': response_data.get('binary_choice', None),
                'prompt_input': response_data.get('full_prompt', ''),
                'llm_response': response_data.get('full_response', '')
            })
    
    # Calcula média para cada número de 1's baseada nas escolhas majoritárias
    averages = {}
    for num_ones, majority_choices in results_by_ones.items():
        if majority_choices:
            avg = np.mean(majority_choices)
            averages[num_ones] = avg
            print(f"  {num_ones} 1's: {len(majority_choices)} configs, majority-based average = {avg:.4f}")
        else:
            print(f"AVISO: Nenhuma resposta válida para configuração com {num_ones} 1's")
            averages[num_ones] = None
    
    return averages, raw_data, extended_raw_data

def save_combined_csv(all_raw_data: List[Dict], num_neighbors: int, variant: str = None, experiment_id: int = None):
    """
    Salva todos os dados brutos combinados em um único arquivo CSV.
    
    Args:
        all_raw_data: Lista de dicionários com todos os dados brutos de todas as temperaturas
        num_neighbors: Número de vizinhos usado no experimento
        variant: Variante específica do prompt (opcional)
        experiment_id: ID do experimento (opcional)
    
    Returns:
        str: Caminho do arquivo CSV salvo
    """
    # Cria diretório para resultados básicos se não existir
    results_dir = os.path.join(os.path.dirname(__file__), 'resultados', 'csv', 'basic')
    os.makedirs(results_dir, exist_ok=True)
    
    # Cria nome do arquivo com timestamp e variante
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Se temos ID do experimento, usa ele no nome
    if experiment_id:
        exp_suffix = f"_exp{experiment_id}"
    else:
        exp_suffix = ""

    csv_filename = os.path.join(results_dir, f"dados_combinados_n{num_neighbors}_iter{NUM_ITERATIONS}_{variant}{exp_suffix}_{timestamp}.csv")
    
    # Define o cabeçalho do CSV
    fieldnames = [
        'temperatura', 
        'configuracao_binaria', 
        'configuracao_letras',
        'num_ones', 
        'escolha'
    ]
    
    # Adiciona coluna de variante se existir nos dados
    if all_raw_data and 'variante' in all_raw_data[0]:
        fieldnames.insert(0, 'variante')
    
    # Escreve os dados no arquivo CSV
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_raw_data:
            writer.writerow(row)
    
    return csv_filename

def save_extended_csv(all_extended_data: List[Dict], num_neighbors: int, variant: str = None, experiment_id: int = None):
    """
    Salva dados estendidos com prompts e respostas em um arquivo CSV.
    
    Args:
        all_extended_data: Lista de dicionários com dados estendidos (incluindo prompts e respostas)
        num_neighbors: Número de vizinhos usado no experimento
        variant: Variante específica do prompt (opcional)
        experiment_id: ID do experimento (opcional)
    
    Returns:
        str: Caminho do arquivo CSV salvo
    """
    # Cria diretório para resultados estendidos se não existir
    results_dir = os.path.join(os.path.dirname(__file__), 'resultados', 'csv', 'extended')
    os.makedirs(results_dir, exist_ok=True)
    
    # Cria nome do arquivo com timestamp e variante
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Se temos ID do experimento, usa ele no nome
    if experiment_id:
        exp_suffix = f"_exp{experiment_id}"
    else:
        exp_suffix = ""

    csv_filename = os.path.join(results_dir, f"dados_estendidos_n{num_neighbors}_iter{NUM_ITERATIONS}_{variant}{exp_suffix}_{timestamp}.csv")
    
    # Define o cabeçalho do CSV estendido
    fieldnames = [
        'temperatura', 
        'configuracao_binaria', 
        'configuracao_letras',
        'num_ones', 
        'escolha',
        'prompt_input',
        'llm_response'
    ]
    
    # Adiciona coluna de variante se existir nos dados
    if all_extended_data and 'variante' in all_extended_data[0]:
        fieldnames.insert(0, 'variante')
    
    # Escreve os dados no arquivo CSV
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_extended_data:
            writer.writerow(row)
    
    return csv_filename

def plot_results(all_results: Dict[float, Dict[int, float]], experiment_id: int = None, num_neighbors: int = None):
    """Plota os resultados para todas as temperaturas"""
    plt.figure(figsize=(10, 6))
    
    # Cores para diferentes temperaturas
    colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, (temperature, results) in enumerate(all_results.items()):
        # Filtra valores None para plotagem
        valid_results = {k: v for k, v in results.items() if v is not None}
        
        if valid_results:
            x_values = list(valid_results.keys())
            y_values = list(valid_results.values())

            # Normaliza o eixo X: divide o número de 1's pelo número total de vizinhos
            # Isso transforma valores absolutos (0, 1, 2, 3...) em proporções (0.0, 0.33, 0.66, 1.0...)
            neighbors_count = num_neighbors if num_neighbors is not None else NUM_NEIGHBORS
            x_values_normalized = [x / neighbors_count for x in x_values]
            
            print(x_values_normalized, x_values)
            # Plota os dados desta temperatura com sua própria cor
            color_index = i % len(colors)
            # Usa marcadores maiores e mais visíveis para garantir que todos os pontos sejam exibidos
            plt.plot(x_values_normalized, y_values, '-', label=f'Temperatura: {temperature}', color=colors[color_index])
            # Adiciona pontos separadamente com marcadores maiores e mais visíveis
            plt.scatter(x_values_normalized, y_values, s=50, color=colors[color_index], zorder=5)
        else:
            print(f"Aviso: Nenhum resultado válido para Temperatura {temperature}")
    
    plt.xlabel('Proporção de vizinhos com opinião \'z\' (Número de 1\'s / Total de Vizinhos)')
    plt.ylabel('Média do output (0=K, 1=Z)')
    neighbors_count = num_neighbors if num_neighbors is not None else NUM_NEIGHBORS
    plt.title(f'Effect of Neighborhood Configuration ({neighbors_count} neighbors)')
    plt.grid(True)
    plt.legend()
    
    # Cria diretório para resultados se não existir
    results_dir = os.path.join(os.path.dirname(__file__), 'resultados', 'plots')
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Se temos ID do experimento, usa ele no nome
    if experiment_id:
        exp_suffix = f"_exp{experiment_id}"
    else:
        exp_suffix = ""
    
    # Salva a figura com variante no nome
    neighbors_count = num_neighbors if num_neighbors is not None else NUM_NEIGHBORS
    fig_filename = os.path.join(results_dir, f"vizinhanca_{PROMPT_VARIANT}_n{neighbors_count}{exp_suffix}_{timestamp}.png")
    
    plt.savefig(fig_filename)
    print(f"\nGráfico salvo em: {fig_filename}")
    
    # Mostra o gráfico apenas se não for modo orquestrador (experiment_id None)
    if experiment_id is None:
        plt.show()
    else:
        # Fecha a figura para liberar memória no modo orquestrador
        plt.close()

def generate_results_report(all_raw_data: List[Dict]):
    """
    Gera um relatório dos resultados por temperatura e configuração binária.
    
    Args:
        all_raw_data: Lista de dicionários com todos os dados brutos
    """
    print("\n=== RELATÓRIO DETALHADO POR TEMPERATURA E CONFIGURAÇÃO ===")
    
    # Agrupa dados por temperatura
    data_by_temp = {}
    for row in all_raw_data:
        temp = row['temperatura']
        config = row['configuracao_binaria']
        escolha = row['escolha']
        
        if temp not in data_by_temp:
            data_by_temp[temp] = {}
        
        if config not in data_by_temp[temp]:
            data_by_temp[temp][config] = {'total': 0, 'escolhas_z': 0, 'num_ones': config.count('1')}
        
        data_by_temp[temp][config]['total'] += 1
        if escolha == 1:  # 1 representa escolha 'Z'
            data_by_temp[temp][config]['escolhas_z'] += 1
    
    # Imprime o relatório para cada temperatura
    for temp in sorted(data_by_temp.keys()):
        print(f"\nTemperatura: {temp}")
        print("Configuração Binária | Nº de 1's | Total de Iterações | Nº de vezes que escolheu \"Z\"")
        print("-" * 80)
        
        # Ordena configurações por ordem binária
def print_final_token_stats():
    """Imprime estatísticas finais de performance baseadas no formato do teste_chamada.py."""
    global TOKEN_STATS
    
    print(f"\n[PERF] === ESTATÍSTICAS FINAIS ===")
    print(f"Total tokens processados: {TOKEN_STATS['total_tokens']:,}")
    print(f"Total requests: {TOKEN_STATS['request_count']:,}")
    print(f"Tempo total: {TOKEN_STATS['total_time']:.2f} segundos")
    
    if TOKEN_STATS['total_time'] > 0:
        avg_tps = TOKEN_STATS['total_tokens'] / TOKEN_STATS['total_time']
        avg_time_per_request = TOKEN_STATS['total_time'] / TOKEN_STATS['request_count'] if TOKEN_STATS['request_count'] > 0 else 0
        avg_tokens_per_request = TOKEN_STATS['total_tokens'] / TOKEN_STATS['request_count'] if TOKEN_STATS['request_count'] > 0 else 0
        
        print(f"TPS médio: {avg_tps:.1f}")
        print(f"Tempo médio por request: {avg_time_per_request:.2f}s")
        print(f"Tokens médios por request: {avg_tokens_per_request:.1f}")
        
        # Log também no arquivo
        if LOG_FILE is not None:
            with open(LOG_FILE, 'a', encoding='utf-8') as f:
                f.write(f"\n[PERF] === ESTATÍSTICAS FINAIS ===\n")
                f.write(f"Total tokens: {TOKEN_STATS['total_tokens']:,}\n")
                f.write(f"Total requests: {TOKEN_STATS['request_count']:,}\n")
                f.write(f"Tempo total: {TOKEN_STATS['total_time']:.2f}s\n")
                f.write(f"TPS médio: {avg_tps:.1f}\n")
                f.write(f"Tempo médio por request: {avg_time_per_request:.2f}s\n")
                f.write(f"Tokens médios por request: {avg_tokens_per_request:.1f}\n")
    else:
        print("❌ Nenhum token foi processado.")

def run_orchestrated_experiment(experiment_data):
    """Executa um experimento específico baseado nos dados da planilha mestre"""
    global MODEL, NUM_ITERATIONS, NUM_NEIGHBORS, PROMPT_VARIANT, LOG_FILE, PROMPT_LOG_FILE
    
    print(f"\n=== EXECUTANDO EXPERIMENTO {experiment_data['id_experimento']} ===")
    print(f"Conjunto: {experiment_data['conjunto_experimento']}")
    print(f"Notas: {experiment_data['notas']}")
    
    # Configura parâmetros globais baseado no experimento
    MODEL = experiment_data['modelo']
    NUM_ITERATIONS = int(experiment_data['num_iteracoes'])
    NUM_NEIGHBORS = int(experiment_data['num_vizinhos'])
    PROMPT_VARIANT = experiment_data['variante_prompt']
    
    # Obtém o modelo específico que será executado
    model_executado = experiment_data.get('modelo_executado', AVAILABLE_MODELS[0])
    
    # Processa temperaturas
    temp_str = str(experiment_data['temperaturas']).strip()
    if ',' in temp_str:
        temperatures = [float(t.strip()) for t in temp_str.split(',')]
    else:
        temperatures = [float(temp_str)]
    
    print(f"Parâmetros: Modelo={MODEL}, Modelo Executado={model_executado}, Vizinhos={NUM_NEIGHBORS}, Iter={NUM_ITERATIONS}")
    print(f"Temperaturas: {temperatures}")
    print(f"Prompt: {PROMPT_VARIANT}")
    
    # Validação: O número de vizinhos deve ser ímpar
    if NUM_NEIGHBORS % 2 == 0:
        raise ValueError(f"O número de vizinhos ({NUM_NEIGHBORS}) deve ser ímpar")
    
    # Obtém o ID do experimento
    experiment_id = experiment_data['id_experimento']
    
    # Setup log files com ID do experimento
    LOG_FILE, PROMPT_LOG_FILE = setup_log_files(experiment_id)
    
    # Executa o experimento
    all_results = {}
    all_raw_data = []
    all_extended_data = []
    
    variant_results = {}
    
    for temp in temperatures:
        print(f"\nExecutando temperatura {temp}...")
        # Passa o número específico de vizinhos deste experimento
        experiment_neighbors = int(experiment_data['num_vizinhos'])
        result_tuple = run_experiment(temp, model_executado, experiment_neighbors)
        
        # Desempacota o resultado
        averages, raw_data, extended_data = result_tuple
        
        variant_results[temp] = averages
        
        # Adiciona informação da variante aos dados brutos
        for row in raw_data:
            row['variante'] = PROMPT_VARIANT
            all_raw_data.append(row)
            
        # Adiciona informação da variante aos dados estendidos
        for row in extended_data:
            row['variante'] = PROMPT_VARIANT
            all_extended_data.append(row)
    
    all_results[PROMPT_VARIANT] = variant_results
    
    # Salva CSV combinado com ID do experimento
    combined_csv_filename = None
    extended_csv_filename = None
    if all_raw_data:
        # Usa o número específico de vizinhos deste experimento
        experiment_neighbors = int(experiment_data['num_vizinhos'])
        combined_csv_filename = save_combined_csv(all_raw_data, experiment_neighbors, PROMPT_VARIANT, experiment_id)
        print(f"CSV básico salvo em: {os.path.abspath(combined_csv_filename)}")
        
    if all_extended_data:
        # Salva CSV estendido com prompts e respostas
        experiment_neighbors = int(experiment_data['num_vizinhos'])
        extended_csv_filename = save_extended_csv(all_extended_data, experiment_neighbors, PROMPT_VARIANT, experiment_id)
        print(f"CSV estendido salvo em: {os.path.abspath(extended_csv_filename)}")
    
    # Gera gráfico dos resultados
    if all_results and PROMPT_VARIANT in all_results:
        print(f"Gerando gráfico para experimento {experiment_id}...")
        try:
            # Usa o número de vizinhos específico deste experimento
            experiment_neighbors = int(experiment_data['num_vizinhos'])
            plot_results(all_results[PROMPT_VARIANT], experiment_id, experiment_neighbors)
            print(f"📈 Gráfico salvo com sucesso!")
        except Exception as e:
            print(f"⚠️  Erro ao gerar gráfico: {e}")
    
    # Imprime estatísticas finais de performance
    print_final_token_stats()
    
    # Retorna caminhos dos arquivos gerados
    return {
        'caminho_csv_saida': os.path.abspath(combined_csv_filename) if combined_csv_filename else '',
        'caminho_log_saida': os.path.abspath(LOG_FILE) if LOG_FILE else '',
        'caminho_log_prompt_saida': os.path.abspath(PROMPT_LOG_FILE) if PROMPT_LOG_FILE else ''
    }

def run_auto_orchestrator(single_experiment=False, start_id=None, end_id=None):
    """Modo orquestrador automático - executa experimentos da planilha mestre"""
    import multiprocessing
    import threading
    import time
    
    if single_experiment:
        print("🎯 === MODO EXPERIMENTO ÚNICO ===")
    else:
        print("🤖 === MODO ORQUESTRADOR AUTOMÁTICO COM PARALELIZAÇÃO ===")
    
    # Mostra range de IDs se especificado
    if start_id is not None or end_id is not None:
        range_str = f"IDs {start_id or 'início'} até {end_id or 'fim'}"
        print(f"🎯 Executando apenas experimentos no range: {range_str}")
    
    manager = ExperimentManager()
    
    # Recupera experimentos órfãos
    print("🔄 Verificando experimentos órfãos...")
    recovered = manager.recover_orphaned_experiments()
    
    if single_experiment:
        # Modo single experiment - executa apenas 1
        experiment_data = manager.get_next_experiment(start_id=start_id, end_id=end_id)
        
        if experiment_data is None:
            if start_id is not None or end_id is not None:
                print(f"\n❌ Não há experimentos pendentes no range {start_id or 'início'} até {end_id or 'fim'}.")
            else:
                print("\n❌ Não há experimentos pendentes.")
            return
        
        experiment_id = experiment_data['id_experimento']
        print(f"\n🎯 Executando experimento único {experiment_id}")
        
        try:
            file_paths = run_orchestrated_experiment(experiment_data)
            manager.update_experiment_status(experiment_id, 'concluido', **file_paths)
            print(f"✅ Experimento {experiment_id} concluído com sucesso!")
        except Exception as e:
            error_message = str(e)
            manager.update_experiment_status(experiment_id, 'erro', notas=f"ERRO: {error_message}")
            print(f"❌ Experimento {experiment_id} falhou: {error_message}")
        
        return
    
    # Modo paralelo - executa até 5 experimentos simultaneamente
    def run_single_experiment(experiment_data):
        """Executa um único experimento em thread separada"""
        experiment_id = experiment_data['id_experimento']
        model_name = experiment_data['modelo_executado']
        
        try:
            print(f"Experimento {experiment_id} iniciado com modelo {model_name.split('/')[-1]}")
            
            file_paths = run_orchestrated_experiment(experiment_data)
            
            manager.update_experiment_status(experiment_id, 'concluido', **file_paths)
            print(f"Experimento {experiment_id} concluído!")
            
        except Exception as e:
            error_message = str(e)
            manager.update_experiment_status(experiment_id, 'erro', notas=f"ERRO: {error_message}")
            print(f"Experimento {experiment_id} falhou: {error_message}")
    
    # Loop principal com paralelização
    active_threads = {}  # modelo -> thread
    experiment_count = 0
    
    print(f"Iniciando orquestração paralela com {len(AVAILABLE_MODELS)} modelos disponíveis")
    
    while True:
        # Remove threads finalizadas
        finished_models = []
        for model, thread in active_threads.items():
            if not thread.is_alive():
                thread.join()
                finished_models.append(model)
        
        for model in finished_models:
            del active_threads[model]
        
        # Tenta pegar novos experimentos para modelos livres
        available_models = [m for m in AVAILABLE_MODELS if m not in active_threads]
        
        if not available_models:
            # Todos os modelos estão ocupados, aguarda um pouco
            if active_threads:
                print(f"Todos os {len(AVAILABLE_MODELS)} modelos estão ocupados. Aguardando...")
                time.sleep(5)
                continue
            else:
                # Nenhum modelo ocupado e nenhum disponível = fim
                break
        
        # Tenta obter experimentos para os modelos disponíveis
        experiments_started = 0
        for model in available_models:
            experiment_data = manager.get_next_experiment(preferred_model=model, start_id=start_id, end_id=end_id)
            
            if experiment_data is None:
                continue  # Não há mais experimentos ou modelo não disponível
            
            experiment_count += 1
            experiment_id = experiment_data['id_experimento']
            
            # Inicia thread para este experimento
            thread = threading.Thread(
                target=run_single_experiment, 
                args=(experiment_data,),
                name=f"Experiment-{experiment_id}-{model}"
            )
            thread.start()
            active_threads[model] = thread
            experiments_started += 1
            
            print(f"[Modelo {model.split('/')[-1]}] Experimento {experiment_id} iniciado (#{experiment_count})")
            
            # Mostra quantos experimentos estão ativos agora
            active_count = len([t for t in active_threads.values() if t.is_alive()])
            if active_count > 1:
                print(f"    Total de experimentos rodando em paralelo: {active_count}")
        
        # Se não conseguiu iniciar nenhum experimento e não há threads ativas, termina
        if experiments_started == 0 and not active_threads:
            if start_id is not None or end_id is not None:
                print(f"\nNão há mais experimentos pendentes no range {start_id or 'início'} até {end_id or 'fim'}. Finalizando orquestrador.")
            else:
                print("\nNão há mais experimentos pendentes. Finalizando orquestrador.")
            break
        
        # Mostra status periódico dos experimentos ativos
        if active_threads:
            active_models = [model.split('/')[-1] for model in active_threads.keys()]
            print(f"⏳ Aguardando conclusão de {len(active_threads)} experimento(s): {', '.join(active_models)}")
        
        # Aguarda um pouco antes de verificar novamente
        time.sleep(3)
    
    # Aguarda todas as threads terminarem
    for model, thread in active_threads.items():
        print(f"⏳ Aguardando conclusão do modelo {model}...")
        thread.join()
    
    # Mostra status final
    status_info = manager.get_experiments_status()
    if status_info:
        print(f"\n=== STATUS FINAL ===")
        print(f"Total de experimentos: {status_info['total']}")
        for status, count in status_info['status_counts'].items():
            print(f"  {status}: {count}")
        print(f"Total de experimentos executados nesta sessão: {experiment_count}")

def show_experiments_status():
    """Mostra o status atual de todos os experimentos"""
    print("=== STATUS DOS EXPERIMENTOS ===")
    
    manager = ExperimentManager()
    status_info = manager.get_experiments_status()
    
    if not status_info:
        print("❌ Erro ao ler status dos experimentos.")
        return
    
    print(f"\nTotal de experimentos: {status_info['total']}")
    print(f"\nResumo por status:")
    for status, count in status_info['status_counts'].items():
        print(f"  {status}: {count}")
    
    # Mostra detalhes dos experimentos em execução
    executando = [exp for exp in status_info['experiments'] if exp['status'] == 'executando']
    if executando:
        print(f"\n🔄 Experimentos em execução ({len(executando)}):")
        for exp in executando:
            modelo_exec = exp.get('modelo_executado', 'N/A')
            print(f"  ID {exp['id_experimento']}: {exp['conjunto_experimento']} - "
                  f"Modelo: {modelo_exec} - PID {exp['process_id']} desde {exp['hora_inicio']}")
    
    # Mostra próximos experimentos pendentes (primeiros 5)
    pendentes = [exp for exp in status_info['experiments'] if exp['status'] == 'pendente']
    if pendentes:
        print(f"\n⏳ Próximos experimentos pendentes ({len(pendentes)} total, mostrando primeiros 5):")
        for exp in pendentes[:5]:
            print(f"  ID {exp['id_experimento']}: {exp['conjunto_experimento']} - {exp['variante_prompt']}")
    
    # Mostra experimentos com erro
    erro = [exp for exp in status_info['experiments'] if exp['status'] == 'erro']
    if erro:
        print(f"\n❌ Experimentos com erro ({len(erro)}):")
        for exp in erro:
            print(f"  ID {exp['id_experimento']}: {exp['conjunto_experimento']} - {exp['notas']}")

# ...existing code...

def main():
    global MODEL, NUM_ITERATIONS, LOG_FILE, NUM_NEIGHBORS, PROMPT_LOG_FILE, TOTAL_PARTICIPANTS, PROMPT_VARIANT
    
    parser = argparse.ArgumentParser(description='Executa experimentos de configuração de vizinhança com diferentes temperaturas')
    parser.add_argument('--temperatures', type=float, nargs='+', default=TEMPERATURES,
                      help=f'Temperaturas para executar experimentos (padrão: {TEMPERATURES})')
    parser.add_argument('--model', type=str, default=MODEL,
                      help=f'Modelo a ser usado para o experimento (padrão: {MODEL})')
    parser.add_argument('--iterations', type=int, default=NUM_ITERATIONS,
                      help=f'Número de iterações por configuração (padrão: {NUM_ITERATIONS})')
    parser.add_argument('--neighbors', type=int, default=NUM_NEIGHBORS,
                      help=f'Número TOTAL de vizinhos no experimento (sempre ímpar) (padrão: {NUM_NEIGHBORS})')
    parser.add_argument('--participants', type=int, default=TOTAL_PARTICIPANTS,
                      help=f'Número total de participantes no experimento (padrão: {TOTAL_PARTICIPANTS})')
    parser.add_argument('--prompt-variant', type=str, default=PROMPT_VARIANT,
                      help=f'Variante de prompt a ser usada (padrão: {PROMPT_VARIANT}, opções disponíveis: {", ".join(VARIANTES_TESTE)})')
    parser.add_argument('--test-all-variants', action='store_true',
                      help='Executar experimento com todas as variantes de prompt disponíveis')
    parser.add_argument('--auto-orchestrate', action='store_true',
                      help='Modo orquestrador: executa experimentos da planilha mestre automaticamente')
    parser.add_argument('--status', action='store_true',
                      help='Mostra o status atual de todos os experimentos')
    parser.add_argument('--single-experiment', action='store_true',
                      help='Executa apenas o próximo experimento pendente e para (não continua com outros)')
    parser.add_argument('--run-experiment-id', type=int, metavar='ID',
                      help='Executa um experimento específico por ID (modo manual)')
    parser.add_argument('--force-model', type=str, choices=AVAILABLE_MODELS, metavar='MODEL',
                      help=f'Força o uso de um modelo específico (opções: {", ".join(AVAILABLE_MODELS)})')
    parser.add_argument('--start-id', type=int, metavar='ID',
                      help='ID mínimo dos experimentos a executar (inclusive)')
    parser.add_argument('--end-id', type=int, metavar='ID',
                      help='ID máximo dos experimentos a executar (inclusive)')
    parser.add_argument('--force-cleanup', action='store_true',
                      help='Força limpeza de todos os experimentos órfãos')
    args = parser.parse_args()
    
    # Modo especial: mostrar status
    if args.status:
        show_experiments_status()
        return
    
    # Modo especial: forçar limpeza de órfãos
    if args.force_cleanup:
        print("🧹 === LIMPEZA FORÇADA DE EXPERIMENTOS ÓRFÃOS ===")
        manager = ExperimentManager()
        recovered = manager.recover_orphaned_experiments()
        if recovered and recovered > 0:
            print(f"✅ Limpeza concluída: {recovered} experimento(s) recuperado(s)")
        else:
            print("✅ Nenhum experimento órfão encontrado")
        return
    
    # Modo especial: executar experimento específico por ID
    if args.run_experiment_id:
        print(f"🎯 === MODO EXECUÇÃO MANUAL - EXPERIMENTO {args.run_experiment_id} ===")
        
        manager = ExperimentManager()
        
        # Tenta obter o experimento específico
        experiment_data = manager.get_experiment_by_id(args.run_experiment_id, args.force_model)
        
        if experiment_data is None:
            print(f"❌ Experimento {args.run_experiment_id} não encontrado, não está pendente, ou modelo não disponível.")
            if args.force_model:
                print(f"   Modelo solicitado: {args.force_model}")
            return
        
        experiment_id = experiment_data['id_experimento']
        model_name = experiment_data['modelo_executado']
        
        print(f"✅ Experimento {experiment_id} encontrado e reservado com modelo {model_name}")
        
        try:
            file_paths = run_orchestrated_experiment(experiment_data)
            manager.update_experiment_status(experiment_id, 'concluido', **file_paths)
            print(f"🎉 Experimento {experiment_id} concluído com sucesso!")
        except Exception as e:
            error_message = str(e)
            manager.update_experiment_status(experiment_id, 'erro', notas=f"ERRO: {error_message}")
            print(f"❌ Experimento {experiment_id} falhou: {error_message}")
        
        return
    
    # Modo especial: orquestrador automático
    if args.auto_orchestrate or args.single_experiment:
        run_auto_orchestrator(single_experiment=args.single_experiment, start_id=args.start_id, end_id=args.end_id)
        return
    
    # Validação: O número de vizinhos deve ser ímpar para que o agente possa estar no centro
    if args.neighbors % 2 == 0:
        print(f"❌ ERRO: O número de vizinhos (--neighbors) deve ser ímpar para este experimento. Você forneceu: {args.neighbors}.")
        return  # Encerra o script se o número for par
    
    # Validação: A variante de prompt deve estar na lista de variantes disponíveis
    if not args.test_all_variants and args.prompt_variant not in VARIANTES_TESTE:
        print(f"❌ ERRO: Variante de prompt '{args.prompt_variant}' não é válida.")
        print(f"Variantes disponíveis: {', '.join(VARIANTES_TESTE)}")
        return
    
    MODEL = args.model
    NUM_ITERATIONS = args.iterations
    NUM_NEIGHBORS = args.neighbors 
    TOTAL_PARTICIPANTS = args.participants
    
    # Determina quais variantes executar
    if args.test_all_variants:
        variantes_para_testar = VARIANTES_TESTE
        print(f"🧪 Executando experimento com TODAS as {len(VARIANTES_TESTE)} variantes de prompt.")
    else:
        variantes_para_testar = [args.prompt_variant]
        PROMPT_VARIANT = args.prompt_variant
        print(f"🧪 Executando experimento com a variante: {args.prompt_variant}")
    
    temperatures = args.temperatures

    print("\n=== EXPERIMENTO DE CONFIGURAÇÃO DE VIZINHANÇA COM TEMPERATURAS VARIÁVEIS ===\n")
    print(f"Modelo: {MODEL}")
    print(f"Iterações por configuração: {NUM_ITERATIONS}")
    print(f"Variantes a testar: {variantes_para_testar}")
    print(f"Temperaturas: {temperatures}")
    
    # Setup log files
    LOG_FILE, PROMPT_LOG_FILE = setup_log_files()
    
    all_results = {}
    all_raw_data = []  # Lista para armazenar todos os dados brutos combinados
    
    # Loop através de cada variante de prompt
    for variant in variantes_para_testar:
        print(f"\n🎯 === EXECUTANDO VARIANTE: {variant} ===")
        
        # Atualiza a variante global para ser usada nas funções
        PROMPT_VARIANT = variant
        
        variant_results = {}
        variant_raw_data = []
        
        for temp in temperatures:
            print(f"\nExecutando temperatura {temp} com variante {variant}...")
            # Agora run_experiment retorna médias, dados brutos e dados estendidos
            result_tuple = run_experiment(temp, AVAILABLE_MODELS[0], NUM_NEIGHBORS)  # Usa o primeiro modelo disponível no modo manual
            
            # Desempacota o resultado
            averages, raw_data, extended_data = result_tuple
            
            variant_results[temp] = averages
            
            # Adiciona informação da variante aos dados brutos
            for row in raw_data:
                row['variante'] = variant
                variant_raw_data.append(row)
                all_raw_data.append(row)
        
        # Salva resultados específicos desta variante
        all_results[variant] = variant_results
    
    # Print summary
    print("\n=== RESUMO DOS RESULTADOS ===")
    print(f"Total de vizinhos: {NUM_NEIGHBORS}, Total de participantes: {TOTAL_PARTICIPANTS}")
    print(f"Configuração: before: {NUM_NEIGHBORS // 2}, after: {(NUM_NEIGHBORS + 1) // 2}")
    
    for variant in variantes_para_testar:
        print(f"\n🎯 Variante: {variant}")
        for temp, results in all_results[variant].items():
            print(f"  Temperatura {temp}:")
            for num_ones, avg in results.items():
                if avg is not None:
                    print(f"    {num_ones} vizinhos 'z': {avg:.4f}")
                else:
                    print(f"    {num_ones} vizinhos 'z': N/A (None)")
    
    # Plota resultados (pode incluir múltiplas variantes se desejado)
    if len(variantes_para_testar) == 1:
        plot_results(all_results[variantes_para_testar[0]])
    else:
        print(f"\nPara visualizar gráficos de múltiplas variantes, execute cada uma individualmente.")
    
    # Salva CSV combinado com todas as variantes
    if all_raw_data:
        # Para CSV combinado, usa a primeira variante se houver apenas uma, senão usa "combinados"
        variant_for_combined = variantes_para_testar[0] if len(variantes_para_testar) == 1 else "combinados"
        combined_csv_filename = save_combined_csv(all_raw_data, NUM_NEIGHBORS, variant_for_combined)
        
        # Gera relatório detalhado
        generate_results_report(all_raw_data)
        
        # Print do caminho do CSV como último output
        print(f"\n📁 CSV combinado com todas as variantes salvo em:")
        print(f"   {os.path.abspath(combined_csv_filename)}")
    else:
        print("Nenhum dado bruto disponível para salvar em CSV.")
    
    # Imprime estatísticas finais de performance
    print_final_token_stats()

if __name__ == "__main__":
    main()
