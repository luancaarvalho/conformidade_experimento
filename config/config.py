"""
Módulo de configuração central para experimentos de conformidade.
Implementa o padrão Singleton para garantir configuração única entre todos os scripts.
"""

import os
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from dotenv import load_dotenv
from openai import OpenAI


class Singleton(type):
    """
    Metaclasse que implementa o padrão Singleton.
    Garante que apenas uma instância da classe de configuração exista.
    """
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


@dataclass
class ConformityConfig(metaclass=Singleton):
    """
    Configuração centralizada para todos os experimentos de conformidade.
    
    Implementa o padrão Singleton e usa dataclasses para organização.
    Todas as configurações comuns entre os scripts são definidas aqui.
    """
    # API e modelo
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.2
    max_tokens: int = 5
    api_call_sleep_time: float = 0.0001  # 0.1 ms de espera após chamadas à API
    
    # Parâmetros do experimento
    num_agents: int = 10
    num_simulations: int = 1  # Número de simulações a serem executadas
    opinions_initial: Dict[int, str] = field(default_factory=lambda: {1: 'k', -1: 'z'})
    opinion_values: Dict[str, int] = field(default_factory=lambda: {'k': 1, 'z': -1})
    max_steps: int = 500  # Será substituído por num_agents * 50 na inicialização
    convergence_threshold: float = 1.0  # 100% consenso
    
    # Diretórios de resultado
    results_dir: str = "results"
    single_experiment_dir: str = "results/single_experiment"
    simples_gpt3_dir: str = "results/simples_gpt3.5"
    simulation_50_agents_dir: str = "results/simulation_50_agents"
    
    # Mensagens do prompt
    system_prompt: str = "Você é um agente em um grupo."
    user_prompt_template: str = """Below you can see the list of all your friends together with the opinion they support.
You must reply with the opinion you want to support: either '{opinion_pos}' or '{opinion_neg}'.
The opinion must be reported between square brackets.

{formatted_opinions}

Reply only with your chosen opinion ('{opinion_pos}' or '{opinion_neg}') between square brackets, like [{opinion_pos}] or [{opinion_neg}]."""
    
    def __post_init__(self) -> None:
        """
        Inicialização após a criação da instância.
        Carrega variáveis de ambiente e configura o cliente.
        """
        # Carregar variáveis de ambiente se ainda não foram carregadas
        load_dotenv()
        
        # Atualizar a API key do ambiente se não foi fornecida
        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY", "")
        
        # Inicializar o cliente OpenAI
        self.client = OpenAI(api_key=self.api_key)
        
        # Ajustar max_steps baseado no número de agentes
        self.max_steps = self.num_agents * 50
    
    def get_user_prompt(self, formatted_opinions: str, opinion_names: Dict[int, str]) -> str:
        """
        Formata o prompt do usuário com as opiniões e nomes atuais.
        
        Args:
            formatted_opinions: String formatada com as opiniões dos agentes
            opinion_names: Dicionário mapeando valores de opinião para nomes
            
        Returns:
            O prompt formatado pronto para envio
        """
        return self.user_prompt_template.format(
            opinion_pos=opinion_names[1],
            opinion_neg=opinion_names[-1],
            formatted_opinions=formatted_opinions
        )
    
    def get_client(self) -> OpenAI:
        """
        Retorna o cliente OpenAI configurado.
        
        Returns:
            Cliente OpenAI
        """
        return self.client
        
    def get_messages(self, formatted_opinions: str, opinion_names: Dict[int, str]) -> List[Dict[str, str]]:
        """
        Cria a lista de mensagens para a API OpenAI.
        
        Args:
            formatted_opinions: String formatada com as opiniões dos agentes
            opinion_names: Dicionário mapeando valores de opinião para nomes
            
        Returns:
            Lista de mensagens formatada para a API
        """
        user_prompt = self.get_user_prompt(formatted_opinions, opinion_names)
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
