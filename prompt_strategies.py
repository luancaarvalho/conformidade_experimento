# prompt_strategies.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple
import json
import yaml
import os


class PromptStrategy(ABC):
    """
    Interface abstrata para uma estratégia de criação de prompt.
    Cada estratégia sabe como formatar os dados de uma maneira específica.
    """
    def __init__(self, config: dict):
        # A estratégia armazena sua própria configuração (system e user template)
        self.system_prompt = config.get('system', '')
        self.user_template = config.get('user', '')

    @abstractmethod
    def format_data_for_prompt(self, **kwargs) -> str:
        """
        Método abstrato. Cada subclasse DEVE implementar este método
        para formatar os dados de entrada em uma string.
        O uso de **kwargs torna a assinatura flexível para diferentes estratégias.
        """
        pass

    def build_prompt(self, **kwargs) -> Tuple[str, str]:
        """
        Método concreto que constrói o prompt final.
        Ele usa o método abstrato para obter a seção formatada.
        """
        formatted_section = self.format_data_for_prompt(**kwargs)
        
        # Prepara argumentos para format, começando com a seção formatada
        format_args = {}
        
        # Detecta qual placeholder usar baseado no template e adiciona aos argumentos
        if '{neighbors_section}' in self.user_template:
            format_args['neighbors_section'] = formatted_section
        elif '{neighborhood_list_section}' in self.user_template:
            format_args['neighborhood_list_section'] = formatted_section
        elif '{relative_map_section}' in self.user_template:
            format_args['relative_map_section'] = formatted_section
        elif '{visual_map_section}' in self.user_template:
            format_args['visual_map_section'] = formatted_section
        elif '{code_section}' in self.user_template:
            format_args['code_section'] = formatted_section
        else:
            # Fallback genérico
            format_args['placeholder_section'] = formatted_section
        
        # Adiciona current_opinion se disponível e necessário
        if '{current_opinion}' in self.user_template:
            format_args['current_opinion'] = kwargs.get('current_opinion', 'k')
        
        # Formata o prompt do usuário com todos os argumentos
        user_prompt = self.user_template.format(**format_args)
            
        return self.system_prompt, user_prompt


class V5OriginalStrategy(PromptStrategy):
    """Estratégia para o formato v5_original com 'left' e 'right'."""
    
    def format_data_for_prompt(self, left: List[str], right: List[str], **kwargs) -> str:
        """
        Formata os dados no estilo 'Left of you:' e 'Right of you:' do v5_original.
        """
        neighbors_section = ""
        
        if left:
            neighbors_section += f"Left of you: {' '.join([f'[{n}]' for n in left])}\n"
        if right:
            neighbors_section += f"Right of you: {' '.join([f'[{n}]' for n in right])}\n"
        
        return neighbors_section


class V6ListaIndicesStrategy(PromptStrategy):
    """Estratégia para o formato de lista com índices, conforme exemplo."""
    
    def format_data_for_prompt(self, neighborhood: List[str], position: int, **kwargs) -> str:
        """
        Formata os dados como uma lista completa com posições indexadas,
        incluindo a opinião atual do agente.
        
        Args:
            neighborhood: Lista completa de opiniões incluindo a do agente
            position: Índice (posição) do agente na lista
        
        Returns:
            String formatada com a vizinhança, posição e opinião atual do agente
        """
        # Extrai a opinião atual do agente com base na sua posição
        current_opinion = neighborhood[position]
        
        # Usa repr() para obter as aspas na representação da lista, exatamente como no exemplo.
        neighborhood_str = f"**Neighborhood (List):** {neighborhood!r}"
        position_str = f"**Your Position (Index):** {position}"
        opinion_str = f"**Your Current Opinion:** '{current_opinion}'"
        
        return f"{neighborhood_str}\n{position_str}\n{opinion_str}"


class V7OffsetsStrategy(PromptStrategy):
    """Estratégia para o formato de offsets relativos."""
    
    def format_data_for_prompt(self, left: List[str], right: List[str], **kwargs) -> str:
        """
        Formata os dados com offsets relativos à posição do agente.
        """
        relative_map = []
        
        # Adiciona vizinhos antes com offsets negativos (em ordem reversa para -1 ficar mais próximo de 0)
        for i, opinion in enumerate(reversed(left)):
            offset = -(i + 1)  # -1, -2, -3, etc.
            relative_map.append(f"offset: {offset}, opinion: [{opinion}]")
        
        # Adiciona a posição do agente
        relative_map.append("offset: 0, opinion: [YOU ARE HERE]")
        
        # Adiciona vizinhos depois com offsets positivos
        for i, opinion in enumerate(right):
            offset = i + 1  # 1, 2, 3, etc.
            relative_map.append(f"offset: +{offset}, opinion: [{opinion}]")
        
        return "\n".join(relative_map)


class V8VisualStrategy(PromptStrategy):
    """Estratégia para o formato visual com mapa."""
    
    def format_data_for_prompt(self, left: List[str], right: List[str], **kwargs) -> str:
        """
        Cria um mapa visual simples para v8_visual.
        """
        visual_map = ""
        
        # Adiciona vizinhos antes
        if left:
            visual_map += " - ".join([f"[{n}]" for n in left])
            visual_map += " -> "
        
        # Adiciona posição do agente
        visual_map += "(YOU)"
        
        # Adiciona vizinhos depois
        if right:
            visual_map += " <- "
            visual_map += " - ".join([f"[{n}]" for n in right])
        
        return visual_map


class V9ListaCompletaMeioStrategy(PromptStrategy):
    """Estratégia baseada em v5_original que passa a lista inteira e diz que você é a opinião do meio."""
    
    def format_data_for_prompt(self, left: List[str], right: List[str], **kwargs) -> str:
        """
        Formata os dados como uma lista completa com você sendo a opinião do meio,
        incluindo a informação da opinião atual.
        """
        # Constrói a lista completa: left + [current_opinion] + right
        current_opinion = kwargs.get('current_opinion', 'k')  # Default 'k' se não fornecido
        full_list = left + [current_opinion] + right
        
        # Formata como lista com aspas simples
        full_list_str = f"**Complete Opinion List:** {full_list!r}"
        middle_str = f"**Your Position:** You are the opinion in the middle"
        current_str = f"**Your Current Opinion:** '{current_opinion}'"
        
        return f"{full_list_str}\n{middle_str}\n{current_str}"


class V10ListaIndiceEspecificoStrategy(PromptStrategy):
    """Estratégia que passa a lista inteira e especifica qual índice representa sua própria opinião."""
    
    def format_data_for_prompt(self, left: List[str], right: List[str], **kwargs) -> str:
        """
        Formata os dados como uma lista completa especificando o índice da sua opinião.
        """
        # Constrói a lista completa: left + [current_opinion] + right
        current_opinion = kwargs.get('current_opinion', 'k')  # Default 'k' se não fornecido
        full_list = left + [current_opinion] + right
        
        # Calcula a posição do agente (0-indexed)
        agent_position = len(left)
        
        # Formata como lista com aspas simples, igual ao v6
        full_list_str = f"**Complete Opinion List:** {full_list!r}"
        position_str = f"**Your Position (Index):** {agent_position}"
        current_str = f"**Your Current Opinion:** '{current_opinion}'"
        
        return f"{full_list_str}\n{position_str}\n{current_str}"


class V11ListaCompletaMeioSemCurrentStrategy(PromptStrategy):
    """Estratégia baseada na v9 que passa a lista inteira e diz que você é a opinião do meio, mas SEM mostrar current opinion."""
    
    def format_data_for_prompt(self, left: List[str], right: List[str], **kwargs) -> str:
        """
        Formata os dados como uma lista completa com você sendo a opinião do meio,
        mas sem revelar qual é sua opinião atual.
        """
        # Constrói a lista completa: left + [current_opinion] + right
        current_opinion = kwargs.get('current_opinion', 'k')  # Default 'k' se não fornecido
        full_list = left + [current_opinion] + right
        
        # Formata como lista com aspas simples
        full_list_str = f"**Complete Opinion List:** {full_list!r}"
        middle_str = f"**Your Position:** You are the opinion in the middle"
        
        return f"{full_list_str}\n{middle_str}"


class V12PythonStrategy(PromptStrategy):
    """Estratégia em Python/Pseudo-Código."""
    
    def format_data_for_prompt(self, neighborhood: List[str] = None, position: int = None, left: List[str] = None, right: List[str] = None, **kwargs) -> str:
        """
        Formata os dados como código Python pseudo-código.
        Aceita tanto o formato de lista completa (neighborhood + position) quanto left/right.
        """
        # Se recebeu neighborhood e position, usa esse formato
        if neighborhood is not None and position is not None:
            full_list = neighborhood
            agent_position = position
        # Senão, constrói a partir de left/right
        elif left is not None and right is not None:
            current_opinion = kwargs.get('current_opinion', 'k')
            full_list = left + [current_opinion] + right
            agent_position = len(left)  # Posição do agente no meio
        else:
            raise ValueError("Deve fornecer ou (neighborhood + position) ou (left + right)")
        
        # Cria a representação em código Python
        code_section = f"neighborhood = {full_list!r}\n"
        code_section += f"my_position = {agent_position}  # My index in the neighborhood list\n"
        code_section += f"my_current_opinion = neighborhood[my_position]  # = '{full_list[agent_position]}'\n"
        code_section += "# Count occurrences of 'k' and 'z' in neighbors (excluding myself) and decide\n"
        code_section += "choice = ???"
        
        return code_section


class V13IncidenceStrategy(PromptStrategy):
    """Estratégia de lista de incidência / pares (offset, opinion)."""
    
    def format_data_for_prompt(self, left: List[str], right: List[str], **kwargs) -> str:
        """
        Formata os dados como pares (offset, opinion).
        """
        pairs = []
        
        # Adiciona vizinhos à esquerda com offsets negativos (ordem crescente de proximidade)
        for i, opinion in enumerate(reversed(left)):
            offset = -(i + 1)  # -1, -2, -3, etc.
            pairs.append(f"({offset}, {opinion})")
        
        # Adiciona vizinhos à direita com offsets positivos
        for i, opinion in enumerate(right):
            offset = i + 1  # +1, +2, +3, etc.
            pairs.append(f"(+{offset}, {opinion})")
        
        return "; ".join(pairs)


class V14JSONStrategy(PromptStrategy):
    """Estratégia de estrutura JSON/YAML."""
    
    def format_data_for_prompt(self, left: List[str], right: List[str], **kwargs) -> str:
        """
        Formata os dados como estrutura JSON.
        """
        json_dict = {}
        
        # Adiciona vizinhos à esquerda com chaves negativas
        for i, opinion in enumerate(reversed(left)):
            offset = -(i + 1)
            json_dict[str(offset)] = opinion
        
        # Adiciona vizinhos à direita com chaves positivas
        for i, opinion in enumerate(right):
            offset = i + 1
            json_dict[f"+{offset}"] = opinion
        
        return json.dumps(json_dict)


class V15CompactSymbolStrategy(PromptStrategy):
    """Estratégia de notação compacta / Chain-of-Symbol."""
    
    def format_data_for_prompt(self, left: List[str], right: List[str], **kwargs) -> str:
        """
        Formata os dados como notação compacta <left|YOU|right>.
        """
        left_str = ",".join(left) if left else ""
        right_str = ",".join(right) if right else ""
        
        return f"<{left_str}|YOU|{right_str}>"


class V16CartesianStrategy(PromptStrategy):
    """Estratégia de representação por coordenadas."""
    
    def format_data_for_prompt(self, left: List[str], right: List[str], **kwargs) -> str:
        """
        Formata os dados como coordenadas unidimensionais.
        """
        coordinates = []
        
        # Adiciona vizinhos à esquerda com coordenadas negativas
        for i, opinion in enumerate(reversed(left)):
            x = -(i + 1)
            coordinates.append(f"x={x}, opinion={opinion}")
        
        # Adiciona vizinhos à direita com coordenadas positivas
        for i, opinion in enumerate(right):
            x = i + 1
            coordinates.append(f"x=+{x}, opinion={opinion}")
        
        return "\n".join(coordinates)


class V17GraphOfThoughtStrategy(PromptStrategy):
    """Estratégia Graph-of-Thought / Tree-of-Thoughts."""
    
    def format_data_for_prompt(self, left: List[str], right: List[str], **kwargs) -> str:
        """
        Formata os dados com análise de hipóteses.
        """
        # Conta opiniões
        all_neighbors = left + right
        num_k = all_neighbors.count('k')
        num_z = all_neighbors.count('z')
        total = len(all_neighbors)
        
        # Monta a representação básica
        neighbors_section = ""
        if left:
            neighbors_section += f"Left of you: {' '.join([f'[{n}]' for n in left])}\n"
        if right:
            neighbors_section += f"Right of you: {' '.join([f'[{n}]' for n in right])}\n"
        
        # Adiciona análise de hipóteses
        neighbors_section += f"\nIf you choose [k], you will agree with {num_k} neighbor(s) and disagree with {num_z}.\n"
        neighbors_section += f"If you choose [z], you will agree with {num_z} neighbor(s) and disagree with {num_k}."
        
        return neighbors_section


class V18RuleStrategy(PromptStrategy):
    """Estratégia de instrução de regra explícita."""
    
    def format_data_for_prompt(self, left: List[str], right: List[str], **kwargs) -> str:
        """
        Formata os dados com regra explícita.
        """
        current_opinion = kwargs.get('current_opinion', 'k')
        
        # Representação tradicional
        neighbors_section = ""
        if left:
            neighbors_section += f"Left of you: {' '.join([f'[{n}]' for n in left])}\n"
        if right:
            neighbors_section += f"Right of you: {' '.join([f'[{n}]' for n in right])}\n"
        
        # Adiciona regra explícita
        neighbors_section += f"\nRule: count how many neighbors support 'k' and how many support 'z'. "
        neighbors_section += f"If 'k' appears more, choose [k]. If 'z' appears more, choose [z]. "
        neighbors_section += f"If the counts are equal, keep your current opinion ({current_opinion})."
        
        return neighbors_section


class V19ListaCompletaMeioComRaciocinio(PromptStrategy):
    """Estratégia baseada na V9 que passa a lista inteira, indica que você é do meio, e solicita raciocínio."""
    
    def format_data_for_prompt(self, left: List[str], right: List[str], **kwargs) -> str:
        """
        Formata os dados como uma lista completa com você sendo a opinião do meio,
        incluindo a informação da opinião atual. Formato exato do exemplo fornecido.
        """
        # Constrói a lista completa: left + [current_opinion] + right
        current_opinion = kwargs.get('current_opinion', 'k')  # Default 'k' se não fornecido
        full_list = left + [current_opinion] + right
        
        # Formata exatamente como no exemplo fornecido
        full_list_str = f"Complete Opinion List: {full_list!r}"
        middle_str = f"Your Position: You are the opinion in the middle"
        current_str = f"Your Current Opinion: {current_opinion}"  # Sem aspas extras
        
        return f"{full_list_str}\n{middle_str}\n{current_str}"


class V20ListaCompletaMeioRaciocinioPrimeiro(PromptStrategy):
    """Estratégia baseada na V19 mas que solicita primeiro o raciocínio e depois a escolha [k] ou [z]."""
    
    def format_data_for_prompt(self, left: List[str], right: List[str], **kwargs) -> str:
        """
        Formata os dados como uma lista completa com você sendo a opinião do meio,
        incluindo a informação da opinião atual. Formato exato do exemplo fornecido.
        """
        # Constrói a lista completa: left + [current_opinion] + right
        current_opinion = kwargs.get('current_opinion', 'k')  # Default 'k' se não fornecido
        full_list = left + [current_opinion] + right
        
        # Formata exatamente como no exemplo fornecido
        full_list_str = f"Complete Opinion List: {full_list!r}"
        middle_str = f"Your Position: You are the opinion in the middle"
        current_str = f"Your Current Opinion: {current_opinion}"  # Sem aspas extras
        
        return f"{full_list_str}\n{middle_str}\n{current_str}"


def get_prompt_strategy(variant_name: str, yaml_path: str = None) -> PromptStrategy:
    """
    Factory que lê o arquivo de configuração e retorna a instância
    da estratégia correta com base no nome da variante.
    
    Args:
        variant_name: Nome da variante (ex: 'v5_original', 'v6_lista_indices')
        yaml_path: Caminho opcional para o arquivo YAML. Se None, usa o padrão.
    
    Returns:
        Instância da estratégia correspondente
        
    Raises:
        FileNotFoundError: Se o arquivo YAML não for encontrado
        ValueError: Se a variante não existir no YAML
        NotImplementedError: Se a estratégia não foi implementada
    """
    if yaml_path is None:
        yaml_path = os.path.join(os.path.dirname(__file__), 'prompt_templates.yaml')
    
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            all_configs = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Arquivo '{yaml_path}' não encontrado.")

    if variant_name not in all_configs:
        available_variants = list(all_configs.keys())
        raise ValueError(
            f"Variante de prompt '{variant_name}' não encontrada no YAML. "
            f"Variantes disponíveis: {available_variants}"
        )

    config = all_configs[variant_name]

    # Mapeamento de nomes de variantes para as classes de Estratégia
    strategy_map = {
        'v5_original': V5OriginalStrategy,
        'v6_lista_indices': V6ListaIndicesStrategy,
        'v7_offsets': V7OffsetsStrategy,
        'v8_visual': V8VisualStrategy,
        'v9_lista_completa_meio': V9ListaCompletaMeioStrategy,
        'v10_lista_indice_especifico': V10ListaIndiceEspecificoStrategy,
        'v11_lista_completa_meio_sem_current': V11ListaCompletaMeioSemCurrentStrategy,
        'v12_python': V12PythonStrategy,
        'v13_incidence': V13IncidenceStrategy,
        'v14_json': V14JSONStrategy,
        'v15_compact_symbol': V15CompactSymbolStrategy,
        'v16_cartesian': V16CartesianStrategy,
        'v17_graph_of_thought': V17GraphOfThoughtStrategy,
        'v18_rule': V18RuleStrategy,
        'v19_lista_completa_meio_com_raciocinio': V19ListaCompletaMeioComRaciocinio,
        'v20_lista_completa_meio_raciocinio_primeiro': V20ListaCompletaMeioRaciocinioPrimeiro,
    }

    if variant_name not in strategy_map:
        implemented_variants = list(strategy_map.keys())
        raise NotImplementedError(
            f"A classe de estratégia para '{variant_name}' não foi implementada. "
            f"Estratégias implementadas: {implemented_variants}"
        )

    # Retorna uma instância da classe de estratégia correta, injetando sua configuração
    return strategy_map[variant_name](config)
