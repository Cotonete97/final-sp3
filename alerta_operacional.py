# Importa Path para montar caminhos de arquivos de forma mais segura entre Windows, Linux e macOS.
from pathlib import Path

# Importa pandas para carregar e manipular o dataset limpo em formato de tabela.
import pandas as pd
# Importa SMOTE para lidar com desbalanceamento criando exemplos sintéticos da classe minoritária.
from imblearn.over_sampling import SMOTE
# Importa o Pipeline do imbalanced-learn, necessário para usar SMOTE dentro do pipeline.
from imblearn.pipeline import Pipeline as ImbPipeline
# Importa ColumnTransformer para aplicar transformações em colunas específicas.
from sklearn.compose import ColumnTransformer
# Importa RandomForestClassifier, o modelo de classificação usado para prever risco de estouro de SLA.
from sklearn.ensemble import RandomForestClassifier
# Importa train_test_split para dividir os dados em treino e teste.
from sklearn.model_selection import train_test_split
# Importa OneHotEncoder para transformar variáveis categóricas em colunas numéricas.
from sklearn.preprocessing import OneHotEncoder

# Define o limiar operacional de alerta: acima de 40%, o chamado deve ser escalonado.
LIMIAR_ALERTA = 0.40

# Define o nome do arquivo processado que será carregado pelo script.
NOME_ARQUIVO_DATASET = "dataset_limpo.csv"

# Define o nome da coluna alvo da classificação: se o chamado estourou SLA ou não.
COLUNA_ALVO_CLASSIFICACAO = "estourou_sla"
# Define o nome da coluna alvo da regressão, que será removida aqui porque este script foca no alerta/classificação.
COLUNA_ALVO_REGRESSAO = "tempo_resolucao_dias"

# Lista as colunas categóricas que precisam passar pelo OneHotEncoder.
COLUNAS_CATEGORICAS = [
    # Nome da distribuidora responsável/relacionada ao chamado.
    "Distribuidora",
    # Natureza geral do chamado.
    "Natureza",
    # Tipologia ou assunto específico do chamado.
    "Tipologia",
    # Canal pelo qual o chamado entrou.
    "CanalEntrada",
]

# Lista colunas que não devem ser usadas como entrada do modelo de classificação.
COLUNAS_PARA_REMOVER_SE_EXISTIREM = [
    # Remove Protocolo porque é apenas um identificador do chamado, não uma variável explicativa.
    "Protocolo",              # ID do chamado
    # Remove o alvo de classificação das variáveis de entrada para não entregar a resposta ao modelo.
    COLUNA_ALVO_CLASSIFICACAO,
    # Remove o alvo de regressão porque este script trabalha com risco de SLA, não previsão de dias.
    COLUNA_ALVO_REGRESSAO,
]


# Define uma função que descobre a pasta raiz do projeto.
def obter_raiz_projeto() -> Path:
    # __file__ aponta para este arquivo; resolve() pega o caminho absoluto; parents[1] sobe de src/ para a raiz do projeto.
    return Path(__file__).resolve().parents[1]


# Define uma função para carregar o dataset limpo.
def carregar_dataset() -> pd.DataFrame:
    # Obtém a pasta raiz do projeto usando a função anterior.
    raiz_projeto = obter_raiz_projeto()
    # Monta o caminho esperado do arquivo data/processed/dataset_limpo.csv.
    caminho_dataset = raiz_projeto / "data" / "processed" / NOME_ARQUIVO_DATASET

    # Verifica se o arquivo dataset_limpo.csv realmente existe nesse caminho.
    if not caminho_dataset.exists():
        # Se o arquivo não existir, interrompe a execução com uma mensagem clara de erro.
        raise FileNotFoundError(
            # Primeira parte da mensagem: informa que o arquivo não foi encontrado.
            "\nArquivo dataset_limpo.csv não encontrado.\n"
            # Mostra o caminho exato onde o script esperava encontrar o arquivo.
            f"Caminho esperado: {caminho_dataset}\n\n"
            # Orienta o usuário a gerar o CSV pelo notebook de limpeza antes de rodar o script.
            "Antes de rodar este script, confirme se o notebook de limpeza gerou "
            # Continua a orientação mostrando o caminho esperado dentro do projeto.
            "o arquivo data/processed/dataset_limpo.csv."
        )

    # Lê o CSV encontrado e devolve o DataFrame carregado.
    return pd.read_csv(caminho_dataset)


# Define uma função para conferir se o dataset possui as colunas necessárias para o alerta.
def validar_colunas_necessarias(df: pd.DataFrame) -> None:
    # Junta as colunas categóricas com a coluna alvo de classificação, formando a lista obrigatória.
    colunas_necessarias = COLUNAS_CATEGORICAS + [COLUNA_ALVO_CLASSIFICACAO]

    # Cria uma lista com as colunas obrigatórias que não foram encontradas no DataFrame.
    colunas_faltando = [
        # Guarda o nome da coluna que estiver faltando.
        coluna for coluna in colunas_necessarias
        # Condição: a coluna entra na lista se não existir em df.columns.
        if coluna not in df.columns
    ]

    # Verifica se a lista de colunas faltando não está vazia.
    if colunas_faltando:
        # Se existir coluna faltando, interrompe a execução com uma mensagem explicativa.
        raise ValueError(
            # Explica o problema geral.
            "\nO dataset não possui todas as colunas esperadas pelo alerta operacional.\n"
            # Mostra exatamente quais colunas estão faltando.
            f"Colunas faltando: {colunas_faltando}\n\n"
            # Orienta a conferir se o CSV foi gerado com a mesma estrutura da modelagem.
            "Verifique se o arquivo data/processed/dataset_limpo.csv foi gerado "
            # Continua a mensagem mencionando o notebook de referência.
            "com a mesma estrutura usada no notebook 03_modelagem.ipynb."
        )


# Define uma função que separa entradas X e alvo y para a classificação.
def preparar_dados_classificacao(df: pd.DataFrame):
    # Primeiro valida se as colunas necessárias existem no dataset.
    validar_colunas_necessarias(df)

    # Monta uma lista de colunas para remover, mas só inclui as que realmente existem no DataFrame.
    colunas_para_remover = [
        # Guarda o nome da coluna que deve ser removida.
        coluna for coluna in COLUNAS_PARA_REMOVER_SE_EXISTIREM
        # Só tenta remover a coluna se ela existir no DataFrame.
        if coluna in df.columns
    ]

    # Cria X removendo ID, alvo de classificação e alvo de regressão das entradas do modelo.
    X = df.drop(columns=colunas_para_remover)
    # Cria y usando apenas a coluna que indica se o chamado estourou SLA.
    y = df[COLUNA_ALVO_CLASSIFICACAO]

    # Devolve as entradas X e o alvo y para serem usados no treino.
    return X, y


# Define uma função para criar o pipeline de classificação completo.
def criar_pipeline_classificacao() -> ImbPipeline:
    # Cria o pré-processador que transforma colunas categóricas e mantém as demais colunas.
    preprocessor = ColumnTransformer(
        # Define a lista de transformações que serão aplicadas.
        transformers=[
            # Aplica OneHotEncoder nas colunas categóricas e ignora categorias novas em dados futuros.
            ("cat", OneHotEncoder(handle_unknown="ignore"), COLUNAS_CATEGORICAS)
        ],
        # Mantém as colunas numéricas sem transformação, como dia, hora e mês de abertura.
        remainder="passthrough"
    )

    # Cria o pipeline final juntando pré-processamento, SMOTE e classificador.
    pipeline_classificacao = ImbPipeline(steps=[
        # Primeiro passo: transformar categorias em números e manter colunas numéricas.
        ("preprocessor", preprocessor),
        # Segundo passo: aplicar SMOTE para lidar com a classe minoritária de atrasos.
        ("smote", SMOTE(random_state=42)),
        # Terceiro passo: treinar o RandomForestClassifier para prever 0 ou 1.
        ("classifier", RandomForestClassifier(
            # Limita a profundidade das árvores para reduzir sobreajuste.
            max_depth=12,
            # Exige pelo menos 3 amostras em cada folha das árvores.
            min_samples_leaf=3,
            # Fixa a aleatoriedade para facilitar reprodução dos resultados.
            random_state=42
        ))
    ])

    # Devolve o pipeline criado para ser treinado depois.
    return pipeline_classificacao


# Define uma função para treinar o modelo de classificação.
def treinar_modelo_classificacao(X: pd.DataFrame, y: pd.Series) -> tuple:
    # Divide os dados em treino e teste.
    X_train, X_test, y_train, y_test = train_test_split(
        # Passa as variáveis de entrada.
        X,
        # Passa o alvo de classificação.
        y,
        # Define que 20% dos dados ficarão para teste.
        test_size=0.20,
        # Fixa a aleatoriedade da divisão para gerar resultados reproduzíveis.
        random_state=42
    )

    # Cria o pipeline de classificação.
    modelo = criar_pipeline_classificacao()
    # Treina o pipeline com os dados de treino.
    modelo.fit(X_train, y_train)

    # Devolve o modelo treinado e os dados de teste para simulação posterior.
    return modelo, X_test, y_test


# Define uma função para descobrir em qual posição está a classe 1 dentro do modelo.
def obter_indice_classe_atraso(modelo: ImbPipeline) -> int:
    # Transforma o array de classes do modelo em uma lista Python.
    classes = list(modelo.classes_)

    # Confere se a classe 1 existe no modelo treinado.
    if 1 not in classes:
        # Se a classe 1 não existir, não dá para calcular probabilidade de atraso.
        raise ValueError(
            # Explica que o modelo não encontrou exemplos da classe 1.
            "O modelo treinado não encontrou a classe 1 em estourou_sla. "
            # Explica a consequência prática do problema.
            "Não é possível calcular o risco de estouro de SLA."
        )

    # Retorna a posição da classe 1 dentro da lista de classes do modelo.
    return classes.index(1)


# Define uma função para escolher um chamado da base de teste e simular um chamado recém-aberto.
def escolher_chamado_para_simulacao(modelo: ImbPipeline, X_test: pd.DataFrame) -> pd.DataFrame:
    # Escolhe aleatoriamente uma linha do conjunto de teste.
    chamado_simulado = X_test.sample(n=1)

    # Devolve o chamado escolhido em formato de DataFrame.
    return chamado_simulado


# Define uma função para calcular a probabilidade de um chamado estourar SLA.
def calcular_probabilidade_estouro_sla(
    # Recebe o modelo treinado.
    modelo: ImbPipeline,
    # Recebe o chamado que será analisado.
    chamado: pd.DataFrame
) -> float:
    """Calcula a probabilidade estimada de o chamado estourar o SLA."""
    # Descobre qual coluna da saída de predict_proba corresponde à classe 1.
    indice_classe_1 = obter_indice_classe_atraso(modelo)

    # Calcula as probabilidades de cada classe para o chamado recebido.
    probabilidades = modelo.predict_proba(chamado)
    # Extrai a probabilidade da classe 1 e converte para float comum.
    probabilidade_estouro = float(probabilidades[0][indice_classe_1])

    # Devolve a probabilidade de estouro de SLA.
    return probabilidade_estouro


# Define uma função para imprimir na tela os dados do chamado simulado.
def exibir_chamado_simulado(chamado: pd.DataFrame) -> None:
    # Imprime um título para a seção de chamado simulado.
    print("\n=== Chamado simulado recebido ===")
    # Explica que o chamado foi retirado da base de teste para simular uso operacional.
    print("Este chamado foi selecionado aleatoriamente da base de teste para simular um chamado recém-aberto.")

    # Pega a primeira linha do DataFrame do chamado simulado.
    linha = chamado.iloc[0]

    # Percorre todas as colunas do chamado.
    for coluna in chamado.columns:
        # Imprime o nome da coluna e o valor correspondente nessa linha.
        print(f"{coluna}: {linha[coluna]}")


# Define uma função para mostrar o resultado da regra operacional.
def exibir_resultado_operacional(probabilidade_estouro: float) -> None:
    # Converte a probabilidade de 0 a 1 para porcentagem de 0 a 100.
    probabilidade_percentual = probabilidade_estouro * 100
    # Converte o limiar de 0.40 para 40%.
    limiar_percentual = LIMIAR_ALERTA * 100

    # Imprime o título da seção de resultado.
    print("\n=== Resultado da análise operacional ===")
    # Mostra a probabilidade estimada de o chamado estourar SLA.
    print(f"Probabilidade estimada de estourar o SLA: {probabilidade_percentual:.2f}%")
    # Mostra o limiar configurado para emitir alerta.
    print(f"Limiar configurado para alerta: {limiar_percentual:.0f}%")

    # Verifica se a probabilidade calculada passou do limiar de alerta.
    if probabilidade_estouro > LIMIAR_ALERTA:
        # Imprime o título de alerta operacional.
        print("\nALERTA OPERACIONAL")
        # Imprime a recomendação de escalonamento para a equipe.
        print(
            # Primeira parte da mensagem de recomendação.
            "Risco acima de 40%. Recomenda-se escalonar o chamado "
            # Segunda parte da mensagem de recomendação.
            "para acompanhamento da equipe."
        )
    # Caso a probabilidade não passe do limiar, entra no fluxo normal.
    else:
        # Imprime que não há alerta operacional.
        print("\nSem alerta operacional")
        # Imprime a recomendação de fluxo normal.
        print(
            # Primeira parte da mensagem de fluxo normal.
            "Risco igual ou abaixo de 40%. O chamado pode seguir "
            # Segunda parte da mensagem de fluxo normal.
            "o fluxo normal de atendimento."
        )


# Define a função principal que organiza a execução do script inteiro.
def main() -> None:
    # Informa ao usuário que o dataset está sendo carregado.
    print("Carregando dataset limpo...")
    # Carrega o dataset limpo em um DataFrame.
    df = carregar_dataset()

    # Informa ao usuário que os dados estão sendo preparados.
    print("Preparando dados de classificação...")
    # Separa variáveis de entrada X e alvo y.
    X, y = preparar_dados_classificacao(df)

    # Informa que o modelo de classificação será treinado.
    print("Treinando modelo de classificação usado no alerta operacional...")
    # Treina o modelo e recebe o conjunto de teste para simulação.
    modelo, X_test, _ = treinar_modelo_classificacao(X, y)

    # Informa que um chamado será escolhido para simular o uso operacional.
    print("Selecionando chamado simulado para teste operacional...")
    # Escolhe aleatoriamente um chamado do conjunto de teste.
    chamado_simulado = escolher_chamado_para_simulacao(modelo, X_test)

    # Calcula a probabilidade de o chamado escolhido estourar SLA.
    probabilidade_estouro = calcular_probabilidade_estouro_sla(
        # Passa o modelo treinado.
        modelo,
        # Passa o chamado simulado.
        chamado_simulado
    )

    # Mostra os dados do chamado simulado.
    exibir_chamado_simulado(chamado_simulado)
    # Mostra o resultado da regra operacional de alerta.
    exibir_resultado_operacional(probabilidade_estouro)


# Verifica se este arquivo está sendo executado diretamente, e não importado por outro arquivo.
if __name__ == "__main__":
    # Executa a função principal do script.
    main()
