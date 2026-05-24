# Importa streamlit para criar a interface web simples do projeto.
import streamlit as st
# Importa pandas para carregar o CSV e montar DataFrames para previsão.
import pandas as pd
# Importa Path para montar caminhos de arquivo de forma segura.
from pathlib import Path
# Importa ColumnTransformer para aplicar transformações específicas em colunas específicas.
from sklearn.compose import ColumnTransformer
# Importa OneHotEncoder para transformar categorias em colunas numéricas.
from sklearn.preprocessing import OneHotEncoder
# Importa os modelos RandomForest de classificação e regressão.
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# Importa o Pipeline do imbalanced-learn para permitir o uso de SMOTE no pipeline de classificação.
from imblearn.pipeline import Pipeline as ImbPipeline
# Importa SMOTE para lidar com o desbalanceamento da classe de atraso.
from imblearn.over_sampling import SMOTE
# Importa o Pipeline padrão do scikit-learn, usado no pipeline de regressão.
from sklearn.pipeline import Pipeline

# Configura a página do Streamlit com título, ícone e layout centralizado.
st.set_page_config(page_title="RAGnarok - SLA Predictor", page_icon="⚡", layout="centered")

# Usa cache para evitar treinar os modelos novamente a cada interação da tela.
@st.cache_resource
# Define a função responsável por carregar dados e treinar os dois modelos.
def treinar_modelos():
    # Monta o caminho do arquivo data/processed/dataset_limpo.csv a partir da localização deste script.
    caminho_dataset = Path(__file__).resolve().parents[1] / "data" / "processed" / "dataset_limpo.csv"
    # Lê o dataset limpo em um DataFrame.
    df = pd.read_csv(caminho_dataset)
    
    # Lista as colunas categóricas que precisam ser convertidas em números.
    COLUNAS_CATEGORICAS = ["Distribuidora", "Natureza", "Tipologia", "CanalEntrada"]
    # Cria X removendo identificador e alvos, deixando apenas variáveis de entrada.
    X = df.drop(columns=["Protocolo", "estourou_sla", "tempo_resolucao_dias"], errors="ignore")
    # Cria o alvo da classificação: risco de estourar SLA.
    y_clf = df["estourou_sla"]
    # Cria o alvo da regressão: tempo de resolução em dias.
    y_reg = df["tempo_resolucao_dias"]
    
    # Cria o pré-processador comum aos dois modelos.
    preprocessor = ColumnTransformer(
        # Define que as colunas categóricas passarão pelo OneHotEncoder.
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), COLUNAS_CATEGORICAS)],
        # Mantém as outras colunas numéricas como estão.
        remainder="passthrough"
    )
    
    # Cria o pipeline de classificação com pré-processamento, SMOTE e RandomForestClassifier.
    pipeline_clf = ImbPipeline(steps=[
        # Primeiro transforma as variáveis categóricas em números.
        ("preprocessor", preprocessor),
        # Depois aplica SMOTE para balancear a classe minoritária no treinamento.
        ("smote", SMOTE(random_state=42)),
        # Por fim treina a floresta aleatória de classificação.
        ("classifier", RandomForestClassifier(max_depth=12, min_samples_leaf=3, random_state=42, n_jobs=-1))
    ])
    
    # Cria o pipeline de regressão com pré-processamento e RandomForestRegressor.
    pipeline_reg = Pipeline(steps=[
        # Primeiro transforma as variáveis categóricas em números.
        ("preprocessor", preprocessor),
        # Depois treina a floresta aleatória de regressão para prever dias.
        ("regressor", RandomForestRegressor(max_depth=12, min_samples_leaf=3, random_state=42, n_jobs=-1))
    ])
    
    # Treina o modelo de classificação usando as entradas X e o alvo y_clf.
    pipeline_clf.fit(X, y_clf)
    # Treina o modelo de regressão usando as entradas X e o alvo y_reg.
    pipeline_reg.fit(X, y_reg)
    
    # Devolve os dois modelos treinados e a base usada para preencher opções da interface.
    return pipeline_clf, pipeline_reg, df

# Mostra o título principal da aplicação.
st.title("⚡ ANEEL: Previsão de Risco de SLA")
# Mostra um texto curto explicando o objetivo da tela.
st.markdown("Abra um novo chamado simulado para avaliar o risco operacional.")

# Mostra um spinner enquanto os modelos são carregados/treinados.
with st.spinner("Carregando e treinando a Inteligência Artificial..."):
    # Chama a função de treino e recebe classificador, regressor e dataset base.
    modelo_clf, modelo_reg, df_base = treinar_modelos()

# Cria um formulário para o usuário preencher os dados do chamado.
with st.form("form_chamado"):
    # Mostra o subtítulo da seção de dados do chamado.
    st.subheader("Dados do Chamado")
    
    # Divide a tela em duas colunas para organizar os campos.
    col1, col2 = st.columns(2)
    
    # Abre o bloco de campos da primeira coluna.
    with col1:
        # Cria uma caixa de seleção com as distribuidoras existentes na base.
        distribuidora = st.selectbox("Distribuidora", df_base["Distribuidora"].unique())
        # Cria uma caixa de seleção com as naturezas existentes na base.
        natureza = st.selectbox("Natureza", df_base["Natureza"].unique())
        # Cria uma caixa de seleção com as tipologias existentes na base.
        tipologia = st.selectbox("Tipologia", df_base["Tipologia"].unique())
        # Cria uma caixa de seleção com os canais de entrada existentes na base.
        canal = st.selectbox("Canal de Entrada", df_base["CanalEntrada"].unique())
        
    # Abre o bloco de campos da segunda coluna.
    with col2:
        # Cria um controle deslizante para o dia da semana da abertura.
        dia_semana = st.slider("Dia da Semana (0=Seg, 6=Dom)", min_value=0, max_value=6, value=0)
        # Cria um controle deslizante para a hora de abertura do chamado.
        hora = st.slider("Hora de Abertura", min_value=0, max_value=23, value=9)
        # Cria um controle deslizante para o mês de abertura do chamado.
        mes = st.slider("Mês de Abertura", min_value=1, max_value=12, value=5)
        
        # Cria um campo numérico para escolher o limiar operacional em porcentagem.
        limiar = st.number_input("Limiar Operacional de Alerta (%)", min_value=10, max_value=90, value=40, step=5)
        
    # Cria o botão que envia o formulário e dispara a previsão.
    submit = st.form_submit_button("🔮 Prever Risco Operacional", use_container_width=True)

# Verifica se o usuário clicou no botão de previsão.
if submit:
    # Monta um DataFrame com uma única linha representando o chamado informado na interface.
    novo_chamado = pd.DataFrame([{
        # Guarda a distribuidora escolhida pelo usuário.
        "Distribuidora": distribuidora,
        # Guarda a natureza escolhida pelo usuário.
        "Natureza": natureza,
        # Guarda a tipologia escolhida pelo usuário.
        "Tipologia": tipologia,
        # Guarda o canal de entrada escolhido pelo usuário.
        "CanalEntrada": canal,
        # Guarda o dia da semana escolhido pelo usuário.
        "dia_semana_abertura": dia_semana,
        # Guarda a hora de abertura escolhida pelo usuário.
        "hora_abertura": hora,
        # Guarda o mês de abertura escolhido pelo usuário.
        "mes_abertura": mes
    }])
    
    # Usa o modelo de regressão para prever o tempo de resolução em dias.
    dias_previstos = modelo_reg.predict(novo_chamado)[0]
    # Usa o modelo de classificação para calcular a probabilidade da classe 1 e converter para porcentagem.
    prob_atraso = modelo_clf.predict_proba(novo_chamado)[0][1] * 100
    
    # Insere uma linha divisória visual na página.
    st.divider()
    # Mostra o subtítulo da área de resultados.
    st.subheader("📊 Resultados da Análise IA")
    
    # Cria duas colunas para exibir as métricas principais.
    col_res1, col_res2 = st.columns(2)
    # Mostra a previsão de tempo de resolução na primeira coluna.
    col_res1.metric(label="Tempo Estimado de Resolução", value=f"{dias_previstos:.1f} dias")
    # Mostra a probabilidade de estouro de SLA na segunda coluna.
    col_res2.metric(label="Risco de Estourar SLA", value=f"{prob_atraso:.1f}%")
    
    # Verifica se a probabilidade de atraso passou do limiar definido pelo usuário.
    if prob_atraso >= limiar:
        # Mostra uma mensagem vermelha de alerta crítico quando o risco supera o limiar.
        st.error(f"⚠️ **ALERTA CRÍTICO!** Risco de {prob_atraso:.1f}% superou o limiar de segurança de {limiar}%. Escalonar imediatamente!")
    # Caso a probabilidade fique abaixo do limiar, entra no fluxo normal.
    else:
        # Mostra uma mensagem verde indicando que o chamado pode seguir normalmente.
        st.success(f"✅ **FLUXO NORMAL.** Risco de {prob_atraso:.1f}% está dentro da margem de segurança configurada.")
