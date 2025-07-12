import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Configuração da página
st.set_page_config(
    page_title="Classificador de Iris - ML Simples",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para melhorar a aparência
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .prediction-box {
        background-color: #e8f5e8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #28a745;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #ffc107;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Carrega o modelo treinado"""
    try:
        model = joblib.load('model/iris_model.pkl')
        model_info = joblib.load('model/model_info.pkl')
        return model, model_info
    except FileNotFoundError:
        st.error("Modelo não encontrado! Execute primeiro o script de treinamento.")
        return None, None

def create_feature_importance_plot(feature_names, feature_importance):
    """Cria gráfico de importância das features"""
    fig = px.bar(
        x=feature_names,
        y=feature_importance,
        title="Importância das Features",
        labels={'x': 'Features', 'y': 'Importância'},
        color=feature_importance,
        color_continuous_scale='Blues'
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        height=400,
        showlegend=False
    )
    return fig

def create_confusion_matrix_plot(y_true, y_pred, target_names):
    """Cria matriz de confusão"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig = px.imshow(
        cm,
        text_auto=True,
        aspect="auto",
        title="Matriz de Confusão",
        labels=dict(x="Predito", y="Real", color="Quantidade"),
        x=target_names,
        y=target_names,
        color_continuous_scale='Blues'
    )
    fig.update_layout(height=400)
    return fig

def main():
    # Título principal
    st.markdown('<h1 class="main-header">🌸 Classificador de Iris com Machine Learning</h1>', unsafe_allow_html=True)
    
    # Carregar modelo
    model, model_info = load_model()
    
    if model is None:
        st.stop()
    
    # Sidebar com informações
    with st.sidebar:
        st.markdown("## 📊 Informações do Modelo")
        st.markdown(f"**Acurácia:** {model_info['accuracy']:.1%}")
        st.markdown(f"**Algoritmo:** Árvore de Decisão")
        st.markdown(f"**Profundidade máxima:** 3")
        
        st.markdown("## 🌸 Espécies de Iris")
        for i, species in enumerate(model_info['target_names']):
            st.markdown(f"**{i}:** {species}")
        
        st.markdown("## 📏 Features")
        for i, feature in enumerate(model_info['feature_names']):
            st.markdown(f"**{i}:** {feature}")
    
    # Explicação do modelo
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("""
    ### 🤖 Sobre este Modelo
    
    Este é um **classificador de árvore de decisão** que identifica automaticamente a espécie de uma flor Iris 
    baseado em suas características físicas (comprimento e largura das sépalas e pétalas).
    
    **Como funciona:** O modelo analisa as medidas da flor e usa regras simples (como "se a pétala tem mais de 2.45cm, 
    então é Iris virginica") para fazer a classificação. É um exemplo perfeito de machine learning interpretável!
    
    **Dataset:** Usamos o famoso dataset Iris de Fisher, que contém 150 amostras de 3 espécies diferentes.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Tabs para diferentes funcionalidades
    tab1, tab2, tab3 = st.tabs(["🔮 Fazer Predição", "📈 Visualizações", "📊 Métricas"])
    
    with tab1:
        st.markdown('<h2 class="sub-header">🔮 Faça sua Predição</h2>', unsafe_allow_html=True)
        
        # Formulário para entrada de dados
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📏 Medidas da Flor")
            sepal_length = st.slider("Comprimento da Sépala (cm)", 4.0, 8.0, 5.4, 0.1)
            sepal_width = st.slider("Largura da Sépala (cm)", 2.0, 4.5, 3.4, 0.1)
            
        with col2:
            st.markdown("### 🌺 Medidas da Pétala")
            petal_length = st.slider("Comprimento da Pétala (cm)", 1.0, 7.0, 4.7, 0.1)
            petal_width = st.slider("Largura da Pétala (cm)", 0.1, 2.5, 1.4, 0.1)
        
        # Botão para fazer predição
        if st.button("🔮 Fazer Predição", type="primary", use_container_width=True):
            # Preparar dados
            input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            
            # Fazer predição
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]
            
            # Mostrar resultado
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown(f"### 🌸 Espécie Predita: **{model_info['target_names'][prediction]}**")
            
            # Mostrar probabilidades
            st.markdown("#### 📊 Probabilidades:")
            for i, (species, prob) in enumerate(zip(model_info['target_names'], prediction_proba)):
                color = "🟢" if i == prediction else "⚪"
                st.markdown(f"{color} **{species}:** {prob:.1%}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Mostrar valores inseridos
            st.markdown("### 📋 Valores Inseridos:")
            input_df = pd.DataFrame({
                'Feature': model_info['feature_names'],
                'Valor': [sepal_length, sepal_width, petal_length, petal_width]
            })
            st.dataframe(input_df, use_container_width=True)
    
    with tab2:
        st.markdown('<h2 class="sub-header">📈 Visualizações</h2>', unsafe_allow_html=True)
        
        # Gráfico de importância das features
        st.markdown("### 🎯 Importância das Features")
        importance_fig = create_feature_importance_plot(
            model_info['feature_names'], 
            model_info['feature_importance']
        )
        st.plotly_chart(importance_fig, use_container_width=True)
        
        # Gráfico de dispersão das features
        st.markdown("### 📊 Distribuição das Features")
        
        # Carregar dados completos para visualização
        from sklearn.datasets import load_iris
        iris = load_iris()
        df_viz = pd.DataFrame(iris.data, columns=iris.feature_names)
        df_viz['species'] = [iris.target_names[i] for i in iris.target]
        
        # Gráfico de dispersão
        fig_scatter = px.scatter(
            df_viz,
            x='petal length (cm)',
            y='petal width (cm)',
            color='species',
            title="Distribuição: Largura vs Comprimento da Pétala",
            labels={'petal length (cm)': 'Comprimento da Pétala (cm)', 
                   'petal width (cm)': 'Largura da Pétala (cm)'}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Histograma das features
        st.markdown("### 📈 Distribuição das Medidas")
        feature_to_plot = st.selectbox(
            "Selecione uma feature para visualizar:",
            model_info['feature_names']
        )
        
        fig_hist = px.histogram(
            df_viz,
            x=feature_to_plot,
            color='species',
            title=f"Distribuição de {feature_to_plot}",
            nbins=20
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with tab3:
        st.markdown('<h2 class="sub-header">📊 Métricas de Performance</h2>', unsafe_allow_html=True)
        
        # Métricas principais
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("Acurácia", f"{model_info['accuracy']:.1%}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # Calcular precisão por classe
            from sklearn.metrics import precision_score
            precision = precision_score(model_info['y_test'], model_info['y_pred'], average='weighted')
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("Precisão", f"{precision:.1%}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            # Calcular recall por classe
            from sklearn.metrics import recall_score
            recall = recall_score(model_info['y_test'], model_info['y_pred'], average='weighted')
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("Recall", f"{recall:.1%}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Matriz de confusão
        st.markdown("### 🎯 Matriz de Confusão")
        cm_fig = create_confusion_matrix_plot(
            model_info['y_test'], 
            model_info['y_pred'], 
            model_info['target_names']
        )
        st.plotly_chart(cm_fig, use_container_width=True)
        
        # Relatório de classificação
        st.markdown("### 📋 Relatório Detalhado")
        from sklearn.metrics import classification_report
        report = classification_report(
            model_info['y_test'], 
            model_info['y_pred'], 
            target_names=model_info['target_names'],
            output_dict=True
        )
        
        # Converter para DataFrame para melhor visualização
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        🌸 Projeto de Machine Learning Simples - Classificador de Iris<br>
        Desenvolvido com Streamlit e Scikit-learn
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 