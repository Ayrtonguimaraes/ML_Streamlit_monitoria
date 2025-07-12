# 🌸 Classificador de Iris - Projeto de Machine Learning Simples

Um projeto completo e funcional que demonstra machine learning de forma clara e acessível, usando o famoso dataset Iris e uma árvore de decisão para classificação.

## 🎯 Sobre o Projeto

Este projeto foi desenvolvido para ser um exemplo prático e educativo de machine learning, focado em:

- **Simplicidade**: Interface clara e intuitiva
- **Explicabilidade**: O modelo é interpretável e fácil de entender
- **Usabilidade**: Experiência do usuário otimizada
- **Educação**: Ideal para iniciantes em ML

### 🤖 O que o Modelo Faz

O classificador usa uma **árvore de decisão** para identificar automaticamente a espécie de uma flor Iris baseado em suas características físicas:

- **Comprimento da Sépala** (cm)
- **Largura da Sépala** (cm) 
- **Comprimento da Pétala** (cm)
- **Largura da Pétala** (cm)

O modelo pode classificar entre 3 espécies:
- 🌸 **Iris Setosa**
- 🌺 **Iris Versicolor** 
- 🌹 **Iris Virginica**

## 📁 Estrutura do Projeto

```
app_simples_monitoria/
├── app/
│   └── main.py              # Aplicação Streamlit principal
├── model/
│   ├── train_model.py       # Script para treinar o modelo
│   ├── iris_model.pkl       # Modelo treinado (gerado automaticamente)
│   └── model_info.pkl       # Informações do modelo (gerado automaticamente)
├── static/                  # Arquivos estáticos (CSS, imagens)
├── templates/               # Templates HTML (se necessário)
├── data/                    # Datasets
├── requirements.txt         # Dependências do projeto
└── README.md               # Este arquivo
```

## 🚀 Como Executar o Projeto

### Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)

### Passo a Passo

1. **Clone ou baixe o projeto**
   ```bash
   # Se estiver usando git
   git clone <url-do-repositorio>
   cd app_simples_monitoria
   ```

2. **Crie um ambiente virtual (recomendado)**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Instale as dependências**
   ```bash
   pip install -r requirements.txt
   ```

4. **Treine o modelo**
   ```bash
   python model/train_model.py
   ```
   Este comando irá:
   - Carregar o dataset Iris
   - Treinar uma árvore de decisão
   - Salvar o modelo em `model/iris_model.pkl`
   - Salvar informações do modelo em `model/model_info.pkl`

5. **Execute a aplicação**
   ```bash
   streamlit run app/main.py
   ```

6. **Acesse a aplicação**
   - Abra seu navegador
   - Vá para `http://localhost:8501`
   - A aplicação será carregada automaticamente

## 🎮 Como Usar a Aplicação

### 🔮 Fazer Predições

1. Vá para a aba **"Fazer Predição"**
2. Ajuste os sliders com as medidas da flor:
   - Comprimento da Sépala (4.0 - 8.0 cm)
   - Largura da Sépala (2.0 - 4.5 cm)
   - Comprimento da Pétala (1.0 - 7.0 cm)
   - Largura da Pétala (0.1 - 2.5 cm)
3. Clique em **"Fazer Predição"**
4. Veja o resultado e as probabilidades

### 📈 Visualizações

Na aba **"Visualizações"** você encontrará:
- **Gráfico de Importância das Features**: Mostra quais características são mais importantes para a classificação
- **Distribuição das Features**: Gráficos de dispersão e histogramas dos dados
- **Análise Interativa**: Selecione diferentes features para visualizar

### 📊 Métricas

Na aba **"Métricas"** você verá:
- **Acurácia do Modelo**: Porcentagem de predições corretas
- **Matriz de Confusão**: Visualização dos acertos e erros
- **Relatório Detalhado**: Precisão, recall e f1-score por classe

## 🛠️ Tecnologias Utilizadas

- **Streamlit**: Interface web interativa
- **Scikit-learn**: Machine learning (árvore de decisão)
- **Pandas**: Manipulação de dados
- **NumPy**: Computação numérica
- **Plotly**: Visualizações interativas
- **Matplotlib/Seaborn**: Gráficos adicionais

## 📊 Sobre o Dataset

O **Dataset Iris** é um dos mais famosos em machine learning:

- **150 amostras** de flores Iris
- **3 espécies** diferentes
- **4 características** medidas para cada flor
- **Dataset balanceado** (50 amostras por espécie)

### Características das Espécies

- **Iris Setosa**: Flores menores, pétalas mais largas
- **Iris Versicolor**: Tamanho médio, características intermediárias  
- **Iris Virginica**: Flores maiores, pétalas mais longas

## 🎓 Conceitos de Machine Learning Demonstrados

1. **Classificação**: Categorizar dados em classes
2. **Árvore de Decisão**: Algoritmo interpretável e visual
3. **Feature Engineering**: Seleção e importância de características
4. **Validação**: Divisão treino/teste e métricas de avaliação
5. **Interpretabilidade**: Entender como o modelo toma decisões

## 🔧 Personalização

### Modificar o Modelo

Para experimentar com outros algoritmos, edite `model/train_model.py`:

```python
# Exemplo: Usar Random Forest
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
```

### Adicionar Novos Datasets

1. Substitua o dataset Iris por outro do scikit-learn
2. Ajuste as features e labels na interface
3. Modifique os ranges dos sliders conforme necessário

## 🐛 Solução de Problemas

### Erro: "Modelo não encontrado"
- Execute primeiro: `python model/train_model.py`

### Erro: "ModuleNotFoundError"
- Verifique se todas as dependências estão instaladas: `pip install -r requirements.txt`

### Erro: "Port already in use"
- Use uma porta diferente: `streamlit run app/main.py --server.port 8502`

## 📝 Licença

Este projeto é de código aberto e pode ser usado para fins educacionais.

## 🤝 Contribuições

Contribuições são bem-vindas! Algumas ideias:
- Adicionar novos algoritmos de ML
- Melhorar a interface
- Adicionar mais visualizações
- Implementar novos datasets

## 📞 Suporte

Se você encontrar problemas ou tiver dúvidas:
1. Verifique se seguiu todos os passos de instalação
2. Confirme que está usando Python 3.8+
3. Verifique se todas as dependências foram instaladas

---

**🌸 Divirta-se explorando machine learning de forma simples e intuitiva!** 