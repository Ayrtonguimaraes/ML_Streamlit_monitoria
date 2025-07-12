import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import warnings

# Suprimir warnings desnecessários
warnings.filterwarnings('ignore')

def train_iris_model():
    """
    Treina um modelo de árvore de decisão para classificação do dataset Iris
    """
    try:
        # Carregar o dataset Iris
        iris = load_iris()
        X = iris.data
        y = iris.target
        
        # Criar DataFrame para melhor visualização
        feature_names = iris.feature_names
        target_names = iris.target_names
        
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        df['species'] = [target_names[i] for i in y]
        
        # Dividir os dados em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Treinar o modelo
        model = DecisionTreeClassifier(
            max_depth=3,  # Limitar profundidade para evitar overfitting
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Fazer predições
        y_pred = model.predict(X_test)
        
        # Calcular acurácia
        accuracy = accuracy_score(y_test, y_pred)
        
        # Criar diretório se não existir
        os.makedirs('model', exist_ok=True)
        
        # Salvar o modelo
        joblib.dump(model, 'model/iris_model.pkl')
        
        # Salvar informações do modelo
        model_info = {
            'feature_names': feature_names,
            'target_names': target_names,
            'accuracy': accuracy,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'feature_importance': model.feature_importances_
        }
        
        joblib.dump(model_info, 'model/model_info.pkl')
        
        print(f"Modelo treinado com sucesso!")
        print(f"Acurácia: {accuracy:.3f}")
        print(f"Features: {feature_names}")
        print(f"Classes: {target_names}")
        
        return model, model_info
        
    except Exception as e:
        print(f"Erro ao treinar modelo: {e}")
        return None, None

if __name__ == "__main__":
    train_iris_model() 