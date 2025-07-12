#!/usr/bin/env python3
"""
Script principal para executar o projeto de Machine Learning - Classificador de Iris
"""

import os
import sys
import subprocess
import importlib.util

def check_python_version():
    """Verifica se a versão do Python é compatível"""
    if sys.version_info < (3, 8):
        print("❌ Erro: Python 3.8 ou superior é necessário!")
        print(f"Versão atual: {sys.version}")
        return False
    print(f"✅ Python {sys.version.split()[0]} detectado")
    return True

def check_dependencies():
    """Verifica se as dependências estão instaladas"""
    required_packages = [
        'streamlit', 'scikit-learn', 'pandas', 'numpy', 
        'matplotlib', 'seaborn', 'plotly', 'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} instalado")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} não encontrado")
    
    if missing_packages:
        print(f"\n⚠️  Pacotes faltando: {', '.join(missing_packages)}")
        print("Execute: pip install -r requirements.txt")
        return False
    
    return True

def train_model():
    """Treina o modelo se não existir"""
    model_path = 'model/iris_model.pkl'
    
    if os.path.exists(model_path):
        print("✅ Modelo já treinado encontrado")
        return True
    
    print("🤖 Treinando o modelo...")
    try:
        from model.train_model import train_iris_model
        train_iris_model()
        print("✅ Modelo treinado com sucesso!")
        return True
    except Exception as e:
        print(f"❌ Erro ao treinar modelo: {e}")
        return False

def run_streamlit():
    """Executa a aplicação Streamlit"""
    print("🚀 Iniciando aplicação Streamlit...")
    print("📱 A aplicação será aberta em: http://localhost:8501")
    print("🔄 Para parar a aplicação, pressione Ctrl+C")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app/main.py",
            "--server.headless", "true",
            "--server.port", "8501"
        ])
    except KeyboardInterrupt:
        print("\n👋 Aplicação encerrada pelo usuário")
    except Exception as e:
        print(f"❌ Erro ao executar Streamlit: {e}")

def main():
    """Função principal"""
    print("🌸 Classificador de Iris - Machine Learning Simples")
    print("=" * 50)
    
    # Verificações iniciais
    if not check_python_version():
        return
    
    if not check_dependencies():
        return
    
    # Treinar modelo
    if not train_model():
        return
    
    # Executar aplicação
    run_streamlit()

if __name__ == "__main__":
    main() 