# CardioIA: Assistente Cardiológico com Visão Computacional

## 🏛️ Instituição
FIAP - Faculdade de Informática e Administração Paulista

## 👨‍🎓 Integrantes
- Bruno Castro - RM558359
- Hugo Mariano - RM560688
- Matheus Castro - RM559293

---

## 📜 Descrição do Projeto

O **CardioIA** é um protótipo de assistente cardiológico que utiliza Visão Computacional para analisar imagens de raio-X de tórax e classificar a presença de pneumonia. O sistema implementa e compara duas abordagens de Redes Neurais Convolucionais (CNN): uma criada do zero (baseline) e outra utilizando Transfer Learning com o modelo VGG16. O objetivo é desenvolver um modelo acurado e, mais importante, com alta sensibilidade (recall) para auxiliar na triagem de pacientes, minimizando o risco de casos não detectados.

---

## 🚀 Guia Rápido de Uso

### 1. **Configuração do Ambiente**

- Recomenda-se o uso de ambiente virtual Python.
- Instale as dependências:
  ```bash
  pip install -r requirements.txt
  ```

### 2. **Análise e Treinamento dos Modelos**

- Abra e execute o notebook `notebooks/cardioai_cnn_analysis.ipynb` em um ambiente Jupyter (como VSCode, Jupyter Lab ou Google Colab).
- O notebook contém todas as etapas do projeto:
  1.  Análise Exploratória dos Dados (EDA)
  2.  Pré-processamento das imagens e Data Augmentation
  3.  Treinamento e avaliação do modelo CNN Baseline
  4.  Treinamento e avaliação do modelo com Transfer Learning (VGG16)
  5.  Comparação detalhada de performance entre os dois modelos.

---

## 📁 Estrutura do Repositório

```
/
├── PLANNING.md                    # Planejamento do projeto
├── TASKS.md                       # Lista de tarefas detalhadas
├── dataset/                       # Dataset Chest X-Ray Pneumonia
│   ├── train/
│   ├── test/
│   └── val/
├── notebooks/                     # Notebooks Jupyter
│   └── cardioai_cnn_analysis.ipynb
├── models/                        # Modelos treinados salvos
│   ├── cnn_baseline_best.keras
│   └── transfer_learning_vgg16_best.keras
├── results/                       # Gráficos, métricas e visualizações
├── requirements.txt
└── README.md
```

---

## 📚 Documentação e Referências

- Toda a análise, implementação e documentação do processo estão consolidadas no notebook `notebooks/cardioai_cnn_analysis.ipynb`.
- O arquivo `relatorio_cardioai.docx` contém o relatório técnico final do projeto.
- O arquivo `PLANNING.md` detalha toda a concepção, planejamento e resultados esperados do projeto.

---

## 🧪 Testes e Validação

- O notebook `cardioai_cnn_analysis.ipynb` inclui seções detalhadas para avaliação de cada modelo, com as seguintes métricas:
  - Acurácia, Precisão, Recall e F1-Score.
  - Matriz de Confusão para análise de Falsos Positivos e Falsos Negativos.
- Uma análise comparativa final recomenda o melhor modelo para aplicação em contexto médico, priorizando o **Recall** (sensibilidade) para detecção de casos de pneumonia.

---

## 🗃 Histórico de Versões

- **v1.0.0 (Dezembro/2024):**
  - Estrutura inicial do projeto com dados, notebooks e planejamento.
- **v1.1.0 (Julho/2025):**
  - Criação do repositório no GitHub.
  - Adição do README.md detalhado.

---

## 📋 Licença

Este projeto segue o modelo educacional FIAP e destina-se a fins acadêmicos.
