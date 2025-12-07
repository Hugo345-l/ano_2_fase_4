# CardioIA - Fase 4: Assistente Cardiológico com Visão Computacional

## 📂 Informações do Projeto

### Caminho do Repositório
```
C:\Users\gugue\Documents\projetos_fiap\ano_2_fase_4\
```

**Estrutura de Pastas Esperada:**
```
C:\Users\gugue\Documents\projetos_fiap\ano_2_fase_4\
├── PLANNING.md                    # Este arquivo (planejamento do projeto)
├── TASKS.md                       # Lista de tarefas detalhadas
├── dataset\                       # Dataset Chest X-Ray Pneumonia
│   ├── train\
│   │   ├── NORMAL\
│   │   └── PNEUMONIA\
│   ├── test\
│   │   ├── NORMAL\
│   │   └── PNEUMONIA\
│   └── val\
│       ├── NORMAL\
│       └── PNEUMONIA\
├── notebooks\                     # Notebooks Jupyter
│   └── cardioai_cnn_analysis.ipynb
├── models\                        # Modelos treinados salvos
│   ├── cnn_baseline.keras
│   └── transfer_learning_vgg16.keras
├── reports\                       # Relatórios e documentação
│   └── relatorio_tecnico.pdf
└── results\                       # Gráficos, métricas e visualizações
    ├── metricas_comparacao.png
    └── matriz_confusao.png
```

---
## 📌 Resumo Executivo do Briefing

### Contexto do Projeto
A CardioIA avança para a análise de dados médicos utilizando **Visão Computacional**, desenvolvendo um protótipo capaz de transformar imagens médicas em informações interpretáveis para auxiliar na tomada de decisão clínica.

### Objetivo Geral
Construir um protótipo de Assistente Cardiológico Virtual que:
- Realize pré-processamento de imagens médicas
- Treine e avalie modelos de CNN para classificar padrões em imagens
- Apresente resultados de forma acessível em notebook interativo

---

## 📦 Pacotes e Dependências Requeridas

### Ambiente
- **Python:** 3.12 (já instalado)
- **IDE:** VSCode (já configurado)
- **Sistema Operacional:** Windows 11
- **Opcional:** Google Colab (se precisar de GPU para treinamento)

### Bibliotecas Principais

**Deep Learning & Visão Computacional:**
```python
tensorflow>=2.15.0          # Framework principal para CNN
keras>=3.0.0               # API de alto nível (incluso no TF)
pillow>=10.0.0             # Manipulação de imagens
opencv-python>=4.8.0       # Processamento de imagens (opcional)
```

**Manipulação de Dados:**
```python
numpy>=1.24.0              # Arrays e operações matemáticas
pandas>=2.0.0              # Análise de dados tabulares
```

**Visualização:**
```python
matplotlib>=3.7.0          # Gráficos e plots
seaborn>=0.12.0            # Visualizações estatísticas
plotly>=5.14.0             # Gráficos interativos (opcional)
```

**Métricas e Avaliação:**
```python
scikit-learn>=1.3.0        # Métricas, matriz de confusão, split
```

**Utilidades:**
```python
jupyter>=1.0.0             # Notebook interativo
ipywidgets>=8.0.0          # Widgets interativos (opcional)
tqdm>=4.65.0               # Barra de progresso
```

### Instalação Rápida
```bash
pip install tensorflow pillow numpy pandas matplotlib seaborn scikit-learn jupyter tqdm
```

**Nota:** Se usar Google Colab, TensorFlow, NumPy, Pandas, Matplotlib e Scikit-learn já vêm pré-instalados.

---

## 🎯 Escopo de Entrega (Trabalho Individual - Entregável Básico)

### Pontuação Total: 10 pontos

| Critério | Pontos |
|----------|--------|
| Pipeline de pré-processamento implementado | 3 |
| Treinamento e avaliação de CNN do zero | 2 |
| Implementação de Transfer Learning funcional | 2 |
| Apresentação dos resultados em protótipo simples | 2 |
| Documentação clara | 1 |

**Nota:** Trabalho em equipe daria +1 ponto extra, mas será feito individual.

---

## 📊 Dataset Escolhido

**Chest X-Ray Images (Pneumonia)**
- **Fonte:** Kaggle - Paul Timothymooney
- **Link:** https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- **Características:**
  - Classificação binária: NORMAL vs PNEUMONIA
  - ~5.863 imagens JPEG
  - Pré-organizado em train/test/val
  - Imagens de raio-X de tórax de pacientes pediátricos

**Justificativa da Escolha:**
- ✅ Classificação binária (mais simples para primeira experiência com CNN)
- ✅ Dataset já organizado em estrutura adequada
- ✅ Tamanho gerenciável (~1.2GB)
- ✅ Contexto médico relevante para o projeto CardioIA
- ✅ Transfer Learning funciona muito bem neste dataset
- ✅ Facilita validação visual dos resultados

---

## 🗺️ Macro Etapas do Projeto

### **ETAPA 1: Exploração e Compreensão dos Dados**
**Objetivo:** Entender a estrutura, qualidade e distribuição do dataset antes de qualquer processamento.

**Atividades:**
- Explorar estrutura de diretórios do dataset
- Verificar quantidade de imagens por classe (NORMAL vs PNEUMONIA)
- Analisar dimensões e formatos das imagens
- Visualizar amostras de cada classe
- Identificar possíveis desbalanceamentos

**✅ CHECKPOINT 1 - Validação:**
- [x] Confirmar que consegue carregar imagens corretamente
- [x] Visualizar pelo menos 5 imagens de cada classe
- [x] Entender a distribuição de classes (quantidade NORMAL vs PNEUMONIA)
- [x] Identificar dimensões predominantes das imagens

**Entregável Parcial:** Notebook com análise exploratória visual

### 🎯 RESULTADOS OBTIDOS - CHECKPOINT 1 ✅ CONCLUÍDO

**Status:** ✅ ETAPA 1 COMPLETADA - Dezembro 2024

**Dados Confirmados do Dataset:**
- **TRAIN:** 5.216 imagens (1.341 NORMAL, 3.875 PNEUMONIA)
- **TEST:** 624 imagens (234 NORMAL, 390 PNEUMONIA)  
- **VAL:** 16 imagens (8 NORMAL, 8 PNEUMONIA)
- **TOTAL:** 5.856 imagens

**Análise de Balanceamento:**
- Proporção PNEUMONIA/NORMAL: **2.89:1**
- Dataset DESBALANCEADO ⚠️
- **Estratégia:** Class weights calculados e serão aplicados no treinamento
  - NORMAL (classe 0): weight = 1.9439
  - PNEUMONIA (classe 1): weight = 0.6726

**Análise de Dimensões (amostra de 100 imagens):**
- Largura média: ~1000-2000 pixels
- Altura média: ~1000-2000 pixels
- Dimensões variadas - redimensionamento para 224x224 necessário

**Visualizações Criadas:**
- ✅ `results/dataset_distribution.png` - Gráfico de distribuição de classes
- ✅ `results/samples_train.png` - Amostras visuais do dataset
- ✅ `results/image_dimensions_distribution.png` - Histogramas de dimensões

**Configurações Definidas:**
- IMG_SIZE: 224x224x3 (RGB)
- BATCH_SIZE: 32
- EPOCHS: 25 (inicial)
- LEARNING_RATE: 0.001
- TensorFlow/Keras configurado com seeds para reprodutibilidade

**Observações Clínicas:**
- Imagens NORMAL mostram pulmões limpos e claros
- Imagens PNEUMONIA apresentam opacidades e infiltrados
- Recall (sensibilidade) será métrica crítica - não podemos perder casos de PNEUMONIA!

---

### **ETAPA 2: Pré-processamento e Preparação dos Dados**
**Objetivo:** Preparar o dataset para treinamento das redes neurais, garantindo formato e qualidade adequados.

**Atividades:**
- Redimensionar imagens para tamanho padrão (ex: 224x224 para VGG16/ResNet)
- Normalizar valores dos pixels (0-255 → 0-1)
- Aplicar data augmentation no conjunto de treino (rotação, zoom, flip)
- Organizar estrutura em train/validation/test
- Criar generators/loaders para os dados

**✅ CHECKPOINT 2 - Validação:**
- [x] Verificar que todas as imagens foram redimensionadas corretamente
- [x] Confirmar que normalização foi aplicada (valores entre 0-1)
- [x] Visualizar exemplos de data augmentation aplicado
- [x] Validar que os conjuntos de treino/validação/teste estão separados
- [x] Testar que os generators carregam batches corretamente

**Entregável Parcial:** Pipeline de pré-processamento funcional e testado

### 🎯 RESULTADOS OBTIDOS - CHECKPOINT 2 🚧 EM DESENVOLVIMENTO

**Status:** 🚧 ETAPA 2 AGUARDANDO IMPLEMENTAÇÃO

---

### **ETAPA 3: CNN Simples do Zero**
**Objetivo:** Implementar e treinar uma rede neural convolucional básica criada do zero.

**Atividades:**
- Desenhar arquitetura de CNN simples (Conv2D + MaxPooling + Dense)
- Compilar modelo com otimizador e função de perda adequados
- Treinar modelo com callbacks (EarlyStopping, ModelCheckpoint)
- Gerar gráficos de loss e accuracy durante treinamento
- Avaliar modelo no conjunto de teste

**Detalhamento Técnico:**
- **Arquitetura sugerida:** 3-4 blocos Conv2D + MaxPooling, seguidos de camadas Dense
- **Input shape:** (224, 224, 3) para RGB
- **Função de ativação:** ReLU nas convolucionais, Sigmoid na saída (binária)
- **Otimizador:** Adam (learning_rate=0.001)
- **Loss:** binary_crossentropy
- **Batch size:** 32
- **Epochs:** 20-30 (com early stopping patience=5)

**✅ CHECKPOINT 3 - Validação:**
- [x] Modelo compila sem erros
- [x] Treinamento executa e mostra progresso epoch por epoch
- [x] Gráficos de loss/accuracy são gerados
- [x] Acurácia no conjunto de validação > 60% (baseline mínimo)
- [x] Modelo consegue fazer predições em imagens novas

**Entregável Parcial:** CNN básica treinada com métricas documentadas

### 🎯 RESULTADOS OBTIDOS - CHECKPOINT 3 ✅ CONCLUÍDO

**Status:** ✅ ETAPA 3 COMPLETADA - Dezembro 2024

**Configuração de Treinamento:**
- Treinado no Kaggle com GPU Tesla T4
- Epochs: 12 (early stopping no epoch 12, best model epoch 5)
- Batch size: 32
- Learning rate: 0.001 (com ReduceLROnPlateau)
- Tempo de treinamento: ~21 minutos

**Resultados no Conjunto de Teste (624 imagens):**

**📊 Métricas Gerais:**
- **Test Accuracy:** 88.46% ✅ (Meta: >70% - SUPERADA!)
- **Test Precision:** 86.81%
- **Test Recall:** 96.15% ✅✅ (Meta: >85% - SUPERADA!)
- **Test AUC:** 0.9437
- **Test Loss:** 0.4529

**🏥 Análise por Classe:**
- **NORMAL (234 imagens):**
  - Acertos: 177 (75.6%)
  - Precision: 92.19%
  - Recall: 75.64%
  - F1-Score: 0.8310

- **PNEUMONIA (390 imagens):**
  - Acertos: 375 (96.2%) ⭐
  - Precision: 86.81%
  - Recall: 96.15% ⭐⭐
  - F1-Score: 0.9124

**📊 Matriz de Confusão:**
```
                Predito
              NORMAL  PNEUMONIA
    NORMAL      177      57     (FP - alarmes falsos)
    PNEUMONIA    15     375     (FN - casos perdidos)
```

**🎯 Análise Crítica dos Resultados:**

**✅ PONTOS FORTES:**
1. **Recall de 96.15% é EXCEPCIONAL!**
   - Apenas 15 falsos negativos de 390 casos reais de PNEUMONIA
   - Taxa de detecção de casos críticos: 96.2%
   - **CRÍTICO para medicina:** O modelo NÃO deixa passar a maioria dos casos graves!

2. **Acurácia geral de 88.46%**
   - Supera amplamente a meta mínima de 70%
   - Resultado sólido para uma CNN do zero

3. **AUC de 0.9437**
   - Excelente capacidade discriminativa do modelo

**⚠️ PONTOS DE ATENÇÃO:**
1. **57 Falsos Positivos (FP)**
   - 57 pacientes saudáveis classificados como PNEUMONIA
   - Em contexto médico: **ACEITÁVEL** - melhor cautela excessiva
   - Esses casos seriam verificados por radiologistas

2. **Recall de NORMAL mais baixo (75.64%)**
   - Modelo tende a ser mais conservador
   - Prioriza não perder casos de PNEUMONIA (correto!)

**🎓 Conclusão:**
O modelo atingiu um **excelente trade-off** entre sensibilidade e especificidade para aplicação médica. A alta taxa de recall para PNEUMONIA (96.15%) é o resultado mais importante, pois minimiza casos não detectados - crítico em saúde!

**Visualizações Geradas:**
- ✅ `results/cnn_baseline_confusion_matrix.png` - Matriz de confusão
- ✅ `results/cnn_baseline_metrics_detailed.png` - Gráfico de métricas detalhadas

---

### **ETAPA 4: Transfer Learning com Modelo Pré-treinado** ✅ CONCLUÍDA
**Objetivo:** Implementar Transfer Learning usando modelo pré-treinado (VGG16 ou ResNet50) para melhorar resultados.

**Atividades:**
- Carregar modelo pré-treinado (VGG16 ou ResNet50) sem a camada de classificação
- Congelar camadas base do modelo
- Adicionar camadas personalizadas de classificação
- Treinar apenas as camadas novas (fine-tuning opcional)
- Comparar resultados com CNN do zero

**Detalhamento Técnico:**
- **Modelo escolhido:** VGG16 ✅
- **Weights:** 'imagenet' (pré-treinado)
- **Include_top:** False (sem camadas de classificação)
- **Camadas customizadas:** GlobalAveragePooling2D + Dense(256) + BatchNorm + Dropout(0.5) + Dense(128) + Dropout(0.3) + Dense(1, sigmoid)
- **Parâmetros:** 14.714.688 não-treináveis (VGG16) + 164.865 treináveis (novas camadas)
- **Epochs:** 21 (early stopping, melhor: epoch 9)

**✅ CHECKPOINT 4 - Validação:**
- [x] Modelo pré-treinado carregou corretamente
- [x] Camadas base estão congeladas (trainable=False)
- [x] Treinamento convergiu mais rápido que CNN do zero
- [x] Acurácia no conjunto de validação > 80% (93.75% no melhor epoch)
- [x] Transfer Learning comparado com CNN básica

**Entregável Parcial:** Modelo de Transfer Learning treinado com comparativo

### 🎯 RESULTADOS OBTIDOS - CHECKPOINT 4 ✅ CONCLUÍDO

**Status:** ✅ ETAPA 4 COMPLETADA - Dezembro 2024

**Configuração de Treinamento:**
- Treinado no Kaggle com GPU Tesla P100
- Epochs: 21 (early stopping, melhor modelo: epoch 9)
- Batch size: 32
- Learning rate: 0.001 (reduzido 3x durante treino)
- Tempo de treinamento: ~32 minutos

**Resultados no Conjunto de Teste (624 imagens):**

**📊 Métricas Gerais:**
- **Test Accuracy:** 90.71% ✅ (Meta: >85% - SUPERADA!)
- **Test Precision:** 93.23% ✅ (Excelente!)
- **Test Recall:** 91.79% (Bom, mas menor que CNN Baseline)
- **Test AUC:** 0.9581
- **Test Loss:** 0.3264

**🏥 Análise por Classe:**
- **NORMAL (234 imagens):**
  - Acertos: 208 (88.9%)
  - Precision: 88.9%
  - Recall: 88.9%

- **PNEUMONIA (390 imagens):**
  - Acertos: 358 (91.8%)
  - Precision: 93.2%
  - Recall: 91.8%

**📊 Matriz de Confusão:**
```
                Predito
              NORMAL  PNEUMONIA
    NORMAL      208      26     (FP - alarmes falsos)
    PNEUMONIA    32     358     (FN - casos perdidos)
```

**Visualizações Geradas:**
- ✅ `results/transfer_learning_confusion_matrix.png`
- ✅ `results/comparison_metrics.png`
- ✅ `results/comparison_confusion_matrices.png`

---

### **ETAPA 5: Avaliação Completa dos Modelos** ✅ CONCLUÍDA
**Objetivo:** Avaliar ambos os modelos com métricas detalhadas e comparar performance.

**Atividades:**
- Calcular métricas: Acurácia, Precisão, Recall, F1-Score
- Gerar matriz de confusão para cada modelo
- Criar relatório comparativo CNN vs Transfer Learning
- Visualizar predições corretas e incorretas
- Analisar casos de erro (falsos positivos/negativos)

**Interpretação das Métricas (contexto médico):**
- **Acurácia:** % total de acertos (geral)
- **Precisão:** De todos que prevemos como PNEUMONIA, quantos realmente eram? (evitar alarmes falsos)
- **Recall (Sensibilidade):** De todos os casos de PNEUMONIA reais, quantos detectamos? (evitar casos perdidos - CRÍTICO em saúde!)
- **F1-Score:** Balanceamento entre Precisão e Recall
- **Matriz de Confusão:** Visualizar FP (falso positivo) vs FN (falso negativo)

**✅ CHECKPOINT 5 - Validação:**
- [x] Todas as métricas foram calculadas para ambos os modelos
- [x] Matrizes de confusão estão legíveis e corretas
- [x] Comparação clara entre os dois modelos
- [x] Análise quantitativa de erros (FN e FP)
- [x] Análise crítica dos resultados documentada

**Entregável Parcial:** Relatório de métricas e análise comparativa

### 🎯 RESULTADOS OBTIDOS - CHECKPOINT 5 ✅ CONCLUÍDO

**Status:** ✅ ETAPA 5 COMPLETADA - Dezembro 2024 (Integrada na Etapa 4)

**📊 COMPARAÇÃO FINAL: CNN Baseline vs Transfer Learning VGG16**

| Métrica | CNN Baseline | Transfer Learning | Diferença | Vencedor |
|---------|--------------|-------------------|-----------|----------|
| **Accuracy** | 88.46% | **90.71%** | +2.25% | Transfer Learning ✅ |
| **Precision** | 86.81% | **93.23%** | +7.39% | Transfer Learning ✅ |
| **Recall** | **96.15%** | 91.79% | -4.36% | **CNN Baseline** ✅ |
| **AUC** | 94.37% | **95.81%** | +1.52% | Transfer Learning ✅ |
| **False Negatives** | **15** | 32 | +17 casos | **CNN Baseline** ✅ |
| **False Positives** | 57 | **26** | -31 casos | Transfer Learning ✅ |

**🏥 ANÁLISE CRÍTICA PARA USO MÉDICO:**

**✅ Vantagens do Transfer Learning:**
1. **Maior acurácia geral** (+2.25%) - Menos erros totais
2. **Maior precisão** (+7.39%) - Menos alarmes falsos (57 → 26 FP)
3. **Melhor AUC** (+1.52%) - Melhor capacidade discriminativa
4. **Convergência mais rápida** - Features pré-treinadas aceleram treinamento

**⚠️ PONTO CRÍTICO - Contexto Médico:**
1. **CNN Baseline é SUPERIOR para detecção de PNEUMONIA!** 🏥
2. **Recall de 96.15% vs 91.79%**: CNN detecta **4.36% mais casos**
3. **15 vs 32 False Negatives**: CNN perde **17 casos A MENOS**
4. **Em medicina, Recall é CRÍTICO** - perder pacientes doentes pode ser fatal

**🎯 RECOMENDAÇÃO FINAL:**

Para **aplicação médica real (triagem de PNEUMONIA)**:
- ✅ **USE CNN BASELINE** - Recall de 96.15% é crucial
- Apenas 3.85% de casos perdidos (15 de 390)
- Aceitar 57 alarmes falsos é preferível a perder 17 casos graves

Para **aplicação geral (precisão prioritária)**:
- ✅ **USE Transfer Learning** - Acurácia 90.71%, Precision 93.23%
- Menos alarmes falsos (26 vs 57)
- Melhor para cenários onde FP tem custo alto

**💡 PRÓXIMOS PASSOS SUGERIDOS:**
1. **Ensemble**: Combinar ambos os modelos para melhor desempenho
2. **Fine-tuning**: Descongelar últimas camadas do VGG16
3. **Ajustar threshold**: Modificar limiar de decisão para aumentar recall do TL
4. **Class weights**: Aumentar peso da classe PNEUMONIA no Transfer Learning

---

### **ETAPA 6: Protótipo de Apresentação (Notebook Interativo)**
**Objetivo:** Organizar o notebook final com apresentação clara e interativa dos resultados.

**Atividades:**
- Estruturar notebook com seções claras e navegáveis
- Adicionar textos explicativos em Markdown
- Criar seção de "Demo" para testar com imagens novas
- Incluir visualizações de resultados (gráficos, imagens, métricas)
- Garantir reprodutibilidade do código

**Estrutura Sugerida do Notebook:**
1. **Introdução e Contexto** (Markdown)
2. **Importação de Bibliotecas**
3. **Exploração dos Dados** (Etapa 1)
4. **Pré-processamento** (Etapa 2)
5. **Modelo 1: CNN do Zero** (Etapa 3)
6. **Modelo 2: Transfer Learning** (Etapa 4)
7. **Comparação de Resultados** (Etapa 5)
8. **Demo Interativa** (upload de imagem e predição)
9. **Conclusões e Próximos Passos**

**✅ CHECKPOINT 6 - Validação Final:**
- [ ] Notebook executa do início ao fim sem erros
- [ ] Todas as seções têm títulos e explicações claras
- [ ] Gráficos e visualizações estão legíveis
- [ ] Função de predição em imagens novas funciona
- [ ] Resultados finais estão destacados e bem apresentados

**Entregável Parcial:** Notebook interativo finalizado

---

### **ETAPA 7: Documentação e Relatório Final**
**Objetivo:** Criar relatório técnico conciso documentando todo o processo e resultados.

**Atividades:**
- Escrever relatório de 1-2 páginas conforme solicitado
- Documentar etapas do pipeline de pré-processamento
- Justificar escolhas técnicas (arquiteturas, hiperparâmetros)
- Apresentar resultados com tabelas e gráficos
- Incluir conclusões e possíveis melhorias futuras

**Estrutura do Relatório:**
1. **Introdução** (contexto e objetivo)
2. **Metodologia:**
   - Dataset utilizado
   - Pipeline de pré-processamento
   - Arquiteturas implementadas
3. **Resultados:**
   - Tabela comparativa de métricas
   - Análise dos resultados
4. **Conclusões:**
   - Modelo mais eficaz e por quê
   - Limitações encontradas
   - Próximos passos

**✅ CHECKPOINT 7 - Entrega Final:**
- [ ] Relatório tem 1-2 páginas conforme especificado
- [ ] Todas as escolhas técnicas estão justificadas
- [ ] Resultados estão apresentados de forma clara
- [ ] Documento está bem formatado e sem erros

**Entregável Final:** Relatório técnico completo

---

## 📦 Entregáveis Finais do Projeto

1. **Notebook Python (Google Colab ou Local)**
   - Código de pré-processamento completo
   - Implementação de CNN do zero
   - Implementação de Transfer Learning
   - Avaliação com métricas detalhadas
   - Protótipo de demonstração interativa

2. **Relatório Técnico (1-2 páginas)**
   - Descrição do pipeline de pré-processamento
   - Justificativa das escolhas técnicas
   - Apresentação dos resultados
   - Análise comparativa dos modelos
   - Conclusões e próximos passos

3. **Prints das Métricas** (integrados no notebook)
   - Acurácia, Precisão, Recall, F1-Score
   - Matrizes de confusão
   - Gráficos de loss e accuracy

---

## 🎓 Observações Importantes

### Critérios de Sucesso
- Pipeline de pré-processamento documentado e funcional
- Duas abordagens implementadas e funcionais (CNN + Transfer Learning)
- Métricas de avaliação completas e corretas
- Notebook organizado e reprodutível
- Documentação clara das escolhas

### Dicas para Execução
- **Validar cada checkpoint antes de avançar** - não pule etapas!
- Salvar modelos treinados para não precisar retreinar (.h5 ou .keras)
- Usar Google Colab se não tiver GPU local (treinamento 10x+ mais rápido)
- Documentar problemas encontrados e soluções aplicadas
- Manter código organizado com comentários claros
- Fazer commits frequentes no Git (versionamento do código)

### Troubleshooting Comum

**Problema: Overfitting (treino >>  validação)**
- Solução: Aumentar Dropout, adicionar mais Data Augmentation, reduzir complexidade do modelo

**Problema: Underfitting (ambos com acurácia baixa)**
- Solução: Aumentar complexidade do modelo, mais epochs, ajustar learning rate

**Problema: Treinamento muito lento**
- Solução: Usar GPU (Colab), reduzir tamanho das imagens, diminuir batch size

**Problema: Out of Memory (OOM)**
- Solução: Reduzir batch size, usar imagens menores, limpar cache (`tf.keras.backend.clear_session()`)

---

**Versão:** 1.2  
**Data de Criação:** Dezembro 2024  
**Última Atualização:** Dezembro 2024  
**Autor:** Hugo - Analista BI Ifood  
**Projeto:** CardioIA - Fase 4 - FIAP
**Repositório:** `C:\Users\gugue\Documents\projetos_fiap\ano_2_fase_4\`
