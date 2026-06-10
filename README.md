Disciplina de Inteligência Artificial - Professor Munif - Unicesumar 2026

## Integrantes
- Gabriel Andrade Garcia  - RA:23271855-2
- Thiago Nunes de Souza - RA: 23000383-2
- Pedro Antônio Paiva - RA: 26010619-2

# Estimativa de Gasto Calórico e Treinos

## 2. Resumo do Projeto

### Contextualização do Tema
O acompanhamento do gasto calórico é fundamental para o planejamento de treinos e rotinas de saúde. Com o avanço dos dispositivos de monitoramento e análise de dados, tornou-se possível estimar o impacto fisiológico das atividades físicas de forma mais precisa. Este projeto foca na intersecção entre dados de saúde e Inteligência Artificial para a estimativa de gasto calórico em treinos, utilizando métricas reais de esforço e biotipo.

### Problema Investigado (Regressão)
Como prever o gasto calórico exato de um indivíduo durante um treino com base em suas características físicas (idade, peso, batimentos cardíacos) e no tipo de exercício realizado?

### Hipótese da Equipe
Modelos de Inteligência Artificial conseguem estimar com alta precisão as calorias queimadas (`Calories_Burned`). A hipótese central é que a duração da sessão (`Session_Duration (hours)`) e os batimentos cardíacos médios (`Avg_BPM`) serão os atributos com maior impacto no resultado, superando características físicas isoladas.

### Descrição do Dataset Utilizado
O dataset **Gym Members Exercise Dataset**, obtido através do Kaggle, contém 973 registros de usuários detalhando atributos como idade, gênero, peso, altura, tipo de treino, frequência cardíaca e duração da sessão. A nossa Variável Alvo (target) é a coluna `Calories_Burned`.
**Link:** [Kaggle - Gym Members Exercise Dataset](https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset)

---

## 3. Metodologia e Pré-Processamento

### Análise Exploratória e Seleção de Características
Antes de treinar os modelos de Inteligência Artificial, realizamos um Estudo de Ablação para identificar quais características físicas e de treino possuíam maior impacto na queima de calorias. 

Utilizamos a função `.corr()` da biblioteca Pandas para calcular o **Coeficiente de Correlação de Pearson** entre todas as variáveis numéricas. Os resultados foram plotados em um Mapa de Calor (Heatmap).

Através desta análise estatística, comprovamos a nossa hipótese inicial: as variáveis de tempo de treino e batimentos cardíacos médios apresentaram a correlação positiva mais forte (valores mais próximos de 1) com a nossa Variável Alvo.

> **[INSERIR IMAGEM DO MAPA DE CALOR AQUI]**

### Tratamento de Dados e Normalização
- **Tradução de Texto para Número (Dummy Encoding):** O comando `pd.get_dummies` foi utilizado para transformar variáveis categóricas (como Gênero e Tipo de Treino) em colunas binárias ("Sim ou Não").
- **Normalização (StandardScaler):** Realizamos a ponderação dos dados utilizando o `StandardScaler`. Sem isso, o modelo daria um "peso" injusto à frequência cardíaca (ex: 150 bpm) sobre o tempo de treino (ex: 1.5 horas) apenas pelo tamanho do número. A normalização equilibrou as escalas.
- **Divisão (Holdout):** Utilizamos o `train_test_split` para separar a base em 80% para treino e 20% para teste.

### Criação dos Lotes de Teste
Para um teste científico estruturado, dividimos as variáveis em 3 lotes de complexidade crescente:
1. **Lote 1 (O básico do esforço):** Apenas Tempo de treino e Batimentos Cardíacos.
2. **Lote 2 (Esforço + Corpo):** Lote 1 somado a Peso, Idade, Gênero e % de Gordura.
3. **Lote 3 (Tudo junto):** Todas as colunas possíveis da tabela simultaneamente.

---

## 4. Métodos de IA Utilizados

### Modelo Parte 1: K-Nearest Neighbors (K-NN)
O K-NN Regressor foi escolhido como método inicial. A lógica consiste em procurar na base de dados os "K" alunos (configurado para 5) com características mais parecidas com a amostra testada, tirando uma média das calorias deles. A ponderação utilizada foi a padrão (uniforme).

### Modelo Parte 2: Support Vector Regression (SVR / SVM)
Para a etapa avançada, aplicamos a variação do SVM para valores contínuos (SVR) utilizando um **Kernel Linear**. Esse modelo desenha um hiperplano matemático (uma linha reta de separação) através do espaço de dados normatizado. Diferente do K-NN, o SVR é extremamente robusto contra dados de alta dimensionalidade (Maldição da Dimensionalidade).

---

## 5. Avaliação dos Modelos e Comparação dos Resultados

Como o nosso objetivo é descobrir um gasto calórico exato, caracterizando um problema de **Regressão**, o uso do F1-Score não é matematicamente aplicável. Por isso, a nossa avaliação foi ancorada no **MAE (Erro Absoluto Médio)** e no **R² (Acurácia de ajuste)**.

> **[INSERIR IMAGEM DO GRÁFICO FINAL DE BARRAS AGRUPADAS DOS 3 LOTES AQUI]**

### Análise Comparativa por Lote:
1. **Sobre o Lote 1 (O Empate Técnico):** Onde usamos apenas 2 variáveis (Tempo e BPM), houve um empate técnico. Em um cenário de baixa complexidade, o cálculo de distância do K-NN funciona super bem.
2. **Sobre o Lote 2 (A Virada do SVM):** Adicionando as características corporais, o erro caiu em ambos. Porém, o SVM já começou a se distanciar, derrubando a margem de erro. O hiperplano matemático se ajustou melhor aos dados físicos.
3. **Sobre o Lote 3 (A Maldição da Dimensionalidade):** O grande destaque do trabalho. Ao introduzir todas as variáveis simultaneamente, o K-NN colapsou. O erro disparou devido à Maldição da Dimensionalidade. Em contrapartida, o SVR com Kernel Linear manteve-se totalmente estável, ignorou o ruído das colunas desnecessárias e manteve um erro baixíssimo de cerca de 38 calorias.

---

## 6. Conclusão
O projeto comprovou a hipótese inicial sobre a forte influência do ritmo cardíaco e tempo de treino no gasto calórico. Através da experimentação estruturada em lotes, ficou provado que algoritmos baseados em distância (K-NN) falham em bases de dados complexas e com muitas colunas de ruído. A aplicação do método avançado (SVR com Kernel Linear) resolveu esse gargalo com perfeição, demonstrando capacidade de generalização e entregando estimativas calóricas precisas e estáveis em todos os cenários testados.
