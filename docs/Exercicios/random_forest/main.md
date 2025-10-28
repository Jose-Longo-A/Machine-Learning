### Introdução

A Random Forest é um método de aprendizado supervisionado baseado em conjunto de árvores de decisão. A ideia central é treinar várias árvores em amostras bootstrap do conjunto de treino, introduzindo aleatoriedade tanto nas amostras quanto na seleção de atributos em cada divisão. A predição final resulta do voto da maioria (classificação) ou da média (regressão).
Essa estratégia reduz overfitting típico de árvores individuais, melhora a generalização e mantém boa interpretabilidade via importância de atributos. Além disso, por ser baseada em árvores, lida naturalmente com relações não lineares e interações entre variáveis e não exige padronização das features para funcionar — embora possamos manter o mesmo pipeline de pré-processamento para consistência com os outros modelos do projeto.


## Exploração dos Dados

#### O Dataset

Para esse projeto foi utilizada o Dataset [Fitness Classification Dataset](https://www.kaggle.com/datasets/muhammedderric/fitness-classification-dataset-synthetic){:target='blank'}. Essa Base de dados posuí 2.000 linhas e 11 colunas. A variável dependente que será usada como objeto de classificação é a is_fit, ela indica se a pessoa é fit (1) ou não fit (0).


#### Análise dos dados

=== "Age"

    Tipo: numérica contínua

    O que é: idade em anos.

    Para que serve: pode relacionar-se com hábitos e condição física.

    Ação necessária: nenhuma obrigatória; só checar faixas implausíveis (não observei no geral).

    ```python exec="on" html="1"
    --8<-- "docs/Exercicios/graficos/age.py"
    ```

=== "height_cm"

    Tipo: numérica contínua

    O que é: altura em centímetros.

    Para que serve: isoladamente costuma ter pouco poder; combinada ao peso forma o BMI.

    Ação necessária: checar valores muito fora do plausível. Sugestão: considerar substituir altura e peso por bmi(Índice de Massa Corporal).

    ```python exec="on" html="1"
    --8<-- "docs/Exercicios/graficos/height_cm.py"
    ```

=== "weight_kg"

    Tipo: numérica contínua

    O que é: peso em quilogramas.

    Para que serve: junto com a altura permite calcular BMI = peso(kg) / (altura(m))², que costuma ser mais informativo para a árvore.

    Ação necessária: manter como numérica ou criar bmi e remover height_cm/weight_kg das features (deixando só o bmi).

    ```python exec="on" html="1"
    --8<-- "docs/Exercicios/graficos/weight_kg.py"
    ```

=== "heart_rate"

    Tipo: numérica contínua

    O que é: frequência cardíaca (bpm).

    Para que serve: indicador de condicionamento cardiovascular; pode ajudar na separação das classes.

    Ação necessária: nenhuma obrigatória; apenas conferir plausibilidade de valores extremos.

    ```python exec="on" html="1"
    --8<-- "docs/Exercicios/graficos/heart_rate.py"
    ```

=== "blood_pressure"

    Tipo: numérica contínua

    O que é: medida sintética de pressão arterial fornecida pelo dataset.

    Para que serve: sinal de saúde geral que pode complementar a predição.

    Ação necessária: nenhuma obrigatória; só verificar extremos muito fora do usual.

    ```python exec="on" html="1"
    --8<-- "docs/Exercicios/graficos/blood_pressure.py"
    ```

=== "Sleep_Hours"

    Tipo: numérica contínua

    O que é: horas de sono por dia.

    Para que serve: hábito de descanso; costuma ter correlação com “estar fit”.

    Ação necessária: possui valores ausentes (160 valores); imputar com a mediana.

    ```python exec="on" html="1"
    --8<-- "docs/Exercicios/graficos/sleep_hours.py"
    ```

=== "Nutrition_quality"

    Tipo: numérica contínua (escala)

    O que é: qualidade da nutrição (escala contínua, ex.: 0–10).

    Para que serve: proxy de alimentação saudável; geralmente relevante.

    Ação necessária: nenhuma; manter como numérica (só garantir faixa válida).

    ```python exec="on" html="1"
    --8<-- "docs/Exercicios/graficos/nutrition_quality.py"
    ```

=== "Activity_index"

    Tipo: numérica contínua (escala)

    O que é: nível de atividade física (escala contínua, ex.: 0–10).

    Para que serve: costuma ser uma das variáveis mais importantes para is_fit.

    Ação necessária: nenhuma; manter como numérica (garantir faixa válida).

    ```python exec="on" html="1"
    --8<-- "docs/Exercicios/graficos/activity_index.py"
    ```

=== "smokes"

    Tipo: categórica binária

    O que é: status de tabagismo (sim/não).

    Para que serve: fator de estilo de vida; pode ajudar a separar perfis.

    Ação necessária: tipos mistos no bruto (“yes/no” e “1/0”). Padronizar para binário numérico (no→0, yes→1) e converter para int.
    
    ```python exec="on" html="1"
    --8<-- "docs/Exercicios/graficos/smokes.py"
    ```

=== "gender"

    Tipo: categórica binária

    O que é: gênero (F/M).

    Para que serve: possível moderador de outros efeitos; em geral fraco sozinho.

    Ação necessária: codificar para numérico (F→0, M→1) e converter para int.

    ```python exec="on" html="1"
    --8<-- "docs/Exercicios/graficos/gender.py"
    ```

=== "is_fit"

    Tipo: categórica binária (target)

    O que é: rótulo de condição física (1 = fit, 0 = não fit).

    Para que serve: variável dependente a ser prevista.

    Ação necessária: checar balanceamento das classes.

    ```python exec="on" html="1"
    --8<-- "docs/Exercicios/graficos/is_fit.py"
    ```

## Pré-processamento

Nesta etapa tratei e preparei os dados para treinar a Árvore de Decisão. Antes do tratamento, a base apresentava valores ausentes em sleep_hours, tipos mistos em smokes (valores como yes/no e 0/1 ao mesmo tempo) e variáveis categóricas em texto (gender com F/M). Abaixo, o que foi feito:

• Padronização de categóricas

- smokes: normalizei rótulos e converti para binário numérico (no→0, yes→1, cobrindo também 0/1 em string).

- gender: converti F→0 e M→1.

• Valores ausentes

- sleep_hours: converti para numérico e imputei a mediana.

• Tipos e consistência

- Garanti que as variáveis contínuas ficaram em formato numérico, sem strings residuais/espaços.

• Criação de nova variável

- Criei a variável BMI (peso(kg) / altura(m)²) para avaliar seu impacto. Na exploração, mantenho height_cm e weight_kg para referência; na modelagem, comparo dois cenários: (A) sem BMI (altura + peso) e (B) com apenas BMI, evitando usar os três juntos no mesmo modelo para não introduzir redundância.


=== "Base original"
    
    ```python exec="on"
    --8<-- "docs/Exercicios/base_original.py"
    ```

=== "Tratamento"

    ```python
    --8<-- "docs/Exercicios/base_tratada.py"
    ```

=== "Base Tratada"

    ```python exec="on"
    --8<-- "docs/Exercicios/base_tratada.py"
    ```

## Divisão dos Dados

Os dados foram divididos em treino (80%) e teste (20%) com o parâmetro random_state=42 para garantir reprodutibilidade e stratify=y para manter a proporção entre as classes is_fit.

Como a Random Forest é composta por árvores de decisão, não há necessidade de padronização das variáveis numéricas, porém a mesma estrutura de features foi mantida para comparação com os demais modelos.

=== "Código de divisão"

    ``` python exec="on" html="1"
    --8<-- "docs/Exercicios/random_forest/divisaodados_rf.py"
    ```

## Treinamento do Modelo

O modelo foi configurado com 300 árvores (n_estimators=300), seleção aleatória de atributos (max_features="sqrt") e profundidade ilimitada (max_depth=None), permitindo que cada árvore se adapte ao padrão dos dados de forma independente.
O treinamento foi realizado sobre o conjunto de treino e avaliado com base na acurácia, balanced accuracy e matriz de confusão.

=== "Matriz de Confusão"

    ``` python exec="on" html="1"
    --8<-- "docs/Exercicios/random_forest/treino_rf.py"
    ```
    
=== "Importância das Variáveis"

    ``` python exec="on" html="1"
    --8<-- "docs/Exercicios/random_forest/importancias_rf.py"
    ```

## Avaliação do Modelo

A Random Forest apresentou desempenho consistente e maior estabilidade em comparação com modelos individuais de árvore ou com o KNN.
A matriz de confusão mostrou uma redução significativa nos erros de classificação, e o valor de Balanced Accuracy indicou boa capacidade de generalização entre as classes.

A análise da importância das variáveis revelou que os fatores mais determinantes para o modelo são:

- activity_index
- nutrition_quality
- bmi
- heart_rate

Essas variáveis refletem, respectivamente, o nível de atividade física, a qualidade da alimentação e os indicadores fisiológicos de saúde.

## Conclusão

O uso da Random Forest no dataset de fitness demonstrou excelente equilíbrio entre desempenho e interpretabilidade.
O modelo foi capaz de identificar padrões consistentes que diferenciam indivíduos “fit” e “não fit”, destacando-se pela robustez e pela redução de sobreajuste em relação à árvore de decisão simples.

A análise confirma que hábitos e parâmetros fisiológicos — especialmente atividade física, nutrição e índice de massa corporal — são fatores-chave para a previsão da condição física.
Como próximos passos, recomenda-se explorar a otimização de hiperparâmetros (max_depth, min_samples_leaf, n_estimators) e realizar validação cruzada para refinar ainda mais a performance do modelo.
