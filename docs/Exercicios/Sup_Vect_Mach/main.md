### Introdução ao SVM

O algoritmo Support Vector Machine (SVM) foi utilizado para resolver a tarefa de classificação da condição física (`is_fit`) a partir dos indicadores de saúde e estilo de vida do dataset. Diferente da árvore de decisão e do KNN, o SVM busca encontrar um hiperplano que separe as classes com a maior margem possível, podendo usar o *kernel trick* para projetar os dados em um espaço de maior dimensão e lidar melhor com fronteiras de decisão não lineares. 

Neste exercício, aplico um SVM com kernel RBF para prever se um indivíduo é “fit” (1) ou “não fit” (0), comparando seu desempenho com os modelos já treinados anteriormente.

### Descrição sobre o banco

Para mais informações, cheque a página sobre [Árvore de decisão](https://jose-longo-a.github.io/Machine-Learning/arvore-de-decisao/main/), aqui tem toda a explicação necessária para compreender as variáveis e as outras coisas.

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

### Pré-processamento

Nesta etapa tratei e preparei os dados para os modelos utilizados nos exercícios anteriores (Árvore de Decisão, KNN e K-Means), e reaproveito exatamente o mesmo tratamento para calcular as métricas de avaliação aqui. Antes do tratamento, a base apresentava valores ausentes em sleep_hours, tipos mistos em smokes (valores como yes/no e 0/1 ao mesmo tempo) e variáveis categóricas em texto (gender com F/M). Abaixo, o que foi feito:

• Padronização de categóricas  

- smokes: normalizei rótulos e converti para binário numérico (no→0, yes→1, cobrindo também 0/1 em string).  
- gender: converti F→0 e M→1.  

• Valores ausentes  

- sleep_hours: converti para numérico e imputei a mediana.  

• Tipos e consistência  

- Garanti que as variáveis contínuas ficaram em formato numérico, sem strings residuais/espaços.  

• Criação de nova variável  

- Criei a variável BMI (peso(kg) / altura(m)²) para avaliar seu impacto. Na exploração, mantenho height_cm e weight_kg para referência; na modelagem, comparo dois cenários: (A) sem BMI (altura + peso) e (B) com apenas BMI, evitando usar os três juntos no mesmo modelo para não introduzir redundância.

Esse mesmo pipeline de pré-processamento é reutilizado tanto no KNN quanto no K-Means, garantindo que as métricas de avaliação sejam comparáveis entre os modelos.

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

### Divisão dos Dados

Para o SVM, mantive exatamente o mesmo esquema de divisão utilizado nos outros modelos supervisionados: 70% dos dados para treino e 30% para teste, com `random_state=42` para reprodutibilidade e `stratify=y` para preservar a proporção entre as classes de `is_fit` em ambas as partições. 

Além disso, como o SVM é sensível à escala das variáveis, as features numéricas foram padronizadas com `StandardScaler` dentro de um `Pipeline`, garantindo que o modelo seja treinado em atributos na mesma ordem de grandeza.

### Treinamento e Métricas – SVM

Nesta seção, treino um modelo de SVM com kernel RBF para prever a variável `is_fit`. O modelo é encapsulado em um `Pipeline` que primeiro padroniza os dados com `StandardScaler` e, em seguida, aplica o `SVC`. Em seguida, avalio o desempenho usando acurácia simples, acurácia balanceada, matriz de confusão e o relatório de classificação (precisão, recall e F1-score por classe).

```python exec="on" html="1"
--8<-- "docs/Exercicios/Sup_Vect_Mach/treino_svm.py"
```

### Visualização da Fronteira de Decisão

Para aproximar a visualização do que foi feito na aula de SVM, reduzi o conjunto de features para duas dimensões por meio de PCA após a padronização. Em seguida, treinei um SVM RBF nesse espaço 2D e plotei a fronteira de decisão, junto com os pontos de treino.

Essa projeção não preserva exatamente todas as relações do espaço original, mas ajuda a visualizar como o SVM separa, aproximadamente, os indivíduos “fit” e “não fit” em um plano.

```python exec="on" html="1"
--8<-- "docs/Exercicios/Sup_Vect_Mach/svm_decision_boundary.py"
```