

### Introdução ao KNN

O algoritmo K-Nearest Neighbors (KNN) foi utilizado como alternativa para realizar tarefas de classificação. Esse método classifica uma nova observação com base nos exemplos mais próximos do conjunto de treino, considerando a similaridade entre seus atributos. Por sua simplicidade e flexibilidade, o KNN não exige pressupostos sobre a distribuição dos dados e oferece uma análise fundamentada na proximidade entre instâncias, funcionando como complemento às previsões obtidas com a árvore de decisão.

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


### Divisão dos Dados
Para o modelo KNN, optei por separar o conjunto em treino (80%) e teste (20%), de forma a avaliar o desempenho do classificador em dados não vistos. O parâmetro random_state=42 foi utilizado para garantir reprodutibilidade, e stratify=y assegurou que a proporção entre as classes (is_fit = 0 e is_fit = 1) fosse preservada em ambas as partições.

Antes da divisão, o pré-processamento já havia sido realizado: imputação da mediana em sleep_hours, padronização de smokes (0/1), codificação de gender (0/1) e criação da variável bmi em substituição às variáveis originais height_cm e weight_kg. Como o KNN é sensível a diferenças de escala, também foi aplicada a padronização dos atributos numéricos após a definição das features finais.

Essa estratégia garante que o modelo seja treinado em um subconjunto representativo e avaliado de maneira justa, evitando que a acurácia reflita apenas o aprendizado sobre o conjunto total de dados.

=== "Código"

    Features: age, height_cm, weight_kg, heart_rate, blood_pressure, sleep_hours, nutrition_quality, activity_index, smokes, gender.
    
    Objetivo: servir de referência para comparar com a versão engenheirada. 
    
    Mesma configuração de split (70/30, random_state=42, stratify=y).

    ``` python exec="0"
    --8<-- "docs/Exercicios/knn/divisaodados.py"
    ```

### Treinamento do Modelo

=== "Modelo"

    ```python exec="on"
    --8<-- "docs/Exercicios/knn/treino1.py"
    ```

=== "Código"

    ```python
    --8<-- "docs/Exercicios/knn/treino1.py"
    ```

### Usando o Scikit-Learn

=== "Resultado"

    ```python exec="on" html="1"
    --8<-- "docs/Exercicios/knn/resultado1.py"
    ```

=== "Código"

    ```python
    --8<-- "docs/Exercicios/knn/resultado1.py"
    ```

### Avaliação do Modelo

O modelo KNN, configurado com k = 5, apresentou uma acurácia de aproximadamente 62% no conjunto de teste, com balanced accuracy em torno de 59%. Isso significa que o desempenho ficou apenas um pouco acima do acaso (50%), refletindo a dificuldade do algoritmo em distinguir corretamente entre indivíduos classificados como “fit” e “não fit”.

A visualização da fronteira de decisão confirma esse resultado: as classes aparecem bastante sobrepostas no espaço de duas dimensões (após PCA), e a divisão entre elas não é nítida. Esse comportamento indica que as variáveis utilizadas (idade, indicadores de saúde, hábitos de sono, nutrição, atividade física, tabagismo, gênero e BMI) possuem alguma relevância, mas não oferecem separação clara o suficiente para que o KNN construa regiões bem definidas para cada classe. Além disso, a irregularidade da fronteira de decisão ressalta a sensibilidade do método a ruídos e à distribuição dos dados, algo esperado nesse tipo de problema.

### Conclusão

O uso do KNN para prever a condição física (fit vs. não fit) demonstrou resultados modestos, mas ainda válidos como exercício de classificação e comparação com outros algoritmos, como a árvore de decisão. Apesar da simplicidade e da interpretação intuitiva do método, a análise gráfica mostrou que as classes são altamente sobrepostas, o que limita a capacidade do modelo de generalizar.

De maneira geral, os resultados indicam que o KNN pode servir como uma abordagem inicial para explorar o dataset, mas melhorias dependem de ajustes nos hiperparâmetros (como o valor de k), da criação de novas features ou da aplicação de modelos mais robustos, capazes de lidar melhor com a complexidade e o ruído dos dados.