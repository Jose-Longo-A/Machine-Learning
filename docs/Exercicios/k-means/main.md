

### Introdução ao K-Means

O algoritmo K-Means foi aplicado com o objetivo de identificar padrões e agrupar os indivíduos em clusters com características semelhantes relacionadas à saúde e ao condicionamento físico. Diferente dos métodos supervisionados, como a árvore de decisão e o KNN, o K-Means é um algoritmo de aprendizado não supervisionado, que organiza os dados em grupos de acordo com a proximidade entre seus atributos, sem utilizar diretamente a variável-alvo is_fit como guia. Dessa forma, é possível verificar se os agrupamentos formados refletem, ao menos parcialmente, a divisão entre pessoas classificadas como “fit” e “não fit”, oferecendo uma perspectiva complementar sobre como os hábitos e indicadores de saúde se distribuem na base.

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
Diferentemente dos modelos supervisionados, o K-Means é um algoritmo não supervisionado e, portanto, não requer a separação em treino e teste. Após o pré-processamento, utilizei todo o conjunto de dados para formar os clusters com base apenas nas variáveis explicativas (idade, frequência cardíaca, pressão arterial, horas de sono, qualidade da nutrição, nível de atividade física, tabagismo, gênero e o BMI). A variável-alvo is_fit foi mantida de lado e utilizada apenas posteriormente para avaliar a correspondência entre os clusters encontrados e as classes reais (“fit” vs. “não fit”).

Como o K-Means é sensível à escala, apliquei padronização (z-score) às variáveis numéricas, garantindo que nenhum atributo dominasse a distância euclidiana. Além disso, para evitar redundância, substituí height_cm e weight_kg pela variável derivada bmi. Dessa forma, os agrupamentos refletem padrões de similaridade nos hábitos e indicadores de saúde, e a validação com is_fit serve apenas como referência externa de qualidade do clustering.

=== "Código"

    Features: age, height_cm, weight_kg, heart_rate, blood_pressure, sleep_hours, nutrition_quality, activity_index, smokes, gender.
    
    Objetivo: servir de referência para comparar com a versão engenheirada. 
    
    Mesma configuração de split (70/30, random_state=42, stratify=y).

    ``` python exec="0"
    --8<-- "docs/Exercicios/k-means/divisaodados.py"
    ```

### Treinamento do Modelo

=== "Modelo"

    ```python exec="on" html="1"
    --8<-- "docs/Exercicios/k-means/treino.py"
    ```

=== "Código"

    ```python
    --8<-- "docs/Exercicios/k-means/treino.py"
    ```

### Avaliação do Modelo

O algoritmo K-Means foi aplicado com k = 2, refletindo a natureza binária da variável is_fit (fit e não fit). A análise gráfica mostra que os clusters foram formados, mas com forte sobreposição entre os grupos. Isso significa que, embora o algoritmo tenha conseguido identificar padrões de proximidade nos dados, a separação entre os perfis de indivíduos com boa condição física e os demais não é nítida.

O posicionamento dos centróides indica regiões médias de concentração, mas a distribuição densa e misturada dos pontos em torno deles revela que as variáveis utilizadas — como BMI, atividade física, nutrição e indicadores de saúde — possuem relevância, mas não criam divisões claras o suficiente para que o K-Means consiga formar agrupamentos bem distintos. Essa característica é comum em bases de dados relacionadas a hábitos e saúde, em que múltiplos fatores se combinam de maneira complexa e não linear.

### Conclusão

O uso do K-Means permitiu uma exploração inicial da base, evidenciando como os indivíduos se distribuem em grupos de acordo com seus atributos de saúde e estilo de vida. Apesar da simplicidade e eficiência do algoritmo, os resultados mostram que os clusters obtidos não correspondem perfeitamente à divisão real entre “fit” e “não fit”.

De forma geral, a análise reforça que, embora o K-Means seja útil para identificar padrões gerais e tendências de proximidade, sua capacidade de representar a variável is_fit de forma fiel é limitada. Para análises mais robustas, seria necessário considerar técnicas adicionais, como algoritmos supervisionados (Árvore de Decisão, KNN ou Random Forest), ou mesmo explorar variações de clustering que lidem melhor com sobreposição de classes. Ainda assim, o exercício com K-Means foi importante para oferecer uma perspectiva não supervisionada da estrutura dos dados e validar a dificuldade inerente da tarefa de classificação no contexto do dataset de fitness.