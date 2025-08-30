
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
    --8<-- "docs/arvore-de-decisao/graficos/age.py"
    ```

=== "height_cm"

    Tipo: numérica contínua

    O que é: altura em centímetros.

    Para que serve: isoladamente costuma ter pouco poder; combinada ao peso forma o BMI.

    Ação necessária: checar valores muito fora do plausível. Sugestão: considerar substituir altura e peso por bmi(Índice de Massa Corporal).

    ```python exec="on" html="1"
    --8<-- "docs/arvore-de-decisao/graficos/height_cm.py"
    ```

=== "weight_kg"

    Tipo: numérica contínua

    O que é: peso em quilogramas.

    Para que serve: junto com a altura permite calcular BMI = peso(kg) / (altura(m))², que costuma ser mais informativo para a árvore.

    Ação necessária: manter como numérica ou criar bmi e remover height_cm/weight_kg das features (deixando só o bmi).

    ```python exec="on" html="1"
    --8<-- "docs/arvore-de-decisao/graficos/weight_kg.py"
    ```

=== "heart_rate"

    Tipo: numérica contínua

    O que é: frequência cardíaca (bpm).

    Para que serve: indicador de condicionamento cardiovascular; pode ajudar na separação das classes.

    Ação necessária: nenhuma obrigatória; apenas conferir plausibilidade de valores extremos.

    ```python exec="on" html="1"
    --8<-- "docs/arvore-de-decisao/graficos/heart_rate.py"
    ```

=== "blood_pressure"

    Tipo: numérica contínua

    O que é: medida sintética de pressão arterial fornecida pelo dataset.

    Para que serve: sinal de saúde geral que pode complementar a predição.

    Ação necessária: nenhuma obrigatória; só verificar extremos muito fora do usual.

    ```python exec="on" html="1"
    --8<-- "docs/arvore-de-decisao/graficos/blood_pressure.py"
    ```

=== "Sleep_Hours"

    Tipo: numérica contínua

    O que é: horas de sono por dia.

    Para que serve: hábito de descanso; costuma ter correlação com “estar fit”.

    Ação necessária: possui valores ausentes (160 valores); imputar com a mediana.

    ```python exec="on" html="1"
    --8<-- "docs/arvore-de-decisao/graficos/sleep_hours.py"
    ```

=== "Nutrition_quality"

    Tipo: numérica contínua (escala)

    O que é: qualidade da nutrição (escala contínua, ex.: 0–10).

    Para que serve: proxy de alimentação saudável; geralmente relevante.

    Ação necessária: nenhuma; manter como numérica (só garantir faixa válida).

    ```python exec="on" html="1"
    --8<-- "docs/arvore-de-decisao/graficos/nutrition_quality.py"
    ```

=== "Activity_index"

    Tipo: numérica contínua (escala)

    O que é: nível de atividade física (escala contínua, ex.: 0–10).

    Para que serve: costuma ser uma das variáveis mais importantes para is_fit.

    Ação necessária: nenhuma; manter como numérica (garantir faixa válida).

    ```python exec="on" html="1"
    --8<-- "docs/arvore-de-decisao/graficos/activity_index.py"
    ```

=== "smokes"

    Tipo: categórica binária

    O que é: status de tabagismo (sim/não).

    Para que serve: fator de estilo de vida; pode ajudar a separar perfis.

    Ação necessária: tipos mistos no bruto (“yes/no” e “1/0”). Padronizar para binário numérico (no→0, yes→1) e converter para int.
    
    ```python exec="on" html="1"
    --8<-- "docs/arvore-de-decisao/graficos/smokes.py"
    ```

=== "gender"

    Tipo: categórica binária

    O que é: gênero (F/M).

    Para que serve: possível moderador de outros efeitos; em geral fraco sozinho.

    Ação necessária: codificar para numérico (F→0, M→1) e converter para int.

    ```python exec="on" html="1"
    --8<-- "docs/arvore-de-decisao/graficos/gender.py"
    ```

=== "is_fit"

    Tipo: categórica binária (target)

    O que é: rótulo de condição física (1 = fit, 0 = não fit).

    Para que serve: variável dependente a ser prevista.

    Ação necessária: checar balanceamento das classes.

    ```python exec="on" html="1"
    --8<-- "docs/arvore-de-decisao/graficos/is_fit.py"
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
    --8<-- "docs/arvore-de-decisao/base_original.py"
    ```

=== "Tratamento"

    ```python
    --8<-- "docs/arvore-de-decisao/base_tratada.py"
    ```

=== "Base Tratada"

    ```python exec="on"
    --8<-- "docs/arvore-de-decisao/base_tratada.py"
    ```


## Divisão dos Dados
Nesta etapa separei o conjunto em treino (70%) e teste (30%) para avaliar a árvore em dados não vistos. Usei random_state=42 para reprodutibilidade e stratify=y para manter a proporção de is_fit em treino e teste. Antes do split, finalizei o pré-processamento (imputação da mediana em sleep_hours, padronização de smokes para 0/1 e de gender para 0/1). 

Nota: em um pipeline mais rígido, a imputação/codificação seria ajustada no treino e aplicada no teste para evitar vazamento; aqui mantive a simplicidade do material.

=== "Cenário A"

    Features: age, height_cm, weight_kg, heart_rate, blood_pressure, sleep_hours, nutrition_quality, activity_index, smokes, gender.
    
    Objetivo: servir de referência para comparar com a versão engenheirada. 
    
    Mesma configuração de split (70/30, random_state=42, stratify=y).

    ``` python exec="0"
    --8<-- "docs/arvore-de-decisao/divisaodadosA.py"
    ```

=== "Cenário B"

    Engenharia: criei bmi = peso(kg)/altura(m)² (cálculo linha a linha, sem vazamento).
    Features: age, bmi, heart_rate, blood_pressure, sleep_hours, nutrition_quality, activity_index, smokes, gender.
    
    Objetivo: avaliar o impacto do BMI na acurácia e na simplicidade da árvore. Mantive height_cm e weight_kg na base tratada para transparência, mas retirei essas colunas apenas na seleção das features do Cenário B para evitar redundância com o bmi.
    
    Comparação justa: ambos os cenários usam o mesmo split (70/30, random_state=42, stratify=y).

    ``` python exec="0"
    --8<-- "docs/arvore-de-decisao/divisaodadosB.py"
    ```


## Primeiro Treinamento do Modelo

=== "Árvore"

    ``` python exec="on" html="1"
    --8<-- "docs/arvore-de-decisao/treino1.py"
    ```

=== "code"

    ``` python exec="0"
    --8<-- "docs/arvore-de-decisao/treino1.py"
    ```

### Avaliação do primeiro modelo

O modelo com todas as variáveis atingiu 68,83% de precisão. As que mais pesaram foram activity_index (23,2%) e nutrition_quality (17,5%), seguidas por bmi (12,4%), age (11,6%) e smokes (10,9%). Já sleep_hours (8,7%), heart_rate (8,2%), blood_pressure (6,5%) e, principalmente, gender (1,0%) tiveram impacto baixo.
Em resumo: o sinal principal vem de atividade física e qualidade da alimentação, com um complemento do BMI e idade.

## Segundo Treinamento do Modelo

=== "Árvore"

    ``` python exec="on" html="1"
    --8<-- "docs/arvore-de-decisao/treino2.py"
    ```


=== "code"

    ``` python exec="0"
    --8<-- "docs/arvore-de-decisao/treino2.py"
    ```

### Avaliação do segundo modelo
    
Ao remover as variáveis mais fracas (gender, blood_pressure e heart_rate) e manter apenas as mais relevantes, a precisão subiu levemente para 69,00% (+0,17 p.p.). A importância ficou ainda mais concentrada em activity_index (26,0%), nutrition_quality (21,2%) e bmi (19,6%), com age (13,5%), smokes (10,9%) e sleep_hours (8,8%) completando o conjunto.

Conclusão desta comparação: tirar as variáveis com pouco sinal reduz ruído e deixa a árvore mais simples, mantendo (ou melhorando) a precisão. Para a avaliação final, faz sentido seguir com o modelo compacto com BMI.


## Relatório Final

Árvores de decisão não pedem normalização, então o que realmente ajudou aqui foi tratar a base (padronizar smokes/gender, imputar sleep_hours) e criar o bmi. Comparei dois modelos e fiquei com o compacto com BMI: é mais simples e bateu ~69,00% de precisão (contra 68,83% do completo). As variáveis que mais fizeram diferença foram activity_index, nutrition_quality e bmi; idade, sono e tabagismo somaram um pouco. Mantive split 70/30 estratificado — reduzir o teste não melhora o modelo e ainda piora a avaliação. Resumo: dados limpos + bons atributos valem mais do que mexer em escala quando o modelo é uma árvore.

