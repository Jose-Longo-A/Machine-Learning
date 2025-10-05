## Grupo 3 / Os Goats do SI

1. José Longo Neto
2. Pedro Almeida Maricate
3. Martim Ponzio
4. Pablo Dimitrof
5. Enzo Malagoli
6. Eduardo Gul

## Introdução
O objetivo deste projeto é aplicar três algoritmos de *Machine Learning* — **Árvore de Decisão**, **K-Nearest Neighbors (KNN)** e **K-Means (Clustering)** — sobre a base de dados do [kagle](https://www.kaggle.com/datasets/zynicide/wine-reviews).  
A base contém aproximadamente **130 mil registros** de vinhos, com informações como país de origem, pontuação atribuída por especialistas, preço, tipo de uva, região produtora e descrição sensorial.

A proposta é explorar o uso de algoritmos de aprendizado supervisionado e não supervisionado para compreender padrões entre as variáveis e identificar possíveis relações entre características como **origem, variedade e pontuação** dos vinhos.

O roteiro segue a mesma estrutura aplicada em outros projetos da disciplina, dividindo o processo em etapas sequenciais e documentadas:

- **Exploração dos Dados (EDA)**: análise geral das variáveis, distribuição de valores e identificação de possíveis inconsistências;  
- **Pré-processamento**: limpeza, padronização e codificação de dados categóricos;  
- **Divisão dos Dados**: separação em conjuntos de treino e teste, quando houver variável-alvo definida;  
- **Treinamento e Avaliação**: aplicação e comparação dos modelos supervisionados (Árvore de Decisão e KNN);  
- **Modelagem Não Supervisionada**: uso do **K-Means** para identificar agrupamentos entre os vinhos sem utilizar o rótulo de pontuação;  
- **Relatório Final e Comparação**: discussão sobre os resultados obtidos e limitações de cada abordagem.

> **Observação:**  
> O dataset não possui uma coluna explicitamente binária de qualidade, mas contém a variável **`points`**, que representa a nota do vinho atribuída por avaliadores.  
> Para fins de classificação, esta variável será transformada em uma **variável-alvo derivada**, categorizando os vinhos conforme sua pontuação (por exemplo, *alta pontuação* ≥ 90).  
> Assim, os modelos supervisionados trabalharão com essa classificação, enquanto o K-Means será utilizado para detectar padrões de agrupamento sem rótulos.


## Exploração dos Dados
A etapa de **Exploração dos Dados (EDA)** tem como objetivo compreender a estrutura, o conteúdo e o significado das variáveis presentes na base **`wine.csv`**.  
Essa análise inicial permite identificar padrões, outliers, distribuições e possíveis problemas de qualidade dos dados, como valores ausentes ou inconsistências.  
As visualizações e descrições abaixo ajudam a construir uma visão geral do conjunto e a orientar as etapas seguintes de pré-processamento e modelagem.

=== "country"

    A coluna country indica o país de origem do vinho.
    Essa variável é importante para observar a distribuição geográfica dos registros e entender a representatividade de cada país no conjunto.

    ```python exec="on" html="1"
    --8<-- "docs/projetos/projeto1/graficos/country.py"
    ```

=== "designation"

    A coluna designation informa a denominação ou nome específico do vinho, dentro da vinícola.
    É uma variável categórica de alta cardinalidade (muitos valores únicos) e pode indicar edições especiais ou lotes de produção.

    ```python exec="on" html="1"
    --8<-- "docs/projetos/projeto1/graficos/designation.py"
    ```

=== "points"

    A coluna points representa a pontuação do vinho atribuída por avaliadores especializados, geralmente variando entre 80 e 100 pontos.
    Essa variável é central no projeto, pois será usada para derivar a variável-alvo de qualidade que alimentará os modelos supervisionados.

    ```python exec="on" html="1"
    --8<-- "docs/projetos/projeto1/graficos/points.py"
    ```

=== "price"

    A coluna price indica o preço do vinho em dólares.
    Ela é uma variável numérica contínua que pode apresentar assimetria devido à presença de vinhos muito caros (outliers).
    A relação entre preço e pontuação será uma das análises mais relevantes desta etapa.

    ```python exec="on" html="1"
    --8<-- "docs/projetos/projeto1/graficos/price.py"
    ```

=== "province"

    A coluna province indica a província ou região produtora do vinho dentro de seu país.
    É uma variável categórica útil para observar a diversidade geográfica da produção.

    ```python exec="on" html="1"
    --8<-- "docs/projetos/projeto1/graficos/province.py"
    ```

=== "region_1"

    A coluna region_1 representa uma sub-região produtora (como “Napa Valley” ou “Bordeaux”).
    Pode ser usada para análises mais detalhadas de terroir e diferenciação regional.

    ```python exec="on" html="1"
    --8<-- "docs/projetos/projeto1/graficos/region1.py"
    ```

=== "region_2"

    A coluna region_2 fornece informações complementares sobre uma segunda subdivisão geográfica, quando disponível.
    Nem todos os registros possuem este campo preenchido, portanto ele pode apresentar alta taxa de valores ausentes.

    ```python exec="on" html="1"
    --8<-- "docs/projetos/projeto1/graficos/region2.py"
    ```

=== "taster_name"

    A coluna taster_name contém o nome do avaliador responsável pela nota e descrição do vinho.
    Ela permite explorar a distribuição de avaliações entre diferentes especialistas e identificar potenciais vieses.

    ```python exec="on" html="1"
    --8<-- "docs/projetos/projeto1/graficos/taster_name.py"
    ```

=== "variety"

    A coluna variety indica o tipo de uva utilizada na produção (por exemplo: Pinot Noir, Chardonnay, Riesling).
    É uma das variáveis mais importantes do conjunto, pois reflete o perfil sensorial e o tipo do vinho.

    ```python exec="on" html="1"
    --8<-- "docs/projetos/projeto1/graficos/variety.py"
    ```

=== "winery"

    A coluna winery identifica a vinícola responsável pela produção.
    É uma variável categórica de alta cardinalidade e pode ser explorada futuramente em análises de desempenho médio por produtor.

    ```python exec="on" html="1"
    --8<-- "docs/projetos/projeto1/graficos/winery.py"
    ```


## Pré-processamento

Após a exploração inicial, aplicamos um conjunto de procedimentos para preparar a base **`wine.csv`** para modelagem:

- **Remoção de colunas irrelevantes**: descartamos campos sem utilidade direta para o modelo ou que serão usados apenas como metadados na documentação (ex.: `Unnamed: 0`, `title`, `description`, `taster_twitter_handle`).  
- **Conversão e limpeza de tipos**: garantimos que `price` e `points` estejam em formato numérico; tratamos valores ausentes com imputações simples (mediana para numéricas; rótulo `"Unknown"` para categóricas).  
- **Criação do alvo (quando aplicável)**: a partir de `points`, derivamos a variável **`quality_high`** (1 se `points ≥ 90`; 0 caso contrário). Para evitar **vazamento de informação**, `points` **não** será usada como *feature* nos modelos supervisionados.  
- **Codificação de categóricas**: para esta versão “base preparada” usada na documentação, aplicamos **Label Encoding** coluna a coluna (adequado para árvore; nas seções de treino do KNN/K-Means faremos *scaling* e codificações apropriadas nos pipelines).  
- **Entrega de uma visão pronta para modelagem**: apresentamos uma amostra da base já limpa e codificada, com `quality_high` disponível para as etapas supervisionadas e os demais campos prontos para uso em pipelines.

=== "Base preparada"

    ```python exec="on"
    --8<-- "docs/projetos/projeto1/base/pre.py"
    ```

=== "code"

    ```python exec="0"
    --8<-- "docs/projetos/projeto1/base/pre.py"
    ```

=== "Base Original"

    ```python exec="on"
    --8<-- "docs/projetos/projeto1/base/base0.py"
    ```


## Divisão dos Dados 

Com a base pré-processada, realizou-se a divisão entre conjuntos de **treinamento** e **teste**.  
O objetivo dessa etapa é garantir que o modelo seja avaliado em dados que ele nunca viu durante o treinamento, permitindo uma medida mais confiável de sua capacidade de generalização.  

Foi utilizada a função `train_test_split` da biblioteca *scikit-learn*, com os seguintes critérios:  
- **70% dos dados** destinados ao treinamento, para que o modelo aprenda os padrões da base;  
- **30% dos dados** destinados ao teste, para avaliar o desempenho em novos exemplos;  
- **Estratificação pelo alvo (`quality_high`)**, garantindo que a proporção entre classes seja mantida em ambos os conjuntos;  
- **Random State** fixado, assegurando reprodutibilidade na divisão.  

```python exec="0"
--8<-- "docs/projetos/projeto1/divisao/div.py"
```


## Treinamento do Modelo — Árvore de Decisão

Nesta etapa treinamos um **DecisionTreeClassifier** utilizando a base pré-processada.  
Mantemos a mesma preparação usada na divisão (remoção de colunas irrelevantes, criação de `quality_high` a partir de `points`, codificação das categóricas) e realizamos o ajuste do modelo com **70/30** de treino/teste e `random_state=27`.

=== "Modelo da Árvore"

    ```python exec="on" html="1"
    --8<-- "docs/projetos/projeto1/modelos/arvore.py"
    ```

=== "code"

    ```python exec="0"
    --8<-- "docs/projetos/projeto1/modelos/arvore.py"
    ```


## Avaliação do Modelo — Árvore de Decisão

Após o treinamento, a árvore foi avaliada no **conjunto de teste (30%)**, garantindo que as métricas reflitam a capacidade de **generalização** do modelo. As saídas principais consideradas são:

- **Acurácia (teste)**: proporção de acertos sobre o total de amostras de teste;  
- **Matriz de Confusão**: distribuição de acertos/erros por classe (0 = qualidade comum, 1 = alta qualidade), útil para inspecionar **erros assimétricos**;  
- **Classification Report**: métricas por classe (**precision**, **recall** e **F1**), evidenciando se o modelo está sacrificando uma classe em detrimento da outra;  
- **Importância das Features**: ranking de variáveis mais influentes na decisão (ex.: `variety`, `province/country`, `price`), auxiliando na interpretação.

### Observações e interpretação
- **Desbalanceamento**: é comum a classe “alta qualidade” (≥ 90 pontos) ser **minoritária**, o que pode reduzir o **recall** dessa classe mesmo com acurácia global razoável.  
- **Overfitting**: árvores muito profundas tendem a memorizar o treino. Se a diferença entre desempenho de treino e teste for alta, recomenda-se restringir **`max_depth`**, **`min_samples_leaf`** e/ou **`min_samples_split`**.  
- **Variáveis proxy**: atributos geográficos (`country`, `province`, `region_*`) e **`variety`** podem capturar padrões de estilo/qualidade; **`price`** costuma aparecer relevante, mas atenção a correlações espúrias e viés de seleção.  
- **Sem vazamento**: a variável **`points`** foi usada apenas para gerar o alvo (`quality_high`) e **não** entrou como *feature* nas previsões.


## Treinamento do Modelo - KNN

Nesta seção foi implementado o algoritmo **KNN** de forma **manual**, a partir do zero, para consolidar o entendimento do funcionamento do método.  
A implementação considera a **distância euclidiana** entre os pontos, identifica os **vizinhos mais próximos** e atribui a classe com **maior frequência**.  
Esse exercício é importante para compreender a lógica por trás do KNN antes de utilizar bibliotecas prontas.

=== "Modelo"

    ```python exec="on" html="1"    
    --8<-- "docs/projetos/projeto1/modelos/modelo.py"
    ```

=== "Código"

    ```python exec="on" html="1"  
    --8<-- "docs/projetos/projeto1/modelos/modelo.py"
    ```

## Usando Scikit-Learn

Aqui repetimos a preparação e treinamos o KNeighborsClassifier do scikit-learn.
Para fins de visualização, usamos PCA (2D) apenas para projetar os dados e exibir a fronteira de decisão do KNN no plano, junto de um gráfico de dispersão dos pontos de treino.

=== "Resultado"

    ```python exec="on" html="1"    
    --8<-- "docs/projetos/projeto1/modelos/setup.py"
    ```

=== "Código"

    ```python exec="0"    
    --8<-- "docs/projetos/projeto1/modelos/setup.py"
    ```

## Avaliação do Modelo - KNN

Após o treinamento, avaliamos a acurácia em teste nas duas abordagens (manual e com scikit-learn).
Como o alvo é quality_high (derivado de points), o desempenho pode ser afetado por desbalanceamento (vinhos com nota ≥ 90 tendem a ser minoria).
Além disso, o KNN é sensível à escala e à escolha de k, o que pode gerar variação de resultados.
A visualização em PCA 2D tende a mostrar sobreposição entre classes, reforçando a dificuldade de separação perfeita nesse domínio.

## Treinamento do Modelo - K-Means

O modelo **K-Means** foi treinado com **3 clusters** como referência pedagógica para **segmentar perfis de vinhos** (ex.: baixa/média/alta qualidade percebida).  
Após o pré-processamento (padronização de variáveis numéricas e *one-hot* para categóricas selecionadas), aplicamos o K-Means e projetamos os dados em **PCA (2D)** para visualização dos grupos e de seus **centróides**.

=== "Modelo"
    ```python exec="on" html="1"    
    --8<-- "docs/projetos/projeto1/modelos/treino.py"
    ```

=== "Código"
    ```python exec="0"
    --8<-- "docs/projetos/projeto1/modelos/treino.py"
    ```

## Avaliação do Modelo

Para avaliar o clustering, mapeamos os clusters para o rótulo derivado quality_high (1 se points ≥ 90, 0 caso contrário) por voto majoritário em cada grupo.
Assim, obtemos métricas de classificação mesmo em um cenário originalmente não supervisionado, incluindo acurácia e matriz de confusão.

=== "Resultado"
    ```python exec="on" html="1"  
    --8<-- "docs/projetos/projeto1/modelos/avaliacao.py"
    ```

=== "Código"
    ```python exec="0" 
    --8<-- "docs/projetos/projeto1/modelos/avaliacao.py"
    ```


## Conclusão Geral do Projeto

O projeto teve como objetivo aplicar e comparar três abordagens clássicas de *Machine Learning* — **Árvore de Decisão**, **KNN (K-Nearest Neighbors)** e **K-Means** — sobre o dataset [kagle](https://www.kaggle.com/datasets/zynicide/wine-reviews) de vinhos, que contém informações como país, região, variedade, provador, preço e pontuação (variável `points`), entre outras.  
A proposta foi compreender como diferentes técnicas de aprendizado supervisionado e não supervisionado lidam com o problema de prever ou agrupar vinhos de alta qualidade (definidos como aqueles com pontuação ≥ 90).

---

### Sobre a Base de Dados

A base apresenta **alta cardinalidade categórica** (muitos valores distintos em colunas como `variety` e `region_1`) e **distribuição desigual** entre países e faixas de pontuação.  
Por esse motivo, foi necessário aplicar **pré-processamentos cuidadosos**, como:
- normalização das variáveis numéricas (`price`);
- *one-hot encoding* para variáveis categóricas;
- e amostragem estratificada para reduzir o custo computacional mantendo representatividade.  

Essa estrutura favoreceu a análise, mas também revelou as limitações naturais do conjunto — com poucos atributos diretamente relacionados à qualidade sensorial do vinho, as previsões tendem a capturar mais o perfil geral do mercado do que a avaliação crítica de especialistas.

---

### Árvore de Decisão

O modelo de **Árvore de Decisão** apresentou uma **acurácia de 75%**, sendo o melhor desempenho entre os métodos supervisionados testados.  
A análise das **importâncias das variáveis** mostrou que:
- o **preço** é o principal preditor (forte correlação com a pontuação),  
- seguido por **winery** e **designation**, que representam o produtor e o rótulo do vinho.  

O modelo conseguiu estruturar regras compreensíveis — como faixas de preço associadas à qualidade —, reforçando o caráter interpretável das árvores.  
Ainda assim, parte do resultado pode refletir **overfitting leve**, já que a árvore aprende padrões muito específicos de produtores e regiões.

---

### KNN (K-Nearest Neighbors)

O **KNN manual** alcançou **≈71,6% de acurácia**, enquanto a versão com *Scikit-Learn* e projeção via **PCA (2D)** obteve cerca de **70,8%**.  
O gráfico de fronteira de decisão mostrou uma **sobreposição significativa** entre classes, especialmente nas regiões de média qualidade.  
Isso reforça o fato de que os atributos disponíveis não separam de forma nítida os vinhos de alta e baixa pontuação.

Apesar disso, o KNN apresentou comportamento consistente e intuitivo:
- bons resultados para amostras com padrões similares de região e variedade;
- sensibilidade a escala e densidade de vizinhança, exigindo *scaling* e escolha adequada de `k`.

---

### K-Means (Clustering)

O **K-Means**, aplicado com **3 clusters**, buscou agrupar vinhos de maneira não supervisionada, representando diferentes **perfis de qualidade**.  
Os grupos formados mostraram **tendência de separação**, mas com **fronteiras difusas**, o que é esperado dado que os dados não possuem rótulos explícitos de classes distintas.  
O mapeamento por voto majoritário atingiu **≈62,9% de acurácia**, com confusão entre vinhos medianos e de alta pontuação.

O resultado indica que, embora o K-Means consiga identificar **padrões estruturais** (como faixas de preço e origem), ele não captura com precisão o conceito subjetivo de “qualidade” — que depende de fatores sensoriais não representados na base.

---

### Considerações Finais

Comparando os modelos:

| Modelo | Tipo de Aprendizado | Acurácia | Observações |
|:-------|:--------------------|:---------|:-------------|
| **Árvore de Decisão** | Supervisionado | **0.75** | Melhor desempenho; interpretável; sensível a overfitting |
| **KNN** | Supervisionado | ~0.71 | Bom em padrões locais; alta sobreposição entre classes |
| **K-Means** | Não supervisionado | ~0.63 | Agrupamento coerente, mas difuso; útil para segmentação exploratória |

Em síntese, o projeto mostrou que:
- modelos supervisionados (Árvore e KNN) se beneficiam do conhecimento prévio das classes e alcançam resultados mais robustos;  
- o K-Means, embora menos preciso, é útil para **análises exploratórias e descoberta de padrões**;
- e que a **qualidade do dado** é tão ou mais determinante do que o algoritmo escolhido — atributos objetivos como preço e origem são insuficientes para explicar completamente uma variável subjetiva como “pontuação de degustação”.

Assim, o estudo reforça a importância do **pré-processamento, seleção de variáveis e análise crítica dos resultados** em qualquer projeto de *Machine Learning*, especialmente em domínios complexos como o enológico, onde os dados quantitativos capturam apenas parte da realidade avaliada por especialistas.

---

### Reflexão Final

O desenvolvimento deste projeto proporcionou uma visão prática do ciclo completo de *Machine Learning*: desde a limpeza e transformação de dados até a avaliação e comparação de modelos.  
Ficou evidente que compreender o **contexto do problema e a natureza dos dados** é essencial para interpretar resultados e extrair conclusões relevantes.  
Mais do que alcançar a maior acurácia possível, o aprendizado principal foi entender como **cada modelo oferece uma lente diferente sobre os mesmos dados**, revelando tanto suas potencialidades quanto suas limitações.