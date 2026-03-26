# Pipeline TOMS com K regressores (um por classe)

Este documento descreve o que o código faz após a alteração para o modo **TOMS multi-regressor**, usado em `run_experiment.py` com `EXP_TYPES = ["TOMS"]`.

## 1. Tempo `t` único por janela

- Cada **janela** é um elemento de `ts_chunks[i]` (um DataFrame com todos os tweets daquele intervalo temporal).
- Para a coluna temporal (`TweetAt` no Covid), convertemos todas as linhas do chunk para **dias desde o epoch UNIX** (mesma regra que `quantifications._time_column_to_X`).
- O **único** `t` associado à janela `i` é a **mediana** desses valores: é robusto se houver ruído e coincide com o valor único quando todos os tweets do chunk partilham a mesma data (caso típico do loader antigo por dia).
- Esse mapa `window_t[i]` é guardado no `TOMSMultiRegressorBundle` e usado em todo o treino e nas matrizes `M` em validação/teste (log).

## 2. Coluna `_window_id` no conjunto de validação

- `val_set` é a concatenação dos chunks `0 .. val_length-1` (função `utils.val_test_split`).
- Atribuímos `val_set["_window_id"]` de modo que cada linha sabe a que janela pertence (`attach_window_ids`), para reutilizar exatamente o `t` mediano dessa janela.

## 3. Treino: K regressores `TimeSeriesMultinomialRegressor`

- Com o classificador HF, obtemos as **probabilidades moles** `Y` em todo o `val_set` (`Classifying.analyzer`).
- Para cada classe `c_k` (índice `k = 0..K-1` na ordem de `classes`):
  - Filtramos apenas as linhas com **rótulo verdadeiro** `label == c_k`.
  - **Entrada**: vetor `t` onde cada amostra usa o escalar `window_t[_window_id da linha]` (o mesmo `t` para todas as linhas da mesma janela, replicado por linha).
  - **Alvo**: as linhas correspondentes de `Y`, ou seja **vetores de dimensão K** (scores do classificador para todas as classes), como no TOMS original — cada regressor continua a prever um simplex sobre **todas** as classes.
- Se uma classe não tiver amostras no `val_set`, é instalado um regressor **uniforme** (`1/K` para todas as classes).
- Os modelos são treinados via `regression.trainingModel.trainer` com `REGRESSOR_NAME` (ex.: `"TSMN"`).

## 4. Matriz `M` (K × K) num instante `t`

- Para um escalar `t_w` (tempo da janela):
  - Para cada `k`, o regressor `k` devolve um vetor de probabilidades de dimensão `K`.
  - Empilhamos essas `K` linhas numa matriz **`M`**, com shape **(K, K)** — linha `k` = saída normalizada do regressor associado à classe `k`.

## 5. Scores para o quantificador na **validação**

- Para **cada linha** do `val_set`, obtemos `t` pela janela, calculamos `M`, e o vetor de scores alimentado ao DyS / DyS-Opt / etc. é a **média das linhas de `M`**, **renormalizada** para somar 1 (mistura equiprovável das previsões dos K regressores).
- O analisador `analyzer_toms_multi_val` **não** volta a chamar o HF para estes scores de validação (apenas foi usado para construir os alvos `Y` no treino dos regressores).

## 6. Scores na fase de **teste**

- Pedido explícito: na quantificação dos chunks de teste, usam-se **apenas os scores do classificador** (HF), como em `Classifying.analyzer` — não os scores dos regressores.
- O ficheiro de log (`regressor/toms_regressor_log.txt`) ainda mostra, para cada janela de teste, a matriz `M` que os regressores **teriam** produzido no `t` dessa janela, e a média do classificador na janela (para comparação), mas o DyS nos testes usa só o classificador.

## 7. Ficheiros de saída

- **`regressor/toms_regressor_log.txt`** (append): cabeçalho por run (dataset, quantificador, seed), bloco de treino por classe (`n`, `t` min/max, amostra de `Y`), e secções de **validação** e **teste** com `M` e vetores médios.
- **`output_regressor/*.csv`**: por janela, colunas `t_window`, entradas `M_r{i}_c{j}`, e `score_mean_{classe}` (média das linhas de `M` renormalizada).

## 8. Notas

- O min–max temporal de cada `TimeSeriesMultinomialRegressor` é aprendido **só** com os `t` das amostras da classe correspondente no treino; em predição, tempos fora desse intervalo continuam a ser extrapolados linearmente na normalização interna do modelo.
- O cache em `quant_results` continua a ser **ignorado** quando há regressor TOMS (comportamento existente em `qtfied_dists`).
