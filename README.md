# Dijkstra multi-demanda para ruteo de cable residencial

Implementación de la adaptación de Dijkstra descrita en el artículo *A multi-demand adaptation of Dijkstra's algorithm for Residential Cable Routing Optimisation*: conectar el tablero de distribución con todos los puntos de demanda sobre una red de conductos fija, con la menor longitud de cable.

## Cómo funciona

El estado de búsqueda es `(v, S)`: el vértice actual y el conjunto de demandas ya conectadas, codificado como máscara de bits. Dos rutas al mismo vértice con distinto `S` no se podan entre sí. La cola de prioridad se ordena por `g`, la suma de pesos de la caminata, y la búsqueda termina al extraer el primer estado con `S = D`.

Se manejan dos costos:

- `g*`: costo de la caminata contando cada cruce de arista. La búsqueda es exacta para este costo.
- `c`: costo reportado, que cuenta cada arista distinta una sola vez. Cumple `c* <= c <= g* <= 2 c*`, donde `c*` es el árbol de Steiner óptimo.

## Estructura

```
algorithms/
  dijkstra_multi_goal.py   algoritmo del artículo (Algoritmo 1)
  steiner_exact.py         óptimo c* por Dreyfus-Wagner, como referencia (hasta 16 terminales)
  takahashi_matsuyama.py   heurística de Steiner 2-aproximada (método de referencia)
  steiner_ilp.py           ILP exacto con flujo multi-commodity, resuelto con HiGHS (método de referencia)
components/
  problem.py, state.py     modelo del grafo
  connectivity.py          validación de conectividad
problems/
  plans.py                 redes de conductos de los planos A y B
  cable_installation.py    plano + nodos demanda -> Problem
  experiments.py           definición de los experimentos 1.1 a 2.4
tests/                     contraejemplo y Proposiciones 1 y 2 del artículo
run_experiments.py         ejecuta Dijkstra multi-demanda y escribe results/experiments.*
run_reference_methods.py   evaluación aislada de Takahashi-Matsuyama e ILP, escribe results/reference_methods.*
```

## Uso

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -t .
python run_experiments.py --only 1.1 2.1 2.3 --exact
python run_experiments.py --time-limit 3600 --memory-limit-gb 16
```

`run_experiments.py` valida que el grafo sea conexo y que el tablero alcance cada demanda antes de buscar. Corre cada experimento en un proceso aparte y lo corta al pasar el límite de tiempo o de memoria; ese caso queda como "No completado" con los estados y la memoria alcanzados. Los resultados se guardan en `results/experiments.json` y `results/experiments.md`.

### Métodos de referencia

`run_reference_methods.py` evalúa por separado Takahashi-Matsuyama y el ILP exacto sobre los mismos planos y demandas. No ejecuta la búsqueda de Dijkstra, de modo que ambas evaluaciones son independientes y se comparan después. Reporta la brecha `100 (c - c*) / c*` de Takahashi-Matsuyama y de la solución humana, con `c*` del ILP. Requiere `scipy` (HiGHS):

```bash
python run_reference_methods.py --ilp-time-limit 600
```

## Experimentos

El tablero es el nodo 0 en ambos planos. El plano A tiene 38 nodos y 45 conductos; el plano B, 62 nodos y 77 conductos.

| Exp | Plano | Nodos demanda |
| --- | --- | --- |
| 1.1 | A | 0, 1, 7, 15, 22, 23, 24 |
| 1.2 | A | 0, 1, 2, 3, 6, 4, 5, 16, 17, 18, 19, 12, 13, 24, 29, 27, 26, 37, 36, 35, 33 |
| 1.3 | A | 0 a 37 |
| 2.1 | B | 0, 1, 6, 2, 15, 24, 3, 4, 25, 14, 26, 27, 28, 5 |
| 2.2 | B | 0 a 28 |
| 2.3 | B | 0, 29, 33, 36, 39, 40, 41, 42, 52, 53, 54, 55, 56, 61 |
| 2.4 | B | 0 y 29 a 61 |

El espacio de estados crece como `|V| * 2^|D|`; los resultados de cada corrida, incluidos los experimentos que no completan, están en `results/experiments.md`.
