"""
Dijkstra Convencional — Camino mínimo de origen a un único destino
==================================================================

Implementación del algoritmo de Dijkstra en su forma canónica (SSSP:
Single-Source Shortest Path). Dado un nodo origen y UN nodo destino,
encuentra el camino de menor costo acumulado que los conecta.

Diferencia fundamental con la versión adaptada para RCRP:
-   El estado de búsqueda es solo el nodo actual (no incluye información
    sobre objetivos cubiertos).
-   El algoritmo termina en cuanto EXTRAE el primer nodo objetivo del heap.
-   El árbol de padres es único y global: cada nodo tiene exactamente un
    predecesor óptimo.
-   No hay noción de "cobertura acumulada"; si hay múltiples objetivos,
    este algoritmo resuelve el problema de forma independiente para cada uno.

Uso dentro del framework:
    Para ejecutar correctamente este algoritmo con CableInstallationProblem,
    usar create_single_objective_problem(target_node) — no el problema
    multi-objetivo — ya que este Dijkstra solo persigue UN objetivo a la vez.

Complejidad:
    Tiempo:  O((V + E) log V)  con heap binario
    Espacio: O(V)
"""

import heapq
import itertools
from components.node import Node
from components.problem import Problem
from components.utils import get_root_node, reconstruct_path


def dijkstra_convencional(problem: Problem, explored_steps: list = None) -> Node | None:
    """
    Busca el camino de menor costo desde el estado inicial hasta el primer
    estado objetivo encontrado.

    El algoritmo mantiene una cola de prioridad (min-heap) ordenada por costo
    acumulado g(n). En cada iteración extrae el nodo de menor costo, lo verifica
    como objetivo y expande sus hijos. Un nodo ya procesado con costo óptimo no
    vuelve a encolarse, garantizando que la primera vez que un nodo objetivo es
    extraído del heap, el costo es mínimo.

    Args:
        problem (Problem): Instancia del problema. Se usa problem.initial como
                           origen y problem.targets[0] como único destino.
                           Si se pasan múltiples targets, solo se detecta el
                           primero que se encuentre durante la expansión.
        explored_steps (list, optional): Lista donde se acumulan las aristas
                           exploradas como tuplas (origen, destino). Útil para
                           trazar el proceso de búsqueda en benchmarks.

    Returns:
        Node | None: Nodo objetivo con la cadena de padres que reconstruye el
                     camino óptimo. Retorna None si no existe camino.
    """

    # ── 1. Nodo raíz ─────────────────────────────────────────────────────────
    raiz = get_root_node(problem)

    # Caso trivial: el nodo inicial ya es el objetivo
    if problem.is_target(raiz.state):
        return raiz

    # ── 2. Estructuras de control ─────────────────────────────────────────────
    # Heap de tuplas (costo_acumulado, contador_desempate, nodo)
    # El contador evita comparar objetos Node cuando los costos son iguales.
    frontera: list[tuple[float, int, Node]] = []
    contador = itertools.count()
    heapq.heappush(frontera, (raiz.cost, next(contador), raiz))

    # Registro del mejor costo conocido para llegar a cada estado.
    # Clave: nombre del nodo (str). Valor: costo mínimo registrado (float).
    # Un nodo solo se reencola si se encuentra un camino estrictamente mejor.
    mejor_g: dict[str, float] = {raiz.state.name: 0.0}

    # ── 3. Bucle principal ────────────────────────────────────────────────────
    while frontera:
        costo_actual, _, nodo = heapq.heappop(frontera)

        # ── 3a. Prueba de objetivo ────────────────────────────────────────────
        # En Dijkstra el cheque se hace al EXTRAER del heap, no al insertar.
        # Esto garantiza que cuando llegamos aquí, costo_actual ES el mínimo
        # posible para este nodo (propiedad fundamental de Dijkstra).
        if problem.is_target(nodo.state):
            return nodo

        # ── 3b. Poda por costo obsoleto ───────────────────────────────────────
        # Si el costo con el que fue encolado es mayor al mejor conocido,
        # este nodo es una entrada obsoleta del heap; se descarta sin expandir.
        if costo_actual > mejor_g.get(nodo.state.name, float('inf')):
            continue

        # ── 3c. Expansión ─────────────────────────────────────────────────────
        for hijo in nodo.expand(problem):
            costo_hijo = hijo.cost  # costo acumulado ya calculado en expand()

            # Solo se encola si mejora el camino conocido hacia este estado
            if costo_hijo < mejor_g.get(hijo.state.name, float('inf')):
                mejor_g[hijo.state.name] = costo_hijo

                # Registro para trazabilidad / benchmarks
                if explored_steps is not None:
                    explored_steps.append((nodo.state.name, hijo.state.name))

                heapq.heappush(frontera, (costo_hijo, next(contador), hijo))

    # ── 4. Sin solución ───────────────────────────────────────────────────────
    print(f"Dijkstra: no existe camino desde '{problem.initial.name}' "
          f"hacia ninguno de los objetivos.")
    return None


# ─── Función auxiliar: ejecutar para CADA objetivo de forma independiente ────

def dijkstra_todos_objetivos(
    cable_problem,
    verbose: bool = True
) -> dict[str, dict]:
    """
    Ejecuta dijkstra_convencional de forma independiente para cada nodo
    objetivo definido en CableInstallationProblem.

    Esto ilustra el comportamiento tradicional: cada ejecución produce un
    camino independiente desde el origen. Las rutas NO comparten tramos,
    por lo que el costo de cable total es la suma de todos los caminos
    individuales (sin reutilización de segmentos).

    Args:
        cable_problem (CableInstallationProblem): Instancia del problema.
        verbose (bool): Si True, imprime los resultados de cada ejecución.

    Returns:
        dict: Diccionario con los resultados por objetivo:
              {
                'objetivo': {
                  'costo':  float,
                  'camino': list[str],
                  'nodo':   Node
                }
              }
    """
    resultados = {}
    costo_total_acumulado = 0.0

    if verbose:
        print("=" * 62)
        print("  Dijkstra Convencional — ejecución por objetivo individual")
        print("=" * 62)
        print(f"  Origen: nodo {cable_problem.network_nodes['0'].name}")
        print(f"  Objetivos: {cable_problem.target_nodes}")
        print("=" * 62)

    for objetivo in cable_problem.target_nodes:
        # Construir un problema de objetivo único para este destino
        problema_unitario = cable_problem.create_single_objective_problem(objetivo)

        # Ejecutar Dijkstra convencional
        pasos = []
        nodo_solucion = dijkstra_convencional(problema_unitario, explored_steps=pasos)

        if nodo_solucion is not None:
            camino = reconstruct_path(nodo_solucion)
            costo  = nodo_solucion.cost
            costo_total_acumulado += costo

            resultados[objetivo] = {
                'costo':  costo,
                'camino': camino,
                'pasos_explorados': len(pasos),
                'nodo':  nodo_solucion,
            }

            if verbose:
                print(f"\n  Objetivo: nodo {objetivo}")
                print(f"  Camino:   {' → '.join(camino)}")
                print(f"  Costo:    {costo:.2f} m")
                print(f"  Nodos explorados: {len(pasos)}")
        else:
            resultados[objetivo] = None
            if verbose:
                print(f"\n  Objetivo: nodo {objetivo}  →  SIN SOLUCIÓN")

    if verbose:
        print("\n" + "=" * 62)
        print(f"  Costo total acumulado (sin compartir tramos): {costo_total_acumulado:.2f} m")
        print("  NOTA: Este costo asume que cada camino instala su propio")
        print("  cable de forma independiente, sin reutilizar segmentos.")
        print("=" * 62)

    return resultados