# Algoritmos de Búsqueda en Grafos

## Resumen

Este documento presenta un framework completo para la implementación, análisis y visualización de algoritmos de búsqueda en grafos. El proyecto incluye implementaciones de siete algoritmos de búsqueda, extensiones para problemas multi-objetivo, y módulos de benchmarking y visualización. Las principales contribuciones del trabajo incluyen: (1) adaptaciones optimizadas de algoritmos para problemas multi-objetivo, (2)  módulo de visualización, y (3) módulo de análisis comparativo de rendimiento.

## 1. Introducción

### 1.1 Contexto y motivación

Los algoritmos de búsqueda en grafos constituyen una base fundamental en ciencias de la computación e inteligencia artificial, con aplicaciones que van desde navegación y logística hasta planificación automatizada y optimización de recursos. Si bien los algoritmos clásicos como BFS, DFS, UCS, A* y Dijkstra están bien establecidos para problemas de objetivo único, las extensiones para problemas multi-objetivo presentan desafíos adicionales de complejidad computacional y optimización.

### 1.2 Objetivos del proyecto

- **Objetivo Principal:** Desarrollar un framework para la implementación y análisis comparativo de algoritmos de búsqueda en grafos
- **Objetivos Específicos:**
  - Implementar algoritmos clásicos de búsqueda no informada e informada
  - Desarrollar extensiones multi-objetivo para UCS y Dijkstra
  - Crear un sistema de visualización interactiva y animaciones educativas
  - Establecer un framework de benchmarking para análisis comparativo de rendimiento

### 1.3 Contribuciones principales

1. **Extensiones multi-objetivo:** Adaptación de UCS y Dijkstra para problemas tipo TSP reducido con manejo optimizado de estados
3. **Arquitectura modular:** Diseño extensible que permite la incorporación de nuevos algoritmos
4. **Sistema de benchmarking:** Análisis automatizado de métricas de rendimiento y comparación algorítmica

## 2. Marco Teórico

### 2.1 Taxonomía de Algoritmos de Búsqueda

Los algoritmos de búsqueda se clasifican tradicionalmente en dos categorías principales:

#### 2.1.1 Búsqueda No Informada (Ciega)
- **BFS (Breadth-First Search):** Exploración sistemática por niveles
- **DFS (Depth-First Search):** Exploración en profundidad
- **UCS (Uniform Cost Search):** Expansión basada en costo acumulado mínimo

#### 2.1.2 Búsqueda Informada (Heurística)
- **A\*:** Combinación optimal de costo real g(n) y estimación heurística h(n)
- **Dijkstra:** Algoritmo de camino más corto con garantías de optimalidad

### 2.2 Problemática Multi-Objetivo

Los problemas multi-objetivo requieren visitar múltiples estados objetivo, generalmente minimizando el costo total del recorrido. Esta variante introduce:

- **Explosión del espacio de estados:** El estado de búsqueda se amplía a tuplas (nodo_actual, objetivos_restantes)
- **Complejidad exponencial:** O(2^n) donde n es el número de objetivos
- **Optimización global:** Necesidad de encontrar la ruta óptima que visite todos los objetivos

## 3. Arquitectura del Sistema

### 3.1 Diseño Modular

El framework está estructurado en cuatro módulos principales:

```
SearchAlgos/
├── algorithms/           # Implementaciones algorítmicas
├── components/          # Componentes fundamentales del framework
├── problems/           # Casos de prueba y validación
└── benchmark.py        # Sistema de análisis de rendimiento
```

### 3.2 Componentes Fundamentales

#### 3.2.1 State (Estado)
```python
class State:
    def __init__(self, name: str):
        self.name = name
```

**Descripción:** Representación inmutable de un estado en el espacio de búsqueda.

**Características:**
- Identificación única mediante atributo `name`
- Implementación de métodos `__eq__` y `__hash__` para comparaciones en memoria
- Soporte para uso en estructuras de datos como sets y diccionarios

#### 3.2.2 Node (Nodo)
```python
class Node:
    def __init__(self, state: State, parent: Optional['Node'] = None, 
                 action: Optional[Action] = None, cost: float = 0.0,
                 visited: Optional[FrozenSet[str]] = None):
```

**Descripción:** Nodo en el árbol de búsqueda con información completa de trazabilidad.

**Atributos principales:**
- `state`: Estado asociado al nodo
- `parent`: Referencia al nodo padre para reconstrucción de ruta
- `action`: Acción que llevó a este nodo
- `cost`: Costo acumulado desde el nodo raíz
- `visited`: Conjunto inmutable de objetivos visitados (para problemas multi-objetivo)
- `heuristic`: Diccionario de valores heurísticos hacia objetivos

**Métodos clave:**
- `expand(problem)`: Genera nodos hijos aplicando todas las acciones válidas

#### 3.2.3 Problem (Problema)
```python
class Problem:
    def __init__(self, initial, targets: List[State], 
                 actions: Dict[str, Dict[str, State]],
                 costs: Dict[str, Dict[str, float]] = None,
                 heuristic: Dict[str, Dict[str, float]] = None):
```

**Descripción:** Definición completa del problema de búsqueda.

**Componentes:**
- `initial`: Estado inicial del problema
- `targets`: Lista de estados objetivo
- `actions`: Mapa de acciones disponibles desde cada estado
- `costs`: Costos asociados a cada acción (default: 1.0)
- `heuristic`: Función heurística para algoritmos informados

**Métodos principales:**
- `is_target(state)`: Verificación de estado objetivo
- `result(state, action)`: Estado resultante de aplicar una acción
- `action_cost(state, action)`: Costo de ejecutar una acción

#### 3.2.4 Sistema de Visualización

El framework incluye un sistema dual de visualización:

**NetworkX + Matplotlib:**
- Visualizaciones estáticas con layout automático
- Resaltado de rutas óptimas

**Plotly:**
- Zoom, pan y hover information
- Exportación a formatos múltiples


**Configuración de Animaciones:**
```python
ANIMATION_SPEEDS = {
    'muy_lento': {'interval': 1000, 'duration': 800},
    'lento': {'interval': 600, 'duration': 400}, 
    'normal': {'interval': 300, 'duration': 200},
    'rapido': {'interval': 100, 'duration': 100},
    'muy_rapido': {'interval': 50, 'duration': 50}
}
```

## 4. Implementaciones Algorítmicas

### 4.1 Algoritmos de Búsqueda No Informada

#### 4.1.1 Búsqueda en Anchura (BFS)

**Complejidad Temporal:** O(b^d) donde b = factor de ramificación, d = profundidad de solución

**Complejidad Espacial:** O(b^d)

**Características:**
- ✅ **Completo:** Garantiza encontrar solución si existe
- ✅ **Óptimo:** Para problemas con costo uniforme de pasos
- ⚠️ **Memoria:** Requiere almacenar todos los nodos en la frontera

**Implementación Clave:**
```python
def bfs(problem: Problem, explored_steps=None) -> Node | None:
    root = get_root_node(problem)
    if problem.is_target(root.state):
        return root
    
    frontier: list[Node] = [root]   # Cola FIFO
    explored: set[str] = set()      # Estados visitados
    
    while frontier:
        node = frontier.pop(0)
        explored.add(node.state.name)
        
        for child in node.expand(problem):
            if child.state.name in explored:
                continue
            if problem.is_target(child.state):
                return child
            frontier.append(child)
    
    return None
```

#### 4.1.2 Búsqueda en Profundidad (DFS)

**Complejidad Temporal:** O(b^m) donde m = profundidad máxima

**Complejidad Espacial:** O(bm) - significativamente mejor que BFS

**Características:**
- ⚠️ **Completo:** Solo en espacios finitos o con detección de ciclos
- ❌ **Óptimo:** No garantiza solución óptima
- ✅ **Memoria:** Eficiente en uso de memoria

**Implementación Recursiva:**
```python
def dfs(problem: Problem, explored_steps=None) -> Node | None:
    root = get_root_node(problem)
    if problem.is_target(root.state):
        return root
    
    explored: set[str] = set()
    
    def _dfs(node: Node) -> Node | None:
        explored.add(node.state.name)
        for child in node.expand(problem):
            if child.state.name in explored:
                continue
            if problem.is_target(child.state):
                return child
            result = _dfs(child)
            if result:
                return result
        return None
    
    return _dfs(root)
```

#### 4.1.3 Búsqueda de Costo Uniforme (UCS)

**Complejidad Temporal:** O(b^⌊C*/ε⌋) donde C* = costo de solución óptima, ε = costo mínimo de paso

**Características:**
- ✅ **Completo:** Si el costo de paso es ≥ ε > 0
- ✅ **Óptimo:** Garantiza solución de costo mínimo
- 🔄 **Cola de Prioridad:** Expande nodos en orden de costo creciente

**Implementación con Heap:**
```python
def ucs(problem: Problem, explored_steps=None) -> Node | None:
    root = get_root_node(problem)
    if problem.is_target(root.state):
        return root
    
    frontier: list[tuple[float, int, Node]] = []
    counter = itertools.count()
    heapq.heappush(frontier, (root.cost, next(counter), root))
    
    explored: dict[str, float] = {}
    
    while frontier:
        cost, _, node = heapq.heappop(frontier)
        
        if explored.get(node.state.name, float('inf')) <= cost:
            continue
        explored[node.state.name] = cost
        
        if problem.is_target(node.state):
            return node
        
        for child in node.expand(problem):
            if explored.get(child.state.name, float('inf')) <= child.cost:
                continue
            heapq.heappush(frontier, (child.cost, next(counter), child))
    
    return None
```

### 4.2 Algoritmos de Búsqueda Informada

#### 4.2.1 Algoritmo A*

**Función de Evaluación:** f(n) = g(n) + h(n)
- g(n): Costo real desde el inicio hasta n
- h(n): Estimación heurística desde n hasta el objetivo

**Características:**
- ✅ **Completo:** Si la heurística es admisible
- ✅ **Óptimo:** Si la heurística es admisible y consistente
- ⚡ **Eficiencia:** Expande el mínimo número de nodos necesarios

**Implementación con Heurística Multi-Objetivo:**
```python
def astar(problem: Problem, explored_steps=None) -> Node | None:
    root = get_root_node(problem)
    targets = {t.name for t in problem.targets}
    
    def heuristic(node: Node) -> float:
        remaining = targets - node.visited
        if not remaining: 
            return 0.0
        return min(node.heuristic.get(t, float('inf')) for t in remaining)
    
    frontier: list[tuple[float, int, Node]] = []
    counter = itertools.count()
    
    f_score = root.cost + heuristic(root)
    heapq.heappush(frontier, (f_score, next(counter), root))
    
    explored: dict[tuple[str, frozenset[str]], float] = {}
    
    while frontier:
        f, _, node = heapq.heappop(frontier)
        
        key = (node.state.name, node.visited)
        if explored.get(key, float("inf")) <= node.cost:
            continue
        explored[key] = node.cost
        
        if node.visited == targets:
            return node
        
        for child in node.expand(problem):
            child_key = (child.state.name, child.visited)
            if explored.get(child_key, float("inf")) <= child.cost:
                continue
            
            f_child = child.cost + heuristic(child)
            heapq.heappush(frontier, (f_child, next(counter), child))
    
    return None
```

#### 4.2.2 Algoritmo Dijkstra

**Características:**
- ✅ **Completo:** Garantiza encontrar solución si existe
- ✅ **Óptimo:** Solución de costo mínimo garantizada
- 📊 **Versatilidad:** No requiere función heurística
- 🔄 **Principio de Relajación:** Actualización iterativa de distancias mínimas

**Implementación Optimizada:**
```python
def dijkstra(problem: Problem, explored_steps=None) -> Node | None:
    root = get_root_node(problem)
    if problem.is_target(root.state):
        return root
    
    frontier: list[tuple[float, int, Node]] = []
    counter = itertools.count()
    heapq.heappush(frontier, (root.cost, next(counter), root))
    
    best_g: dict[str, float] = {root.state.name: 0.0}
    
    while frontier:
        cost, _, node = heapq.heappop(frontier)
        
        if problem.is_target(node.state):
            return node
        
        for child in node.expand(problem):
            g2 = child.cost
            if g2 < best_g.get(child.state.name, float('inf')):
                best_g[child.state.name] = g2
                heapq.heappush(frontier, (g2, next(counter), child))
    
    return None
```

### 4.3 Extensiones Multi-Objetivo

#### 4.3.1 UCS Multi-Objetivo

**Innovación Principal:** Extensión del espacio de estados para problemas tipo TSP reducido.

**Estado Ampliado:** (nodo_actual, objetivos_restantes)

**Características:**
- 🎯 **Problema TSP Reducido:** Visita todos los objetivos exactamente una vez
- ✅ **Optimalidad:** Mantiene garantías de UCS original
- 🔄 **Gestión de Estado:** Uso de frozenset para objetivos restantes
- ⚠️ **Complejidad:** Exponencial en número de objetivos

**Implementación:**
```python
def multi_ucs(problem: Problem, explored_steps=None) -> Node | None:
    root = get_root_node(problem)
    targets = {t.name for t in problem.targets}
    
    if root.visited == targets:
        return root
    
    frontier: list[Node] = [root]
    explored: dict[tuple[str, frozenset[str]], float] = {}
    
    while frontier:
        frontier.sort(key=lambda n: n.cost)
        node = frontier.pop(0)
        key = (node.state.name, node.visited)
        
        if explored.get(key, float("inf")) <= node.cost:
            continue
        explored[key] = node.cost
        
        if node.visited == targets:
            return node
        
        for child in node.expand(problem):
            child_key = (child.state.name, child.visited)
            
            if explored.get(child_key, float("inf")) <= child.cost:
                continue
            
            existing = next((n for n in frontier 
                           if (n.state.name, n.visited) == child_key), None)
            if existing:
                if child.cost < existing.cost:
                    frontier.remove(existing)
                else:
                    continue
            
            frontier.append(child)
    
    return None
```

#### 4.3.2 Dijkstra Multi-Objetivo (Contribución Principal)

**Mejoras Implementadas:**

1. **Optimización con heapq:** Uso de cola de prioridad binaria para mejor rendimiento
2. **Evitación de Comparaciones Problemáticas:** Contador único para evitar comparaciones entre objetos State
3. **Reconstrucción Optimizada:** Construcción eficiente de la cadena de nodos enlazados
4. **Registro Detallado:** Información completa de trazabilidad para análisis

**Algoritmo:**
```python
def dijkstra_multi_goal(problem, explored_steps=None):
    start = problem.initial
    goals = set(problem.targets)
    nodes_dict = {s.name: s for s in [problem.initial] + [t for t in problem.targets]}
    
    # Registro de nodos disponibles para reconstrucción
    for src, adjs in problem.actions.items():
        for a, t in adjs.items():
            nodes_dict[t.name] = t
    
    heap = []
    counter = count()
    heappush(heap, (0, next(counter), start, frozenset(goals), [start]))
    visited = {}

    while heap:
        cost, _, node, goals_left, path = heappop(heap)
        state = (node, goals_left)
        
        if state in visited and visited[state] <= cost:
            continue
        visited[state] = cost

        # Actualizar objetivos restantes
        if node in goals_left:
            goals_left = goals_left - {node}
        
        # Verificar si se completaron todos los objetivos
        if not goals_left:
            # Reconstrucción de cadena enlazada para compatibilidad
            prev = None
            total_cost = 0
            path_names = [s.name for s in path]
            
            for i in range(len(path_names)):
                name = path_names[i]
                if i == 0:
                    n = Node(state=nodes_dict[name], parent=None, action=None, cost=0)
                else:
                    src = path_names[i-1]
                    dst = path_names[i]
                    # Buscar costo de la acción
                    found = False
                    for a, t in problem.actions[src].items():
                        if t.name == dst:
                            step_cost = problem.costs[src][a]
                            total_cost += step_cost
                            found = True
                            break
                    if not found:
                        raise ValueError(f"No se encontró acción de {src} a {dst}")
                    n = Node(state=nodes_dict[dst], parent=prev, action=None, cost=total_cost)
                prev = n
            
            prev.cost = cost
            return prev

        # Expansión a vecinos
        for action, neighbor in problem.actions[node.name].items():
            step_cost = problem.costs[node.name][action]
            heappush(heap, (cost + step_cost, next(counter), neighbor, goals_left, path + [neighbor]))
    
    return None
```

**Ventajas de la Implementación:**

1. **Rendimiento Mejorado:** 
   - Uso de heapq reduce complejidad de operaciones de cola de prioridad de O(n) a O(log n)
   - Evita ordenamientos repetidos de la frontera completa

2. **Robustez:**
   - Manejo correcto de comparaciones entre objetos
   - Prevención de errores por objetos no comparables

3. **Trazabilidad:**
   - Registro completo de pasos explorados
   - Información detallada para visualización y debugging

4. **Compatibilidad:**
   - Retorna objetos Node compatibles con el resto del framework
   - Mantiene interfaz consistente con otros algoritmos

## 5. Casos de Estudio y Validación

### 5.1 Batería de Problemas de Prueba

#### Problem 1: Grafo Básico (Validación de construcción)
```
Nodos: A, B, C, D, E, F, G, H, I, J
Objetivo Único: A → G
Multi-Objetivo: Visitar A, D, G, J
Propósito: Validación de implementaciones básicas
```

#### Problem 2: Red Intermedia (Análisis de escalabilidad)
```
Nodos: 10+ estados interconectados
Costos: Variables con múltiples caminos alternativos
Propósito: Comparación de eficiencia algorítmica
```

#### Problem 3: Grafo Complejo (Ánalisis de problemas complejos)
```
Nodos: A-T (20 estados)
Conexiones: Densas con costos heterogéneos
Multi-Objetivo: Visitar M, P, T desde A
Propósito: Análisis de rendimiento en grafos complejos
```

#### Problem 4: Documento de Tesis (Caso de uso con 37 nodos)
```
Nodos: 0-37 (representando fuentes de energía demanda)
Escenario: Conexión de nodos minimizando costo
Objetivos: Nodos '0', '1', '7', '15', '22', '23', '24'
Interpretación: Optimización de ruta
Métricas: Tiempo total de búsqueda de información
```

#### Problema 5: Documento de Tesis (Caso de uso con 61 nodos)
```
Nodos: 1-61 (representando fuentes de energía demanda)
Escenario: Conexión de nodos minimizando costo
Objetivos: Nodos '1', '6', '50', '2', '3', '4', '5', '14', '15', '24', '25', '26', '27', '28'
Interpretación: Optimización de ruta
Métricas: Tiempo total de búsqueda de información
```


## 6. Sistema de benchmarking y análisis

### 6.1 Framework de evaluación

El módulo de benchmarking automatizado proporciona análisis comparativo:

```python
def run_benchmark(problem: Problem, algos: Dict[str, Callable]) -> List[Dict]:
    results = []
    for name, fn in algos.items():
        start = time.perf_counter()
        node = fn(problem)
        elapsed = time.perf_counter() - start

        if node:
            path = reconstruct_path(node)
            cost = node.cost
            route = " → ".join(path)
        else:
            cost = float('inf')
            route = ""

        results.append({
            'Algoritmo': name,
            'Tiempo (s)': round(elapsed, 6),
            'Costo': cost,
            'Ruta': route
        })
    return results
```

### 6.2 Métricas de evaluación

**Métricas Primarias:**
- **Tiempo de ejecución:** Medición de tiempo con `time.perf_counter()`
- **Costo de solución:** Suma total de costos de aristas en la ruta óptima
- **Longitud de ruta:** Número de nodos visitados
- **Ruta completa:** Secuencia detallada de estados visitados

### 6.3 Visualización de Resultados

El framework genera automáticamente tablas comparativas y permite análisis interactivo:

```python
def print_benchmark_table(results: List[Dict]):
    df = pd.DataFrame(results)
    print(tabulate(df, headers='keys', tablefmt='grid', floatfmt='.6f'))
```

## 7. Módulo de visualización

### 7.1 Arquitectura 

#### 7.1.1 NetworkX + Matplotlib
```python
def plot_graph_and_path_nx(nodes, actions, costs, path=None):
    G = nx.DiGraph()
    # Construcción del grafo dirigido
    for src, targets in actions.items():
        for action, target in targets.items():
            weight = costs[src][action]
            G.add_edge(src, target.name, weight=weight)
    
    # Layout automático para distribución de nodos
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Renderizado con resaltado de ruta óptima
    if path:
        path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, 
                              edge_color='red', width=3)
```

**Ventajas:**
- Layout automático inteligente
- Integración nativa con algoritmos de grafos
- Capacidades de exportación a múltiples formatos

#### 7.1.2 Plotly Interactive
```python
def plot_graph_and_path(nodes, actions, costs, path=None):
    # Posicionamiento circular para claridad visual
    n = len(nodes)
    angle = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = {name: (np.cos(a), np.sin(a)) for name, a in zip(nodes, angle)}
    
    # Construcción de trazas interactivas
    edge_trace = go.Scatter(x=edge_x, y=edge_y,
                           line=dict(width=1, color='#888'),
                           hoverinfo='none', mode='lines')
```

### 7.2 Módulo de animaciones

#### 7.2.1 Configuracion de velocidad
```python
ANIMATION_SPEEDS = {
    'muy_lento': {'interval': 1000, 'duration': 800},
    'lento': {'interval': 600, 'duration': 400}, 
    'normal': {'interval': 300, 'duration': 200},
    'rapido': {'interval': 100, 'duration': 100},
    'muy_rapido': {'interval': 50, 'duration': 50}
}
```

#### 7.2.2 Animación paso a paso
```python
def run_and_animate_nx(problem, algorithm, nodes, actions, costs, speed='normal'):
    explored_steps = []
    result_node = algorithm(problem, explored_steps)
    
    if not result_node:
        print("No se encontró solución")
        return
    
    path = reconstruct_path(result_node)
    animate_search_nx(nodes, actions, costs, explored_steps, path, speed)
```

**Características de Animación:**
- **Codificación por Colores:**
  - Azul: Nodos no explorados
  - Amarillo: Nodos en exploración actual
  - Verde: Nodos en ruta óptima final
  - Rojo: Nodos explorados sin éxito

- **Información Contextual:**
  - Contador de pasos de exploración
  - Costo acumulado en tiempo real
  - Progreso hacia objetivos

- **Control de Reproducción:**
  - Pausa/reanudación
  - Velocidad ajustable dinámicamente
  - Navegación paso a paso


## Referencias y Fuentes

### Literatura
- Russell, S. & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2009). *Introduction to Algorithms* (3rd ed.). MIT Press.
- Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). A formal basis for the heuristic determination of minimum cost paths. *IEEE Transactions on Systems Science and Cybernetics*, 4(2), 100-107.

### Implementaciones de Referencia
- NetworkX Documentation: https://networkx.org/
- Python heapq Module: https://docs.python.org/3/library/heapq.html
- Matplotlib Animation: https://matplotlib.org/stable/api/animation_api.html


