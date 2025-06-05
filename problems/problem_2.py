from components.state import State

# Crear los nodos una sola vez y usarlos en todo el problema
define_nodes = list('ABCDEFGHIJK')
nodes = {name: State(name) for name in define_nodes}

# Definir las acciones usando solo los nodos creados
actions = {
    'A': {'toB': nodes['B'], 'toD': nodes['D']},
    'B': {'toA': nodes['A'], 'toC': nodes['C']},
    'C': {'toB': nodes['B'], 'toF': nodes['F'], 'toK': nodes['K'], 'toG': nodes['G']},
    'D': {'toA': nodes['A'], 'toE': nodes['E'], 'toF': nodes['F']},
    'E': {'toD': nodes['D'], 'toF': nodes['F'], 'toI': nodes['I']},
    'F': {'toI': nodes['I'], 'toJ': nodes['J']},
    'G': {'toJ': nodes['J']},
    'H': {'toJ': nodes['J']},
    'I': {'toJ': nodes['J']},
    'J': {},
    'K': {'toC': nodes['C'], 'toH': nodes['H'], 'toJ': nodes['J']}
}

# Para usar en los algoritmos, solo se debe pasar nodes y actions como parámetros.
# Ejemplo de uso en un algoritmo:
# resultado = algoritmo_busqueda(nodes, actions, nodo_inicial=nodes['A'], nodo_objetivo=nodes['J'])

# No se crean instancias duplicadas de State ni variables A, B, C, ...