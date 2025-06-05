from components.state import State
from components.problem import Problem
from algorithms.bfs import bfs
from algorithms.ucs import ucs
from algorithms.astar import astar
from algorithms.ucs_multi_goal import multi_ucs

nodes = {name: State(name) for name in list('ABCDEFGHIJ')}
actions = {
    'A': {'toB': nodes['B'], 'toC': nodes['C'], 'toD': nodes['D']},
    'B': {'toE': nodes['E'], 'toF': nodes['F']},
    'C': {'toF': nodes['F'], 'toG': nodes['G']},
    'D': {'toG': nodes['G'], 'toH': nodes['H']},
    'E': {'toI': nodes['I']},
    'F': {'toI': nodes['I'], 'toJ': nodes['J']},
    'G': {'toJ': nodes['J']},
    'H': {'toJ': nodes['J']},
    'I': {'toJ': nodes['J']},
    'J': {}
}
costs = {
    'A': {'toB': 4, 'toC': 3, 'toD': 7}, 
    'B': {'toE': 5, 'toF': 2}, 
    'C': {'toF': 4, 'toG': 6}, 
    'D': {'toG': 1, 'toH': 3}, 
    'E': {'toI': 7}, 
    'F': {'toI': 2, 'toJ': 9}, 
    'G': {'toJ': 5}, 
    'H': {'toJ': 4}, 
    'I': {'toJ': 3}, 
    'J': {}
}

heuristic = {
    'A': {'J': 9}, 
    'B': {'J': 8}, 
    'C': {'J': 7}, 
    'D': {'J': 5}, 
    'E': {'J': 6}, 
    'F': {'J': 4}, 
    'G': {'J': 5}, 
    'H': {'J': 4}, 
    'I': {'J': 3}, 
    'J': {'J': 0}
}

prob_single = Problem(
    initial=nodes['A'],
    targets=[nodes['G']],
    actions=actions,
    costs=costs,
    heuristic=heuristic
)


prob_multi = Problem(
    initial=nodes['A'],
    targets=[nodes['A'], nodes['D'],nodes['G'], nodes['J']],
    actions=actions,
    costs=costs,
    heuristic=heuristic
)

