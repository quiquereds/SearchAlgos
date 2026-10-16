"""
Problema 4: Optimización de Instalación de Cables - Análisis Integral
=====================================================================

Descripción del Problema:
Este problema simula la instalación óptima de cables en una red de telecomunicaciones.
Los nodos representan puntos de conexión y las aristas representan rutas donde se pueden
instalar cables. El objetivo es conectar todos los nodos objetivo minimizando la cantidad
total de cable utilizado.

Características Especiales:
- Los costos representan metros de cable necesarios
- Una vez instalado un cable entre dos nodos, puede reutilizarse sin costo adicional
- Se optimiza para evitar costos de regreso en rutas ya establecidas

Nodos Objetivo: 0, 1, 2, 3, 4, 5, 16, 17, 18, 19, 12, 13, 24, 29, 27, 26, 37, 36, 35, 33
Grafo: 38 nodos (0-37) con conexiones ponderadas
"""

from algorithms.astar import astar
from algorithms.dijkstra_multi_goal import dijkstra_multi_goal
from algorithms.ucs_multi_goal import multi_ucs
from algorithms.bfs import bfs
from algorithms.dfs import dfs
from algorithms.ucs import ucs
from algorithms.dijkstra import dijkstra
from components.benchmark_v2 import execute_benchmark, generate_report
from components.problem import Problem
from components.state import State


class CableInstallationProblem:
    """Clase para gestionar el problema de instalación de cables"""
    
    def __init__(self):
        self.network_nodes = {str(node_id): State(str(node_id)) for node_id in range(0, 38)}
        self.target_nodes = [str(i) for i in range(0, 38)]
        self.network_topology = [
                ('0', '1', 2.0),
                ('0', '20', 3.8),
                ('1', '2', 2.8),
                ('1', '7', 3.8),
                ('2', '8', 4.0),
                ('3', '9', 3.65),
                ('4', '10', 3.4),
                ('5', '11', 3.55),
                ('6', '8', 2.5),
                ('7', '15', 3.70),
                ('8', '9', 2.1),
                ('8', '15', 1.5),
                ('9', '10', 2.3),
                ('9', '16', 3.5),
                ('9', '22', 2.2),
                ('10', '11', 2.4),
                ('10', '17', 3.65),
                ('11', '14', 3.2),
                ('11', '18', 3.6),
                ('11', '23', 4.3),
                ('12', '23', 2.5),
                ('13', '23', 2.8),
                ('14', '19', 3.45),
                ('15', '22', 2.8),
                ('19', '25', 4.2),
                ('20', '21', 2.85),
                ('21', '22', 3.6),
                ('21', '33', 5.2),
                ('22', '27', 3.85),
                ('22', '30', 4.5),
                ('22', '32', 6.0),
                ('23', '24', 3.1),
                ('23', '28', 3.85),
                ('25', '29', 2.0),
                ('26', '30', 3.6),
                ('28', '32', 3.6),
                ('29', '32', 3.0),
                ('30', '31', 3.0),
                ('30', '36', 3.5),
                ('31', '32', 3.2),
                ('31', '34', 2.5),
                ('32', '34', 3.5),
                ('32', '37', 4.1),
                ('33', '35', 2.0),
                ('35', '36', 3.0),
        ]
        
        self.actions, self.costs = self._build_network_structure()
        self.heuristic = self._create_distance_heuristic()
    
    def _build_network_structure(self):
        """Construye la estructura de acciones y costos de la red"""
        actions = {node_name: {} for node_name in self.network_nodes}
        costs = {node_name: {} for node_name in self.network_nodes}
        
        for source_node, target_node, cable_cost in self.network_topology:
            # Conexión bidireccional
            actions[source_node][f"to{target_node}"] = self.network_nodes[target_node]
            costs[source_node][f"to{target_node}"] = cable_cost
            actions[target_node][f"to{source_node}"] = self.network_nodes[source_node]
            costs[target_node][f"to{source_node}"] = cable_cost
        
        return actions, costs
    
    def _create_distance_heuristic(self):
        """Crea heurística basada en distancia Manhattan entre nodos"""
        heuristic = {}
        for node_name in self.network_nodes:
            heuristic[node_name] = {}
            for target_name in self.target_nodes:
                node_id = int(node_name)
                target_id = int(target_name)
                # Heurística simple basada en diferencia de IDs
                heuristic[node_name][target_name] = abs(node_id - target_id) * 0.5
        return heuristic
    
    def create_multi_objective_problem(self):
        """Crea el problema multi-objetivo de instalación de cables"""
        return Problem(
            initial=self.network_nodes['0'],
            targets=[self.network_nodes[node_id] for node_id in self.target_nodes],
            actions=self.actions,
            costs=self.costs,
            heuristic=self.heuristic
        )
    
    def get_problem_summary(self):
        """Retorna un resumen del problema"""
        return {
            'total_nodes': len(self.network_nodes),
            'total_connections': len(self.network_topology),
            'target_nodes': len(self.target_nodes),
            'target_list': self.target_nodes,
            'average_cable_cost': sum(cost for _, _, cost in self.network_topology) / len(self.network_topology)
        }
        
    def create_single_objective_problem(self, target_node: str = "24"):
        """Problema de control para algoritmos tradicionales (objetivo único)."""
        single_heuristic = {}
        for node_name in self.network_nodes:
            single_heuristic[node_name] = {
                target_node: abs(int(node_name) - int(target_node)) * 0.5
            }

        return Problem(
            initial=self.network_nodes["0"],
            targets=[self.network_nodes[target_node]],
            actions=self.actions,
            costs=self.costs,
            heuristic=single_heuristic
        )

def execute_cable_installation_analysis():
    """Ejecuta comparación válida + demostración de caso no válido."""
    cable_problem = CableInstallationProblem()
    multi_problem = cable_problem.create_multi_objective_problem()
    single_problem = cable_problem.create_single_objective_problem("24")
    problem_summary = cable_problem.get_problem_summary()

    # Algoritmos adaptados
    multi_algorithms = {
        "Dijkstra Multi-Goal": dijkstra_multi_goal,
    }

    # Algoritmos tradicionales 
    traditional_algorithms = {
        "Dijkstra Tradicional": dijkstra,
    }

    print("PROBLEMA")
    print("=" * 80)
    print(
        f"Nodos en la red: {problem_summary['total_nodes']}\n"
        f"Conexiones: {problem_summary['total_connections']}\n"
        f"Objetivos multi: {problem_summary['target_nodes']} ({', '.join(problem_summary['target_list'])})\n"
        f"Costo promedio de cable: {problem_summary['average_cable_cost']:.2f} metros"
    )
    print("=" * 80)

    valid_multi_results = execute_benchmark(
        problem_instance=multi_problem,
        algorithm_suite=multi_algorithms,
        enable_realtime_monitoring=True,
        verbose_output=True,
    )
    generate_report(
        performance_metrics=valid_multi_results,
        clear_screen=False
    )


    invalid_traditional_results = execute_benchmark(
        problem_instance=multi_problem,
        algorithm_suite=traditional_algorithms,
        enable_realtime_monitoring=True,
        verbose_output=True,
    )
    generate_report(
        performance_metrics=invalid_traditional_results,
        clear_screen=False
    )

    control_traditional_results = execute_benchmark(
        problem_instance=single_problem,
        algorithm_suite=traditional_algorithms,
        enable_realtime_monitoring=True,
        verbose_output=True,
    )
    generate_report(
        performance_metrics=control_traditional_results,
        clear_screen=False
    )

    return {
        "valid_multi_results": valid_multi_results,
        "invalid_traditional_results": invalid_traditional_results,
        "control_traditional_results": control_traditional_results,
    }
    

if __name__ == "__main__":
    # Ejecutar análisis
    results = execute_cable_installation_analysis()
    
    print("=" * 80)
