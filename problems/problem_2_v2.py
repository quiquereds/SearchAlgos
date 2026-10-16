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

Nodos Objetivo: 0, 1, 7, 15, 22, 23, 24
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
        self.network_nodes = {str(node_id): State(str(node_id)) for node_id in range(0, 62)}
        self.target_nodes = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28']
        self.network_topology = [
                ('0', '6', 4.8),
                ('0', '17', 6.2),
                ('0', '7', 3.8),
                ('0', '2', 5.2),
                ('1', '6', 3.5),
                ('2', '0', 5.2),
                ('2', '7', 4.7),
                ('3', '8', 4.8),
                ('4', '9', 5.2),
                ('5', '13', 4.2),
                ('6', '0', 4.8),
                ('6', '16', 4.5),
                ('7', '0', 3.8),
                ('7', '17', 3.2),
                ('7', '8', 3.4),
                ('8', '3', 4.8),
                ('8', '18', 3.2),
                ('8', '9', 4.1),
                ('9', '4', 5.2),
                ('9', '19', 3.2),
                ('9', '10', 2.5),
                ('10', '20', 3.2),
                ('10', '21', 5.8),
                ('10', '11', 3.8),
                ('11', '12', 3.5),
                ('12', '22', 3.5),
                ('12', '13', 3.1),
                ('13', '5', 4.2),
                ('13', '23', 3.5),
                ('14', '21', 3.8),
                ('15', '16', 4.3),
                ('16', '38', 4.5),
                ('16', '29', 5.5),
                ('17', '0', 6.2),
                ('17', '24', 3.3),
                ('17', '30', 3.8),
                ('17', '18', 3.5),
                ('18', '19', 4.2),
                ('19', '25', 3.7),
                ('19', '20', 2.5),
                ('21', '22', 3.7),
                ('21', '26', 3.7),
                ('22', '27', 3.5),
                ('22', '23', 3.2),
                ('23', '28', 4.2),
                ('23', '37', 3.6),
                ('29', '30', 3.4),
                ('30', '43', 5.9),
                ('30', '31', 4.5),
                ('31', '44', 5.3),
                ('31', '32', 3.9),
                ('32', '34', 3.8),
                ('32', '33', 4.2),
                ('34', '35', 3.5),
                ('35', '46', 5.8),
                ('36', '37', 3.7),
                ('37', '48', 5.6),
                ('38', '49', 5.2),
                ('39', '43', 3.6),
                ('40', '45', 3.5),
                ('41', '48', 4.2),
                ('42', '43', 3.9),
                ('43', '50', 3.5),
                ('43', '44', 4.3),
                ('44', '51', 3.6),
                ('44', '45', 4.5),
                ('45', '46', 3.5),
                ('45', '57', 4.2),
                ('46', '58', 4.3),
                ('46', '47', 3.8),
                ('47', '59', 3.7),
                ('47', '48', 3.2),
                ('48', '60', 3.7),
                ('49', '53', 4.8),
                ('49', '50', 5.2),
                ('50', '54', 4.2),
                ('50', '51', 4.4),
                ('52', '59', 3.8),
                ('54', '55', 3.9),
                ('56', '57', 3.8),
                ('57', '58', 3.5),
                ('58', '59', 4.2),
                ('59', '60', 3.2),
                ('60', '61', 2.5),
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
            initial=self.network_nodes['1'],
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
        
    def create_single_objective_problem(self, target_node: str = "7"):
        """Problema de control para algoritmos tradicionales (objetivo único)."""
        single_heuristic = {}
        for node_name in self.network_nodes:
            single_heuristic[node_name] = {
                target_node: abs(int(node_name) - int(target_node)) * 0.5
            }

        return Problem(
            initial=self.network_nodes["1"],
            targets=[self.network_nodes[target_node]],
            actions=self.actions,
            costs=self.costs,
            heuristic=single_heuristic
        )

def execute_cable_installation_analysis():
    """Ejecuta comparación válida + demostración de caso no válido."""
    cable_problem = CableInstallationProblem()
    multi_problem = cable_problem.create_multi_objective_problem()
    single_problem = cable_problem.create_single_objective_problem("7")
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
