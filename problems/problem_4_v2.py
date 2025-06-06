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
from components.benchmark_v2 import execute_comprehensive_benchmark, generate_comprehensive_report
from components.problem import Problem
from components.state import State


class CableInstallationProblem:
    """Clase para gestionar el problema de instalación de cables"""
    
    def __init__(self):
        self.network_nodes = {str(node_id): State(str(node_id)) for node_id in range(0, 38)}
        self.target_nodes = ['0', '1', '7', '15', '22', '23', '24']
        self.network_topology = [
                ('0', '1', 2),
                ('0', '20', 3.8),
                ('1', '7', 3.8),
                ('1', '2', 2.8),
                ('2', '8', 4),
                ('3', '9', 3.65),
                ('4', '10', 3.4),
                ('5', '11', 3.55),
                ('6', '8', 2.5),
                ('7', '15', 3.7),
                ('8', '9', 2.1),
                ('8', '15', 1.5),
                ('9', '3', 3.65),
                ('9', '10', 2.3),
                ('9', '16', 3.5),
                ('9', '22', 2.2),
                ('10', '4', 3.4),
                ('10', '11', 2.4),
                ('10', '17', 3.65),
                ('11', '5', 3.55),
                ('11', '14', 3.2),
                ('11', '18', 3.6),
                ('11', '23', 4.3),
                ('12', '23', 2.5),
                ('13', '23', 2.8),
                ('14', '19', 3.45),
                ('15', '22', 2.8),
                ('16', '9', 3.5),
                ('17', '10', 3.65),
                ('18', '11', 3.6),
                ('19', '25', 4.2),
                ('20', '21', 2.85),
                ('21', '22', 3.6),
                ('21', '33', 5.2),
                ('22', '27', 3.85),
                ('22', '32', 6),
                ('22', '30', 4.5),
                ('23', '24', 3.1),
                ('23', '28', 3.85),
                ('25', '29', 2),
                ('26', '30', 3.6),
                ('28', '32', 3.6),
                ('29', '32', 3),
                ('30', '31', 3),
                ('30', '36', 3.5),
                ('31', '32', 3.2),
                ('31', '34', 2.5),
                ('32', '37', 4.1),
                ('32', '34', 3.5),
                ('33', '35', 2),
                ('35', '36', 3),
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


def execute_cable_installation_analysis():
    """Función principal para ejecutar el análisis integral del problema"""
    
    # Configuración del problema
    cable_problem = CableInstallationProblem()
    problem_instance = cable_problem.create_multi_objective_problem()
    problem_summary = cable_problem.get_problem_summary()
    
    # Configuración de algoritmos de optimización
    optimization_algorithms = {
        'A* (A-Star)': astar,
        'Uniform Cost Search Multi-Goal': multi_ucs,
        'Dijkstra Multi-Objetivo': dijkstra_multi_goal
    }
    
    # Información del problema
    problem_description = (
        f"Optimización del uso de cable en residenciales\n"
        f"Nodos en la red doméstica: {problem_summary['total_nodes']}\n"
        f"Conexiones disponibles: {problem_summary['total_connections']}\n"
        f"Puntos objetivo a conectar: {problem_summary['target_nodes']} ({', '.join(problem_summary['target_list'])})\n"
        f"Costo promedio de cable: {problem_summary['average_cable_cost']:.2f} metros"
    )
    
    print("🏗️  ANÁLISIS")
    print("=" * 80)
    print(problem_description)
    print("=" * 80)
    print("\n🔧 Iniciando evaluación de algoritmos de optimización...")
    
    # Ejecutar benchmark integral
    performance_results = execute_comprehensive_benchmark(
        problem_instance=problem_instance,
        algorithm_suite=optimization_algorithms,
        enable_realtime_monitoring=True,
        verbose_output=True
    )
    
    # Generar reporte profesional
    generate_comprehensive_report(
        performance_metrics=performance_results,
        problem_description="Optimización del uso de cable en residenciales"
    )
    
    # Información técnica adicional
    print("\n\n📋 INFORMACIÓN TÉCNICA DETALLADA")
    print("=" * 80)
    print("🔹 Algoritmos evaluados:")
    print("   • A*: Búsqueda heurística con función de evaluación f(n) = g(n) + h(n)")
    print("   • UCS Multi-Goal: Búsqueda de costo uniforme adaptada para múltiples objetivos")
    print("   • Dijkstra Multi-Objetivo: Algoritmo de Dijkstra modificado para problemas multi-objetivo")
    print("\n🔹 Optimizaciones implementadas:")
    print("   • Cálculo de costo real de cables únicos (sin duplicar costos de regreso)")
    print("   • Seguimiento de nodos explorados para análisis de eficiencia")
    print("   • Monitoreo de memoria en tiempo real")
    print("   • Heurística optimizada para distancias en grafo")
    print("\n🔹 Métricas de evaluación:")
    print("   • Tiempo de ejecución")
    print("   • Costo de la solución (metros de cable)")
    print("   • Eficiencia (nodos explorados/segundo)")
    print("   • Uso de memoria (pico máximo)")
    print("   • Longitud del camino de solución")
    
    return performance_results


if __name__ == "__main__":
    # Ejecutar análisis
    results = execute_cable_installation_analysis()
    
    print("=" * 80)
