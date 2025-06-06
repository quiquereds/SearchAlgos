"""
Problema 5: Optimización de Rutas en Red de Distribución - Análisis Profesional
===============================================================================

Descripción del Problema:
Este problema modela una red de distribución compleja con 61 nodos interconectados.
El objetivo es encontrar rutas óptimas tanto para objetivos únicos como múltiples,
simulando diferentes escenarios de logística y distribución.

Características del Sistema:
- Red de gran escala: 61 nodos numerados del 1 al 61
- Conexiones ponderadas que representan distancias/costos de transporte
- Análisis comparativo entre algoritmos de objetivo único y múltiple
- Optimización para minimizar costos totales de distribución

Escenarios de Análisis:
1. Objetivo Único: Ruta del nodo 1 al nodo 60
2. Objetivo Múltiple: Conectar múltiples puntos de distribución estratégicos

Nodos Objetivo Múltiple: 1, 2, 3, 4, 5, 6, 14, 15, 24, 25, 26, 27, 28, 50
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
        self.network_nodes = {str(node_id): State(str(node_id)) for node_id in range(1, 62)}
        self.target_nodes = ['1', '2', '3', '4', '5', '6', '14', '15', '24', '25', '26', '27', '28', '50']
        self.network_topology = [
                ('1', '6', 3.5),
                ('2', '50', 5.2),
                ('2', '7', 4.7),
                ('3', '8', 4.8),
                ('4', '9', 5.2),
                ('5', '13', 4.2),
                ('6', '50', 4.8),
                ('6', '16', 4.5),
                ('7', '50', 3.8),
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
                ('17', '50', 6.2),
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
