import time
import os
import sys
import threading
from typing import Callable, Dict, List, Optional
from dataclasses import dataclass
import psutil
import tracemalloc
from components.utils import reconstruct_path
from components.problem import Problem


@dataclass
class PerformanceMetrics:
    """Métricas de rendimiento para análisis de algoritmos"""
    algorithm_name: str
    execution_successful: bool
    execution_time_seconds: float
    solution_cost: float
    actual_cable_cost: float
    solution_path: str
    nodes_explored_count: int
    peak_memory_mb: float
    path_length: int
    convergence_iterations: Optional[int] = None
    error_description: Optional[str] = None


class AlgorithmMonitor:
    """Monitor para seguimiento de algoritmos en tiempo real"""
    
    def __init__(self, algorithm_name: str, verbose: bool = True):
        self.algorithm_name = algorithm_name
        self.start_timestamp = time.perf_counter()
        self.nodes_processed = 0
        self.current_best_solution_cost = float('inf')
        self.is_monitoring_active = True
        self.verbose = verbose
        self.monitor_thread = None
        self.update_interval = 0.5  # segundos
    
    def initiate_monitoring(self):
        """Inicia el monitoreo en tiempo real del algoritmo"""
        if self.verbose:
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
    
    def _monitoring_loop(self):
        """Bucle principal del monitor que muestra progreso periódicamente"""
        while self.is_monitoring_active:
            elapsed_time = time.perf_counter() - self.start_timestamp
            progress_info = (
                f"\r🔍 {self.algorithm_name}: "
                f"{elapsed_time:.1f}s | "
                f"Nodos procesados: {self.nodes_processed:,} "
            )
            print(progress_info, end="", flush=True)
            time.sleep(self.update_interval)
    
    def update_algorithm_progress(self, nodes_count: int, current_solution_cost: float = None):
        """Actualiza las métricas de progreso del algoritmo"""
        self.nodes_processed = nodes_count
        if current_solution_cost is not None and current_solution_cost < self.current_best_solution_cost:
            self.current_best_solution_cost = current_solution_cost
    
    def terminate_monitoring(self):
        """Finaliza el monitoreo y limpia la salida"""
        self.is_monitoring_active = False
        if self.monitor_thread and self.verbose:
            self.monitor_thread.join(timeout=0.1)
            print()  # Nueva línea para limpiar el output


def clear_terminal_screen():
    """Limpia la pantalla del terminal de manera multiplataforma"""
    if os.name == 'nt':  # Windows
        os.system('cls')
    else:  # Unix/Linux/macOS
        os.system('clear')


def format_execution_time(seconds: float) -> str:
    """Formatea el tiempo de ejecución de manera profesional y legible"""
    if seconds < 1e-6:
        return f"{seconds * 1e9:.0f}ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.1f}μs"
    elif seconds < 1:
        return f"{seconds * 1e3:.2f}ms"
    elif seconds < 60:
        return f"{seconds:.4f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.2f}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        remaining_seconds = seconds % 60
        return f"{hours}h {remaining_minutes}m {remaining_seconds:.1f}s"


def format_memory_usage(bytes_value: float) -> str:
    """Formatea el uso de memoria de manera legible y precisa"""
    if bytes_value < 1024:
        return f"{bytes_value:.0f}B"
    elif bytes_value < 1024**2:
        return f"{bytes_value/1024:.1f}KB"
    elif bytes_value < 1024**3:
        return f"{bytes_value/(1024**2):.2f}MB"
    else:
        return f"{bytes_value/(1024**3):.2f}GB"


def execute_comprehensive_benchmark(
    problem_instance: Problem,
    algorithm_suite: Dict[str, Callable[[Problem], object]],
    enable_realtime_monitoring: bool = True,
    verbose_output: bool = True
) -> List[PerformanceMetrics]:
    """
    Ejecuta un benchmark integral con monitoreo avanzado y métricas detalladas
    
    Args:
        problem_instance: Instancia del problema a resolver
        algorithm_suite: Diccionario de algoritmos a evaluar
        enable_realtime_monitoring: Habilitar monitoreo en tiempo real
        verbose_output: Mostrar salida detallada
    
    Returns:
        Lista de métricas de rendimiento para cada algoritmo
    """
    performance_results = []
    
    if verbose_output:
        print("INICIANDO EVALUACIÓN DE ALGORITMOS [...]")
        print("=" * 70)
        print(f"📊 Algoritmos a evaluar: {len(algorithm_suite)}")
        print(f"🎯 Nodos objetivo: {len(problem_instance.targets)}")
        print("=" * 70)
    
    for algorithm_index, (algorithm_name, algorithm_function) in enumerate(algorithm_suite.items(), 1):
        if verbose_output:
            print(f"\n[{algorithm_index:02d}/{len(algorithm_suite):02d}] Evaluando: {algorithm_name}")
            print("-" * 50)
        
        # Configuración del monitoreo de memoria
        tracemalloc.start()
        
        # Inicialización del monitor de progreso
        progress_monitor = AlgorithmMonitor(algorithm_name, enable_realtime_monitoring) 
        if enable_realtime_monitoring:
            progress_monitor.initiate_monitoring()
        
        try:
            # Medición precisa del tiempo de ejecución
            execution_start_time = time.perf_counter()
            
            # Definir callback para progreso si el algoritmo lo soporta
            def progress_update_callback(processed_nodes: int, current_cost: float):
                if progress_monitor:
                    progress_monitor.update_algorithm_progress(processed_nodes, current_cost)
            
            # Ejecutar el algoritmo con manejo de compatibilidad
            try:
                solution_node = algorithm_function(problem_instance, progress_callback=progress_update_callback)
            except TypeError:
                # Algoritmo no compatible con callback de progreso
                solution_node = algorithm_function(problem_instance)
            
            execution_end_time = time.perf_counter()
            total_execution_time = execution_end_time - execution_start_time
            
            # Finalizar monitoreo
            if progress_monitor:
                progress_monitor.terminate_monitoring()
            
            # Obtener métricas de memoria
            current_memory, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Procesar resultados
            if solution_node:
                solution_path = reconstruct_path(solution_node)
                solution_cost = solution_node.cost
                formatted_route = " → ".join(solution_path)
                
                # Calcular costo real de cables para problemas multi-objetivo
                actual_cable_cost = solution_cost
                if hasattr(solution_node, 'visited') and len(problem_instance.targets) > 1:
                    from components.utils import calculate_cable_cost
                    actual_cable_cost = solution_cost  # Ya calculado previamente
                
                metrics = PerformanceMetrics(
                    algorithm_name=algorithm_name,
                    execution_successful=True,
                    execution_time_seconds=total_execution_time,
                    solution_cost=solution_cost,
                    actual_cable_cost=actual_cable_cost,
                    solution_path=formatted_route,
                    nodes_explored_count=getattr(progress_monitor, 'nodes_processed', 0),
                    peak_memory_mb=peak_memory / (1024 * 1024),
                    path_length=len(solution_path),
                    convergence_iterations=None
                )
                
                if verbose_output:
                    print(f"✅ Ejecución completada")
                    print(f"   ⏱️  Tiempo: {format_execution_time(total_execution_time)}")
                    print(f"   💰 Costo de la solución: {solution_cost:.3f}")
                    print(f"   🛤️  Longitud del camino: {len(solution_path)} nodos")
                    print(f"   💾 Memoria pico usada: {format_memory_usage(peak_memory)}")
                    if metrics.nodes_explored_count > 0:
                        print(f"   🔍 Nodos explorados: {metrics.nodes_explored_count:,}")
                
            else:
                metrics = PerformanceMetrics(
                    algorithm_name=algorithm_name,
                    execution_successful=False,
                    execution_time_seconds=total_execution_time,
                    solution_cost=float('inf'),
                    actual_cable_cost=float('inf'),
                    solution_path="Sin solución encontrada",
                    nodes_explored_count=getattr(progress_monitor, 'nodes_processed', 0),
                    peak_memory_mb=peak_memory / (1024 * 1024),
                    path_length=0,
                    error_description="Algoritmo no encontró solución válida"
                )
                
                if verbose_output:
                    print(f"❌ Sin solución en {format_execution_time(total_execution_time)}")
                
        except Exception as algorithm_exception:
            if progress_monitor:
                progress_monitor.terminate_monitoring()
            tracemalloc.stop()
            
            metrics = PerformanceMetrics(
                algorithm_name=algorithm_name,
                execution_successful=False,
                execution_time_seconds=0,
                solution_cost=float('inf'),
                actual_cable_cost=float('inf'),
                solution_path="Error en ejecución",
                nodes_explored_count=0,
                peak_memory_mb=0,
                path_length=0,
                error_description=str(algorithm_exception)
            )
            
            if verbose_output:
                print(f"💥 Error en ejecución: {str(algorithm_exception)}")
        
        performance_results.append(metrics)
        
        # Pausa breve para mejor visualización
        if verbose_output:
            time.sleep(0.2)
    
    return performance_results


def print_enhanced_results(results: List[PerformanceMetrics], problem_name: str = ""):
    """
    Imprime los resultados de manera clara y formateada
    """
    clear_terminal_screen()
    
    print("🎯 RESULTADOS DEL BENCHMARK")
    if problem_name:
        print(f"📋 Problema: {problem_name}")
    print("=" * 80)
    
    # Separar exitosos y fallidos
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    if successful:
        # Ordenar por costo y luego por tiempo
        successful.sort(key=lambda x: (x.cost, x.execution_time))
        
        print("\n✅ ALGORITMOS EXITOSOS:")
        print("-" * 80)
        
        for i, result in enumerate(successful, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
            
            print(f"{medal} {result.algorithm}")
            print(f"   ⏱️  Tiempo: {format_execution_time(result.execution_time)}")
            print(f"   💰 Costo: {result.cost:.2f}")
            print(f"   🛤️  Ruta ({result.path_length} nodos): {result.route}")
            print(f"   💾 Memoria pico: {format_memory_usage(result.memory_peak * 1024 * 1024)}")
            if result.nodes_explored > 0:
                print(f"   🔍 Nodos explorados: {result.nodes_explored}")
            print()
    
    if failed:
        print("❌ ALGORITMOS FALLIDOS:")
        print("-" * 40)
        for result in failed:
            print(f"   {result.algorithm}: {result.error_message or 'Sin solución'}")
        print()
    
    # Estadísticas generales
    if successful:
        best_cost = min(r.cost for r in successful)
        fastest_time = min(r.execution_time for r in successful)
        avg_time = sum(r.execution_time for r in successful) / len(successful)
        
        print("📊 ESTADÍSTICAS:")
        print("-" * 40)
        print(f"   🏆 Mejor costo: {best_cost:.2f}")
        print(f"   ⚡ Tiempo más rápido: {format_execution_time(fastest_time)}")
        print(f"   📈 Tiempo promedio: {format_execution_time(avg_time)}")
        
        # Encontrar el más eficiente (mejor balance costo-tiempo)
        best_efficiency = min(successful, key=lambda x: x.cost + x.execution_time * 10)
        print(f"   🎯 Más eficiente: {best_efficiency.algorithm}")


def run_problem_comparison(problems: Dict[str, Problem], algorithms: Dict[str, Callable]):
    """
    Ejecuta comparación entre múltiples problemas
    """
    all_results = {}
    
    for prob_name, problem in problems.items():
        print(f"\n🧩 Procesando: {prob_name}")
        results = execute_comprehensive_benchmark(problem, algorithms)
        all_results[prob_name] = results
        
        # Mostrar resultados inmediatamente
        print_enhanced_results(results, prob_name)
        input("\nPresiona Enter para continuar...")
    
    return all_results


def generate_comprehensive_report(
    performance_metrics: List[PerformanceMetrics], 
    problem_description: str = "",
    export_format: str = "console"
) -> None:
    """
    Genera un reporte integral y profesional de los resultados del benchmark
    
    Args:
        performance_metrics: Lista de métricas de rendimiento
        problem_description: Descripción del problema evaluado
        export_format: Formato de salida ("console", "markdown", "latex")
    """
    clear_terminal_screen()
    
    # Encabezado principal
    print("📊 REPORTE INTEGRAL DE EVALUACIÓN DE ALGORITMOS")
    print("=" * 80)
    
    if problem_description:
        print(f"🎯 Problema: {problem_description}")
        print("-" * 80)
    
    # Separar resultados exitosos y fallidos
    successful_algorithms = [m for m in performance_metrics if m.execution_successful]
    failed_algorithms = [m for m in performance_metrics if not m.execution_successful]
    
    # Análisis de algoritmos exitosos
    if successful_algorithms:
        # Ordenar por costo de solución y tiempo de ejecución
        successful_algorithms.sort(key=lambda x: (x.solution_cost, x.execution_time_seconds))
        
        print("\n✅ ANÁLISIS DE ALGORITMOS EXITOSOS")
        print("=" * 80)
        
        for rank, metrics in enumerate(successful_algorithms, 1):
            # Determinar medallería
            if rank == 1:
                medal = "🥇 PRIMER LUGAR"
            elif rank == 2:
                medal = "🥈 SEGUNDO LUGAR"
            elif rank == 3:
                medal = "🥉 TERCER LUGAR"
            else:
                medal = f"🏅 PUESTO #{rank}"
            
            print(f"\n{medal}: {metrics.algorithm_name}")
            print("─" * 60)
            print(f"  📈 Métricas de Rendimiento:")
            print(f"     • Tiempo de ejecución: {format_execution_time(metrics.execution_time_seconds)}")
            print(f"     • Costo de la solución: {metrics.solution_cost:.4f} unidades")
            print(f"     • Longitud del camino: {metrics.path_length} nodos")
            print(f"     • Memoria máxima utilizada: {format_memory_usage(metrics.peak_memory_mb * 1024 * 1024)}")
            
            if metrics.nodes_explored_count > 0:
                efficiency = metrics.nodes_explored_count / metrics.execution_time_seconds if metrics.execution_time_seconds > 0 else 0
                print(f"     • Nodos explorados: {metrics.nodes_explored_count:,}")
                print(f"     • Eficiencia de exploración: {efficiency:.0f} nodos/segundo")
            
            print(f"  🛤️  Ruta de la solución:")
            print(f"     {metrics.solution_path}")
    
    # Análisis de algoritmos fallidos
    if failed_algorithms:
        print(f"\n\n❌ ALGORITMOS SIN SOLUCIÓN VÁLIDA ({len(failed_algorithms)})")
        print("=" * 80)
        for metrics in failed_algorithms:
            print(f"  • {metrics.algorithm_name}: {metrics.error_description or 'Sin solución encontrada'}")
    
    # Estadísticas comparativas
    if len(successful_algorithms) > 1:
        print(f"\n\n📊 ANÁLISIS ESTADÍSTICO COMPARATIVO")
        print("=" * 80)
        
        # Métricas de costos
        costs = [m.solution_cost for m in successful_algorithms]
        best_cost = min(costs)
        worst_cost = max(costs)
        avg_cost = sum(costs) / len(costs)
        
        print(f"  💰 Análisis de Costos:")
        print(f"     • Mejor costo: {best_cost:.4f}")
        print(f"     • Peor costo: {worst_cost:.4f}")
        print(f"     • Costo promedio: {avg_cost:.4f}")
        print(f"     • Variación: {((worst_cost - best_cost) / best_cost * 100):.1f}%")
        
        # Métricas de tiempo
        times = [m.execution_time_seconds for m in successful_algorithms]
        fastest_time = min(times)
        slowest_time = max(times)
        avg_time = sum(times) / len(times)
        
        print(f"\n  ⏱️  Análisis de Tiempos:")
        print(f"     • Más rápido: {format_execution_time(fastest_time)}")
        print(f"     • Más lento: {format_execution_time(slowest_time)}")
        print(f"     • Tiempo promedio: {format_execution_time(avg_time)}")
        
        # Métricas de memoria
        memories = [m.peak_memory_mb for m in successful_algorithms]
        min_memory = min(memories)
        max_memory = max(memories)
        avg_memory = sum(memories) / len(memories)
        
        print(f"\n  💾 Análisis de Memoria:")
        print(f"     • Menor uso: {format_memory_usage(min_memory * 1024 * 1024)}")
        print(f"     • Mayor uso: {format_memory_usage(max_memory * 1024 * 1024)}")
        print(f"     • Uso promedio: {format_memory_usage(avg_memory * 1024 * 1024)}")
        
        # Algoritmo más equilibrado (mejor balance costo-tiempo-memoria)
        def calculate_efficiency_score(metrics):
            # Normalizar métricas (menor es mejor)
            norm_cost = metrics.solution_cost / best_cost
            norm_time = metrics.execution_time_seconds / fastest_time
            norm_memory = metrics.peak_memory_mb / min_memory
            return norm_cost + norm_time + norm_memory
        
        most_balanced = min(successful_algorithms, key=calculate_efficiency_score)
        
        print(f"\n  🎯 Algoritmo Más Equilibrado:")
        print(f"     • {most_balanced.algorithm_name}")
        print(f"     • Puntaje de eficiencia: {calculate_efficiency_score(most_balanced):.2f}")
    
    # Recomendaciones
    print(f"\n\n💡 RECOMENDACIONES DE USO")
    print("=" * 80)
    
    if successful_algorithms:
        best_cost_algo = min(successful_algorithms, key=lambda x: x.solution_cost)
        fastest_algo = min(successful_algorithms, key=lambda x: x.execution_time_seconds)
        most_memory_efficient = min(successful_algorithms, key=lambda x: x.peak_memory_mb)
        
        print(f"  • Para MEJOR COSTO: {best_cost_algo.algorithm_name}")
        print(f"  • Para MAYOR VELOCIDAD: {fastest_algo.algorithm_name}")
        print(f"  • Para MENOR USO DE MEMORIA: {most_memory_efficient.algorithm_name}")
        
        if len(successful_algorithms) > 1:
            most_balanced = min(successful_algorithms, key=calculate_efficiency_score)
            print(f"  • Para USO GENERAL: {most_balanced.algorithm_name}")


def execute_problem_comparison_suite(
    problem_collection: Dict[str, Problem], 
    algorithm_collection: Dict[str, Callable],
    detailed_analysis: bool = True
) -> Dict[str, List[PerformanceMetrics]]:
    """
    Ejecuta comparación integral entre múltiples problemas
    
    Args:
        problem_collection: Diccionario de problemas a evaluar
        algorithm_collection: Diccionario de algoritmos a probar
        detailed_analysis: Mostrar análisis detallado por problema
    
    Returns:
        Diccionario con resultados por problema
    """
    comprehensive_results = {}
    
    print("🧪 SUITE INTEGRAL DE EVALUACIÓN MULTI-PROBLEMA")
    print("=" * 80)
    print(f"📊 Problemas a evaluar: {len(problem_collection)}")
    print(f"🔬 Algoritmos por problema: {len(algorithm_collection)}")
    print("=" * 80)
    
    for problem_index, (problem_name, problem_instance) in enumerate(problem_collection.items(), 1):
        print(f"\n🧩 [{problem_index}/{len(problem_collection)}] EVALUANDO: {problem_name}")
        print("─" * 70)
        
        # Ejecutar benchmark para este problema
        problem_results = execute_comprehensive_benchmark(
            problem_instance, 
            algorithm_collection,
            enable_realtime_monitoring=True,
            verbose_output=True
        )
        
        comprehensive_results[problem_name] = problem_results
        
        # Mostrar resultados inmediatamente si se solicita
        if detailed_analysis:
            generate_comprehensive_report(problem_results, problem_name)
            print("\n" + "="*60)
            input("📋 Presiona ENTER para continuar con el siguiente problema...")
    
    return comprehensive_results
