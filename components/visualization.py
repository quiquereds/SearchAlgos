import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import plotly.graph_objects as go
from components.utils import reconstruct_path

# Configuración de velocidad de animación
ANIMATION_SPEEDS = {
    'muy_lento': {'interval': 1000, 'duration': 800},
    'lento': {'interval': 600, 'duration': 400}, 
    'normal': {'interval': 300, 'duration': 200},
    'rapido': {'interval': 100, 'duration': 100},
    'muy_rapido': {'interval': 50, 'duration': 50}
}

def set_animation_speed(speed='normal'):
    """
    Configura la velocidad de animación.
    Args:
        speed (str): Velocidad deseada ('muy_lento', 'lento', 'normal', 'rapido', 'muy_rapido')
    Returns:
        dict: Configuración de velocidad con 'interval' y 'duration'
    """
    return ANIMATION_SPEEDS.get(speed, ANIMATION_SPEEDS['normal'])

def plot_graph_and_path(nodes, actions, costs, path=None):
    """
    Visualiza el grafo y resalta la ruta encontrada usando Plotly.
    Args:
        nodes (dict): Diccionario de nodos.
        actions (dict): Diccionario de acciones.
        costs (dict): Diccionario de costos.
        path (list): Ruta óptima (lista de nombres de nodos).
    """
    n = len(nodes)
    angle = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = {name: (np.cos(a), np.sin(a)) for name, a in zip(nodes, angle)}
    edge_x = []
    edge_y = []
    for src, adjs in actions.items():
        for action, dst in adjs.items():
            x0, y0 = pos[src]
            x1, y1 = pos[dst.name]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines')
    node_x = [pos[name][0] for name in nodes]
    node_y = [pos[name][1] for name in nodes]
    node_text = list(nodes.keys())
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition='top center',
        marker=dict(size=20, color='lightblue', line=dict(width=2, color='darkblue')),
        hoverinfo='text')
    if path:
        path_x = [pos[n][0] for n in path]
        path_y = [pos[n][1] for n in path]
        path_trace = go.Scatter(
            x=path_x, y=path_y,
            mode='lines+markers',
            line=dict(width=4, color='red'),
            marker=dict(size=24, color='red', symbol='circle-open'),
            name='Ruta óptima',
            hoverinfo='none')
        data = [edge_trace, path_trace, node_trace]
    else:
        data = [edge_trace, node_trace]
    fig = go.Figure(data=data)
    fig.update_layout(
        showlegend=False,
        title='Visualización del grafo y ruta óptima',
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        hovermode='closest',
        width=800, height=800
    )
    fig.show()

def animate_search(nodes, actions, costs, explored_steps, path=None, speed='normal'):
    """
    Anima el proceso de búsqueda mostrando los pasos explorados y la ruta final (si existe).
    Args:
        nodes (dict): Diccionario de nodos.
        actions (dict): Diccionario de acciones.
        costs (dict): Diccionario de costos.
        explored_steps (list): Lista de pares (padre, hijo) explorados en orden.
        path (list): Ruta óptima (opcional).
        speed (str): Velocidad de animación ('muy_lento', 'lento', 'normal', 'rapido', 'muy_rapido').
    """
    n = len(nodes)
    angle = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = {name: (np.cos(a), np.sin(a)) for name, a in zip(nodes, angle)}
    all_edges = set()
    for src, adjs in actions.items():
        for action, dst in adjs.items():
            all_edges.add((src, dst.name))
    frames = []
    explored_set = set()
    for i, (src, dst) in enumerate(explored_steps):
        explored_set.add((src, dst))
        edge_x = []
        edge_y = []
        for e_src, e_dst in all_edges:
            x0, y0 = pos[e_src]
            x1, y1 = pos[e_dst]
            color = 'red' if (e_src, e_dst) in explored_set else '#888'
            width = 3 if (e_src, e_dst) in explored_set else 1
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=width, color=color),
            hoverinfo='none',
            mode='lines')
        node_x = [pos[name][0] for name in nodes]
        node_y = [pos[name][1] for name in nodes]
        node_text = list(nodes.keys())
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition='top center',
            marker=dict(size=20, color='lightblue', line=dict(width=2, color='darkblue')),
            hoverinfo='text')
        data = [edge_trace, node_trace]
        if path and i == len(explored_steps) - 1:
            path_x = [pos[n][0] for n in path]
            path_y = [pos[n][1] for n in path]
            path_trace = go.Scatter(
                x=path_x, y=path_y,
                mode='lines+markers',
                line=dict(width=4, color='green'),
                marker=dict(size=24, color='green', symbol='circle-open'),
                name='Ruta óptima',
                hoverinfo='none')
            data.append(path_trace)
        frames.append(go.Frame(data=data, name=str(i)))
    edge_x = []
    edge_y = []
    for e_src, e_dst in all_edges:
        x0, y0 = pos[e_src]
        x1, y1 = pos[e_dst]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines')
    node_x = [pos[name][0] for name in nodes]
    node_y = [pos[name][1] for name in nodes]
    node_text = list(nodes.keys())
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition='top center',
        marker=dict(size=20, color='lightblue', line=dict(width=2, color='darkblue')),
        hoverinfo='text')
    data = [edge_trace, node_trace]
    
    # Obtener configuración de velocidad
    speed_config = set_animation_speed(speed)
    
    fig = go.Figure(
        data=data,
        frames=frames,
        layout=go.Layout(
            updatemenus=[dict(
                type='buttons',
                showactive=False,
                buttons=[dict(label='Play', method='animate', args=[None, {'frame': {'duration': speed_config['duration'], 'redraw': True}, 'fromcurrent': True}]),
                         dict(label='Pause', method='animate', args=[[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate', 'transition': {'duration': 0}}])]
            )],
            title='Animación del proceso de búsqueda',
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            hovermode='closest',
            width=800, height=800
        )
    )
    fig.show()

def plot_graph_and_path_nx(nodes, actions, costs, path=None):
    """
    Visualiza el grafo y resalta la ruta encontrada usando NetworkX y Matplotlib.
    Args:
        nodes (dict): Diccionario de nodos.
        actions (dict): Diccionario de acciones.
        costs (dict): Diccionario de costos.
        path (list): Ruta óptima (lista de nombres de nodos).
    """
    G = nx.DiGraph()
    for src, adjs in actions.items():
        for action, dst in adjs.items():
            G.add_edge(src, dst.name, weight=costs.get(src, {}).get(action, 1))
    pos = nx.circular_layout(G)
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='#888', node_size=800, font_size=14)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    if path:
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=3)
        nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='red', node_size=900, alpha=0.5)
    plt.title('Visualización del grafo y ruta óptima')
    plt.axis('off')
    plt.show()

def animate_search_nx(nodes, actions, costs, explored_steps, path=None, speed='normal'):
    """
    Anima el proceso de búsqueda mostrando los pasos explorados y la ruta final (si existe) usando NetworkX y Matplotlib.
    Args:
        nodes (dict): Diccionario de nodos.
        actions (dict): Diccionario de acciones.
        costs (dict): Diccionario de costos.
        explored_steps (list): Lista de pares (padre, hijo) explorados en orden.
        path (list): Ruta óptima (opcional).
        speed (str): Velocidad de animación ('muy_lento', 'lento', 'normal', 'rapido', 'muy_rapido').
    """
    G = nx.DiGraph()
    for src, adjs in actions.items():
        for action, dst in adjs.items():
            G.add_edge(src, dst.name, weight=costs.get(src, {}).get(action, 1))
    pos = nx.circular_layout(G)
    fig, ax = plt.subplots(figsize=(10, 10))
    def update(num):
        ax.clear()
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='#888', node_size=800, font_size=14, ax=ax)
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, ax=ax)
        # Dibuja aristas exploradas hasta el frame actual
        explored = explored_steps[:num+1]
        if explored:
            nx.draw_networkx_edges(G, pos, edgelist=explored, edge_color='red', width=3, ax=ax)
            nodes_in_explored = set([n for edge in explored for n in edge])
            nx.draw_networkx_nodes(G, pos, nodelist=list(nodes_in_explored), node_color='red', node_size=900, alpha=0.5, ax=ax)
        # Dibuja la ruta óptima al final
        if path and num == len(explored_steps)-1:
            path_edges = list(zip(path, path[1:]))
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='green', width=4, ax=ax)
            nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='green', node_size=900, alpha=0.5, ax=ax)
        ax.set_title(f'Animación del proceso de búsqueda (paso {num+1}/{len(explored_steps)})')
        ax.axis('off')
    # Si explored_steps está vacío, no hay animación
    if not explored_steps:
        print("No hay pasos explorados para animar.")
        return
    explored_steps = [
        (str(e[0]), str(e[1])) if isinstance(e, tuple) and len(e) == 2 else e
        for e in explored_steps
    ]
    # Asegurar que la ruta óptima se muestre al final
    if path:
        path_edges = list(zip(path, path[1:]))
        # Solo agregar los edges de la ruta óptima si no están ya al final
        if not all(edge in explored_steps for edge in path_edges):
            explored_steps.extend(path_edges)
    # Obtener configuración de velocidad
    speed_config = set_animation_speed(speed)
    
    # Puedes cambiar interval para controlar la velocidad: 
    # interval=1000 (muy lento), interval=600 (lento), interval=300 (normal), interval=100 (rápido), interval=50 (muy rápido)
    ani = animation.FuncAnimation(fig, update, frames=len(explored_steps), interval=speed_config['interval'], repeat=False)
    plt.show()

def run_and_animate(problem, algorithm, nodes, actions, costs, speed='normal'):
    """
    Ejecuta un algoritmo sobre un problema y anima el proceso de búsqueda con Plotly.
    Args:
        problem (Problem): El problema a resolver.
        algorithm (callable): Algoritmo de búsqueda (debe aceptar explored_steps).
        nodes, actions, costs: Definición del grafo.
        speed (str): Velocidad de animación ('muy_lento', 'lento', 'normal', 'rapido', 'muy_rapido').
    """
    explored_steps = []
    result = algorithm(problem, explored_steps=explored_steps)
    path = reconstruct_path(result) if result else None
    animate_search(nodes, actions, costs, explored_steps, path, speed)
    return result

def run_and_animate_nx(problem, algorithm, nodes, actions, costs, speed='normal'):
    """
    Ejecuta un algoritmo sobre un problema y anima el proceso de búsqueda con NetworkX y Matplotlib.
    Args:
        problem (Problem): El problema a resolver.
        algorithm (callable): Algoritmo de búsqueda (debe aceptar explored_steps).
        nodes, actions, costs: Definición del grafo.
        speed (str): Velocidad de animación ('muy_lento', 'lento', 'normal', 'rapido', 'muy_rapido').
    """
    explored_steps = []
    result = algorithm(problem, explored_steps=explored_steps)
    path = reconstruct_path(result) if result else None
    animate_search_nx(nodes, actions, costs, explored_steps, path, speed)
    return result
