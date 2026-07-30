import json
import os
import networkx as nx
import matplotlib.pyplot as plt

def build_task_dependency_graph(task_json_str):
    """
    Build and visualize dependency graph from atomic tasks JSON string
    
    Args:
        task_json_str (str): JSON string containing atomic tasks with dependencies
        
    Returns:
        nx.DiGraph: NetworkX directed graph representing task dependencies
    """
    # Parse JSON data
    data = json.loads(task_json_str)
    atomic_tasks = data['atomic_tasks']
    
    # Extract task IDs and build dependency edges
    task_ids = [task['task_id'] for task in atomic_tasks]
    edges = []
    
    # Create dependency edges (from dependency to dependent task)
    for task in atomic_tasks:
        task_id = task['task_id']
        dependencies = task['dependencies']
        
        for dep in dependencies:
            edges.append((dep, task_id))
    
    # Build directed graph
    graph = nx.DiGraph()
    graph.add_nodes_from(task_ids)
    graph.add_edges_from(edges)
    
    return graph

def has_cycle(graph):
    """
    Check if the dependency graph contains cycles
    
    Args:
        graph (nx.DiGraph): Directed graph to check
        
    Returns:
        bool: True if cycle exists, False otherwise
    """
    try:
        cycles = list(nx.find_cycle(graph, orientation="original"))
        if cycles:
            print("Cycle detected in dependency graph:", cycles)
            return True
        return False
    except nx.NetworkXNoCycle:
        return False

def draw_task_dependency_graph(graph, task_json_str=None, layout_type='dot', save_path=None, show_time=2):
    """
    Draw task dependency graph using different Graphviz layouts
    
    Args:
        graph (nx.DiGraph): Dependency graph
        task_json_str (str, optional): JSON string to extract task names
        layout_type (str): Graphviz layout type ('dot', 'neato', 'fdp', 'sfdp', 'circo', 'twopi')
        save_path (str, optional): Directory path to save the graph image
        show_time (float): Time in seconds to display the graph before auto-closing (0 to disable auto-close)
    
    Returns:
        str: Path of the saved image file (if save_path is provided)
    """
    plt.figure(figsize=(14, 10))
    
    # Try different layout methods
    pos = None
    layout_used = "fallback"
    
    try:
        # Method 1: Use pygraphviz (recommended)
        pos = nx.nx_agraph.graphviz_layout(graph, prog=layout_type)
        layout_used = f"pygraphviz_{layout_type}"
    except ImportError:
        try:
            # Method 2: Use pydot
            pos = nx.nx_pydot.graphviz_layout(graph, prog=layout_type)
            layout_used = f"pydot_{layout_type}"
        except ImportError:
            print("Warning: Both pygraphviz and pydot are unavailable, using fallback layout")
            
    # Fallback layout options
    if pos is None:
        if layout_type == 'dot':
            # Manual hierarchical layout implementation
            pos = create_hierarchical_layout(graph)
            layout_used = "manual_hierarchical"
        else:
            # Use NetworkX built-in layout
            pos = nx.spring_layout(graph, k=3, iterations=50)
            layout_used = "spring_layout"
    
    # Create labels
    labels = {}
    if task_json_str:
        data = json.loads(task_json_str)
        atomic_tasks = data['atomic_tasks']
        for task in atomic_tasks:
            task_id = task['task_id']
            task_name = task['name']
            # Truncate long names for better display
            if len(task_name) > 25:
                task_name = task_name[:22] + "..."
            labels[task_id] = f"{task_id}\n{task_name}"
    else:
        labels = {node: node for node in graph.nodes()}
    
    # Draw the graph
    nx.draw(graph, pos, 
            labels=labels,
            with_labels=True, 
            node_color="lightblue", 
            node_size=2500, 
            font_size=10, 
            font_weight='bold', 
            arrows=True,
            arrowsize=20,
            edge_color="gray",
            node_shape="o",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.title(f"Task Dependency Graph - Layout: {layout_used}", fontsize=16, fontweight='bold')
    
    # Save the graph if save_path is provided
    saved_file_path = None
    if save_path:
        saved_file_path = _save_graph_with_unique_name(save_path)
        plt.savefig(saved_file_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')

    # if show_time > 0:
    #     # Use matplotlib's built-in timer for thread-safe auto-close
    #     plt.show(block=False)
    #     current_fig = plt.gcf()
        
    #     def safe_close():
    #         try:
    #             if plt.fignum_exists(current_fig.number):
    #                 plt.close(current_fig)
    #         except Exception:
    #             pass 
        
    #     # Use matplotlib's timer (thread-safe)
    #     timer = current_fig.canvas.new_timer(interval=int(show_time * 1000))
    #     timer.add_callback(safe_close)
    #     timer.single_shot = True
    #     timer.start()

    #     import atexit
    #     atexit.register(lambda: timer.stop() if hasattr(timer, 'stop') else None)
    # else:
    #     plt.show()
    
    return saved_file_path

def _save_graph_with_unique_name(save_path):
    """
    Generate a unique filename to prevent overwriting existing files
    
    Args:
        save_path (str): Directory path to save the file
        
    Returns:
        str: Full path of the unique filename
    """
    # Ensure the directory exists
    os.makedirs(save_path, exist_ok=True)
    
    # Generate base filename with timestamp
    base_filename = f"dependency_graph"
    
    # Check for existing files and add counter if needed
    counter = 0
    while True:
        if counter == 0:
            filename = f"{base_filename}.png"
        else:
            filename = f"{base_filename}_{counter}.png"
        
        full_path = os.path.join(save_path, filename)
        if not os.path.exists(full_path):
            return full_path
        counter += 1

def create_hierarchical_layout(graph):
    """
    Create a manual hierarchical layout for dependency graphs
    
    Args:
        graph (nx.DiGraph): Directed graph
        
    Returns:
        dict: Position dictionary for nodes
    """
    pos = {}
    if len(graph.nodes()) == 0:
        return pos
    
    # Calculate execution levels for positioning
    remaining_nodes = set(graph.nodes())
    execution_levels = []
    
    while remaining_nodes:
        current_level = []
        for node in remaining_nodes:
            predecessors = set(graph.predecessors(node))
            if not predecessors.intersection(remaining_nodes):
                current_level.append(node)
        execution_levels.append(sorted(current_level))
        remaining_nodes -= set(current_level)
    
    # Position nodes based on levels
    y_spacing = 2.0
    for level_idx, level in enumerate(execution_levels):
        y_pos = len(execution_levels) - level_idx - 1  # Top to bottom
        x_spacing = 3.0 if len(level) > 1 else 0
        for node_idx, node in enumerate(level):
            x_pos = (node_idx - (len(level) - 1) / 2) * x_spacing
            pos[node] = (x_pos, y_pos * y_spacing)
    
    return pos

def analyze_task_dependencies(task_json_str, draw_graph=False, save_path=None, 
                            layout_type='dot', show_time=2):
    """
    Complete analysis of task dependencies including graph building, cycle detection, and optional visualization
    
    Args:
        task_json_str (str): JSON string containing atomic tasks
        draw_graph (bool): Whether to display the dependency graph visualization
        save_path (str, optional): Directory path to save the graph image
        layout_type (str): Graphviz layout type for the graph
        show_time (float): Time in seconds to display the graph before auto-closing (0 to disable auto-close)
        
    Returns:
        dict: Analysis results including graph, topological order, cycle status, and saved image path
    """
    # Build dependency graph
    graph = build_task_dependency_graph(task_json_str)
    
    # Check for cycles
    has_cycles = has_cycle(graph)
    
    # Get topological order and analyze execution levels
    topo_order = None
    execution_levels = None
    if not has_cycles:
        topo_order = list(nx.topological_sort(graph))
        
        # Calculate execution levels (tasks that can run in parallel)
        execution_levels = []
        remaining_nodes = set(graph.nodes())
        
        while remaining_nodes:
            # Find nodes with no dependencies among remaining nodes
            current_level = []
            for node in remaining_nodes:
                predecessors = set(graph.predecessors(node))
                if not predecessors.intersection(remaining_nodes):
                    current_level.append(node)
            
            execution_levels.append(sorted(current_level))
            remaining_nodes -= set(current_level)
        
        print("Possible topological order:", ", ".join(topo_order))
        print("Execution levels (parallel execution possible):")
        for i, level in enumerate(execution_levels):
            if len(level) == 1:
                print(f"  Level {i+1}: {level[0]}")
            else:
                print(f"  Level {i+1}: {' || '.join(level)} (can run in parallel)")
    else:
        print("Cannot determine execution order due to cycles in dependencies")
    
    # Visualize graph if requested
    saved_image_path = None
    if draw_graph:
        saved_image_path = draw_task_dependency_graph(
            graph, task_json_str, layout_type, save_path, show_time
        )
    
    return {
        'graph': graph,
        'has_cycles': has_cycles,
        'topological_order': topo_order,
        'execution_levels': execution_levels,
        'num_tasks': len(graph.nodes()),
        'num_dependencies': len(graph.edges()),
        'saved_image_path': saved_image_path
    }