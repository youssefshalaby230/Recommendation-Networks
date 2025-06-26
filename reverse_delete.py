import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import numpy as np
import time
import os
from collections import deque

def load_graph_from_csv(filename):
    """Load a weighted, undirected graph from a CSV or edge-list file."""
    if not os.path.exists(filename):
        print(f"File {filename} not found.")
        return None
    
    G = nx.Graph()
    chunk_size = 100000  # For large files like rec-movielens-user-movies-10m.csv
    
    # Check if file is an edge-list (MovieLens) or CSV (Amazon)
    with open(filename, 'r') as f:
        first_line = f.readline().strip()
        is_edge_list = first_line.startswith('%') or filename.endswith('movielens.csv')
    
    try:
        if is_edge_list:
            # Handle MovieLens edge-list format (e.g., edge_id user_id movie_id weight timestamp or user_id movie_id rating)
            with open(filename, 'r') as f:
                lines = [line.strip() for line in f if not line.startswith('%')]
            edges = []
            for line in lines:
                parts = line.split()
                if len(parts) >= 4:  # edge_id user_id movie_id weight [timestamp]
                    user_id, movie_id, weight = parts[1], parts[2], parts[3]
                    edges.append([user_id, movie_id, float(weight)])
                elif len(parts) == 3:  # user_id movie_id rating (e.g., rec-movielens.csv)
                    user_id, movie_id, weight = parts[0], parts[1], parts[2]
                    edges.append([user_id, movie_id, float(weight)])
                elif len(parts) == 2:  # user_id movie_id (assume weight=1.0)
                    user_id, movie_id = parts[0], parts[1]
                    edges.append([user_id, movie_id, 1.0])
            # Process in chunks for large files
            for i in range(0, len(edges), chunk_size):
                chunk = edges[i:i+chunk_size]
                for edge in chunk:
                    G.add_edge(edge[0], edge[1], weight=edge[2])
        else:
            # Handle Amazon CSV format (e.g., user_id, item_id, rating, timestamp)
            df = pd.read_csv(filename, header=None, names=['user_id', 'item_id', 'rating', 'timestamp'], 
                             usecols=[0, 1, 2], nrows=5)
            print(f"Detected columns in {filename}: {df.columns.tolist()}")
            
            for chunk in pd.read_csv(filename, header=None, names=['user_id', 'item_id', 'rating', 'timestamp'], 
                                    usecols=[0, 1, 2], chunksize=chunk_size):
                for _, row in chunk.iterrows():
                    G.add_edge(row['user_id'], row['item_id'], weight=float(row['rating']))
                
    except Exception as e:
        print(f"Error parsing {filename}: {e}")
        return None
    
    # Use largest connected component if graph is not connected
    if G.number_of_nodes() == 0:
        print(f"Graph from {filename} is empty.")
        return None
    if not nx.is_connected(G):
        print(f"Graph from {filename} is not connected. Using largest component.")
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    
    return G

def is_connected_bfs(G):
    """Check if graph is connected using BFS (non-recursive)."""
    if not G.nodes:
        return False
    
    visited = set()
    queue = deque([next(iter(G.nodes))])
    visited.add(queue[0])
    
    while queue:
        node = queue.popleft()
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return len(visited) == len(G.nodes)

def reverse_delete_mst(G):
    """Implement Reverse-Delete Algorithm for MST."""
    mst = G.copy()
    edges = sorted(mst.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
    total_cost = sum(data['weight'] for _, _, data in mst.edges(data=True))
    mst_edges = []
    cost_evolution = [total_cost]
    
    for i, (u, v, data) in enumerate(edges):
        if i % 100 == 0:  # Progress indicator for large graphs
            print(f"Processing edge {i}/{len(edges)}")
        
        mst.remove_edge(u, v)
        if is_connected_bfs(mst):
            total_cost -= data['weight']
        else:
            mst.add_edge(u, v, weight=data['weight'])
            mst_edges.append((u, v, data['weight']))
        cost_evolution.append(total_cost)
    
    return mst, mst_edges, cost_evolution

def animate_mst_construction(G, mst_edges, output_file):
    """Create animation of MST construction (Output: MP4 video)."""
    if len(G.nodes) > 100:
        print(f"Skipping animation for large graph ({len(G.nodes)} nodes)")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)  # Fixed seed for consistent layout
    
    def update(frame):
        ax.clear()
        nx.draw(G, pos, ax=ax, node_size=50, node_color='lightblue', 
                edge_color='gray', alpha=0.3, with_labels=False)
        current_edges = [(u, v) for u, v, _ in mst_edges[:frame+1]]
        nx.draw_networkx_edges(G, pos, edgelist=current_edges, ax=ax, 
                              edge_color='red', width=2)
        ax.set_title(f"Reverse-Delete Step {frame}: Building MST")
    
    ani = FuncAnimation(fig, update, frames=len(mst_edges), interval=500, repeat=False)
    
    try:
        ani.save(output_file, writer='ffmpeg')
        print(f"Animation saved as {output_file}")
    except Exception as e:
        print(f"FFmpeg failed: {e}")
        gif_file = output_file.replace('.mp4', '.gif')
        try:
            ani.save(gif_file, writer='pillow')
            print(f"Animation saved as GIF: {gif_file}")
        except Exception as e2:
            print(f"Animation saving failed: {e2}")
    
    plt.close()

def plot_computational_cost(datasets, node_counts, edge_counts, times):
    """Plot computational cost growth (Output: reverse_delete_cost_growth.png)."""
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.scatter(node_counts, times, color='blue')
    plt.plot(node_counts, times, 'b--', alpha=0.7)
    plt.xlabel('Number of Nodes')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time vs Nodes')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.scatter(edge_counts, times, color='red')
    plt.plot(edge_counts, times, 'r--', alpha=0.7)
    plt.xlabel('Number of Edges')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time vs Edges')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.bar(range(len(datasets)), times, color='green')
    plt.xlabel('Dataset')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time by Dataset')
    plt.xticks(range(len(datasets)), [os.path.basename(d)[:15] + '...' if len(os.path.basename(d)) > 15 else os.path.basename(d) for d in datasets], rotation=45)
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    efficiency = [t/e if e > 0 else 0 for t, e in zip(times, edge_counts)]
    plt.bar(range(len(datasets)), efficiency, color='orange')
    plt.xlabel('Dataset')
    plt.ylabel('Time per Edge (seconds)')
    plt.title('Algorithm Efficiency')
    plt.xticks(range(len(datasets)), [os.path.basename(d)[:15] + '...' if len(os.path.basename(d)) > 15 else os.path.basename(d) for d in datasets], rotation=45)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('mst_videos/reverse_delete_cost_growth.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Specify your CSV files here
    datasets = [
        r"C:\Users\yyous\Downloads\rec-amz-Video-Games.csv",
        r"C:\Users\yyous\Downloads\rec-movielens-tag-movies-10m.csv",
        r"C:\Users\yyous\Downloads\rec-movielens-user-movies-10m.csv",
        r"C:\Users\yyous\Downloads\rec-movielens.csv"
    ]
    
    # Filter to only existing files
    existing_datasets = [d for d in datasets if os.path.exists(d)]
    if not existing_datasets:
        print("No CSV files found. Please check file names and paths.")
        return
    
    print(f"Found {len(existing_datasets)} datasets: {[os.path.basename(d) for d in existing_datasets]}")
    
    output_dir = "mst_videos"
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for dataset in existing_datasets:
        print(f"\n{'='*50}")
        print(f"Processing {os.path.basename(dataset)}...")
        print(f"{'='*50}")
        
        # Load graph
        G = load_graph_from_csv(dataset)
        if G is None:
            print(f"Skipping {dataset} due to loading error.")
            continue
        
        node_count = G.number_of_nodes()
        edge_count = G.number_of_edges()
        
        print(f"Loaded graph: {node_count} nodes, {edge_count} edges")
        
        # Skip very large graphs to avoid memory issues
        if edge_count > 1000000:
            print(f"Skipping {dataset} - too large ({edge_count} edges)")
            continue
        
        # Compute MST
        start_time = time.time()
        mst, mst_edges, cost_evolution = reverse_delete_mst(G)
        execution_time = time.time() - start_time
        
        # Store results
        results.append({
            'dataset': os.path.basename(dataset),
            'node_count': node_count,
            'edge_count': edge_count,
            'execution_time': execution_time,
            'mst_edges': len(mst_edges),
            'total_cost': cost_evolution[-1] if cost_evolution else 0
        })
        
        # Output 1: Console Output
        print(f"Results for {os.path.basename(dataset)}:")
        print(f"  Nodes: {node_count}")
        print(f"  Edges: {edge_count}")
        print(f"  MST Edges: {len(mst_edges)}")
        print(f"  Total MST Cost: {cost_evolution[-1] if cost_evolution else 0}")
        print(f"  Execution Time: {execution_time:.4f} seconds")
        
        # Output 2: MST Visualization Video (for small graphs)
        if edge_count < 1000:
            output_file = os.path.join(output_dir, f"mst_reverse_delete_{os.path.basename(dataset).replace('.csv', '')}.mp4")
            animate_mst_construction(G, mst_edges, output_file)
        
        # Output 3: Cost Evolution Plot
        if cost_evolution:
            plt.figure(figsize=(10, 6))
            plt.plot(cost_evolution, label="Total Cost", linewidth=2)
            plt.xlabel("Algorithm Step")
            plt.ylabel("Total Edge Weight")
            plt.title(f"MST Cost Evolution (Reverse-Delete) - {os.path.basename(dataset)}")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f"cost_evolution_{os.path.basename(dataset).replace('.csv', '')}.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    # Output 4: Results CSV and Computational Cost Plot
    if results:
        node_counts = [r['node_count'] for r in results]
        edge_counts = [r['edge_count'] for r in results]
        times = [r['execution_time'] for r in results]
        dataset_names = [r['dataset'] for r in results]
        
        # Generate computational cost plot
        plot_computational_cost(dataset_names, node_counts, edge_counts, times)
        
        # Save results to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(output_dir, "mst_results.csv"), index=False)
        
        # Output 5: Summary Table in Console
        print(f"\n{'='*50}")
        print("SUMMARY OF ALL DATASETS")
        print(f"{'='*50}")
        print(results_df.to_string(index=False))
        print(f"\nResults saved to: {os.path.join(output_dir, 'mst_results.csv')}")
        print(f"Computational cost plot saved to: {os.path.join(output_dir, 'reverse_delete_cost_growth.png')}")
        
    else:
        print("No datasets were processed successfully.")

if __name__ == "__main__":
    main()