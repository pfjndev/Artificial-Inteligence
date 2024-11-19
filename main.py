import heapq
import matplotlib.pyplot as plt
import networkx as nx

# Romania Map Data
romania_map = nx.Graph()

# Cities and distances
edges = [
    ("Arad", "Zerind", 75),
    ("Arad", "Sibiu", 140),
    ("Arad", "Timisoara", 118),
    ("Zerind", "Oradea", 71),
    ("Oradea", "Sibiu", 151),
    ("Sibiu", "Fagaras", 99),
    ("Sibiu", "Rimnicu Vilcea", 80),
    ("Rimnicu Vilcea", "Pitesti", 97),
    ("Rimnicu Vilcea", "Craiova", 146),
    ("Fagaras", "Bucharest", 211),
    ("Pitesti", "Bucharest", 101),
    ("Craiova", "Pitesti", 138),
    ("Timisoara", "Lugoj", 111),
    ("Lugoj", "Mehadia", 70),
    ("Mehadia", "Drobeta", 75),
    ("Drobeta", "Craiova", 120),
    ("Bucharest", "Giurgiu", 90),
    ("Bucharest", "Urziceni", 85),
    ("Urziceni", "Hirsova", 98),
    ("Hirsova", "Eforie", 86),
    ("Urziceni", "Vaslui", 142),
    ("Vaslui", "Iasi", 92),
    ("Iasi", "Neamt", 87),
]

romania_map.add_weighted_edges_from(edges)

# Node positions for visualization
positions = {
    "Arad": (0, 4),
    "Zerind": (-1, 5),
    "Oradea": (-2, 6),
    "Sibiu": (1, 5),
    "Fagaras": (3, 5),
    "Rimnicu Vilcea": (2, 4),
    "Pitesti": (3, 3),
    "Timisoara": (-1, 3),
    "Lugoj": (-2, 2),
    "Mehadia": (-3, 1),
    "Drobeta": (-4, 0),
    "Craiova": (-2, 0),
    "Bucharest": (4, 2),
    "Giurgiu": (4, 1),
    "Urziceni": (5, 3),
    "Hirsova": (6, 4),
    "Eforie": (7, 3),
    "Vaslui": (6, 5),
    "Iasi": (5, 6),
    "Neamt": (4, 7),
}

# Heuristic values (straight-line distances to Bucharest)
heuristic = {
    "Arad": 366, "Zerind": 374, "Oradea": 380, "Sibiu": 253, "Fagaras": 178,
    "Rimnicu Vilcea": 193, "Pitesti": 100, "Timisoara": 329, "Lugoj": 244,
    "Mehadia": 241, "Drobeta": 242, "Craiova": 160, "Bucharest": 0,
    "Giurgiu": 77, "Urziceni": 80, "Hirsova": 151, "Eforie": 161,
    "Vaslui": 199, "Iasi": 226, "Neamt": 234,
}

########################################### Search Algorithms ###########################################

# A* Search Algorithm
def astar_search(graph, start, goal, heuristic):
    frontier = []
    heapq.heappush(frontier, (0 + heuristic[start], start, 0, [start]))
    explored = set()
    steps = []
    node_counter = 1  # To number the nodes as they are expanded

    while frontier:
        f, current, g, path = heapq.heappop(frontier)
        steps.append((node_counter, current, g, f, list(frontier)))
        node_counter += 1

        if current in explored:
            continue
        explored.add(current)

        if current == goal:
            return path, g, steps, True  # Path found, optimal

        for neighbor in graph.neighbors(current):
            if neighbor not in explored:
                new_g = g + graph[current][neighbor]["weight"]
                new_f = new_g + heuristic[neighbor]
                heapq.heappush(frontier, (new_f, neighbor, new_g, path + [neighbor]))

    return None, float('inf'), steps, False  # No solution found

# Greedy Best-First Search Algorithm
def greedy_best_first_search(graph, start, goal, heuristic):
    frontier = []
    heapq.heappush(frontier, (heuristic[start], start, [start]))
    explored = set()
    steps = []
    node_counter = 1

    while frontier:
        h, current, path = heapq.heappop(frontier)
        steps.append((node_counter, current, h, list(frontier)))
        node_counter += 1

        if current in explored:
            continue
        explored.add(current)

        if current == goal:
            return path, sum_path_cost(graph, path), steps, False  # Greedy doesn't guarantee optimal

        for neighbor in graph.neighbors(current):
            if neighbor not in explored:
                heapq.heappush(frontier, (heuristic[neighbor], neighbor, path + [neighbor]))

    return None, float('inf'), steps, False

# Uniform Cost Search Algorithm
def uniform_cost_search(graph, start, goal):
    frontier = []
    heapq.heappush(frontier, (0, start, [start]))
    explored = set()
    steps = []
    node_counter = 1

    while frontier:
        cost, current, path = heapq.heappop(frontier)
        steps.append((node_counter, current, cost, list(frontier)))
        node_counter += 1

        if current in explored:
            continue
        explored.add(current)

        if current == goal:
            return path, cost, steps, True  # Path found, optimal

        for neighbor in graph.neighbors(current):
            if neighbor not in explored:
                new_cost = cost + graph[current][neighbor]["weight"]
                heapq.heappush(frontier, (new_cost, neighbor, path + [neighbor]))

    return None, float('inf'), steps, False

# Breadth-First Search Algorithm
def breadth_first_search(graph, start, goal):
    frontier = [(start, [start])]
    explored = set()
    steps = []
    node_counter = 1

    while frontier:
        current, path = frontier.pop(0)
        steps.append((node_counter, current, list(frontier)))
        node_counter += 1

        if current in explored:
            continue
        explored.add(current)

        if current == goal:
            return path, sum_path_cost(graph, path), steps, True  # Path found

        for neighbor in graph.neighbors(current):
            if neighbor not in explored:
                frontier.append((neighbor, path + [neighbor]))

    return None, float('inf'), steps, False

# Helper function to calculate path cost
def sum_path_cost(graph, path):
    return sum(graph[path[i]][path[i+1]]["weight"] for i in range(len(path)-1))

########################################### Visualization Function ###########################################

def visualize_search_tree(graph, steps, final_path, title, algorithm_type):
    plt.figure(figsize=(14, 10))
    nx.draw_networkx_edges(graph, pos=positions, edge_color="gray", alpha=0.5)
    
    # Draw edge labels (distances)
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos=positions, edge_labels=edge_labels, font_size=8, label_pos=0.3)

    # Draw nodes
    nx.draw_networkx_nodes(graph, pos=positions, node_size=700, node_color="lightblue")
    
    # Prepare node labels with heuristic values
    node_labels = {node: f"{node}\nh={heuristic[node]}" for node in graph.nodes()}
    nx.draw_networkx_labels(graph, pos=positions, labels=node_labels, font_size=9, font_color="black", verticalalignment='bottom')

    # Highlight final path
    if final_path:
        path_edges = list(zip(final_path, final_path[1:]))
        nx.draw_networkx_edges(
            graph, pos=positions, edgelist=path_edges, edge_color="red", width=2.5, label='Final Path'
        )

    # Annotate steps
    for step in steps:
        node_number, node, *rest = step
        x, y = positions[node]
        plt.text(x, y + 0.3, f"#{node_number}", fontsize=8, ha="center", color="black")

    plt.title(title, fontsize=16)
    plt.axis("off")
    plt.legend()
    plt.show()

########################################### CLI and Main Function ###########################################

def display_results(algorithm_name, path, cost, steps, is_optimal):
    print(f"\n============================ {algorithm_name} ============================")
    print("Algorithm Steps:")
    for step in steps:
        if algorithm_name == "A* Search":

            node_number, node, g, f, frontier = step
            frontier_contents = [f"{n[1]}(g={n[2]}, f={n[3]})" for n in frontier]
            print(f"Step {node_number}: Node={node}, g={g}, f={f}, Frontier={frontier_contents}\n")

        elif algorithm_name == "Greedy Best-First Search":
            
            node_number, node, h, frontier = step            
            frontier_contents = [f"{n[1]}(h={n[0]})" for n in frontier]
            print(f"Step {node_number}: Node={node}, h={h}, Frontier={frontier_contents}\n")
        
        elif algorithm_name == "Uniform Cost Search":
            
            node_number, node, cost, frontier = step            
            frontier_contents = [f"{n[0]}(cost={n[1]})" for n in frontier]
            print(f"Step {node_number}: Node={node}, Cost={cost}, Frontier={frontier_contents}\n")

        elif algorithm_name == "Breadth-First Search":
            
            node_number, node, frontier = step            
            frontier_contents = [f"{n[0]}(path={n[1]})" for n in frontier]
            print(f"Step {node_number}: Node={node}, Frontier={frontier_contents}\n")
    
    print("\nFinal Solution:")
    if path:
        print(f"Path: {' → '.join(path)}")
        print(f"Cost: {cost} km")
        if is_optimal:
            print("The solution is optimal.")
        else:
            print("The solution is not guaranteed to be optimal.")
    else:
        print("No path found.")

def main():
    algorithms = {
        "1": "A* Search",
        "2": "Greedy Best-First Search",
        "3": "Uniform Cost Search",
        "4": "Breadth-First Search"
    }

    while True:
        print("\nSelect algorithms to run (separated by commas):")
        for key, name in algorithms.items():
            print(f"{key}. {name}")
        print("5. Exit")
        choices = input("Enter choices (e.g., 1,3): ").split(',')

        selected = [choice.strip() for choice in choices]
        if "5" in selected:
            print("Exiting the program.")
            break

        for choice in selected:
            if choice not in algorithms:
                print(f"Invalid choice: {choice}. Skipping.")
                continue

            algorithm_name = algorithms[choice]
            if algorithm_name == "A* Search":
                path, cost, steps, is_optimal = astar_search(romania_map, "Arad", "Bucharest", heuristic)
                display_results(algorithm_name, path, cost, steps, is_optimal)
                visualize_search_tree(romania_map, steps, path, f"{algorithm_name} (Arad to Bucharest)", algorithm_name)

            elif algorithm_name == "Greedy Best-First Search":
                path, cost, steps, is_optimal = greedy_best_first_search(romania_map, "Arad", "Bucharest", heuristic)
                display_results(algorithm_name, path, cost, steps, is_optimal)
                visualize_search_tree(romania_map, steps, path, f"{algorithm_name} (Arad to Bucharest)", algorithm_name)

            elif algorithm_name == "Uniform Cost Search":
                path, cost, steps, is_optimal = uniform_cost_search(romania_map, "Arad", "Bucharest")
                display_results(algorithm_name, path, cost, steps, is_optimal)
                visualize_search_tree(romania_map, steps, path, f"{algorithm_name} (Arad to Bucharest)", algorithm_name)

            elif algorithm_name == "Breadth-First Search":
                path, cost, steps, is_optimal = breadth_first_search(romania_map, "Arad", "Bucharest")
                display_results(algorithm_name, path, cost, steps, is_optimal)
                visualize_search_tree(romania_map, steps, path, f"{algorithm_name} (Arad to Bucharest)", algorithm_name)

        print("\nAll selected algorithms have been executed.")

if __name__ == "__main__":
    main()
