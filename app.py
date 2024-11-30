import streamlit as st
import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point
import matplotlib.pyplot as plt
import pandas as pd

# Import utility functions (assumed to be implemented in utils.py)
from utils import (
    find_intersection_points, 
    decompose_line, 
    find_all_cycles_from_start, 
    calculate_perimeter, 
    unique_cycles, 
    calculate_perimeter_missing, 
    are_cycles_equal, 
    plot_cycles_with_perimeters
)



# Streamlit app
def app():
    st.title("Cycle Perimeter Calculator")
    
    # Example input: lines for graph edges
    lines = [((1, 1), (1, 2)),
         ((1, 2), (1.8, 2)),
         ((1.8, 2), (1.8, 1)),
         ((1.8, 1), (1, 1)),
         ((1.2, 1), (1.2, 2)),
         ((1.6, 1), (1.6, 2)),
         ((1, 1.5), (1.6, 1.8)),
         ((1, 1.3), (1.8, 1.7)),
         ((1.2, 1.2), (1.8, 1.5)),]
    st.write("Problem : Finding all the closed cycles in the set of lines ")
    st.write("Set of lines ")
    st.write_stream(lines)
    
    # Create the graph from the initial lines
    Gn = nx.Graph()
    for line in lines:
        Gn.add_edge(line[0], line[1])
    
    # Plot the graph from the set of lines 
    st.title("Inital Graph")
    st.write("Here is the visualization of how the set of lines looks")
    pos = {node: node for node in Gn.nodes()}  
    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw(
        Gn, pos, with_labels=True, node_size=500, 
        node_color="lightblue", ax=ax
    )
    st.pyplot(fig)  
    
    # Finding all the verticies and edges
    st.write("Finding all the vertices that are made from the set of lines ")
    st.write("Logic : ")
    lines = [LineString([edge[0], edge[1]]) for edge in lines]
    edges = []
    final_edge_array = set()

    for nodes in Gn.nodes():                                                                            #Iterator over all the nodes
        start = nodes                                                                                       # Visit each neighbor
        decomposed_edges = []                                                                               # Initalize emply list to store decomposed edges
        for neighbor in Gn.neighbors(start):                                                                # Iterateover all the neightbors 
            main_line = LineString([start, neighbor])                                                           # Line from current postion to the adjacent neighbor
            intersection_points = find_intersection_points(main_line, lines)                                    # Find intersections with all other lines (returns only unique values)
            decomposed_lines = decompose_line(main_line, intersection_points)                                   # Decompose the main line based on intersection points that lie on the line
            decomposed_edges.append([list(segment.coords) for segment in decomposed_lines])                     # Add decomposed lines to decomposed_edges  
        for edge_group in decomposed_edges:                                                                 # Add decomposed edges to final_edge_array
            for edge in edge_group:                                                                             #For each detected edge
                sorted_edge = tuple(sorted(edge))                                                                   # Normalize edge and sort the coordinates)
                final_edge_array.add(sorted_edge)                                                                   
        edges.append(decomposed_edges)
    final_edge_array = [list(edge) for edge in final_edge_array]                                        # Convert the unique edges from the set back to a list of lists of coordinates

    # Create the second graph with all the vertices in the figure 
    G = nx.Graph()
    for line in final_edge_array:
        G.add_edge(line[0], line[1])
    
    st.title("Graph with all Detected Vertices")
    st.write("Here is the visualization of how the set of lines looks after all the vertices have been discovered ")
    pos = {node: node for node in G.nodes()}  
    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw(
        G, pos, with_labels=True, node_size=500, 
        node_color="lightblue", ax=ax
    )
    st.pyplot(fig)  

    # Calculating the cyles in the set of line 

    #Calculating the master cycle which stores information about all the closed cyles created while travelling from vetrice to vertice
    master_cycle = []
    for nodes in list(G.nodes()): 
        start = nodes                                                                       #(1.0, 1.3)  # Starting node for cycle search
        cycles = find_all_cycles_from_start(G, start)                                       # Find all cycles starting from the 'start' node
        master_cycle.append(cycles)

    #Definfing a min perimter cycle that stores the cycles having minimum pertimter from the master cycle array
    min_perimeter_cycles = []                                                               # Note : Master cycle is a 3D list that contains combination of cycle arrays from each vertice 

    for cycle_list in master_cycle:                                                         # Loop through each set of cycles in master_cycle (3D list)
        min_perimeter = float('inf')                                                        # Start with a very high value for comparison
        min_cycle = None                                                                    # To store the cycle with the minimum perimeter in this set
        
        for cycle in cycle_list:                                                            # Iterate over each cycle in the cycle_list
            try:
                perimeter = calculate_perimeter(cycle)                                      # Calculate the perimeter for the current cycle
                if perimeter < min_perimeter:                                               # If this perimeter is smaller than the current min
                    min_perimeter = perimeter                                               # Update the minimum perimeter
                    min_cycle = cycle                                                       # Update the cycle with the minimum perimeter
            except ValueError as e:
                print(f"Error in cycle {cycle}: {e}")

        if min_cycle is not None:                                                           # After processing the cycle_list, store the cycle with the minimum perimeter
            min_perimeter_cycles.append(min_cycle)
        else:
            print("No valid cycles found in this set.")

    unique_cycles_list = unique_cycles(min_perimeter_cycles)                                # Get the unique cycles from the min_perimeter_cycles list

    # Flatten the cycles and create edges from them
    cycle_edges = set()
    for cycle in unique_cycles_list:
        # Create edges for each pair of consecutive points in the cycle
        for i in range(len(cycle) - 1):
            cycle_edges.add(frozenset([cycle[i], cycle[i+1]]))
        # Add edge between the last and first point to close the cycle
        cycle_edges.add(frozenset([cycle[-1], cycle[0]]))

    # Create a set of edges from the graph G
    graph_edges = set(frozenset(edge) for edge in G.edges())

    # Find missing edges that are in the graph but not in the unique cycles
    missing_edges = graph_edges - cycle_edges

    # Initialize the master_missing_array to collect relevant cycles
    master_missing_array = []

    # Loop through the master_cycle array
    for val in master_cycle:
        relevant_cycles = []
        for cycle in val:
            # Create the set of edges for the current cycle
            cycle_edges = set(
                frozenset([cycle[i], cycle[i + 1]]) for i in range(len(cycle) - 1)
            )
            cycle_edges.add(frozenset([cycle[-1], cycle[0]]))  # Close the cycle

            # Check if any missing edge is in the cycle
            if any(edge in cycle_edges for edge in missing_edges):
                relevant_cycles.append(cycle)

        # Append the relevant cycles to the master_missing_array
        master_missing_array.append(relevant_cycles)

    # Create the figure of all the smallest closed cycles found by traversing in the figure vertice by vertice
    st.title("Smallest Cycles from all the vertices ")
    fig, ax = plt.subplots(figsize=(10, 8))  # Adjust figure size if needed

    # Assign a unique color for each cycle
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_cycles_list)))  # Generate a color map

    # Plot each cycle
    for idx, cycle in enumerate(unique_cycles_list):
        cycle = np.array(cycle)
        ax.plot(cycle[:, 0], cycle[:, 1], marker='o', color=colors[idx], label=f'Cycle {idx+1}')

    # Customize the plot
    ax.set_title('Unique Cycles with Minimum Perimeter')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.grid(True)
    ax.legend(loc='upper right')

    # Display the plot in Streamlit
    st.pyplot(fig)
    st.write(" We can see that there are missing cycles from certain vertices so we will find the cycles with the smallest perimeter from the master cycle array that has the missing line segment in it, then find the smallest perimeter in the returned set of cycles")

    cycle_edges = set()
    for cycle in unique_cycles_list:
        # Create edges for each pair of consecutive points in the cycle
        for i in range(len(cycle) - 1):
            cycle_edges.add(frozenset([cycle[i], cycle[i+1]]))
        # Add edge between the last and first point to close the cycle
        cycle_edges.add(frozenset([cycle[-1], cycle[0]]))

    # Create a set of edges from the graph G
    graph_edges = set(frozenset(edge) for edge in G.edges())

    # Find missing edges that are in the graph but not in the unique cycles
    missing_edges = graph_edges - cycle_edges

    # Initialize the master_missing_array to collect relevant cycles
    master_missing_array = []

    # Loop through the master_cycle array
    for val in master_cycle:
        relevant_cycles = []
        for cycle in val:
            # Create the set of edges for the current cycle
            cycle_edges = set(
                frozenset([cycle[i], cycle[i + 1]]) for i in range(len(cycle) - 1)
            )
            cycle_edges.add(frozenset([cycle[-1], cycle[0]]))  # Close the cycle

            # Check if any missing edge is in the cycle
            if any(edge in cycle_edges for edge in missing_edges):
                relevant_cycles.append(cycle)

        # Append the relevant cycles to the master_missing_array
        master_missing_array.append(relevant_cycles)

    # From the master_missing_array find the cycles that has a line btween the points in each missing edge
    # Initialize a dictionary to store cycles corresponding to each missing edge
    cycles_with_missing_edges = {edge: [] for edge in missing_edges}

    # Loop through the missing edges
    for missing_edge in missing_edges:
        for relevant_cycles in master_missing_array:
            for cycle in relevant_cycles:
                # Create the set of edges for the current cycle
                cycle_edges = set(
                    frozenset([cycle[i], cycle[i + 1]]) for i in range(len(cycle) - 1)
                )
                cycle_edges.add(frozenset([cycle[-1], cycle[0]]))  # Close the cycle

                # Check if the missing edge is in the cycle
                if missing_edge in cycle_edges:
                    cycles_with_missing_edges[missing_edge].append(cycle)

    # Find the cycle with the least perimeter for each missing edge
    least_perimeter_cycles = {}

    for missing_edge, cycles in cycles_with_missing_edges.items():
        min_perimeter = float('inf')
        best_cycle = None

        for cycle in cycles:
            # Calculate the perimeter of the current cycle
            perimeter = calculate_perimeter_missing(cycle)
            if perimeter < min_perimeter:
                min_perimeter = perimeter
                best_cycle = cycle

        # Store the cycle with the least perimeter
        least_perimeter_cycles[missing_edge] = {
            "cycle": best_cycle,
            "perimeter": min_perimeter,
        }

    # Update the unique_cycles_list with calculated perimeters if not present
    unique_cycles_with_perimeters = []
    for cycle in unique_cycles_list:
        # Calculate the perimeter for each cycle in unique_cycles_list
        perimeter = calculate_perimeter(cycle)
        unique_cycles_with_perimeters.append({'cycle': cycle, 'perimeter': perimeter})
      
    # Prepare the least perimeter cycles data in the desired format
    least_perimeter_cycles_data = [{'cycle': data['cycle'], 'perimeter': data['perimeter']} for data in least_perimeter_cycles.values()]

    # Create the plot

    all_cycles_with_perimeter = unique_cycles_with_perimeters + least_perimeter_cycles_data
    df = pd.DataFrame(all_cycles_with_perimeter)
    # Initialize an empty dictionary to hold unique cycles grouped by perimeter
    unique_cycles_by_perimeter = {}

    # Iterate through the groups in the DataFrame
    for perimeter, group in df.groupby('perimeter'):
        unique_cycles_missing = []

        # Iterate through each row in the group (each cycle)
        for _, row in group.iterrows():
            cycle = row['cycle']
            is_unique = True

            # Compare with already added unique cycles
            for unique_cycle in unique_cycles_missing:
                if are_cycles_equal(cycle, unique_cycle['cycle']):  # Compare 'cycle' part of the dict
                    is_unique = False
                    break

            if is_unique:
                unique_cycles_missing.append({'cycle': cycle, 'perimeter': perimeter})  # Add cycle with perimeter as a dict

        # Add the list of unique cycles for this perimeter to the dictionary
        unique_cycles_by_perimeter[perimeter] = unique_cycles_missing
        
    st.dataframe(df)
    # Convert the dictionary to a list of dictionaries as per your requirement
    output_list = []
    for perimeter, cycles in unique_cycles_by_perimeter.items():
        for cycle_dict in cycles:
            output_list.append(cycle_dict)

    st.write(len(output_list))
    st.title("All Cycles with Perimeters")
    fig, ax = plt.subplots(figsize=(10, 8))

    # Assign a unique color for each cycle
    colors = plt.cm.jet(np.linspace(0, 1, len(output_list)))

    # Plot each cycle
    for idx, cycle_data in enumerate(output_list):
        cycle = np.array(cycle_data['cycle'])
        perimeter = cycle_data['perimeter']
        color = colors[idx]  # Get a unique color for the current cycle

        # Close the cycle by adding the first point at the end
        cycle_closed = np.vstack([cycle, cycle[0]])

        # Plot the cycle with the corresponding color
        ax.plot(cycle_closed[:, 0], cycle_closed[:, 1], marker='o', color=color, label=f'Cycle {idx+1}')

        # Add perimeter label at the center of the cycle
        centroid = np.mean(cycle, axis=0)  # Get the centroid of the cycle to place the label
        ax.text(centroid[0], centroid[1], f'{perimeter:.2f}', color=color, fontsize=10, ha='center', va='center')

    # Customize the plot
    ax.set_title('All Cycles with Perimeters')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.grid(True)
    ax.legend(loc='upper right')

    # Display the plot in Streamlit
    st.pyplot(fig)


# Run the Streamlit app
if __name__ == "__main__":
    app()
