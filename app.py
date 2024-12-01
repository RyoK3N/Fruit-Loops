import streamlit as st
import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point
import matplotlib.pyplot as plt
import pandas as pd

# Import utility functions (implemented in utils.py)
from utils import (
    find_intersection_points, 
    decompose_line, 
    find_all_cycles_from_start, 
    calculate_perimeter, 
    unique_cycles, 
    calculate_perimeter_missing, 
    are_cycles_equal
)

# Utility function for precision control
def set_precision(value, precision=4):
    """
    Round the value to the specified precision.
    Handles both single values and lists/tuples of tuples.
    """
    if isinstance(value, (list, tuple)):
        return [tuple(np.round(point, precision)) for point in value]
    elif isinstance(value, np.ndarray):
        return np.round(value, precision)
    return round(value, precision)

def app():
    st.title("Cycle Perimeter Calculator with Fixed Precision")

    st.write("**Problem:** Finding all the closed cycles in the set of lines.")

    # User input for lines
    st.write("### Input Lines")
    st.write("Please input the set of lines as a list of coordinate pairs.")
    st.write("**Example:** [((1,1), (1,2)), ((1,2), (1.8,2)), ...]")

    lines_input = st.text_area(
        "Input lines",
        value=str(
            [
                ((1, 1), (1, 2)),
                ((1, 2), (1.8, 2)),
                ((1.8, 2), (1.8, 1)),
                ((1.8, 1), (1, 1)),
                ((1.2, 1), (1.2, 2)),
                ((1.6, 1), (1.6, 2)),
                ((1, 1.5), (1.6, 1.8)),
                ((1, 1.3), (1.8, 1.7)),
                ((1.2, 1.2), (1.8, 1.5)),
            ]
        ),
        height=200,
    )

    # Parse and validate user input
    try:
        lines = eval(lines_input)
        if not isinstance(lines, list):
            raise ValueError("Input must be a list of coordinate pairs.")
        for line in lines:
            if not (
                isinstance(line, tuple)
                and len(line) == 2
                and all(isinstance(point, tuple) and len(point) == 2 for point in line)
            ):
                raise ValueError("Each line must be a tuple of two coordinate tuples.")
    except Exception as e:
        st.error(f"**Error parsing lines input:** {e}")
        st.stop()

    # Round all input coordinates to 4 decimal places
    lines = [
        (
            (set_precision(pt1[0], 4), set_precision(pt1[1], 4)),
            (set_precision(pt2[0], 4), set_precision(pt2[1], 4))
        )
        for pt1, pt2 in lines
    ]

    # Create the graph from the initial lines
    Gn = nx.Graph()
    for line in lines:
        Gn.add_edge(line[0], line[1])

    # Plot the initial graph
    st.title("Initial Graph")
    st.write("Here is the visualization of how the set of lines looks.")
    st.write("""
    **Logic:** The set of lines are stored as an undirected graph with nodes as all the point sets in the lines and their edges.
    This is done to store the data in a structured and easy-to-retrieve format, making our calculations easier.
    """)
    pos = {node: node for node in Gn.nodes()}  
    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw(
        Gn, pos, with_labels=True, node_size=500, 
        node_color="lightblue", ax=ax
    )
    st.pyplot(fig)  

    # Finding all the vertices and edges
    st.write("### Next Step: Finding All the Vertices from the Set of Lines")

    lines_shapely = [LineString([edge[0], edge[1]]) for edge in lines]
    final_edge_set = set()

    for node in Gn.nodes():  # Iterate over all the nodes
        for neighbor in Gn.neighbors(node):
            main_line = LineString([node, neighbor])
            intersection_points = find_intersection_points(main_line, lines_shapely)
            # Round intersection points to 4 decimal places
            intersection_points = [
                Point(set_precision(p.x, 4), set_precision(p.y, 4)) 
                for p in intersection_points
            ]
            decomposed_lines = decompose_line(main_line, intersection_points)
            for segment in decomposed_lines:
                coords = list(segment.coords)
                if len(coords) == 2:
                    # Round segment coordinates to 4 decimal places
                    rounded_coords = [
                        (set_precision(coords[0][0], 4), set_precision(coords[0][1], 4)),
                        (set_precision(coords[1][0], 4), set_precision(coords[1][1], 4))
                    ]
                    sorted_edge = tuple(sorted(rounded_coords))
                    final_edge_set.add(sorted_edge)

    final_edge_array = [list(edge) for edge in final_edge_set]  # Convert set to list

    # Create the second graph with all the vertices in the figure 
    G = nx.Graph()
    for line in final_edge_array:
        G.add_edge(tuple(line[0]), tuple(line[1]))
    
    st.title("Graph with All Detected Vertices")
    st.write("Here is the visualization of how the set of lines looks after all the vertices have been discovered.")
    st.write("""
    **Logic:** Finding all the vertices is an essential step in figuring out what closed cycles can be made using a single vertex.
    If there is a hidden vertex created by some intersection of the lines from our initial set of lines, we wouldn't be able to find all the cycles corresponding to the hidden point if that vertex is not discovered first.
    Hence, we perform this step.
    """)
    pos = {node: node for node in G.nodes()}  
    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw(
        G, pos, with_labels=True, node_size=500, 
        node_color="lightblue", ax=ax
    )
    st.pyplot(fig)  
    st.write("### Next Step: Finding Smallest Cycles from All the Vertices")

    # Calculating the cycles in the set of lines
    # Calculating the master cycle which stores information about all the closed cycles
    master_cycle = []
    for node in list(G.nodes()): 
        cycles = find_all_cycles_from_start(G, node)
        # Round each cycle's coordinates to 4 decimal places
        rounded_cycles = [set_precision(cycle, 4) for cycle in cycles]
        master_cycle.append(rounded_cycles)

    # Defining min perimeter cycles
    min_perimeter_cycles = []
    for cycle_list in master_cycle:
        min_perimeter = float('inf')
        min_cycle = None
        for cycle in cycle_list:
            try:
                perimeter = calculate_perimeter(cycle)
                perimeter = set_precision(perimeter, 4)  # Round perimeter to 4 decimal places
                if perimeter < min_perimeter:
                    min_perimeter = perimeter
                    min_cycle = cycle
            except ValueError as e:
                st.write(f"Error in cycle {cycle}: {e}")

        if min_cycle is not None:
            min_perimeter_cycles.append(min_cycle)
        else:
            st.write("No valid cycles found in this set.")

    unique_cycles_list = unique_cycles(min_perimeter_cycles)

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
            cycle_edges_in_cycle = set(
                frozenset([cycle[i], cycle[i + 1]]) for i in range(len(cycle) - 1)
            )
            cycle_edges_in_cycle.add(frozenset([cycle[-1], cycle[0]]))  # Close the cycle

            # Check if any missing edge is in the cycle
            if any(edge in cycle_edges_in_cycle for edge in missing_edges):
                relevant_cycles.append(cycle)

        # Append the relevant cycles to the master_missing_array
        master_missing_array.append(relevant_cycles)

    # Create the figure of all the smallest closed cycles
    st.title("Smallest Cycles from All the Vertices")
    st.write("Here is the visualization of the smallest cycles from all the vertices which have been discovered so far.")
    st.write("""
    **Logic:** Finding the smallest cycle from each vertex in the figure will give us the required result. 
    Keep in mind that since the figures have common edges, there might be cases where a closed loop never gets found because of its adjacent smaller loop.
    """)
    fig, ax = plt.subplots(figsize=(10, 8))

    # Assign a unique color for each cycle
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_cycles_list)))

    # Plot each cycle
    for idx, cycle in enumerate(unique_cycles_list):
        # Ensure cycle is a list of tuples with 4 decimal places
        cycle = set_precision(cycle, 4)
        cycle = np.array(cycle)

        if cycle.ndim != 2 or cycle.shape[1] != 2:
            st.write(f"Invalid cycle structure for plotting: {cycle}")
            continue

        # Close the cycle by adding the first point at the end
        cycle_closed = np.vstack([cycle, cycle[0]])

        # Plot the cycle with the corresponding color
        ax.plot(
            cycle_closed[:, 0], cycle_closed[:, 1], marker='o', color=colors[idx], label=f'Cycle {idx+1}'
        )

    # Customize the plot
    ax.set_title('Unique Cycles with Minimum Perimeter')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

    # Display the plot in Streamlit
    st.pyplot(fig)
    st.write("**Inference:** We can see that there are missing cycles from certain vertices.")
    st.write("""
    **Next Step:** We will find the missing edges by comparing this graph to our initial graph. 
    Then, we can find the smallest cycle made by this missing line.
    """)

    # Recalculate missing edges
    cycle_edges = set()
    for cycle in unique_cycles_list:
        cycle = set_precision(cycle, 4)
        for i in range(len(cycle) - 1):
            cycle_edges.add(frozenset([cycle[i], cycle[i+1]]))
        cycle_edges.add(frozenset([cycle[-1], cycle[0]]))  # Close the cycle

    missing_edges = graph_edges - cycle_edges

    # Initialize the master_missing_array to collect relevant cycles
    master_missing_array = []

    # Loop through the master_cycle array
    for val in master_cycle:
        relevant_cycles = []
        for cycle in val:
            cycle = set_precision(cycle, 4)
            cycle_edges_in_cycle = set(
                frozenset([cycle[i], cycle[i + 1]]) for i in range(len(cycle) - 1)
            )
            cycle_edges_in_cycle.add(frozenset([cycle[-1], cycle[0]]))  # Close the cycle

            if any(edge in cycle_edges_in_cycle for edge in missing_edges):
                relevant_cycles.append(cycle)

        master_missing_array.append(relevant_cycles)

    # From the master_missing_array find the cycles that have a line between the points in each missing edge
    cycles_with_missing_edges = {edge: [] for edge in missing_edges}

    for missing_edge in missing_edges:
        for relevant_cycles in master_missing_array:
            for cycle in relevant_cycles:
                cycle = set_precision(cycle, 4)
                cycle_edges_in_cycle = set(
                    frozenset([cycle[i], cycle[i + 1]]) for i in range(len(cycle) - 1)
                )
                cycle_edges_in_cycle.add(frozenset([cycle[-1], cycle[0]]))  # Close the cycle

                if missing_edge in cycle_edges_in_cycle:
                    cycles_with_missing_edges[missing_edge].append(cycle)

    # Find the cycle with the least perimeter for each missing edge
    least_perimeter_cycles = {}

    for missing_edge, cycles in cycles_with_missing_edges.items():
        min_perimeter = float('inf')
        best_cycle = None

        for cycle in cycles:
            perimeter = calculate_perimeter_missing(cycle)
            perimeter = set_precision(perimeter, 4)  # Round perimeter to 4 decimal places
            if perimeter < min_perimeter:
                min_perimeter = perimeter
                best_cycle = cycle

        if best_cycle is not None:
            least_perimeter_cycles[missing_edge] = {
                "cycle": best_cycle,
                "perimeter": min_perimeter,
            }

    # Update the unique_cycles_list with calculated perimeters if not present
    unique_cycles_with_perimeters = []
    for cycle in unique_cycles_list:
        perimeter = calculate_perimeter(cycle)
        perimeter = set_precision(perimeter, 4)  # Round perimeter to 4 decimal places
        unique_cycles_with_perimeters.append({'cycle': cycle, 'perimeter': perimeter})

    # Prepare the least perimeter cycles data in the desired format
    least_perimeter_cycles_data = [
        {'cycle': data['cycle'], 'perimeter': data['perimeter']} 
        for data in least_perimeter_cycles.values()
    ]

    # Combine all cycles
    all_cycles_with_perimeter = unique_cycles_with_perimeters + least_perimeter_cycles_data
    df = pd.DataFrame(all_cycles_with_perimeter)

    # Initialize an empty dictionary to hold unique cycles grouped by perimeter
    unique_cycles_by_perimeter = {}

    # Iterate through the groups in the DataFrame
    for perimeter, group in df.groupby('perimeter'):
        unique_cycles_missing = []

        for _, row in group.iterrows():
            cycle = row['cycle']
            is_unique = True

            for unique_cycle in unique_cycles_missing:
                if are_cycles_equal(cycle, unique_cycle['cycle']):
                    is_unique = False
                    break

            if is_unique:
                unique_cycles_missing.append({'cycle': cycle, 'perimeter': perimeter})

        unique_cycles_by_perimeter[perimeter] = unique_cycles_missing

    # Convert the dictionary to a list of dictionaries
    output_list = []
    for perimeter, cycles in unique_cycles_by_perimeter.items():
        for cycle_dict in cycles:
            output_list.append(cycle_dict)

    # Create DataFrame for display
    temp = pd.DataFrame(output_list)

    # Round 'cycle' values to 3 decimal places for display
    temp['cycle'] = temp['cycle'].apply(lambda cycles: [
        (round(x, 3), round(y, 3)) for x, y in cycles
    ])
    temp['perimeter'] = temp['perimeter'].round(4)

    st.dataframe(temp)
    #st.dataframe(pd.DataFrame(output_list))
    st.title("All Cycles with Perimeters")
    st.write("Here is the visualization of all the unique smallest cycles from all the vertices which have been discovered.")
    st.write(f"**Total number of unique cycles:** {len(output_list)}")

    st.write("""
    **Logic:** Finding the smallest cycle from each missing line gives us the cycles that are missing from the figure. 
    By taking the unique cycles, we eliminate any redundancy.
    """)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Assign a unique color for each cycle
    colors = plt.cm.jet(np.linspace(0, 1, len(output_list)))

    # Plot each cycle
    for idx, cycle_data in enumerate(output_list):
        cycle = cycle_data['cycle']
        perimeter = cycle_data['perimeter']

        # Ensure cycle is a list of tuples with correct precision
        cycle = set_precision(cycle, 4)
        cycle = np.array(cycle)

        if cycle.ndim != 2 or cycle.shape[1] != 2:
            st.write(f"Invalid cycle structure for plotting: {cycle}")
            continue

        # Close the cycle by adding the first point at the end
        cycle_closed = np.vstack([cycle, cycle[0]])

        # Plot the cycle with the corresponding color
        ax.plot(
            cycle_closed[:, 0], cycle_closed[:, 1], marker='o', color=colors[idx], label=f'Cycle {idx+1}'
        )

        # Add perimeter label at the centroid of the cycle
        centroid = np.mean(cycle, axis=0)
        centroid = set_precision(centroid, 4)  # Ensure precision
        ax.text(
            centroid[0], centroid[1], f'{perimeter:.2f}', 
            color=colors[idx], fontsize=10, ha='center', va='center'
        )

    # Customize the plot
    ax.set_title('All Cycles with Perimeters')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

    # Display the plot in Streamlit
    st.pyplot(fig)

    # Calculate and display the multiplicative product of the perimeters
    perimeters = [cycle_data['perimeter'] for cycle_data in output_list]
    perimeter_terms = ' * '.join([f"{p:.4f}" for p in perimeters])
    product = np.prod(perimeters)
    product = set_precision(product, 4)  # Round product to 4 decimal places
    st.write(f"**Inference:** We can see that all the closed cycles have been detected and there are **{len(output_list)}** unique cycles in the figure.")
    st.write(f"**Final Multiplicative Product of All the Cycles Found:** {perimeter_terms} = {product:.4f}")

# Run the Streamlit app
if __name__ == "__main__":
    app()
