import streamlit as st
import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point
import matplotlib.pyplot as plt
import pandas as pd
import math
from concurrent.futures import ThreadPoolExecutor, as_completed


def authenticate(username, password):
    """Validate user credentials."""
    credentials = st.secrets["credentials"]
    for user in credentials:
        if user["username"] == username:
            if user["password"] == password:
                return True
            else:
                return False
    return False

def login():
    """Render the login form in the sidebar."""
    st.sidebar.title("Login")
    with st.sidebar.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            if authenticate(username, password):
                st.session_state['authenticated'] = True
                st.session_state['login_success'] = True
            else:
                st.sidebar.error("Invalid username or password.")

# -----------------------------
# Utility Functions
# -----------------------------

def are_cycles_equal(cycle1, cycle2):
    """Determine if two cycles are identical based on their unique points."""
    set1 = set(cycle1[:-1])  # Exclude the last point to avoid duplication
    set2 = set(cycle2[:-1])
    return set1 == set2

def find_intersection_points(line, lines, tolerance=1e-9):
    """
    Identify all unique intersection points of a given line with other lines.
    """
    intersections = []
    for other_line in lines:
        if line != other_line and line.intersects(other_line):
            intersection = line.intersection(other_line)
            if intersection.geom_type == "Point":
                # Ensure the intersection point is unique within the tolerance
                if not any(
                    np.isclose(intersection.x, p.x, atol=tolerance) and 
                    np.isclose(intersection.y, p.y, atol=tolerance) 
                    for p in intersections
                ):
                    intersections.append(intersection)
    # Sort the intersection points along the line
    return sorted(intersections, key=lambda p: line.project(p))

def decompose_line(line, intersections):
    """
    Split a line into smaller segments based on intersection points.
    """
    segments = []
    points = [Point(line.coords[0])] + intersections + [Point(line.coords[1])]
    for i in range(len(points) - 1):
        segment = LineString([points[i], points[i + 1]])
        if segment.length > 0:
            segments.append(segment)
    return segments

def find_all_cycles_from_start(G, start):
    """
    Discover all cycles in graph G that begin and end at the specified start node.
    """
    cycles = []
    queue = [(start, [start])]

    while queue:
        current_node, path = queue.pop(0)
        for neighbor in G.neighbors(current_node):
            if neighbor == start and len(path) > 2:
                cycle = path + [start]
                cycles.append(cycle)
            elif neighbor not in path:
                queue.append((neighbor, path + [neighbor]))
    return cycles

def calculate_perimeter(cycle):
    """Compute the perimeter of a given cycle."""
    perimeter = 0
    n = len(cycle)
    
    for i in range(n - 1):
        if len(cycle[i]) != 2 or len(cycle[i + 1]) != 2:
            raise ValueError(f"Expected 2D coordinates, but found {cycle[i]} or {cycle[i + 1]}")
        x1, y1 = cycle[i]
        x2, y2 = cycle[i + 1]
        perimeter += math.hypot(x2 - x1, y2 - y1)
    
    # Closing the cycle
    if len(cycle[-1]) != 2 or len(cycle[0]) != 2:
        raise ValueError("Expected 2D coordinates for the last or first point in cycle")
    x1, y1 = cycle[-1]
    x2, y2 = cycle[0]
    perimeter += math.hypot(x2 - x1, y2 - y1)
    
    return perimeter

def unique_cycles(cycles):
    """Filter and return a list of unique cycles."""
    seen = set()
    unique = []
    for cycle in cycles:
        cycle_tuple = tuple(map(tuple, cycle))
        if cycle_tuple not in seen:
            seen.add(cycle_tuple)
            unique.append(cycle)
    return unique

def calculate_distance_missing(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return math.hypot(point1[0] - point2[0], point1[1] - point2[1])

def calculate_perimeter_missing(cycle):
    """Compute the perimeter of a cycle."""
    perimeter = 0
    for i in range(len(cycle) - 1):
        perimeter += calculate_distance_missing(cycle[i], cycle[i + 1])
    perimeter += calculate_distance_missing(cycle[-1], cycle[0])
    return perimeter

def set_precision(value, precision=4):
    """
    Round values to the specified precision.
    Handles individual values, lists, tuples, and NumPy arrays.
    """
    if isinstance(value, (list, tuple)):
        return [tuple(np.round(point, precision)) for point in value]
    elif isinstance(value, np.ndarray):
        return np.round(value, precision)
    return round(value, precision)

# -----------------------------
# Cycle Detection with Parallel Processing
# -----------------------------

def find_unique_cycles_parallel(G, lines_shapely):
    """
    Identify all unique smallest cycles in graph G using parallel processing.
    Returns a list of unique cycles and the master_cycle list.
    """
    master_cycle = []
    unique_cycles_list = []

    def find_cycles_for_node(node):
        return find_all_cycles_from_start(G, node)

    with ThreadPoolExecutor() as executor:
        future_to_node = {executor.submit(find_cycles_for_node, node): node for node in G.nodes()}
        for future in as_completed(future_to_node):
            node = future_to_node[future]
            try:
                cycles = future.result()
                rounded_cycles = [set_precision(cycle, 4) for cycle in cycles]
                master_cycle.append(rounded_cycles)
            except Exception as e:
                st.error(f"Error finding cycles for node {node}: {e}")

    # Identify the cycle with the minimum perimeter for each cycle list
    for cycle_list in master_cycle:
        min_perimeter = float('inf')
        min_cycle = None
        for cycle in cycle_list:
            try:
                perimeter = calculate_perimeter(cycle)
                perimeter = set_precision(perimeter, 4)
                if perimeter < min_perimeter:
                    min_perimeter = perimeter
                    min_cycle = cycle
            except ValueError:
                continue  # Skip invalid cycles
        if min_cycle:
            unique_cycles_list.append(min_cycle)

    unique_cycles_list = unique_cycles(unique_cycles_list)
    return unique_cycles_list, master_cycle

# -----------------------------
# Compute Missing Edges
# -----------------------------

def compute_missing_edges(G, unique_cycles_list):
    """
    Determine which edges are not part of any unique cycles.
    Returns a set of missing edges and all cycle edges.
    """
    cycle_edges = set()
    for cycle in unique_cycles_list:
        for i in range(len(cycle) - 1):
            cycle_edges.add(frozenset([cycle[i], cycle[i + 1]]))
        cycle_edges.add(frozenset([cycle[-1], cycle[0]]))  # Close the cycle

    graph_edges = set(frozenset(edge) for edge in G.edges())
    missing_edges = graph_edges - cycle_edges

    return missing_edges, cycle_edges

# -----------------------------
# Find Least Perimeter Cycles
# -----------------------------

def find_least_perimeter_cycles(G, master_cycle, missing_edges):
    """
    For each missing edge, find the cycle with the least perimeter that includes it.
    Returns a list of cycles with their perimeters.
    """
    master_missing_array = []

    # Collect cycles relevant to missing edges
    for cycle_list in master_cycle:
        relevant_cycles = []
        for cycle in cycle_list:
            cycle_edges = set(frozenset([cycle[i], cycle[i + 1]]) for i in range(len(cycle) - 1))
            cycle_edges.add(frozenset([cycle[-1], cycle[0]]))  # Close the cycle
            if any(edge in cycle_edges for edge in missing_edges):
                relevant_cycles.append(cycle)
        master_missing_array.append(relevant_cycles)

    # Map each missing edge to its corresponding cycles
    cycles_with_missing_edges = {edge: [] for edge in missing_edges}
    for missing_edge in missing_edges:
        for relevant_cycles in master_missing_array:
            for cycle in relevant_cycles:
                cycle_edges = set(frozenset([cycle[i], cycle[i + 1]]) for i in range(len(cycle) - 1))
                cycle_edges.add(frozenset([cycle[-1], cycle[0]]))  # Close the cycle
                if missing_edge in cycle_edges:
                    cycles_with_missing_edges[missing_edge].append(cycle)

    # Identify the cycle with the least perimeter for each missing edge
    least_perimeter_cycles = []
    for missing_edge, cycles in cycles_with_missing_edges.items():
        min_perimeter = float('inf')
        best_cycle = None
        for cycle in cycles:
            try:
                perimeter = calculate_perimeter_missing(cycle)
                perimeter = set_precision(perimeter, 4)
                if perimeter < min_perimeter:
                    min_perimeter = perimeter
                    best_cycle = cycle
            except ValueError:
                continue  # Skip invalid cycles
        if best_cycle:
            least_perimeter_cycles.append({'cycle': best_cycle, 'perimeter': min_perimeter})

    return least_perimeter_cycles

# -----------------------------
# Process Lines Function
# -----------------------------

def process_lines(lines):
    """
    Process input lines by finding intersection points, decomposing lines, and creating graphs.
    Returns the initial graph Gn and the refined graph G.
    """
    # Round all input coordinates to 4 decimal places
    rounded_lines = [
        (
            (set_precision(pt1[0], 4), set_precision(pt1[1], 4)),
            (set_precision(pt2[0], 4), set_precision(pt2[1], 4))
        )
        for pt1, pt2 in lines
    ]

    # Create the initial graph Gn
    Gn = nx.Graph()
    Gn.add_edges_from(rounded_lines)

    # Convert lines to Shapely LineStrings
    lines_shapely = [LineString([edge[0], edge[1]]) for edge in rounded_lines]
    final_edge_set = set()

    # Decompose lines based on intersection points
    for node in Gn.nodes():
        for neighbor in Gn.neighbors(node):
            main_line = LineString([node, neighbor])
            intersection_points = find_intersection_points(main_line, lines_shapely)
            # Round intersection points
            rounded_intersections = [
                Point(set_precision(p.x, 4), set_precision(p.y, 4)) 
                for p in intersection_points
            ]
            decomposed_lines = decompose_line(main_line, rounded_intersections)
            for segment in decomposed_lines:
                coords = list(segment.coords)
                if len(coords) == 2:
                    rounded_coords = [
                        (set_precision(coords[0][0], 4), set_precision(coords[0][1], 4)),
                        (set_precision(coords[1][0], 4), set_precision(coords[1][1], 4))
                    ]
                    sorted_edge = tuple(sorted(rounded_coords))
                    final_edge_set.add(sorted_edge)

    final_edge_array = list(final_edge_set)

    # Create the refined graph G
    G = nx.Graph()
    G.add_edges_from(final_edge_array)

    return Gn, G, lines_shapely

# -----------------------------
# Main Application Logic
# -----------------------------

def main_app():
    """Render the main application interface after successful login."""
    
    # Display success message once after login
    if st.session_state.get('login_success'):
        st.success("Logged in successfully!")
        st.session_state['login_success'] = False  # Reset the flag

    st.title("Cycle Perimeter Calculator with Fixed Precision")
    st.write("**Objective:** Identify all closed cycles within a set of lines.")

    # -----------------------------
    # User Input for Lines
    # -----------------------------
    st.header("### Input Lines")
    st.write("Enter the set of lines as a list of coordinate pairs.")
    st.write("**Example:** [((1,1), (1,2)), ((1,2), (1.8,2)), ...]")

    default_lines = [
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

    lines_input = st.text_area(
        "Input Lines",
        value=str(default_lines),
        height=200,
        help="Enter lines in the format: [((x1,y1), (x2,y2)), ...]"
    )

    # -----------------------------
    # Parse and Validate User Input
    # -----------------------------
    try:
        lines = eval(lines_input)
        if not isinstance(lines, list):
            raise ValueError("Input must be a list of coordinate pairs.")
        for line in lines:
            if not (
                isinstance(line, tuple) and
                len(line) == 2 and
                all(isinstance(point, tuple) and len(point) == 2 for point in line)
            ):
                raise ValueError("Each line must be a tuple of two coordinate tuples.")
    except Exception as e:
        st.error(f"**Error parsing lines input:** {e}")
        st.stop()

    # -----------------------------
    # Process Lines and Create Graphs
    # -----------------------------
    with st.spinner("Processing lines and building graphs..."):
        try:
            Gn, G, lines_shapely = process_lines(lines)
        except Exception as e:
            st.error(f"**Error during processing lines:** {e}")
            st.stop()

    # -----------------------------
    # Plot the Initial Graph
    # -----------------------------
    st.header("### Initial Graph")
    st.write("Visualization of the input lines as an undirected graph.")
    st.write("""
    **Explanation:** The lines are represented as an undirected graph where nodes are the endpoints of the lines and edges represent the connections between them.
    """)

    pos = {node: node for node in Gn.nodes()}  # Positioning nodes based on their coordinates
    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw(
        Gn, pos,
        with_labels=True,
        node_size=500,
        node_color="lightblue",
        edge_color="gray",
        linewidths=1,
        font_size=10,
        ax=ax
    )
    st.pyplot(fig)

    # -----------------------------
    # Plot the Refined Graph with All Detected Vertices
    # -----------------------------
    st.header("### Refined Graph with All Detected Vertices")
    st.write("Visualization of the graph after identifying all intersection points.")
    st.write("""
    **Explanation:** Intersection points are detected and treated as additional vertices to ensure all possible cycles are identified.
    """)

    pos_refined = {node: node for node in G.nodes()}
    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw(
        G, pos_refined,
        with_labels=True,
        node_size=500,
        node_color="lightgreen",
        edge_color="blue",
        linewidths=1,
        font_size=10,
        ax=ax
    )
    st.pyplot(fig)

    st.markdown("---")
    st.header("### Step 2: Identifying Smallest Cycles from All Vertices")

    # -----------------------------
    # Find Unique Cycles Using Parallel Processing
    # -----------------------------
    with st.spinner("Finding unique cycles..."):
        try:
            unique_cycles_list, master_cycle = find_unique_cycles_parallel(G, lines_shapely)
        except Exception as e:
            st.error(f"**Error during finding unique cycles:** {e}")
            st.stop()

    # -----------------------------
    # Compute Missing Edges
    # -----------------------------
    try:
        missing_edges, cycle_edges = compute_missing_edges(G, unique_cycles_list)
    except Exception as e:
        st.error(f"**Error during computing missing edges:** {e}")
        st.stop()

    # -----------------------------
    # Find Least Perimeter Cycles for Missing Edges
    # -----------------------------
    try:
        least_perimeter_cycles = find_least_perimeter_cycles(G, master_cycle, missing_edges)
    except Exception as e:
        st.error(f"**Error during finding least perimeter cycles:** {e}")
        st.stop()

    # -----------------------------
    # Combine All Cycles
    # -----------------------------
    unique_cycles_with_perimeters = []
    for cycle in unique_cycles_list:
        try:
            perimeter = calculate_perimeter(cycle)
            perimeter = set_precision(perimeter, 4)
            unique_cycles_with_perimeters.append({'cycle': cycle, 'perimeter': perimeter})
        except ValueError:
            continue  # Skip invalid cycles

    all_cycles_with_perimeter = unique_cycles_with_perimeters + least_perimeter_cycles
    df = pd.DataFrame(all_cycles_with_perimeter)

    # -----------------------------
    # Identify and Group Unique Cycles by Perimeter
    # -----------------------------
    unique_cycles_by_perimeter = {}
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

    # -----------------------------
    # Prepare Data for Display
    # -----------------------------
    output_list = []
    for cycles in unique_cycles_by_perimeter.values():
        for cycle_dict in cycles:
            output_list.append(cycle_dict)

    temp_df = pd.DataFrame(output_list)

    # Round cycle coordinates for better readability
    temp_df['cycle'] = temp_df['cycle'].apply(lambda cycles: [
        (round(x, 3), round(y, 3)) for x, y in cycles
    ])
    temp_df['perimeter'] = temp_df['perimeter'].round(4)

    st.write("#### All Identified Unique Cycles with Their Perimeters")
    st.dataframe(temp_df)

    st.header("### Visualization of All Unique Cycles")
    st.write("Each unique cycle is plotted with its perimeter labeled at the centroid.")

    fig, ax = plt.subplots(figsize=(10, 8))

    if len(output_list) == 0:
        st.write("No cycles to display.")
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, len(output_list)))  # Use a colormap with enough distinct colors

        for idx, cycle_data in enumerate(output_list):
            cycle = cycle_data['cycle']
            perimeter = cycle_data['perimeter']

            # Convert cycle to NumPy array for plotting
            cycle_np = np.array(cycle)

            if cycle_np.ndim != 2 or cycle_np.shape[1] != 2:
                st.write(f"Invalid cycle structure for plotting: {cycle}")
                continue

            # Close the cycle by appending the first point at the end
            cycle_closed = np.vstack([cycle_np, cycle_np[0]])

            # Plot the cycle
            ax.plot(
                cycle_closed[:, 0], cycle_closed[:, 1],
                marker='o', color=colors[idx], label=f'Cycle {idx+1}'
            )

            # Calculate centroid for labeling
            centroid = np.mean(cycle_np, axis=0)
            centroid = set_precision(centroid, 4)

            # Add perimeter label
            ax.text(
                centroid[0], centroid[1], f'{perimeter:.2f}',
                color=colors[idx], fontsize=10, ha='center', va='center'
            )

    # Customize the plot
    ax.set_title('All Unique Cycles with Perimeters')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

    st.pyplot(fig)

    # -----------------------------
    # Display Final Inference
    # -----------------------------
    if len(output_list) > 0:
        perimeters = [cycle_data['perimeter'] for cycle_data in output_list]
        perimeter_terms = ' * '.join([f"{p:.4f}" for p in perimeters])
        product = np.prod(perimeters)
        product = set_precision(product, 4)
        st.write(f"**Inference:** All closed cycles have been detected. There are **{len(output_list)}** unique cycles in the figure.")
        st.write(f"**Final Multiplicative Product of All the Cycles Found:** {perimeter_terms} = {product:.4f}")
    else:
        st.write("**Inference:** No cycles detected in the figure.")

# -----------------------------
# Run the Application with Authentication
# -----------------------------

def app():
    """Control the flow of the application based on authentication status."""
    # Initialize session state variables if they don't exist
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
    if 'login_success' not in st.session_state:
        st.session_state['login_success'] = False

    if not st.session_state['authenticated']:
        login()
    else:
        main_app()

# Run the Streamlit app
if __name__ == "__main__":
    app()
