import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point
import matplotlib.pyplot as plt 
import math
import pandas as pd

# Function to check if two cycles are the same based on their unique points
def are_cycles_equal(cycle1, cycle2):
    set1 = set(cycle1[:-1])  # Exclude last element (if any) for comparison
    set2 = set(cycle2[:-1])
    return set1 == set2

def find_intersection_points(line, lines, tolerance=1e-9):
    """
    Find all unique intersection points of a given line with other lines.
    """
    intersections = []
    for other_line in lines:
        if line != other_line:  # Skip the same line
            if line.intersects(other_line):
                intersection = line.intersection(other_line)
                if intersection.geom_type == "Point":
                    # Check for uniqueness with a small tolerance
                    if not any(np.isclose(intersection.x, p.x, atol=tolerance) and 
                               np.isclose(intersection.y, p.y, atol=tolerance) 
                               for p in intersections):
                        intersections.append(intersection)
    # Sort points along the line
    return sorted(intersections, key=lambda p: line.project(p))

def decompose_line(line, intersections):
    """
    Decompose a line into smaller segments based on intersection points.
    """
    segments = []
    points = [Point(line.coords[0])] + intersections + [Point(line.coords[1])]
    for i in range(len(points) - 1):
        segment = LineString([points[i], points[i + 1]])
        if segment.length > 0:  # Ensure non-zero length segments
            segments.append(segment)
    return segments

def find_all_cycles_from_start(G, start):
    cycles = []  # List to store all the found cycles
    queue = [(start, [start])]  # Initialize queue with the start node and an empty path

    while queue:
        current_node, path = queue.pop(0)  # Get the next node and the path leading to it
        #print(f"Exploring path: {path}")

        # Explore all neighbors of the current node
        for neighbor in G.neighbors(current_node):
            #print(f"  Checking neighbor: {neighbor}")

            if neighbor == start and len(path) > 2:
                # If the neighbor is the start node and the path length is greater than 2, it's a cycle
                cycle = path + [start]  # Add the start node at the end to complete the cycle
                cycles.append(cycle)
                #print(f"  Cycle found: {cycle}")
            elif neighbor not in path:  # Avoid revisiting nodes in the current path
                # Extend the path and add the neighbor to the queue
                #print(f"  Path extended to: {path + [neighbor]}")
                queue.append((neighbor, path + [neighbor]))

    return cycles

def calculate_perimeter(cycle):
    # Initialize the perimeter
    perimeter = 0
    n = len(cycle)
    
    # Iterate through each pair of consecutive points in the cycle
    for i in range(n - 1):
        # Ensure each cycle element contains the expected number of values (e.g., 2D coordinates)
        if len(cycle[i]) != 2 or len(cycle[i + 1]) != 2:
            raise ValueError(f"Expected 2D coordinates, but found {cycle[i]} or {cycle[i + 1]}")
        
        x1, y1 = cycle[i]
        x2, y2 = cycle[i + 1]
        perimeter += ((x2 - x1)**2 + (y2 - y1)**2)**0.5  # Euclidean distance
        
    # Close the cycle: calculate the distance between the last and first point
    if len(cycle[-1]) != 2 or len(cycle[0]) != 2:
        raise ValueError(f"Expected 2D coordinates for last or first point in cycle")
    
    x1, y1 = cycle[-1]
    x2, y2 = cycle[0]
    perimeter += ((x2 - x1)**2 + (y2 - y1)**2)**0.5  # Closing the cycle
    return perimeter

def unique_cycles(cycles):
    seen = set()
    unique_cycles = []
    for cycle in cycles:
        cycle_tuple = tuple(map(tuple, cycle))  # Convert each cycle into a tuple of tuples (to be hashable)
        if cycle_tuple not in seen:
            seen.add(cycle_tuple)
            unique_cycles.append(cycle)
    return unique_cycles


# Function to calculate the Euclidean distance between two points
def calculate_distance_missing(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


# Function to calculate the perimeter of a cycle
def calculate_perimeter_missing(cycle):
    perimeter = 0
    for i in range(len(cycle) - 1):
        perimeter += calculate_distance_missing(cycle[i], cycle[i + 1])
    # Add the distance to close the cycle
    perimeter += calculate_distance_missing(cycle[-1], cycle[0])
    return perimeter


def plot_cycles_with_perimeters(cycles, perimeters, label_prefix='Cycle'):
    # Generate a color map for unique colors for each cycle
    colors = plt.cm.jet(np.linspace(0, 1, len(cycles)))
    
    for idx, (cycle, perimeter) in enumerate(zip(cycles, perimeters)):
        cycle = np.array(cycle)
        color = colors[idx]  # Get a unique color for the current cycle
        
        # Plot the cycle with the corresponding color
        plt.plot(cycle[:, 0], cycle[:, 1], marker='o', color=color, label=f'{label_prefix} {idx+1}')
        
        # Add perimeter label at the center of the cycle if perimeter is not None
        if perimeter is not None:
            centroid = np.mean(cycle, axis=0)  # Get the centroid of the cycle to place the label
            plt.text(centroid[0], centroid[1], f'{perimeter:.2f}', color=color, fontsize=10, ha='center', va='center')


# Function to calculate the perimeter of a cycle
def calculate_perimeter_all(cycle):
    cycle = np.array(cycle)
    perimeter = 0
    for i in range(len(cycle)):
        p1 = cycle[i]
        p2 = cycle[(i + 1) % len(cycle)]  # Connect the last point to the first
        perimeter += np.linalg.norm(np.array(p2) - np.array(p1))  # Euclidean distance
    return perimeter

def plot_cycles_with_perimeters(cycles, label_prefix='Cycle'):
    # Generate a color map for unique colors for each cycle
    colors = plt.cm.jet(np.linspace(0, 1, len(cycles)))
    
    for idx, cycle_data in enumerate(cycles):
        cycle = np.array(cycle_data['cycle'])
        perimeter = cycle_data['perimeter']
        color = colors[idx]  # Get a unique color for the current cycle
        
        # Plot the cycle with the corresponding color
        plt.plot(cycle[:, 0], cycle[:, 1], marker='o', color=color, label=f'{label_prefix} {idx+1}')
        
        # Add perimeter label at the center of the cycle
        centroid = np.mean(cycle, axis=0)  # Get the centroid of the cycle to place the label
        plt.text(centroid[0], centroid[1], f'{perimeter:.2f}', color=color, fontsize=10, ha='center', va='center')
