import json
import networkx as nx
import matplotlib.pyplot as plt

def add_nodes_and_edges(graph, data, parent=None, edge_label=""):
    """
    Recursively add nodes and edges to the graph from the JSON data.
    """
    if isinstance(data, dict):
        # If the current node has a "prediction", it's a leaf node
        if "prediction" in data:
            node_label = str(data["prediction"])  # Use only the prediction value
        else:
            # Otherwise, it's a decision node
            node_label = f'Feature {data["feature"]}'  # Simplify the feature label
        
        # Add the current node to the graph
        current_node = len(graph)
        graph.add_node(current_node, label=node_label)
        
        if parent is not None:
            graph.add_edge(parent, current_node, label=edge_label)
        
        # Add child nodes recursively
        if "true" in data:
            add_nodes_and_edges(graph, data["true"], current_node, "True")
        if "false" in data:
            add_nodes_and_edges(graph, data["false"], current_node, "False")
    else:
        raise ValueError("Unsupported JSON structure.")

def compute_positions(graph, node, pos, x=0, y=0, layer_width=1.0, depth=1.0):
    """
    Compute positions for nodes in a tree structure.
    """
    pos[node] = (x, y)
    children = list(graph.successors(node))
    if children:
        # Spread children evenly across the x-axis
        step = layer_width / len(children)
        next_x = x - layer_width / 2 + step / 2
        for child in children:
            compute_positions(graph, child, pos, next_x, y - depth, layer_width / len(children), depth)
            next_x += step

def draw_decision_tree(graph, tree_index, filename=None):
    """
    Draw the decision tree using networkx and matplotlib. Optionally save to a file.
    """
    pos = {}
    compute_positions(graph, 0, pos)  # Start with root node (node 0)
    labels = nx.get_node_attributes(graph, "label")
    edge_labels = nx.get_edge_attributes(graph, "label")
    
    plt.figure(figsize=(12, 8))
    nx.draw(graph, pos, with_labels=True, labels=labels, node_size=3000, node_color="lightblue", font_size=10, font_weight="bold", arrows=False)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)
    
    # Add text "Decision Tree: [tree index]" in the top right corner
    plt.text(0.95, 0.95, f"Decision Tree: {tree_index}", ha='right', va='top', transform=plt.gcf().transFigure, fontsize=12, fontweight='bold')
    
    if filename:
        plt.savefig(filename, format="png", bbox_inches="tight")
    else:
        plt.show()
    
    plt.close()  # Close the plot to free memory and avoid overlap

import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

def PlotTreeFarmsDecisionTreeErrors(AllErrors, order_errors=True):
    """
    Plot misclassification errors with TreeIndex values.
    
    Parameters:
        AllErrors (list): List of classification errors.
        order_errors (bool): Whether to sort errors by value (default True).
    """
    # Create the DataFrame
    pdAllErrors = pd.DataFrame(AllErrors, columns=["ClassificationError"])
    pdAllErrors["TreeIndex"] = range(0, len(AllErrors))

    # Round errors to three digits
    pdAllErrors["RoundedError"] = pdAllErrors["ClassificationError"].round(3)

    # Sort the DataFrame if order_errors is True
    if order_errors:
        pdAllErrors = pdAllErrors.sort_values(by="ClassificationError").reset_index(drop=True)

    # Plot the scatter plot
    fig, ax = plt.subplots(figsize=(20, 6))  # Increased figure width

    # Prepare the error counts (from Counter)
    error_counts = Counter(pdAllErrors["RoundedError"])

    # Format the error counts as a string with headers
    caption_header = "Error    Count"
    caption_rows = "\n".join([f"{error}    {count}" for error, count in error_counts.items()])

    # Combine header and rows
    caption = caption_header + "\n" + caption_rows

    # Display the caption vertically on the right side of the plot
    plt.text(1.02, 0.5, caption, ha='left', va='center', transform=ax.transAxes, fontsize=10)

    # Add small circles with the TreeIndex on each point (without scatter plot points)
    for i, row in pdAllErrors.iterrows():
        # Convert TreeIndex to int to remove the decimal .0
        ax.text(i, row['ClassificationError'], str(int(row['TreeIndex'])), color='black', fontsize=8, ha='center', va='center', 
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='circle,pad=0.3'))

    # Adjust the x-axis to provide more spacing between the circles
    ax.set_xlim(-1, len(pdAllErrors) + 1)  # Extend x-axis limits
    ax.set_ylim(0, 0.3)  # Extend y-axis limits
    ax.set_xlabel("Index of Trees in TREEFarms")
    ax.set_ylabel("Misclassification error")

    plt.show()

### # For loop for generating the decision trees ###
# for tree in range(0, Model.get_tree_count()):
#     G = nx.DiGraph()
#     add_nodes_and_edges(G, json.loads(Model[tree].json()))
#     draw_decision_tree(G, tree_index=tree, filename=f"/Users/simondn/Documents/RashomonActiveLearning/ResearchUpdates/Dec2/Iteration1_Trees/Tree{tree}.png")