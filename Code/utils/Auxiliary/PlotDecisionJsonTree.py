### Packages ###
import json
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

### Graph Tree ###
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx

def add_nodes_and_edges(graph, data, df_columns, parent=None, edge_label=""):
    """
    Recursively add nodes and edges to the graph from the JSON data, using column names from the DataFrame
    and replacing underscores with equals signs in the output.

    Parameters:
        graph (networkx.DiGraph): The graph to add nodes and edges to.
        data (dict): The tree JSON data.
        df_columns (list): List of column names from the DataFrame.
        parent (int): Parent node ID.
        edge_label (str): Label for the edge ("True" or "False").
    """
    if isinstance(data, dict):
        # Check if the node is a leaf
        if "prediction" in data:
            node_label = str(data["prediction"])  # Use only the prediction value
            is_leaf = True
        else:
            # Map the feature index to the column name and replace "_" with "="
            feature_index = data["feature"]
            node_label = df_columns[feature_index].replace("_", "=")  # Replace underscores with equals signs
            is_leaf = False
        
        # Add the current node to the graph
        current_node = len(graph)
        graph.add_node(current_node, label=node_label, is_leaf=is_leaf)  # Add is_leaf attribute
        
        if parent is not None:
            graph.add_edge(parent, current_node, label=edge_label)
        
        # Add child nodes recursively
        if "true" in data:
            add_nodes_and_edges(graph, data["true"], df_columns, current_node, "True")
        if "false" in data:
            add_nodes_and_edges(graph, data["false"], df_columns, current_node, "False")
    else:
        raise ValueError("Unsupported JSON structure.")
def compute_positions(graph, node, pos, x=0, y=0, layer_width=1.0, depth=1.0):
    """
    Compute positions for nodes in a tree structure.
    layer_width: Controls horizontal spacing (smaller = less spread out).
    depth: Controls vertical spacing (smaller = more compact vertically).
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
def draw_decision_tree(graph, groups, group_colors, tree_index, filename=None):
    """
    Draw the decision tree using networkx and matplotlib. Optionally save to a file.
    """
    pos = {}
    compute_positions(graph, 0, pos, layer_width=2.0, depth=1.5)  # Increased spacing for better layout
    labels = nx.get_node_attributes(graph, "label")
    edge_labels = nx.get_edge_attributes(graph, "label")
    
    plt.figure(figsize=(10, 12))  # Adjust figure size
    ax = plt.gca()  # Get current axes for custom drawing

    # Separate leaf and feature nodes
    leaf_nodes = [n for n, attr in graph.nodes(data=True) if attr["is_leaf"]]
    feature_nodes = [n for n, attr in graph.nodes(data=True) if not attr["is_leaf"]]

    # Draw edges
    nx.draw_networkx_edges(
        graph,
        pos,
        width=2,  # Thicker edge lines
        edge_color="gray",
    )

    # Draw edge labels
    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=edge_labels,
        font_size=15,
        font_weight="bold",
        rotate=False  # Keep labels horizontal
    )

    # Draw leaf nodes (circles with red/green)
    leaf_colors = ["red" if labels[n] == "0" else "green" for n in leaf_nodes]  # Red for 0, Green for 1
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=leaf_nodes,
        node_shape="o",  # Circle shape
        node_size=800,  # Reduced size for better fit
        node_color=leaf_colors,
        edgecolors="black",
    )
    # Add labels ("0" or "1") inside the circles
    for node in leaf_nodes:
        x, y = pos[node]
        ax.text(
            x, y, labels[node],  # Label corresponds to the prediction ("0" or "1")
            ha="center", va="center",
            fontsize=20, fontweight="bold", color="white", zorder=5  # White text for visibility
        )

    # Draw feature nodes as rectangles
    for node in feature_nodes:
        x, y = pos[node]
        label = labels[node]
        
        # Calculate rectangle dimensions based on the label length
        rect_width = 0.04 * len(label)  # Width scales with label length
        rect_height = 0.2  # Fixed height
        
        # Draw the rectangle
        rect = patches.Rectangle(
            (x - rect_width / 2, y - rect_height / 2),  # Bottom-left corner
            rect_width,
            rect_height,
            edgecolor="black",
            facecolor="lightblue",
            zorder=3,
        )
        ax.add_patch(rect)

        # Add the label inside the rectangle
        ax.text(
            x, y, label,
            ha="center", va="center",
            fontsize=20, fontweight="bold",
            zorder=4,
        )

    # Assign group based on tree index
    group_name = ""
    for group, trees in groups.items():
        if tree_index in trees:
            group_name = group
            break

    # Get the color for the group's bbox
    group_color = group_colors.get(group_name, 'lightgray')  # Default to lightgray if not found

    # Add title in the center of the plot with a gray box outlined with black
    plt.text(
        0.5, 0.85,  # x=0.5 for centering, y=0.95 for slightly near the top
        f"Decision Tree: {tree_index} ({group_name})",
        ha='center',  # Horizontal alignment in the center
        va='center',  # Vertical alignment in the center
        transform=plt.gcf().transFigure,  # Use the figure coordinates for positioning
        fontsize=20,
        fontweight='bold',
        bbox=dict(facecolor=group_color, edgecolor='black', boxstyle='round,pad=0.5')  # Use group color for bbox
    )

    # Dynamically adjust axis limits based on node positions
    x_values = [p[0] for p in pos.values()]
    y_values = [p[1] for p in pos.values()]
    ax.set_xlim(min(x_values) - 0.15, max(x_values) + 0.15)  # Add a bit of padding, but keep it tight
    ax.set_ylim(min(y_values) - 0.15, max(y_values) + 0.5)  # Add a bit of padding, but keep it tight

    ax.axis("off")  # Turn off axis lines and ticks

    # Save or show the plot
    if filename:
        plt.savefig(filename, format="png", bbox_inches="tight")
    else:
        plt.show()
    
    plt.close()  # Close the plot to free memory and avoid overlap

### UNREAL vs. DUREAL Plot ###
def PlotTreeFarmsDecisionTreeErrorsWithGroups(AllErrors, order_errors=True):
    """
    Plot misclassification errors with TreeIndex values and group labels.

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
    fig, ax = plt.subplots(figsize=(10, 1210))  # Adjust figure size

    # Scatter plot for points with equal size
    ax.scatter(
        pdAllErrors.index,  # X-axis values
        pdAllErrors["ClassificationError"],  # Y-axis values
        s=200,  # Size of points
        color="white",  # Point color
        edgecolor="black",  # Outline color
        zorder=2,  # Layer above gridlines
    )

    # Add text labels for TreeIndex
    for i, row in pdAllErrors.iterrows():
        ax.text(
            i, row["ClassificationError"],  # Position
            str(int(row["TreeIndex"])),  # Text (TreeIndex as integer)
            color="black", fontsize=20, ha="center", va="center"
        )

    # Add group labels below the specified indices
    # Group 1
    group_1_x = [0, 1, 2, 3, 6, 7, 8]
    ax.text(
        sum(group_1_x) / len(group_1_x)-1, 0.033,  # Center position for the group
        'Group 1',
        color="black",
        fontsize=20, ha="center", va="center",
        bbox=dict(facecolor="cyan", alpha=0.5, edgecolor="black"),
        fontweight='bold'
    )

    # Group 2
    group_2_x = [4, 5, 9, 12, 13]
    ax.text(
        sum(group_2_x) / len(group_2_x)+.5 , 0.04,  # Center position for the group
        'Group 2',
        color="black",
        fontsize=20, ha="center", va="center",
        bbox=dict(facecolor="cyan", alpha=0.5, edgecolor="black"),
        fontweight='bold'
    )

    # Group 3
    group_3_x = [10, 11]
    ax.text(
        sum(group_3_x) / len(group_3_x)+2, 0.048,  # Center position for the group
        'Group 3',
        color="black",
        fontsize=20, ha="center", va="center",
        bbox=dict(facecolor="cyan", alpha=0.5, edgecolor="black"),
        fontweight='bold'
    )

    # Arrow from Group 1 to Unique Ensemble
    ax.annotate(
        "", xy=(sum(group_1_x) / len(group_1_x)-1, 0.023), xytext=(9.1, -0.049+0.025),  # Adjust the positions for arrows
        arrowprops=dict(facecolor="cyan", edgecolor="cyan", arrowstyle="<-", lw=2),
    )

    # Arrow from Group 2 to Unique Ensemble
    ax.annotate(
        "", xy=(sum(group_2_x) / len(group_2_x)+.5 , 0.03), xytext=(9.1, -0.05+0.025),  # Adjust the positions for arrows
        arrowprops=dict(facecolor="cyan", edgecolor="cyan", arrowstyle="<-", lw=2),
    )

    # Arrow from Group 3 to Unique Ensemble
    ax.annotate(
        "", xy=(sum(group_3_x) / len(group_3_x)+2, 0.038), xytext=(9.2, -0.05+0.025),  # Adjust the positions for arrows
        arrowprops=dict(facecolor="cyan", edgecolor="cyan", arrowstyle="<-", lw=2),
    )
    
    # Add arrows and annotations for Unique and Duplicate ensembles
    ax.annotate(
        "Unique Ensemble (UNREAL)",
        xy=(4, 0.02), xytext=(9.1, -0.06+0.025),
        fontsize=20, color="black", ha="center", va="center",
        bbox=dict(facecolor="cyan", edgecolor="black"),
        fontweight='bold'
    )
    ax.annotate(
        "Duplicate Ensemble (DUREAL)",
        xy=(9.5, 0.1), xytext=(9.5, 0.15),
        fontsize=20, color="black", ha="center", va="center",
        bbox=dict(facecolor="orange", edgecolor="black"),
        fontweight='bold'
    )

    # Define the points for the line
    line_x = [0, 1, 2, 3, 4, 5, 6]
    line_y = [0.072] * len(line_x)  # Set the y-values to 0 for a horizontal line
    ax.plot(line_x, line_y, color="orange", linewidth=2, marker="|", markersize=10)  # Use '|' markers for the points
    ax.text(sum(line_x) / len(line_x), 0.081, "7 trees", color="orange", ha="center", va="center", fontsize=20, fontweight='bold')

    # Define the points for the line
    line_x = [7,8,9,10,11]
    line_y = [0.079] * len(line_x)  # Set the y-values to 0 for a horizontal line
    ax.plot(line_x, line_y, color="orange", linewidth=2, marker="|", markersize=10)  # Use '|' markers for the points
    ax.text(sum(line_x) / len(line_x), 0.088, "5 trees", color="orange", ha="center", va="center", fontsize=20, fontweight='bold')

    # Define the points for the line
    line_x = [12, 13]
    line_y = [0.086] * len(line_x)  # Set the y-values to 0 for a horizontal line
    ax.plot(line_x, line_y, color="orange", linewidth=2, marker="|", markersize=10)  # Use '|' markers for the points
    ax.text(sum(line_x) / len(line_x), 0.095, "2 trees", color="orange", ha="center", va="center", fontsize=20, fontweight='bold')

    # Draw lines from each of the groups to DUREAL (duplicate ensemble)
    # Line from Group 1 to DUREAL
    ax.annotate(
        "", xy=(sum(group_1_x) / len(group_1_x)-1, 0.033+.055), xytext=(9.05, 0.14),  # Starting point from Group 1, ending at DUREAL
        arrowprops=dict(facecolor="orange", edgecolor="orange", arrowstyle="<-", lw=2),
    )

    # Line from Group 2 to DUREAL
    ax.annotate(
        "", xy=(sum(group_2_x) / len(group_2_x)+.5 , 0.04+.055), xytext=(9.05, 0.14),  # Starting point from Group 2, ending at DUREAL
        arrowprops=dict(facecolor="orange", edgecolor="orange", arrowstyle="<-", lw=2),
    )

    # Line from Group 3 to DUREAL
    ax.annotate(
        "", xy=(sum(group_3_x) / len(group_3_x)+2, 0.048+.055), xytext=(9.05, 0.14),  # Starting point from Group 3, ending at DUREAL
        arrowprops=dict(facecolor="orange", edgecolor="orange", arrowstyle="<-", lw=2),
    )

    # Prepare the error counts (from Counter)
    error_counts = Counter(pdAllErrors["RoundedError"])
    caption_header = "Error    Count"
    caption_rows = "\n".join([f"{error}    {count}" for error, count in error_counts.items()])
    caption = caption_header + "\n" + caption_rows

    # Add the caption inside the plot at the top-left corner
    ax.text(
        0.05, 0.95,  # Relative position in the axes (5% from left, 95% from top)
        caption,
        ha="left", va="top",  # Align text to the top-left corner
        transform=ax.transAxes,  # Use axes coordinates for positioning
        fontsize=20, family="monospace", bbox=dict(facecolor="white", alpha=0.8)
    )

    # Customize the plot appearance
    ax.set_xlim(-1, len(pdAllErrors)+2)  # Extend x-axis limits for spacing
    ax.set_ylim(-0.06, .2)  # Extend y-axis limits
    ax.set_xlabel("Index of Trees in TREEFarms")
    ax.set_ylabel("Misclassification Error")
    ax.set_xticks([])  # Remove x-axis ticks for a clean look

    plt.show()
