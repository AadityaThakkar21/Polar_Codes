import matplotlib.pyplot as plt

def build_tree(epsilon, x, y, level, max_level, nodes, edges):
    nodes.append((x, y, epsilon))
    if level == max_level:
        return
    delta_x = 1 / (2 ** (level + 2))
    left_x = x - delta_x
    right_x = x + delta_x
    next_y = y + 1
    left_epsilon = 2 * epsilon - epsilon ** 2
    right_epsilon = epsilon ** 2
    edges.append(((x, y), (left_x, next_y)))
    edges.append(((x, y), (right_x, next_y)))
    build_tree(left_epsilon, left_x, next_y, level + 1, max_level, nodes, edges)
    build_tree(right_epsilon, right_x, next_y, level + 1, max_level, nodes, edges)

# Parameters
initial_epsilon = 0.5  # Initial erasure probability
max_level = 5           # Depth of the tree (root is level 0)

# Generate nodes and edges
nodes = []
edges = []
build_tree(initial_epsilon, 0.5, 0, 0, max_level, nodes, edges)

# Create the plot
plt.figure(figsize=(10, 6))

# Draw edges
for edge in edges:
    (x1, y1), (x2, y2) = edge
    plt.plot([x1, x2], [y1, y2], 'k-', linewidth=1)

# Draw nodes and annotate with epsilon values
for x, y, eps in nodes:
    plt.plot(x, y, 'o', markersize=30, color='lightblue', mec='k')
    plt.text(x, y, f'{eps:.3f}', ha='center', va='center', fontsize=9)

# Adjust plot settings
plt.gca().invert_yaxis()  # Invert y-axis to place root at the top
plt.axis('off')
plt.title('Channel Polarization Tree with Erasure Probabilities', pad=20)
plt.tight_layout()
plt.show()