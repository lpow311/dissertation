import networkx as nx
import random
import plotly.graph_objects as go
import pickle

# Parameters
num_start_nodes = 10
num_exit_nodes = 6
num_connection_nodes = 30

# Create a directed graph
G = nx.Graph()

# Add start nodes, exit nodes, and connection nodes
start_nodes = [(f"Start_{i}", {"type": "start"}) for i in range(1, num_start_nodes + 1)]
exit_nodes = [(f"Exit_{i}", {"type": "exit"}) for i in range(1, num_exit_nodes + 1)]
connection_nodes = [(f"Connection_{i}", {"type": "connection"}) for i in range(1, num_connection_nodes + 1)]

G.add_nodes_from(start_nodes)
G.add_nodes_from(exit_nodes)
G.add_nodes_from(connection_nodes)

# Randomly connect start nodes to connection nodes
for s, _ in start_nodes:
    targets = random.sample(connection_nodes, k=random.randint(2, 4))
    for t, _ in targets:
        G.add_edge(s, t)

# Randomly connect connection nodes to other connection nodes or exit nodes
for c, _ in connection_nodes:
    # Each connection node will have 1-3 outgoing edges
    connection_node_names = [con for con, _ in connection_nodes]
    exit_nodes_names = [e for e, _ in exit_nodes]
    possible_targets = connection_node_names + exit_nodes_names
    possible_targets.remove(c)  # Avoid self-loop
    targets = random.sample(possible_targets, k=random.randint(1, 3))
    for t in targets:
        G.add_edge(c, t)

# # Optionally connect some start nodes directly to exits
# for s in random.sample(start_nodes, k=random.randint(2, 5)):
#     G.add_edge(s, random.choice(exit_nodes))

# Ensure all exits connected to A connection node.
for e, _ in exit_nodes:
    neighbours = list(G.neighbors(e))
    if len(neighbours) == 0:
        connection_node_names = [con for con, _ in connection_nodes]
        connection = random.choice(connection_node_names)
        G.add_edge(e, connection)

# Get positions for nodes using spring layout
pos = nx.spring_layout(G, seed=42)

# Prepare data for Plotly visualization
edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=1, color='#888'),
    hoverinfo='none',
    mode='lines'
)

node_x = []
node_y = []
node_text = []
for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    node_text.append(node)

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers+text',
    text=node_text,
    textposition="top center",
    hoverinfo='text',
    marker=dict(
        size=10,
        color=['blue' if node.startswith('Start') else 'green' if node.startswith('Exit') else 'orange' for node in G.nodes()],
        line_width=2
    )
)

fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    # title='Network Graph with Start, Exit, and Connection Nodes',
                    # titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False),
                    yaxis=dict(showgrid=False, zeroline=False)
                ))

# Save the plot as JSON and PNG
pickle.dump(G, open('small_city_graph.pickle', 'wb'))
fig.write_image("small_city_graph.png")

print("Graph created with 10 start nodes, 6 exit nodes, and 30 connection nodes. Visualization saved as network_graph.json and network_graph.png.")