import matplotlib.pyplot as plt
import networkx as nx


def plot_rail_network(rail_graph: nx.Graph):
    nx.draw(
        rail_graph.to_undirected(),
        rail_graph.nodes.data("position"),
        with_labels=True,
        node_color=dict(rail_graph.nodes.data(data="node_color")).values(),
        edge_color="gray",
        node_size=3,
        font_size=5,
    )
    plt.show()
