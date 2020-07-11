# This file contains all kinds of utilities. 

from collections import defaultdict

class simple_graph(object):

    """
    The idea is to group the features/descriptors under different labels using a very simple graph.
    Although the overall approach might seem trivial, it helps with keeping track of all the different definitions, especially when there are a lot.
    """

    def __init__(self, features: list):
        self.features = features
        self.graph_dict = defaultdict(dict)

    def add_node(self, node):
        if node not in self.graph_dict:
            self.graph_dict[node] = {}

    def add_edge(self, node, neighbor):
        self.graph_dict[node][neighbor] = 1
        self.graph_dict[neighbor][node] = 1

    def delete_node(self, node):

    def delete_edge(self, edge):

        
    def get_nodes(self):
        return list(self.graph_dict.keys())
        
    # def extract(self, node):
    #     return list(self.graph_dict[node].keys())
    
    # def extract_intersection(self, nodes):
    #     results = set(self.extract(nodes[0]))
    #     for node in nodes[1:]:
    #         results = results & set(self.extract(node))
    #     return list(results)
    
    # def extract_union(self, nodes):
    #     results = set(self.extract(nodes[0]))
    #     for node in nodes[1:]:
    #         results = results | set(self.extract(node))
    #     return list(results)

