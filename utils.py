# This file contains all kinds of utilities. 

import numpy as np
from collections import defaultdict

class simple_graph(object):

    """
    The idea is to group the features/descriptors under different labels using a very simple graph.
    Although the overall approach might seem trivial, it helps with keeping track of all the different definitions, especially when there are a lot.
    """

    def __init__(self, features):
        self.graph_dict = defaultdict(dict)
        self.features = features
        self.feature_ind = {}
        for i, f in enumerate(features):
            self.feature_ind[f] = i
    
    def _sort(self, extracted):
        return sorted(extracted, key = lambda x: self.feature_ind[x])

    def add_node(self, node):
        if node not in self.graph_dict:
            self.graph_dict[node] = {}

    def add_edge(self, node, neighbor):
        self.graph_dict[node][neighbor] = 1
        self.graph_dict[neighbor][node] = 1

    def delete_node(self, node):
        _ = self.graph_dict.pop(node, None)
        for key in self.graph_dict:
            _ = self.graph_dict[key].pop(node, None)

    def delete_edge(self, node_1, node_2):
        _ = self.graph_dict[node_1].pop(node_2, None)
        _ = self.graph_dict[node_2].pop(node_1, None)

    def get_graph(self, translation=None):
        self = create_feature_graph(self, translation=translation)

    def get_neighbors(self, node):
        return list(self.graph_dict[node].keys())

    def get_nodes(self):
        return list(self.graph_dict.keys())
        
    def extract(self, node):
        return list(self.graph_dict[node].keys())
    
    def extract_intersection(self, nodes):
        results = set(self.extract(nodes[0]))
        for node in nodes[1:]:
            results &= set(self.extract(node))
        return self._sort(list(results))
    
    def extract_union(self, nodes):
        results = set(self.extract(nodes[0]))
        for node in nodes[1:]:
            results |= set(self.extract(node))
        return self._sort(list(results))
    
def create_feature_graph(graph: simple_graph, translation=None) -> simple_graph:

    """
    Creates a very simple graph that connects the all the features and the groups that they belong to. 
    Assuming all the feature names follow the same pattern A_B_C, which means the feature belongs to groups A, B and C.
    The parameter translation should be a dictionary. For example,
    translation = {'1': 'first'}
    means feature names with '1' in them should be labeled 'first'.
    """

    for feature in graph.features:
        labels = feature.split('_')
        for label in labels:
            if translation and label in translation:
                label = translation[label]
            graph.add_edge(feature, label)
    
    return graph