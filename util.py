# This file contains all kinds of utilities. 

class simple_graph(object):

    """
    The idea is to group the features/descriptors under different labels using a very simple graph.
    Although the overall approach might seem trivial, it helps with keeping track of all the different definitions, especially when there are a lot.
    """

    def __init__(self, features: list):
        self.features = features
        self.graph_dict = {}


