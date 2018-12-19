# -*- coding: utf-8 -*-


import sys
from beautifultable import BeautifulTable
import numpy as numpy
# =============================================================================


class Graph(object):
    user_defined_vertices = []
    dfs_timer = 0
    time = 0

    def __init__(self, vertices, edges):
        super(Graph, self).__init__()
        n = len(vertices)
        self.matrix = [[0 for x in range(n)] for y in range(n)]
        self.vertices = vertices
        self.edges = edges
        for edge in edges:
            x = vertices.index(edge[0])
            y = vertices.index(edge[1])
            self.matrix[x][y] = edge[2]

    def display(self):
        print(self.vertices)
        for i, v in enumerate(self.vertices):
            print(v, self.matrix[i])

    # Method to find the adjacent vertices of the given node
    def neighbour(self, node):
        n = len(self.matrix[0])
        row = self.vertices.index(node)
        neighbour = list()
        for i in range(n):
            if self.matrix[row][i] != 0:
                neighbour.append(self.vertices[i])
        return neighbour

    # Method to relax the edges
    def relax(self, root, node, weight, parent):
        node_index = self.vertices.index(node)
        root_index = self.vertices.index(root)

        if weight[node_index] > weight[root_index] + \
                self.matrix[root_index][node_index]:
            weight[self.vertices.index(node)] =\
                weight[root_index] + self.matrix[root_index][node_index]
            parent[node_index] = root

    # method is an algorithm that computes shortest paths from a single
    # source vertex for graph with negative, Zero  or positive weight
    def bellman_ford(self, source):
        n = len(self.vertices)
        weight = [sys.maxsize] * n
        parent = [None] * n
        visited_nodes = set()
        visited_nodes.add(source)
        iterations = 0
        weight[self.vertices.index(source)] = 0
        print('Initial')
        self.print_d_and_pi(iterations, weight, parent)
        for i in range(n - 1):
            for edge in self.edges:
                visited_nodes.add(edge[0])
                self.relax(edge[0], edge[1], weight, parent)
            iterations += 1
            self.print_d_and_pi(iterations, weight, parent)
        self.hop_number('z')

        for edge in self.edges:
            edge_1 = self.vertices.index(edge[1])
            edge_0 = self.vertices.index(edge[0])
            if weight[edge_1] > weight[edge_0] + \
                    self.matrix[edge_0][edge_1]:
                print("Negative cycle loop, no cost effective solution")

    def print_d_and_pi(self, iteration, d, pi):
        table = BeautifulTable()
        table.column_headers = ["Iteration", "Vertices", "Weight", "Parent"]
        for i, v in enumerate(self.vertices):
            table.append_row([iteration, v, d[i], pi[i]])
        print(table)

        print(table)

    def print_hop(self, discover):
        table = BeautifulTable()
        table.column_headers = ["vertex", "Hop"]
        for i, v in enumerate(self.vertices):
            table.append_row([v, discover[i]])
        print(table)


def main():

    graph = Graph(['u', 'v', 'w', 'x', 'y', 'z'],
                  [('u', 'v', 3),
                   ('v', 'w', 3),
                   ('w', 'z', 6),
                   ('z', 'y', 3),
                   ('y', 'x', 6),
                   ('x', 'u', 1),
                   ('w', 'y', 2),
                   ('w', 'x', 4),
                   ('v', 'x', 1),
                   ('u', 'w', 9),
                   ('v', 'u', 3),
                   ('w', 'v', 3),
                   ('z', 'w', 6),
                   ('y', 'z', 3),
                   ('x', 'y', 6),
                   ('u', 'x', 1),
                   ('y', 'w', 2),
                   ('x', 'w', 4),
                   ('x', 'v', 1),
                   ('w', 'u', 9)
                   ])
    graph.bellman_ford('z')

    graph = Graph(['u', 'v', 'w', 'x', 'y', 'z'],
                  [('u', 'v', 3),
                   ('v', 'w', 3),
                   ('w', 'z', 6),
                   ('z', 'y', 3),
                   ('y', 'x', 6),
                   ('x', 'u', 1),
                   ('w', 'y', 2),
                   ('w', 'x', 2),
                   ('v', 'x', 1),
                   ('u', 'w', 9),
                   ('v', 'u', 3),
                   ('w', 'v', 3),
                   ('z', 'w', 6),
                   ('y', 'z', 3),
                   ('x', 'y', 6),
                   ('u', 'x', 1),
                   ('y', 'w', 2),
                   ('x', 'w', 2),
                   ('x', 'v', 1),
                   ('w', 'u', 9)
                   ])
    graph.bellman_ford('z')


if __name__ == '__main__':
    main()
