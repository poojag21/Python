# -*- coding: utf-8 -*-


import sys
import numpy as numpy
from queue import PriorityQueue
# =============================================================================


class Graph(object):
    """docstring for Graph"""
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

    def transpose(self):
        n = len(self.matrix)
        m = len(self.matrix[0])
        print('Transpose of the given matrix')
        result = numpy.zeros([n, m], dtype=int)
        for i in range(n):
            for j in range(m):
                result[j][i] = self.matrix[i][j]
        self.matrix = result

    # Method to calculate the in degree of the graph

    def in_degree(self):
        print("In degree of the graph:")
        n = len(self.matrix)
        m = len(self.matrix[0])
        count = numpy.zeros(n)
        for i in range(n):
            for j in range(m):
                if self.matrix[i][j] != 0:
                    count[j] = count[j] + 1
        self.print_degree(count)

    # Method to calculate the out degree of the graph

    def out_degree(self):
        print("Out degree of the graph:")
        n = len(self.matrix)
        m = len(self.matrix[0])
        count = numpy.zeros(n)
        for i in range(m):
            for j in range(n):
                if self.matrix[i][j] != 0:
                    count[i] = count[i] + 1
        self.print_degree(count)

    # Method to find the adjacent vertices of the given node
    def neighbours(self, node):
        n = len(self.matrix[0])
        row = self.vertices.index(node)
        neighbour = list()
        for i in range(n):
            if self.matrix[row][i] != 0:
                neighbour.append(self.vertices[i])
        return neighbour

    # Method to traverse a graph as far as possible along the given node
    # and estimate their time of first discovery and
    # the finish time defines the number of nodes finished or
    # discovered before finishing expansion of the node
    def dfs_traverse(self, node, discover, finish):
        self.time = self.time + 1
        index = self.vertices.index(node)
        discover[index] = self.time
        for neighbour in self.neighbours(node):
            if discover[self.vertices.index(neighbour)] == 0:
                self.dfs_traverse(neighbour, discover, finish)
        self.time = self.time + 1
        finish[index] = self.time

    # Method to implement depth first search on graph
    def dfs_on_graph(self):
        n = len(self.matrix)
        matrix_sorted = [[0 for x in range(n)] for y in range(n)]
        self.vertices.sort()
        for edge in self.edges:
            x = self.vertices.index(edge[0])
            y = self.vertices.index(edge[1])
            matrix_sorted[x][y] = edge[2]
        self.matrix = matrix_sorted

        discover = numpy.zeros(n)
        finish = numpy.zeros(n)
        for vertex in self.vertices:
            if discover[self.vertices.index(vertex)] == 0:
                self.dfs_traverse(vertex, discover, finish)
        self.print_discover_and_finish_time(discover, finish)

    # Method to find the vertices with minimum weight
    # to connect it to the graph
    def minimum_weight(self, weight, visited):
        min_value = sys.maxsize
        min_position = -1
        for i in range(len(self.vertices)):
            if visited[i] is False and weight[i] < min_value:
                min_value = weight[i]
                min_position = i
        return min_position

    # Method to find the minimum spanning tree for a weighted undirected graph
    def prim(self, root):
        iteration = 0
        n = len(self.vertices)
        weight = [sys.maxsize] * n
        parent = [None] * n
        visited = [False] * n
        weight[self.vertices.index(root)] = 0
        print('Initial')
        self.print_d_and_pi(iteration, weight, parent)
        for v in range(n - 1):
            next_min_position = self.minimum_weight(weight, visited)
            min_vertex = self.vertices[next_min_position]
            visited[next_min_position] = True
            for neighbour in self.neighbours(min_vertex):
                index_neighbour = self.vertices.index(neighbour)
                index_minimum = self.vertices.index(min_vertex)
                if visited[index_neighbour] is False and \
                        self.matrix[index_minimum][index_neighbour] \
                        < weight[index_neighbour]:
                    parent[index_neighbour] = min_vertex
                    weight[index_neighbour] = \
                        self.matrix[index_minimum][index_neighbour]
            iteration += 1
            self.print_d_and_pi(iteration, weight, parent)

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
    # source vertex for graph with negative or positive weight
    def bellman_ford(self, source):
        n = len(self.vertices)
        weight = [sys.maxsize] * n
        parent = [None] * n
        iterations = 0
        weight[self.vertices.index(source)] = 0
        print('Initial')
        self.print_d_and_pi(iterations, weight, parent)
        for i in range(n - 1):
            for edge in self.edges:
                self.relax(edge[0], edge[1], weight, parent)
            iterations += 1
            self.print_d_and_pi(iterations, weight, parent)
        for edge in self.edges:
            edge_1 = self.vertices.index(edge[1])
            edge_0 = self.vertices.index(edge[0])
            if weight[edge_1] > weight[edge_0] + \
                    self.matrix[edge_0][edge_1]:
                print("Negative cycle loop, no solution")

    # method to update the value of the weight of the edges and heapify them
    def update_heap(self, pq, weight):
        n = pq.qsize()
        node = set()
        for i in range(n):
            u = pq.get()
            p = u[1]
            node.add(p)
        for j, vertex in enumerate(node):
            pq.put((weight[self.vertices.index(vertex)], vertex))

    # method is an algorithm that computes shortest paths from a
    # single source vertex for graph with negative
    def dijkstra(self, source):
        for edge in self.edges:
            if edge[2] < 0:
                print('No Solution, the weight of edge is negative')
                return 0
            else:
                iterations = 0
                n = len(self.vertices)
                weight = [sys.maxsize] * n
                parent = [None] * n
                node = set()
                pq = PriorityQueue()
                weight[self.vertices.index(source)] = 0
                for i, vertex in enumerate(self.vertices):
                    pq.put((weight[self.vertices.index(vertex)], vertex))

        while not pq.empty():
            u = pq.get()
            p = u[1]
            node.add(p)
            iterations += 1
            for neighbour in self.neighbours(p):
                self.relax(p, neighbour, weight, parent)
            self.update_heap(pq, weight)
            self.print_d_and_pi(iterations, weight, parent)

    def print_d_and_pi(self, iteration, d, pi):
        assert((len(d) == len(self.vertices)) and
               (len(pi) == len(self.vertices)))

        print("Iteration: {0}".format(iteration))
        for i, v in enumerate(self.vertices):
            print("Vertex: {0}\td: {1}\tpi: {2}".format
                  (v, 'inf' if d[i] == sys.maxsize else d[i], pi[i]))

    def print_discover_and_finish_time(self, discover, finish):
        assert((len(discover) == len(self.vertices)) and
               (len(finish) == len(self.vertices)))
        for i, v in enumerate(self.vertices):
            print("Vertex: {0}\tDiscovered: {1}\tFinished: {2}".format(
                    v, discover[i], finish[i]))

    def print_degree(self, degree):
        assert(len(degree) == len(self.vertices))
        for i, v in enumerate(self.vertices):
            print("Vertex: {0}\tDegree: {1}".format(v, degree[i]))


def main():
    graph = Graph(['1', '2', '3', '4', '5', '6'],
                    [('1', '2', 6),
                     ('2', '5', 3),
                     ('5', '6', 6),
                     ('6', '4', 2),
                     ('4', '1', 5),
                     ('3', '1', 1),
                     ('3', '2', 5),
                     ('3', '5', 6),
                     ('3', '6', 4),
                     ('3', '4', 5)])
    graph.transpose()
    graph.display()

    graph = Graph(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
                    [('a', 'b', 1),
                     ('a', 'f', 1),
                     ('b', 'c', 1),
                     ('b', 'i', 1),
                     ('b', 'g', 1),
                     ('c', 'd', 1),
                     ('c', 'i', 1),
                     ('d', 'e', 1),
                     ('d', 'g', 1),
                     ('d', 'h', 1),
                     ('d', 'i', 1),
                     ('e', 'f', 1),
                     ('e', 'h', 1),
                     ('f', 'g', 1),
                     ('g', 'h', 1)])
    graph.dfs_on_graph();

    graph = Graph(['A', 'B', 'C', 'D', 'E'],
                    [('A', 'B', -1),
                     ('A', 'C', 4),
                     ('B', 'C', 3),
                     ('B', 'D', 2),
                     ('B', 'E', 2),
                     ('D', 'B', 1),
                     ('D', 'C', 5),
                     ('E', 'D', -3)])
    graph.bellman_ford('A')

    graph = Graph(['A', 'B', 'C', 'D', 'E'],
                    [('A', 'B', -1),
                     ('A', 'C', 4),
                     ('B', 'C', 3),
                     ('B', 'D', 2),
                     ('B', 'E', 2),
                     ('D', 'B', -1),
                     ('D', 'C', 5),
                     ('E', 'D', -3)])
    graph.bellman_ford('A')

    graph = Graph(['1', '2', '3', '4', '5', '6'],
                    [('1', '2', 6),
                     ('2', '5', 3),
                     ('5', '6', 6),
                     ('6', '4', 2),
                     ('4', '1', 5),
                     ('3', '1', 1),
                     ('3', '2', 5),
                     ('3', '5', 6),
                     ('3', '6', 4),
                     ('3', '4', 5)])
    graph.dijkstra('3')

    graph = Graph(['q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'],
                  [('q', 's', 1),
                   ('s', 'v', 1),
                   ('v', 'w', 1),
                   ('w', 's', 1),
                   ('q', 'w', 1),
                   ('q', 't', 1),
                   ('t', 'x', 1),
                   ('x', 'z', 1),
                   ('z', 'x', 1),
                   ('t', 'y', 1),
                   ('y', 'q', 1),
                   ('r', 'y', 1),
                   ('r', 'u', 1),
                   ('u', 'y', 1)])

    graph.display()
    graph.transpose()
    graph.display()
    graph.transpose()
    graph.display()
    graph.in_degree()
    graph.out_degree()

    graph = Graph(['q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'],
                  [('q', 's', 1),
                   ('s', 'v', 1),
                   ('v', 'w', 1),
                   ('w', 's', 1),
                   ('q', 'w', 1),
                   ('q', 't', 1),
                   ('t', 'x', 1),
                   ('x', 'z', 1),
                   ('z', 'x', 1),
                   ('t', 'y', 1),
                   ('y', 'q', 1),
                   ('r', 'y', 1),
                   ('r', 'u', 1),
                   ('u', 'y', 1)])
    graph.display()
    graph.dfs_on_graph()

    graph = Graph(['z', 's', 'r', 't', 'u', 'v', 'w', 'x', 'y', 'q'],
                  [('q', 's', 1),
                   ('s', 'v', 1),
                   ('v', 'w', 1),
                   ('w', 's', 1),
                   ('q', 'w', 1),
                   ('q', 't', 1),
                   ('t', 'x', 1),
                   ('x', 'z', 1),
                   ('z', 'x', 1),
                   ('t', 'y', 1),
                   ('y', 'q', 1),
                   ('r', 'y', 1),
                   ('r', 'u', 1),
                   ('u', 'y', 1)])
    graph.dfs_on_graph()

    graph = Graph(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
                    [('a', 'b', 1),
                     ('a', 'f', 1),
                     ('b', 'c', 1),
                     ('b', 'i', 1),
                     ('b', 'g', 1),
                     ('c', 'd', 1),
                     ('c', 'i', 1),
                     ('d', 'e', 1),
                     ('d', 'g', 1),
                     ('d', 'h', 1),
                     ('d', 'i', 1),
                     ('e', 'f', 1),
                     ('e', 'h', 1),
                     ('f', 'g', 1),
                     ('g', 'h', 1)])
    graph.dfs_on_graph()

    graph = Graph(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
                  [('A', 'H', 6),
                   ('H', 'A', 6),
                   ('A', 'B', 4),
                   ('B', 'A', 4),
                   ('B', 'H', 5),
                   ('H', 'B', 5),
                   ('B', 'C', 9),
                   ('C', 'B', 9),
                   ('G', 'H', 14),
                   ('H', 'G', 14),
                   ('F', 'H', 10),
                   ('H', 'F', 10),
                   ('B', 'E', 2),
                   ('E', 'B', 2),
                   ('G', 'F', 3),
                   ('F', 'G', 3),
                   ('E', 'F', 8),
                   ('F', 'E', 8),
                   ('D', 'E', 15),
                   ('E', 'D', 15)])
    graph.prim('G')

    graph = Graph(['s', 't', 'x', 'y', 'z'],
                  [('t', 'x', 5),
                   ('t', 'y', 8),
                   ('t', 'z', -4),
                   ('x', 't', -2),
                   ('y', 'x', -3),
                   ('y', 'z', 9),
                   ('z', 'x', 7),
                   ('z', 's', 2),
                   ('s', 't', 6),
                   ('s', 'y', 7)])
    graph.bellman_ford('z')

    graph = Graph(['a', 'b', 'c', 'd'],
                  [('a', 'b', 4),
                   ('c', 'b', -10),
                   ('d', 'c', 3),
                   ('b', 'd', 5),
                   ('a', 'd', 5)])
    graph.bellman_ford('a')

    graph = Graph(['s', 't', 'x', 'y', 'z'],
                  [('t', 'x', 5),
                   ('t', 'y', 8),
                   ('t', 'z', -4),
                   ('x', 't', -2),
                   ('y', 'x', -3),
                   ('y', 'z', 9),
                   ('z', 'x', 4),
                   ('z', 's', 2),
                   ('s', 't', 6),
                   ('s', 'y', 7)])
    graph.bellman_ford('s')

    graph = Graph(['s', 't', 'x', 'y', 'z'],
                  [('s', 't', 3),
                   ('s', 'y', 5),
                   ('t', 'x', 6),
                   ('t', 'y', 2),
                   ('x', 'z', 2),
                   ('y', 't', 1),
                   ('y', 'x', 4),
                   ('y', 'z', 6),
                   ('z', 's', 3),
                   ('z', 'x', 7)])
    graph.dijkstra('s')

    graph = Graph(['1', '2', '3', '4', '5', '6'],
                    [('1', '2', 6),
                     ('2', '5', 3),
                     ('5', '6', 6),
                     ('6', '4', 2),
                     ('4', '1', 5),
                     ('3', '1', 1),
                     ('3', '2', 5),
                     ('3', '5', 6),
                     ('3', '6', 4),
                     ('3', '4', 5)])

    graph.dijkstra('3')

    graph = Graph(['s', 't', 'x', 'y', 'z'],
                  [('s', 't', -10),
                   ('s', 'y', 5),
                   ('t', 'x', 1),
                   ('t', 'y', -2),
                   ('x', 'z', 4),
                   ('y', 't', 3),
                   ('y', 'x', 9),
                   ('y', 'z', 2),
                   ('z', 's', 7),
                   ('z', 'x', 6)])
    graph.dijkstra('s')


if __name__ == '__main__':
    main()
