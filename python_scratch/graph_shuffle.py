#!/bin/bash

import numpy as np
import copy as cp
import matplotlib.pyplot as plt

class SoA:
    def __init__(self):
        self.points = np.random.uniform(0.0, 1.0, 2)
        self.type = 0
        self.connections = -np.ones(2, int)

    def __str__(self):
        return "{}\n{}\n{}\n".format(self.points, self.type, self.connections)

class Graph:
    def __init__(self, ppmt, nmt):
        self.ppmt = ppmt
        self.nmt = nmt
        self.size = ppmt*nmt
        self.graph = [SoA() for x in range(0, self.size)]

    def show(self):
        for soa in self.graph:
            print(soa)

    def shuffle_correct(self):
        graph_temp = cp.deepcopy(self.graph)

        original = np.arange(0, self.size)
        permute  = np.arange(0, self.size)
        np.random.shuffle(permute)

        for idx, soa in zip(permute, self.graph):
            graph_temp[idx] = cp.deepcopy(soa)
        #system_temp.show()

        for idx, soa in zip(permute, graph_temp):
            for jdx, con in zip(np.arange(0, 2), soa.connections):
                if(con != -1):
                    soa.connections[jdx] = permute[con]
        self.graph = cp.deepcopy(graph_temp)
        del graph_temp

    def shuffle_incorrect(self):
        graph_temp = cp.deepcopy(self.graph)

        original = np.arange(0, self.size)
        permute  = np.arange(0, self.size)
        np.random.shuffle(permute)

        for idx, soa in zip(permute, self.graph):
            graph_temp[idx] = cp.deepcopy(soa)
        #system_temp.show()

        self.graph = cp.deepcopy(graph_temp)
        del graph_temp

def initialize_mt(system):
    for i in range(0, system.size, system.ppmt):
        c_x = np.random.uniform(1.0, 9.0)
        c_y = np.random.uniform(1.0, 9.0)
        t_s = np.random.uniform(0.0, 2.0*np.pi)
        length_max = 0.5
        r_s = np.random.uniform(0.5*length_max, 0.55*length_max)
        x1 = r_s*np.cos(t_s)+c_x
        y1 = r_s*np.sin(t_s)+c_y
        x3 = c_x-r_s*np.cos(t_s)
        y3 = c_y-r_s*np.sin(t_s)

        system.graph[i].type = 0
        system.graph[i].connections[0] = i+1
        system.graph[i].points[0] = x1
        system.graph[i].points[1] = y1

        system.graph[i+1].type = 1
        system.graph[i+1].connections[0] = i
        system.graph[i+1].connections[1] = i+2
        system.graph[i+1].points[0] = c_x
        system.graph[i+1].points[1] = c_y

        system.graph[i+2].type = 2
        system.graph[i+2].connections[0] = i+1
        system.graph[i+2].points[0] = x3
        system.graph[i+2].points[1] = y3

def display_mt(system, file=""):
    color_set = ['r', 'b', 'k']
    node_set = []
    edge_set = []
    for soa in system.graph:
        node_set.append(plt.Circle((soa.points[0], soa.points[1]), radius=0.1, color=color_set[soa.type]))
        for con in soa.connections:
            if(con != -1):
                x1 = soa.points[0]
                y1 = soa.points[1]
                x3 = system.graph[con].points[0]
                y3 = system.graph[con].points[1]
                edge_set.append(plt.Line2D((x1, x3), (y1, y3), lw=2, color='k'))

    plt.figure(figsize=[12.8, 9.6])
    plt.title(file)
    for circle in node_set:
        plt.gca().add_patch(circle)
    for edge in edge_set:
        plt.gca().add_line(edge)
    plt.xlim([0, 10])
    plt.ylim([0, 10])
    if(file):
        plt.savefig("{}.png".format(file))
        print("saved as {}.png\n".format(file))
    plt.show()

def main():
    ppmt = 3
    nmt  = 15
    system = Graph(ppmt, nmt)
    initialize_mt(system)
    display_mt(system, "unshuffled")
    system.shuffle_correct()
    display_mt(system, "shuffle correct")
    system.shuffle_incorrect()
    display_mt(system, "shuffle incorrect")


if __name__ == '__main__':
    main()
