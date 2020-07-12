import cirq
import numpy as np
from typing import Tuple
from networkx import nx

# from qiskit import
from src import SurfaceCodeGraph

# visualization tools
# %matplotlib inline
import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit


class CirqSurfaceCodeCircuit():

    def __init__(self, sigma: Tuple[Tuple[int]], alpha: Tuple[Tuple[int]]):
        # super().__init__()
        self.sigma = sigma
        self.alpha = alpha

        self.scgraph = SurfaceCodeGraph(self.sigma, self.alpha)

        '''
        Compute the permutation corresponding to phi and create a 
        'surface code circuit' based on a (multi)graph 'surface_code_graph'
        given by sigma, alpha, and phi
        Create quantum and classical registers based on the number of nodes in G
        '''
        # f = self.scgraph.compute_phi()
        self.phi = self.scgraph.phi

        self.qubits = [cirq.NamedQubit(str(node)) for node in self.scgraph.code_graph.nodes]
        self.circuit = cirq.Circuit()

        self.node_info = self.scgraph.node_info
        self.sigma_dict, self.alpha_dict, self.phi_dict = self.node_info

        for cycle in self.sigma:
            self.circuit.append(cirq.H(cirq.NamedQubit(str(cycle))))

        for cycle in self.phi:
            self.circuit.append(cirq.H(cirq.NamedQubit(str(cycle))))


    def x_measurement(self, qubit):
        """Measure 'qubit' in the X-basis
        :param qubit: a name to designate a cirq.NamedQubit(str(qubit))
        :return None
        """
        self.circuit.append(cirq.H(cirq.NamedQubit(str(qubit))))
        self.circuit.append(cirq.measure(cirq.NamedQubit(str(qubit))))
        self.circuit.append(cirq.H(cirq.NamedQubit(str(qubit))))

    def star_syndrome_measure(self, vertex: Tuple[int]):
        """
        Applies CX gates to surrounding qubits of a star then measures star qubit in X-basis
        :param vertex: a vertex node given by a cycle of sigma
        :return:
        """

        for node in self.scgraph.code_graph.neighbors(vertex):
            self.circuit.append(cirq.CNOT(cirq.NamedQubit(str(vertex)), cirq.NamedQubit(str(node))))
        self.x_measurement(vertex)

    def face_syndrome_measure(self, vertex: Tuple[int]):
        """
        Applies CZ gates to surrounding qubits on the boundary of a face then measures face qubit in X-basis
        :param vertex: a face node given by a cycle in phi
        :return:
        """

        for node in self.scgraph.code_graph.neighbors(vertex):
            self.circuit.append(cirq.CZ(cirq.NamedQubit(str(vertex)), cirq.NamedQubit(str(node))))

        self.x_measurement(vertex)

    def X_1_chain(self, edges):
        """
        Pauli product X operator for arbitrary 1-cochain given by
        a list of edges
        """
        for edge in edges:
            self.circuit.append(cirq.X(cirq.NamedQubit(str(edge))))

    def Z_1_chain(self, edges):
        """
        Pauli product Z operator for arbitrary 1-chain given by
        a list of edges
        """
        for edge in edges:
            self.circuit.append(cirq.Z(cirq.NamedQubit(str(edge))))

    def product_Z(self, faces):
        """
        Pauli product Z operator for arbitrary 2-chain boundary
        """

        boundary_nodes = self.scgraph.del_2(faces)
        for node in boundary_nodes:
            self.circuit.append(cirq.Z(cirq.NamedQubit(str(node))))

    def product_X(self, stars):
        """
        Pauli product X operator for arbitrary 0-cochain coboundary
        """
        coboundary_nodes = self.scgraph.delta_1(stars)
        for node in coboundary_nodes:
            self.circuit.append(cirq.X(cirq.NamedQubit(str(node))))



    def draw_graph(self, node_type='', layout=''):
        if layout == 'spring':
            pos = nx.spring_layout(self.scgraph.code_graph)
        if layout == 'spectral':
            pos = nx.spectral_layout(self.scgraph.code_graph)
        if layout == 'planar':
            pos = nx.planar_layout(self.scgraph.code_graph)
        if layout == 'shell':
            pos = nx.shell_layout(self.scgraph.code_graph)
        if layout == 'circular':
            pos = nx.circular_layout(self.scgraph.code_graph)
        if layout == 'spiral':
            pos = nx.spiral_layout(self.scgraph.code_graph)
        if layout == 'random':
            pos = nx.random_layout(self.scgraph.code_graph)
        if node_type == 'cycles':
            self.scgraph.draw('cycles', layout)
        if node_type == 'dict':
            self.scgraph.draw('dict', layout)

    def draw_circ(self, type = ''):
        if type not in ['SVG', '']:
            raise ValueError('node_type can be "SVG" or "plain"')
        if type == 'SVG':
            SVGCircuit(self.circuit)
        elif type == 'plain':
            print(self.circuit)
