import random
import networkx as nx
import matplotlib.pyplot as plt


class SALSA:
    def __init__(self):
        self.currentLeftNodeVisits = dict()
        self.currentRightNodeVisits = dict()
        self.totalRightNodeVisits = dict()
        self.graph = {'a': ['0', '1', '3', '5'], 'b': ['3', '5'], 'c': ['0', '5'],
                      '0': ['a', 'c'], '1': ['a', 'b'], '3': ['a', 'b'], '5': ['b', 'c']}

    def show(self):
        B = nx.Graph()
        authorities = ['a', 'a', 'a', 'b', 'b', 'c', 'c', 'a', 'c']
        hubs =        ['0', '1', '3', '3', '5', '0', '5', '5', '5']

        B.add_nodes_from(authorities, bipartite=1)
        B.add_nodes_from(hubs, bipartite=0)
        B.add_edges_from([(hubs[idx], authorities[idx]) for idx in range(len(authorities))])
        pos = {node: [0, i] for i, node in enumerate(authorities)}
        pos.update({node: [1, i] for i, node in enumerate(hubs)})
        nx.draw(B, pos, with_labels=True)
        plt.show()

    def train(self, root, visits, n_iterations):
        is_left_to_right = True
        self.currentLeftNodeVisits[root] = visits
        for i in range(n_iterations):
            self.left() if is_left_to_right else self.right()
            is_left_to_right = not is_left_to_right

    def calc_prob(self):
        normalizing_sum = 0
        probs = []
        for k, v in self.totalRightNodeVisits.items():
            normalizing_sum += v
        for k, v in self.totalRightNodeVisits.items():
            probs.append(v / normalizing_sum)
        print(probs)

    def left(self):
        for node in self.currentLeftNodeVisits.keys():
            visits = self.currentLeftNodeVisits[node]
            edges = self.graph[node]
            for i in range(visits):
                if len(edges) > 0:  # avoid node with zero connections
                    # Select 1 random connection
                    random_position = random.randint(0, len(edges)-1)
                    edge = edges[random_position]
                    if edge not in self.currentRightNodeVisits:
                        self.currentRightNodeVisits[edge] = 0
                    if edge not in self.totalRightNodeVisits:
                        self.totalRightNodeVisits[edge] = 0
                    self.currentRightNodeVisits[edge] += 1
                    self.totalRightNodeVisits[edge] += 1
        self.currentLeftNodeVisits.clear()

    def right(self):
        for node in self.currentRightNodeVisits.keys():
            visits = self.currentRightNodeVisits[node]
            edges = self.graph[node]
            for i in range(visits):
                if len(edges) > 0:
                    random_position = random.randint(0, len(edges)-1)
                    edge = edges[random_position]
                    if edge not in self.currentLeftNodeVisits:
                        self.currentLeftNodeVisits[edge] = 0
                    self.currentLeftNodeVisits[edge] += 1
        self.currentRightNodeVisits.clear()


if __name__ == '__main__':
    salsa = SALSA()
    salsa.train('a', 4, 1000)
    salsa.show()
    print(salsa.totalRightNodeVisits)
    salsa.calc_prob()
