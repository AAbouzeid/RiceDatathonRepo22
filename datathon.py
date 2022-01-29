import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

import csv
import numpy as np
import pandas as pd
from collections import defaultdict
from stellargraph.data import BiasedRandomWalk
from stellargraph import StellarGraph
from gensim.models import Word2Vec
import multiprocessing
import argparse

def train(p):
    file = open('bill_challenge_datasets/Training/training_graph.csv')
    csvreader = csv.reader(file)
    edges = []
    for row in csvreader:
        if row[0][0] != "n":
            edges.append(row)

    training_edges = []
    test_true_edges = []
    test_false_edges = []
    graph = defaultdict(set)
    training_graph = defaultdict(set)
    for i, row in enumerate(edges):
        if i % 10 == 0:
            test_true_edges.append(row)
        else:
            training_edges.append(row)
            training_graph[row[0]].add(row[1])
            training_graph[row[1]].add(row[0])
        graph[row[0]].add(row[1])
        graph[row[1]].add(row[0])

    for i in range(0, len(edges) - 65000, 2):
        if i % 2 == 0:
            if edges[i][0] not in edges[i + 1] and edges[i][0] not in graph[edges[i + 1][0]]:
                test_false_edges.append([edges[i][0], edges[i+1][0]])

    pdgraph = pd.DataFrame(
        {"source": [x[0] for x in training_edges], "target": [x[1] for x in training_edges]}
    )
    sgraph = StellarGraph(edges=pdgraph)
    print(sgraph.info())

    rw = BiasedRandomWalk(sgraph)

    walks = rw.run(
        nodes=list(sgraph.nodes()),  # root nodes
        length=50,  # maximum length of a random walk
        n=10,  # number of random walks per root node
        p=p,  # Defines (unormalised) probability, 1/p, of returning to source node
        q=1.0,  # Defines (unormalised) probability, 1/q, for moving away from source node
    )
    print("Number of random walks: {}".format(len(walks)))

    str_walks = [[str(n) for n in walk] for walk in walks]
    model = Word2Vec(str_walks, size=20, min_count=0, sg=1, workers=multiprocessing.cpu_count(), iter=3)

    best_acc = -1
    best_threshold = None
    for threshold_dist in [1, 2, 3, 4, 5, 6, 7]:
        amount_correct = 0
        amount_incorrect = 0
        for row in test_true_edges:
            try:
                # if distance less than threshold, we predict an edge
                if np.linalg.norm(model.wv[row[0]] - model.wv[row[1]]) < threshold_dist:
                    amount_correct += 1
                else:
                    amount_incorrect += 1
            except:
                continue
            
        for row in test_false_edges:
            try:
                # if distance less than threshold, we predict an edge
                if np.linalg.norm(model.wv[row[0]] - model.wv[row[1]]) < threshold_dist:
                    amount_incorrect += 1
                else:
                    amount_correct += 1
            except:
                continue
        acc = amount_correct / (amount_correct + amount_incorrect)
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold_dist

    result = f"Best Accuracy for p={p} it acc={best_acc} at threshold={best_threshold}\n"
    f = open(f"results_p{p}.txt", "w")
    f.write(result)
    f.close()

def main():
    print("Starting")
    parser = argparse.ArgumentParser()
    parser.add_argument('p', action="store", metavar="<p>",
                        help="")
    args = parser.parse_args()
    train(float(args.p))

main()