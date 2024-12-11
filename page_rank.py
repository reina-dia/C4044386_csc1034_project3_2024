import sys
import os
import time
import argparse
import random
import networkx as nx
import numpy as np
from direct.task.TaskTester import counter

from progress import Progress


def load_graph(args):
    """Load graph from text file

    Parameters:
    args -- arguments named tuple

    Returns:
    A dict mapling a URL (str) to a list of target URLs (str).
    """

    #creating an adjacenecy list to store graph data
    # Create dictionary to store connections in
    #pageRank_dict = {}
    # Iterate through the file line by line
    #for line in args.datafile:
        # And split each line into two URLs
    #    node, target = line.split()

    #    if node in pageRank_dict:
    #        pageRank_dict[node].append(target)
    #     else:
    #        pageRank_dict[node] = [target]
    #return pageRank_dict

    #using networkx to store graph data
    pageRank_dict = nx.DiGraph()

    for line in args.datafile:

        node, target = line.split()
        pageRank_dict.add_edge(node, target)



def print_stats(graph):
        """Print number of nodes and edges in the given graph"""
        raise RuntimeError("This function is not implemented yet.")


def stochastic_page_rank(graph, args):
    """Stochastic PageRank estimation

    Parameters:
    graph -- a graph object as returned by load_graph()
    args -- arguments named tuple



    Returns:
    A dict that assigns each page its hit frequency

    This function estimates the Page Rank by counting how frequently
    a random walk that starts on a random node will after n_steps end
    on each node of the given graph.
    """

    steps = args.steps * args.repeats

    nodes = list(graph.nodes)
    hit_count = [0 for _ in range(graph.number_of_nodes())]
    first_node = random.choice(nodes)
    hit_count[first_node] += 1
    out = list(graph.out_edges(first_node))

    counter = 0
    while counter < steps:
        if len(out) == 0:
            current_node = random.choice(nodes)
        else:
            new_node = random.choice(out)
            current_node = new_node[1]

        hit_count[current_node] += 1
        out = list(graph.out_edges(current_node))


    #raise RuntimeError("This function is not implemented yet.")


def distribution_page_rank(graph, args):
    """Probabilistic PageRank estimation

    Parameters:
    graph -- a graph object as returned by load_graph()
    args -- arguments named tuple

    Returns:
    A dict that assigns each page its probability to be reached

    This function estimates the Page Rank by iteratively calculating
    the probability that a random walker is currently on any node.
    """



    raise RuntimeError("This function is not implemented yet.")





parser = argparse.ArgumentParser(description="Estimates page ranks from link information")
parser.add_argument('datafile', nargs='?', type=argparse.FileType('r'), default=sys.stdin,
                    help="Textfile of links among web pages as URL tuples")
parser.add_argument('-m', '--method', choices=('stochastic', 'distribution'), default='stochastic',
                    help="selected page rank algorithm")
parser.add_argument('-r', '--repeats', type=int, default=1_000_000, help="number of repetitions")
parser.add_argument('-s', '--steps', type=int, default=100, help="number of steps a walker takes")
parser.add_argument('-n', '--number', type=int, default=20, help="number of results shown")


if __name__ == '__main__':
    args = parser.parse_args()
    algorithm = distribution_page_rank if args.method == 'distribution' else stochastic_page_rank

    graph = load_graph(args)

    print_stats(graph)

    start = time.time()
    ranking = algorithm(graph, args)
    stop = time.time()
    time = stop - start

    top = sorted(ranking.items(), key=lambda item: item[1], reverse=True)
    sys.stderr.write(f"Top {args.number} pages:\n")
    print('\n'.join(f'{100*v:.2f}\t{k}' for k,v in top[:args.number]))
    sys.stderr.write(f"Calculation took {time:.2f} seconds.\n")
