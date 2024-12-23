import sys
import os
import time
import argparse
import random
import networkx as nx

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

    # adjacency list is similar to networkx but has less uses due to networkx having more methods that can be used


    #advanced adjacency list
    #pageRank_dict = {}
    #for line in args.datafile:
    #    node, target = line.split()
    #    if node not in pageRank_dict:
    #        pageRank_dict[node] = {}
    #    pageRank_dict[node][target] = (node,"->",target)
    #return pageRank_dict

    #this method creates a dictionary of dictionary to store the graph data. It is more complicted than need, and would
    #be more useful if the nodes where weighted, which in this exercise they are not



    #vertex set and edge dictionary
    #V = []
    #E = {}
    #for line in args.datafile:
    #    node, target = line.split()
    #    if node not in V:
    #        V.append(node)
    #    E[node] = target
    #return E, V
    #This method uses pre-existing methods. Nothing unique to python, creates a longer dictionary for edge list
    #Which is less efficient



    #using networkx to store graph data
    pageRank_dict = nx.DiGraph()

    for line in args.datafile:

        node, target = line.split()
        pageRank_dict.add_node(node)
        pageRank_dict.add_edge(node, target)
    return pageRank_dict
    #networkx has a library of methods that are useful in the page_rank methods


def print_stats(graph):
        """Print number of nodes and edges in the given graph"""

        print("number of nodes:", len(graph.nodes))
        print("number of edges:", len(graph.edges))



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

    repeats = args.repeats

    nodes = list(graph.nodes)
    hit_count  = {}
    for node in nodes:
        hit_count[node] = 0
        #initializes the hit_count dictionary
    first_node = random.choice(nodes)
    hit_count[first_node] += 1
    out = list(graph.out_edges(first_node))
    #obtains the first node, and updates the required fields


    counter = 0
    while counter < repeats:
        if len(out) == 0:
            current_node = random.choice(nodes)
            #updates the system for a "dead end" node
        else:
            new_node = random.choice(out)
            current_node = new_node[1]
            #updates the system for a node with neighbors

        hit_count[current_node] += 1
        out = list(graph.out_edges(current_node))
        counter += 1
        #updates necessary fields
    return hit_count

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


    steps = args.steps
    counter = 0
    node_prob = {}
    next_prob = {}
    for node in graph.nodes:
        node_prob[node] = 1/len(graph.nodes)
        #initializes the node_prob dictionary
    while counter < steps:
        for node in graph.nodes:
            next_prob[node] = float(0)
            #initializes on first loop the next_prob dictionary, resets it on following loops
        for node in graph.nodes:
            p = node_prob[node]/graph.out_degree(node)
            for target in graph.neighbors(node):
                next_prob[target] = next_prob[target] + p
                #updates the next_prob dictionary for each node leading into the target node
        for node in graph.nodes:
            node_prob[node] = next_prob[node]
            #copies the next_prob dictionary into the node_prob dictionary

        counter += 1

    return node_prob

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
