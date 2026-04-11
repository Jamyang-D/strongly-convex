# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 16:34:06 2024

@author: dyx123
"""
import itertools
import networkx as nx
import copy
import random
import matplotlib.pyplot as plt
from pgmpy.readwrite import BIFReader
import time

####################
# Randomly generate a connected CG
def generate_random_lwf(n, edge_density):
    """
    Input:
        n: int - Number of nodes in the CG.
        edge_density: float - Probability of edge creation between any two nodes.

    Output:
        graph: DiGraph - A directed acyclic graph (CG) generated randomly.
    """

    graph = nx.DiGraph()
    nodes = ['v' + str(i) for i in range(1, n + 1)]
    for node in nodes:
        graph.add_node(node)
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < edge_density:
                graph.add_edge(nodes[i], nodes[j])
    return graph


# Check connected components in a CG
def connect_components(graph):
    """
    Input:
        graph: DiGraph - A directed acyclic graph (CG).

    Output:
        None (modifies the input graph by connecting components if needed).
    """

    components = list(nx.weakly_connected_components(graph))
    num_components = len(components)
    if num_components > 1:
        for i in range(num_components - 1):
            root1 = next(iter(components[i]))
            root2 = next(iter(components[i + 1]))
            graph.add_edge(root1, root2)


## Randomly generate a connected CG
def Generate_lwf(n, edge_density, p_undirected):
    """
    Input:
        n: int - Number of nodes in the CG.
        edge_density: float - Probability of edge creation between any two nodes.

    Output:
        graph: dict - A connected CG represented as a dictionary.
    """

    random_cg = generate_random_lwf(n, edge_density)
    connect_components(random_cg)
    graph = {node: {} for node in random_cg.nodes}
    for node in random_cg.nodes:
        neighbors = list(random_cg.neighbors(node))
        neighbors = [neighbor for neighbor in neighbors if neighbor != node]
        graph[node] = {neighbor: 'b' for neighbor in neighbors}
    # 单向边变双向边
    for u in graph:
        for v in graph[u]:
            if random.random() < p_undirected:
                graph[v][u] = graph[u][v]
    return graph


#########################################
# Convert CG format
def convert_cg(cg_graph):  # dictionary to list
    """
    Input:
        cg_graph: dict - A CG represented as a dictionary.  CG={'1': {'2': 'b', '3': 'b'}, '2': {'3': 'b'}}

    Output:
        new_cg_graph: dict - A CG represented as a  list. CG={'1': ['2', '3'], '2': ['3']}
    """
    return {node: list(neighbors) for node, neighbors in cg_graph.items()}

################
# Output the reversed graoh
def reverse_graph(graph):
    """
    Input:
        graph: dict - A CG represented as a dictionary.

    Output:
        reversed_graph: dict - The reversed CG, where all edges are reversed.
    """

    reversed_graph = {node: {} for node in graph}
    for u, neighbors in graph.items():
        for v, weight in neighbors.items():
            reversed_graph.setdefault(v, {})[u] = weight
    return reversed_graph



# Plot
def plot_result(result, title=None):
    """
    Input:
        result: dict - A CG represented as a dictionary.
        file_path: str - The path where the plot will be saved.
        title: str or None - The title of the plot.

    Output:
        None (saves a plot of the CG to the specified file path).
    """
    G = nx.DiGraph()
    # Add nodes
    for node in result:
        G.add_node(node)
        # Add edges and annotation information

    edge_colors = []
    for node, neighbors in result.items():
        for neighbor, info in neighbors.items():
            G.add_edge(node, neighbor)
            edge_colors.append(info)
            # plot
    pos = nx.circular_layout(G)  # Use the circular_layout layout method
    # Create an image of size 8x6 inches with dpi set to 150
    plt.figure(figsize=(16, 12), dpi=150)
    nx.draw_networkx(G, pos, with_labels=True, node_color='lightblue',
                     node_size=2000, font_size=28, edge_color=edge_colors)

    if title:
        plt.title(title)
    else:
        plt.title("initial")

    plt.show()
    # plt.close()


#########################
# Children set of a vertex
def ch(CG, vertice, reverse_CG=None):
    """
    Input:
        CG: dict - A CG represented as a dictionary.
        vertice: str - A vertex in the CG.
        reverse_CG: dict or None - The reversed CG, if already computed.

    Output:
        set - The set of children of the given vertex.
    """
    CH = set()
    for v in CG[vertice]:
        if vertice not in CG[v]:
            CH.add(v)
    return CH


# Parent set of  a vertex
def pa(CG, vertice, reverse_CG=None):
    """
    Input:
        CG: dict - A CG represented as a dictionary.
        vertices: str - A vertex in the CG.
        reverse_CG: dict or None - The reversed CG, if already computed.

    Output:
        set - The set of parents of the given vertex.
    """
    PA = set()
    if reverse_CG is None:
        reverse_CG = reverse_graph(CG)
    for v in reverse_CG[vertice]:
        if vertice not in reverse_CG[v]:
            PA.add(v)
    return PA


def ne(CG, vertice, reverse_CG=None):
    """
    Input:
        CG: dict - A CG represented as a dictionary.
        node: str - A vertex in the CG.
        reverse_CG: dict or None - The reversed CG, if already computed.

    Output:
        set - The set of neighbors of the given vertex.
    """
    NE = set()
    for v in CG[vertice]:
        if vertice in CG[v]:
            NE.add(v)
    return NE


def tch(CG, vertice, reverse_CG=None):
    """
    Input:
        CG: dict - A CG represented as a dictionary.
        vertice: str - A vertex in the CG.
        reverse_CG: dict or None - The reversed CG, if already computed.

    Output:
        set - The set of spouses of the given vertex.
    """

    if reverse_CG is None:
        reverse_CG = reverse_graph(CG)
    ch1 = set()
    ch_add = ch(CG, vertice, reverse_CG)
    ch1 = ch1 | ch_add
    while ch_add:
        ch_add1 = set()
        for v in ch_add:
            ch_add1 = ch_add1 | ne(CG, v, reverse_CG)
        ch_add = ch_add1 - ch1
        ch1 = ch1 | ch_add
    tch_value = set()
    for i in ch1:
        tch_value = tch_value.union(pa(CG, i, reverse_CG))
    tch_value.discard(vertice)
    # tch_value = tch_value - ch1
    return tch_value


# Spouse set of vertices
def set_tch(CG, nodes, reverse_CG=None):
    """
    Input:
        CG: dict - A CG represented as a dictionary.
        nodes: list - A list of vertices in the CG.
        reverse_CG: dict or None - The reversed CG, if already computed.

    Output:
        set - The set of spouses of the given vertices.
    """
    if reverse_CG is None:
        reverse_CG = reverse_graph(CG)
    tch_set = set()
    for i in nodes:
        tch_set = tch_set | tch(CG, i, reverse_CG)
    tch_set = tch_set - set(nodes)
    return tch_set


# Children set of vertices
def set_ch(CG, nodes, reverse_CG=None):
    """
    Input:
        CG: dict - A CG represented as a dictionary.
        nodes: list - A list of vertices in the CG.

    Output:
        set - The set of children of the given vertices.
    """
    if reverse_CG is None:
        reverse_CG = reverse_graph(CG)
    CHS = set()
    for i in nodes:
        CHS = CHS | ch(CG, i, reverse_CG)
    CHS = CHS - set(nodes)
    return CHS


# Parent set of vertices
def set_pa(CG, nodes, reverse_CG=None):
    """
    Input:
        CG: dict - A CG represented as a dictionary.
        nodes: list - A list of vertices in the CG.
        reverse_CG: dict or None - The reversed CG, if already computed.

    Output:
        set - The set of parents of the given vertices.
    """
    if reverse_CG is None:
        reverse_CG = reverse_graph(CG)
    PAS = set()
    for i in nodes:
        PAS = PAS | pa(CG, i, reverse_CG)
    PAS = PAS - set(nodes)
    return PAS


# Neighbor set of vertices
def set_ne(CG, nodes, reverse_CG=None):
    """
    Input:
        CG: dict - A CG represented as a dictionary.
        nodes: list - A list of vertices in the CG.
        reverse_CG: dict or None - The reversed CG, if already computed.

    Output:
        set - The set of neighbors of the given vertices.
    """

    if reverse_CG is None:
        reverse_CG = reverse_graph(CG)
    NES = set()
    for i in nodes:
        NES = NES | ne(CG, i, reverse_CG)
    NES = NES - set(nodes)
    return NES


# Ancestor set of vertices
def ant(CG, nodes, reverse_CG=None):
    """
    Input:
        CG: dict - A CG represented as a dictionary.
        nodes: list - A list of vertices in the CG.
        reverse_CG: dict or None - The reversed CG, if already computed.

    Output:
        set - The set of ancestors of the given vertices.
    """

    if reverse_CG is None:
        reverse_CG = reverse_graph(CG)
    visited = set()
    stack = list(nodes)
    ancestors = set()
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            ancestors.add(node)
            stack.extend(reverse_CG.get(node, {}).keys())
    return ancestors


# Markov blanket of  a vertex
def markov_blanket(CG, vertices, reverse_CG=None):
    """
    Input:
        CG: dict - A CG represented as a dictionary.
        nodes: list - A list of vertices in the CG.

    Output:
        set - The set of markov blanket of a given vertex.
    """

    if reverse_CG is None:
        reverse_CG = reverse_graph(CG)
    pa_ = pa(CG, vertices, reverse_CG)
    ch_ = ch(CG, vertices, reverse_CG)
    ne_ = ne(CG, vertices, reverse_CG)
    tch_ = tch(CG, vertices, reverse_CG)
    markov = ch_ | pa_ | tch_ | ne_
    return markov


# Markov blanket of  vertices
def set_markov_blanket(CG, nodes, reverse_CG=None):
    """
    Input:
        CG: dict - A CG represented as a dictionary.
        nodes: list - A list of vertices in the CG.

    Output:
        set - The set of markov blanket of  given vertices.
    """

    if reverse_CG is None:
        reverse_CG = reverse_graph(CG)
    set_pa_ = set_pa(CG, nodes, reverse_CG)
    set_ch_ = set_ch(CG, nodes, reverse_CG)
    set_ne_ = set_ne(CG, nodes, reverse_CG)
    set_tch_ = set_tch(CG, nodes, reverse_CG)
    set_markov = set_ch_ | set_pa_ | set_ne_ | set_tch_
    return set_markov


# G1-G2
def dict_difference(dict1, dict2):
    """
    Calculate the difference between two dictionaries.

    Input:
    - dict1: The first dictionary.
    - dict2: The second dictionary to compare with the first.

    Output:
    - difference: A dictionary containing the elements in dict1 that are not in dict2 or that differ between dict1 and dict2.
    """

    difference = {node: {} for node in dict1}
    for key, sub_dict1 in dict1.items():
        if key not in dict2:
            # Entire entry is new
            difference[key] = sub_dict1
        else:
            sub_dict2 = dict2[key]
            # Collect differing or missing sub-keys
            diff_sub = {
                sub_key: val
                for sub_key, val in sub_dict1.items()
                if sub_key not in sub_dict2 or sub_dict2[sub_key] != val
            }
            if diff_sub:
                difference[key] = diff_sub

    return difference

###################

# Moralization on a CG  ?
def moralize_cg(CG, reverse_CG=None):
    """
    Moralize a CG, connecting all parents of each node.

    Input:
    - cg: The original CG.
    - cg_reverse: (Optional) The reversed CG. If not provided, it will be computed.

    Output:
    - moralized_cg: The moralized CG.
    """

    if reverse_CG is None:
        reverse_CG = reverse_graph(CG)
    CG_add = dict()
    for parents_i in CG:
        tch_v = tch(CG, parents_i, reverse_CG)
        for parents_j in tch_v:
            if parents_i not in CG_add:
                CG_add[parents_i] = {}
            if parents_j not in CG_add:
                CG_add[parents_j] = {}
            CG_add[parents_i][parents_j] = 'b'
            CG_add[parents_j][parents_i] = 'b'
    moralized_cg = dict()
    graphs_to_merge = [CG, CG_add, reverse_CG]
    for graph in graphs_to_merge:
        for node, neighbors in graph.items():
            moralized_cg.setdefault(node, {}).update(neighbors)
    return moralized_cg


# Obtain a induced graph from a CG
def sub_cg(CG, H):
    """
    Obtain a subgraph induced by a set of vertices from the CG.

    Input:
    - CG: The original CG.
    - H: A set of vertices to induce the subgraph.

    Output:
    - sub_CG: The induced subgraph.
    """

    H_set = set(H)
    return {
        u: {v: w for v, w in CG[u].items() if v in H_set}
        for u in CG
        if u in H_set
    }


# Determine if two vertices are connected in a UG
def is_connected(graph, start, target):
    """
    Determine if two vertices are connected in an undirected graph.

    Input:
    - graph: The undirected graph.
    - start: The starting vertex.
    - target: The target vertex.

    Output:
    - Boolean value indicating whether there is a path connecting the start and target vertices.
    """

    visited = set()
    stack = [start]
    while stack:
        node = stack.pop()
        visited.add(node)
        if node == target:
            return True
        for neighbor in graph.get(node, {}):
            if neighbor not in visited:
                stack.append(neighbor)
    return False


# Check if there exists an inducing path between two vertices in a CG
def is_inducing_path(CG, u, v, M, reverse_CG=None, re_MF=False):
    """
    Check if there exists an inducing path between two vertices in a CG.

    Input:
    - CG: The original CG.
    - u: The first vertex.
    - v: The second vertex.
    - M: A set of vertices to consider for the inducing path.
    - reverse_CG: (Optional) The reversed CG. If not provided, it will be computed.
    - re_MF: (Optional) A boolean to determine if the function should return additional details.   
    Output:
    - Boolean value indicating whether there exists an inducing path between u and v.
      If re_MF is True, a tuple containing the vertices and the boolean value is returned.
      """

    if reverse_CG is None:
        reverse_CG = reverse_graph(CG)
    nodes = [u, v]
    ANT = ant(CG, nodes, reverse_CG)
    CG_ANT = sub_cg(CG, list(ANT))  # ancestral graph
    MCG_ANT = moralize_cg(CG_ANT)  # ancestral graph after moralizing
    MCG_M_u_v = sub_cg(MCG_ANT, M + nodes)  # E
    flag = is_connected(MCG_M_u_v, u, v)  # V+E
    return (nodes, flag) if re_MF else flag


# Check t-removability
def ITRSA1_(CG, M, R1=None, reverse_CG=None):
    """
    Check t-removability of a set of vertices in a CG and return the flag and list of inducing paths.

    Input:
    - CG: The original CG.
    - M: The set of vertices to check for t-removability.
    - R1: (Optional) The set of vertices not in M. If not provided, it will be computed.
    - reverse_CG: (Optional) The reversed CG. If not provided, it will be computed.

    Output:
    - A tuple containing a boolean indicating t-removability and a list of mf-pairs .
    """

    # if reverse_CG is None:
    #     reverse_CG = reverse_graph(CG)
    # if R1 is None:
    #     R1 = [x for x in CG if x not in M]
    # MF = list()
    # f = True
    # for i in range(1, len(R1)):
    #     for j in range(i):  #
    #         if R1[j] not in CG[R1[i]] and R1[i] not in CG[R1[j]]:
    #             nodes, flag = is_inducing_path(CG, R1[i], R1[j], M, reverse_CG, re_MF=True)
    #             if flag:
    #                 f = False
    #                 MF.append(nodes)
    # return f, MF

    M_set = set(M)

    if R1 is None:
        R1 = [x for x in CG if x not in M_set]
    else:
        R1 = list(R1)

    # Build reverse_CG if needed
    if reverse_CG is None:
        reverse_CG = reverse_graph(CG)

    # Ensure adjacency lists in CG are sets for fast lookup
    CG_adj = {}
    for node, neighbors in CG.items():
        if isinstance(neighbors, (set, frozenset)):
            CG_adj[node] = neighbors
        else:
            CG_adj[node] = set(neighbors)

    MF = []
    f = True
    n = len(R1)

    for i in range(1, n):
        u = R1[i]
        u_neighbors = CG_adj[u]
        for j in range(i):
            v = R1[j]
            # Skip if u and v are directly connected (no inducing path possible)
            if v in u_neighbors or u in CG_adj[v]:
                continue

            # Check for an inducing path through M_set
            nodes, has_inducing_path = is_inducing_path(
                CG_adj, u, v, M_set, reverse_CG, re_MF=True
            )
            if has_inducing_path:
                f = False
                MF.append(nodes)

    return f, MF



# Check t-removability
def ITRSA_(CG, M, R1=None, reverse_CG=None):
    """
    Check t-removability of a set of vertices in a CG.

    Input:
    - CG: The original CG.
    - M: The set of vertices to check for t-removability.
    - R1: (Optional) The set of vertices not in M. If not provided, it will be computed.
    - reverse_CG: (Optional) The reversed CG. If not provided, it will be computed.

    Output:
    - Boolean value indicating t-removability.
    """
    # M_set = set(M)
    # if reverse_CG is None:
    #     reverse_CG = reverse_graph(CG)
    # if R1 is None:
    #     R1 = [x for x in CG if x not in M_set]
    # for i in range(1, len(R1)):
    #     for j in range(i):  # V^2
    #         if R1[j] not in CG[R1[i]] and R1[i] not in CG[R1[j]]:
    #             if is_inducing_path(CG, R1[i], R1[j], M_set, reverse_CG):
    #                 return False
    # return True
    M_set = set(M)

    # Preprocess adjacency lists to sets for O(1) membership tests
    CG_adj = {}
    for node, neighbors in CG.items():
        if isinstance(neighbors, (set, frozenset)):
            CG_adj[node] = neighbors
        else:
            CG_adj[node] = set(neighbors)

    # Compute R1 if not provided
    if R1 is None:
        R1 = [node for node in CG if node not in M_set]
    else:
        R1 = list(R1)  # Ensure it's a list for indexing

    # Compute reverse_CG if not provided
    if reverse_CG is None:
        reverse_CG = reverse_graph(CG_adj)  # Pass preprocessed CG for efficiency

    n = len(R1)
    for i in range(1, n):
        u = R1[i]
        u_neighbors = CG_adj[u]
        for j in range(i):
            v = R1[j]
            # Skip if u and v are directly connected (in either direction)
            if v in u_neighbors or u in CG_adj.get(v, ()):
                continue

            # Check for inducing path through M_set
            if is_inducing_path(CG_adj, u, v, M_set, reverse_CG):
                return False

    return True


# Check t-removability
def ITRSA(CG, M, reverse_CG=None, re_MF=False):
    """
    Check the t-removability of a set of vertices.

    Input:
    - CG: The original Directed Acyclic Graph (CG).
    - M: The set of vertices to check for t-removability.
    - re_MF: (Optional) If True, returns detailed information about the mf-pairs; 
             otherwise, returns only a boolean value.

    Output:
    - If re_MF is True, returns a tuple with a boolean indicating t-removability and a list of mf-pairs;
      otherwise, returns a boolean indicating whether the set of vertices is t-removable.
    """
    R = [x for x in CG if x not in M]
    if reverse_CG is None:
        reverse_CG = reverse_graph(CG)
    set_markov = set_markov_blanket(CG, M, reverse_CG)
    R1 = [node for node in R if node in set_markov]
    if re_MF:
        return ITRSA1_(CG, M, R1, reverse_CG)
    else:
        return ITRSA_(CG, M, R1, reverse_CG)


def c_Proximal_separator(CG, u, v, reverse_CG=None):
    if reverse_CG is None:
        reverse_CG = reverse_graph(CG)
    nodes = [u, v]
    ANT = ant(CG, nodes, reverse_CG)
    CG_ANT = sub_cg(CG, ANT)  # ancestral graph
    MCG_ANT = moralize_cg(CG_ANT)  # ancestral graph after moralizing
    u_ne = set(MCG_ANT[u])
    c_p_s = set()
    for node in u_ne:
        sub = ANT - {u, } - u_ne | {node, }
        G_sub = sub_cg(MCG_ANT, sub)
        if is_connected(G_sub, node, v):
            c_p_s.add(node)
    return c_p_s


# Check if vertices of pa(w) are adjacent unless both of them belongs to R
def pd(G, Ma, pa1):  ##########################
    """
    Check if the parent nodes of a vertex in the graph are adjacent, unless both belong to the set R.

    Input:
    - G: The original graph.
    - Ma: The set of vertices to check.
    - pa1: The set of parent nodes of Ma.

    Output:
    - Returns True if the parent nodes are adjacent or if both belong to R; otherwise, returns False.
    """
    PA = set()
    Ma = list(Ma)
    # pa1 = list(pa1)
    for i in range(1, len(Ma)):
        for j in range(i):
            if Ma[i] not in pa1 or Ma[j] not in pa1:
                if Ma[i] not in G[Ma[j]] and Ma[j] not in G[Ma[i]]:
                    PA.add(Ma[i])
                    PA.add(Ma[j])
    PA -= pa1
    return PA


def condition(DAG, M, R, reverse_DAG=None):  #############################
    """
    Check the condition for c-removability in a Directed Acyclic Graph (DAG).

    Input:
    - DAG: The original Directed Acyclic Graph (DAG).
    - M: The set of vertices to check for c-removability.
    - R: The set of remaining vertices after excluding M.
    - reverse_DAG: (Optional) The reversed DAG. If not provided, it will be computed.

    Output:
    - Returns True if the condition for c-removability is met; otherwise, returns False.
    """
    PA = set()
    if reverse_DAG is None:
        reverse_DAG = reverse_graph(DAG)
    AN = ant(DAG, R, reverse_DAG)
    CH = set_ch(DAG, M, reverse_DAG) | set(M)
    w = AN & CH
    for i in w:  # V
        paa = pa(DAG, i, reverse_DAG)
        PA |= pd(DAG, paa, paa & set(R))
    return PA


# Check  c-removability
def is_cremoved(G, vertices, reverse_DAG=None):
    """
    Check if a vertex is c-removable from the DAG.
    
    Input:
    - G: The original Directed Acyclic Graph (DAG).
    - vertices: The vertex to check for c-removability.
    - reverse_DAG: (Optional) The reversed DAG. If not provided, it will be computed.
    
    Output:
    - Boolean value indicating whether the vertex is c-removable.
    """

    if reverse_DAG is None:
        reverse_DAG = reverse_graph(G)
    Ma = list(markov_blanket(G, vertices, reverse_DAG)) + [vertices]
    pa_set = pa(G, vertices, reverse_DAG)
    for i in range(1, len(Ma)):
        for j in range(i):
            if Ma[i] not in pa_set or Ma[j] not in pa_set:
                if Ma[i] not in G[Ma[j]] and Ma[j] not in G[Ma[i]]:
                    return False
    return True


# Check sequentially c-removability
def is_set_cremoved(G, M):
    """
    Check if a set of vertices is sequentially c-removable from the DAG.
    
    Input:
    - G: The original DAG.
    - M: A list of vertices to check for sequential c-removability.
    
    Output:
    - A tuple containing a boolean indicating if the entire set is c-removable and a list of c-removable vertices in sequence.
    """
    G1 = copy.deepcopy(G)
    G2 = reverse_graph(G1)
    M1 = copy.deepcopy(M)
    k = 1
    M_T = []
    while M1:
        del1 = {}
        for i in M1:
            if is_cremoved(G1, i, G2):
                M_T.append(i)
                M1.remove(i)
                del1[i] = G1[i]
                for j in G2[i]:
                    if j not in del1:
                        del1[j] = {}
                    del1[j][i] = G2[i][j]
                G1 = dict_difference(G1, del1)
                G2 = dict_difference(G2, reverse_graph(del1))
                break
        if k != len(M_T):
            break
        else:
            k = k + 1
    return len(M_T) == len(M), M_T


##极小分离子找c凸包
def CMCSA111(CG, R, reverse_CG=None):
    R = set(R)
    M = [x for x in CG if x not in R]
    if reverse_CG is None:
        reverse_CG = reverse_graph(CG)
    _, Q = ITRSA(CG, M, reverse_CG, re_MF=True)
    PA = condition(CG, M, R, reverse_CG)
    H = copy.deepcopy(R)
    while Q or PA:
        for u, v in Q:
            u_c = c_Proximal_separator(CG, u, v, reverse_CG=reverse_CG)
            v_c = c_Proximal_separator(CG, u, v, reverse_CG=reverse_CG)
            H |= u_c | v_c
        H |= PA
        M = [x for x in CG if x not in H]
        _, Q = ITRSA(CG, M, reverse_CG, re_MF=True)
        PA = condition(CG, M, H, reverse_CG)
    return H


# 极小分离子找t凸包
def CMCSA(CG, R, reverse_CG=None):
    R = set(R)
    M = [x for x in CG if x not in R]
    if reverse_CG is None:
        reverse_CG = reverse_graph(CG)
    f, Q = ITRSA(CG, M, reverse_CG=reverse_CG, re_MF=True)
    H = copy.deepcopy(R)
    while not f:
        for u, v in Q:
            u_c = c_Proximal_separator(CG, u, v, reverse_CG=reverse_CG)
            v_c = c_Proximal_separator(CG, u, v, reverse_CG=reverse_CG)
            H |= u_c | v_c
        M = [x for x in CG if x not in H]
        f, Q = ITRSA(CG, M, reverse_CG=reverse_CG, re_MF=True)
    return H


# 找uv之间的导出路
def uv_DCL(CG, u, v, M, reverse_CG=None):
    """
    Check if there exists an inducing path between two vertices in a CG.

    Input:
    - CG: The original CG.
    - u: The first vertex.
    - v: The second vertex.
    - M: A set of vertices to consider for the inducing path.
    - reverse_CG: (Optional) The reversed CG. If not provided, it will be computed.
    - re_MF: (Optional) A boolean to determine if the function should return additional details.
    Output:
    - Boolean value indicating whether there exists an inducing path between u and v.
      If re_MF is True, a tuple containing the vertices and the boolean value is returned.
      """

    if reverse_CG is None:
        reverse_CG = reverse_graph(CG)
    nodes = [u, v]
    ANT = ant(CG, nodes, reverse_CG)
    CG_ANT = sub_cg(CG, list(ANT))  # ancestral graph
    MCG_ANT = moralize_cg(CG_ANT)  # ancestral graph after moralizing
    MCG_M_u_v = sub_cg(MCG_ANT, M + nodes)  # E

    DCL = BFS_shortest_path(MCG_M_u_v, u, v)  # V+E
    return DCL


# 找集合R中全部的导出路
def all_DCL(CG, M, reverse_CG=None):
    """
    Check the t-removability of a set of vertices.

    Input:
    - CG: The original Directed Acyclic Graph (CG).
    - M: The set of vertices to check for t-removability.
    - re_MF: (Optional) If True, returns detailed information about the mf-pairs;
             otherwise, returns only a boolean value.

    Output:
    - If re_MF is True, returns a tuple with a boolean indicating t-removability and a list of mf-pairs;
      otherwise, returns a boolean indicating whether the set of vertices is t-removable.
    """
    R = [x for x in CG if x not in M]
    if reverse_CG is None:
        reverse_CG = reverse_graph(CG)
    set_markov = set_markov_blanket(CG, M, reverse_CG)
    R1 = [node for node in R if node in set_markov]

    all_D = set()
    for i in range(1, len(R1)):
        for j in range(i):  #
            if R1[j] not in CG[R1[i]] and R1[i] not in CG[R1[j]]:
                DLC = uv_DCL(CG, R1[i], R1[j], M, reverse_CG)
                all_D |= set(DLC)
    return all_D


# 导出路吸收方法找t凸包
def CMCSA_new(CG, R, reverse_CG=None):
    R = set(R)
    M = [x for x in CG if x not in R]
    if reverse_CG is None:
        reverse_CG = reverse_graph(CG)
    All_D = all_DCL(CG, M, reverse_CG)
    H = copy.deepcopy(R)
    while All_D:
        H |= All_D
        M = [x for x in CG if x not in H]
        All_D = all_DCL(CG, M, reverse_CG)
    return H


# 导出路吸收方法找c凸包
def CMCSA111_new(CG, R, reverse_CG=None):
    R = set(R)
    M = [x for x in CG if x not in R]
    if reverse_CG is None:
        reverse_CG = reverse_graph(CG)
    All_D = all_DCL(CG, M, reverse_CG)
    PA = condition(CG, M, R, reverse_CG)
    H = copy.deepcopy(R)
    while All_D or PA:
        H |= All_D
        H |= PA
        M = [x for x in CG if x not in H]
        All_D = all_DCL(CG, M, reverse_CG)
        PA = condition(CG, M, H, reverse_CG)
    return H


def BFS_shortest_path(graph, start, goal):  # 最短路
    # 初始化队列，用于存储当前探索的节点及其路径
    queue = [(start, [start])]

    # 初始化已访问集合
    visited = set()

    while queue:
        # 从队列中取出第一个元素
        (vertex, path) = queue.pop(0)

        # 如果当前节点还没有被访问过
        if vertex not in visited:
            # 将其标记为已访问
            visited.add(vertex)

            # 遍历当前节点的所有邻接节点
            for next_vertex in graph[vertex]:
                # 如果找到了目标节点
                if next_vertex == goal:
                    # 返回到达目标节点的完整路径
                    return path + [next_vertex]

                # 否则，将邻接节点及其路径加入队列以供后续探索
                else:
                    queue.append((next_vertex, path + [next_vertex]))

    # 如果队列空了且没有找到目标节点，则返回 None 表示不存在这样的路径
    return []


#####################
def in_degrees(CG, reverse_CG=None):  # DAG 的入度
    if reverse_CG is None:
        reverse_CG = reverse_graph(CG)
    in_dag = dict()
    for node in reverse_CG:
        in_dag[node] = len(reverse_CG[node])
    return in_dag


def topoSort(CG, reverse_CG=None):  # DAG 的拓扑序
    if reverse_CG is None:
        reverse_CG = reverse_graph(CG)

    in_dag = in_degrees(CG, reverse_CG)

    Q = [u for u in in_dag if in_dag[u] == 0]  # 筛选入度为0的顶点
    Seq = []

    while Q:
        u = Q.pop()  # 默认从最后一个删除
        Seq.append(u)
        for v in CG[u]:
            in_dag[v] -= 1  # 移除子节点的入边
            if in_dag[v] == 0:
                Q.append(v)  # 再次加入下一次迭代生成的入度为0的顶点
    if len(Seq) == len(CG):  # 输出的顶点数是否与图中的顶点数相等
        return Seq
    else:
        print("G is not the directed acyclic graph.")
        return None


# C分解
def EDC(CG, reverse_CG=None):
    if reverse_CG is None:
        reverse_CG = reverse_graph(CG)
    topo = topoSort(CG, reverse_CG)
    D_G = []
    for v in topo[::-1]:
        # v = topo.pop(-1)
        PA = pa(CG, v, reverse_CG)
        fa = PA | {v}
        if all(not fa.issubset(h) for h in D_G):
            H = CMCSA111_new(CG, fa, reverse_CG)
            D_G.append(H)
    return D_G


def EDC1(CG, reverse_CG=None):
    if reverse_CG is None:
        reverse_CG = reverse_graph(CG)
    topo = topoSort(CG, reverse_CG)
    D_G = []
    for v in topo[::1]:
        # v = topo.pop(-1)
        PA = pa(CG, v, reverse_CG)
        fa = PA | {v}
        if all(not fa.issubset(h) for h in D_G):
            H = CMCSA111_new(CG, fa, reverse_CG)
            D_G.append(H)
    return D_G


# t分解
def MDC(CG, reverse_CG=None):
    if reverse_CG is None:
        reverse_CG = reverse_graph(CG)
    topo = topoSort(CG, reverse_CG)
    D_G = []
    for v in topo[::-1]:
        # v = topo.pop(-1)
        PA = pa(CG, v, reverse_CG)
        fa = PA | {v}
        if all(not fa.issubset(h) for h in D_G):
            H = CMCSA(CG, fa, reverse_CG)
            D_G.append(H)
    return D_G


# 导出路找t分解
def MDC111(CG, reverse_CG=None):
    if reverse_CG is None:
        reverse_CG = reverse_graph(CG)
    topo = topoSort(CG, reverse_CG)
    D_G = []
    for v in topo[::-1]:
        # v = topo.pop(-1)
        PA = pa(CG, v, reverse_CG)
        fa = PA | {v}
        if all(not fa.issubset(h) for h in D_G):
            H = CMCSA_new(CG, fa, reverse_CG)
            D_G.append(H)
    return D_G


def build_junction_tree(c_decomposition):
    """
    Build a junction tree from C-decomposition.
    
    Args:
        c_decomposition (list): List of C-convex sets (subgraphs).
        
    Returns:
        junction_tree (nx.Graph): The constructed junction tree.
    """
    # Create a graph where nodes are C-convex sets and edges are weighted by intersection size
    G = nx.Graph()

    # Add nodes (each node is a frozenset representing a C-convex set)
    for h in c_decomposition:
        G.add_node(frozenset(h))

    # Add edges with weights = intersection size
    for i in range(len(c_decomposition)):
        for j in range(i + 1, len(c_decomposition)):
            intersection = c_decomposition[i] & c_decomposition[j]
            if intersection:
                G.add_edge(
                    frozenset(c_decomposition[i]),
                    frozenset(c_decomposition[j]),
                    weight=len(intersection)
                )

    # Compute the maximal spanning tree (MST) to ensure RIP
    junction_tree = nx.maximum_spanning_tree(G)

    return junction_tree


def dict_to_networkx(graph_dict, is_directed=True):
    """
    Convert a dictionary representing a graph into a NetworkX graph object.

    Parameters:
        graph_dict (dict): The graph represented as a dictionary.
        is_directed (bool): Whether the resulting NetworkX graph should be directed.

    Returns:
        G (networkx.Graph or networkx.DiGraph): The NetworkX graph object.
    """
    if is_directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    # Add nodes to the graph
    for node in graph_dict:
        G.add_node(node)

    # Add edges to the graph
    for u, neighbors in graph_dict.items():
        for v in neighbors:
            # Check if edge already exists to prevent adding duplicates
            if not G.has_edge(u, v):
                G.add_edge(u, v)

    return G


def find_subsets(sets):
    # 创建一个字典用于存储每个集合及其索引
    set_dict = {frozenset(s): i for i, s in enumerate(sets)}

    # 检查是否有子集关系
    for i, s in enumerate(sets):
        for j, other in enumerate(sets):
            if i != j and s.issubset(other) and s != other:
                print(f"集合 {i} 是集合 {j} 的子集")
                return True
    return False


def has_running_intersection_property(sets):
    # 如果集合列表为空或只有一个集合，则直接返回 True，因为它们自然满足传交性
    if not sets or len(sets) <= 1:
        return True
    sets = [set(x) for x in sets]  # set(x) list(x) tuple(x)
    # print(sets)
    E_T = list()
    # 遍历每个集合，从第二个集合开始（索引为1）
    for t in range(1, len(sets)):

        current_set = sets[t]  # 当前检查的集合 U_t
        # 计算所有之前集合的并集 (U_1 到 U_(t-1)) 的并集
        previous_union = set().union(*sets[:t])
        # 计算当前集合与之前所有集合并集的交集
        intersection = current_set.intersection(previous_union)

        found = False  # 标记是否找到符合条件的 U_p
        # 在之前的集合中寻找一个集合 U_p，使得交集是 U_p 的子集
        for p in range(t):  # 检查 U_p 其中 p < t
            if intersection.issubset(sets[p]):  # 如果交集是 U_p 的子集
                found = True  # 找到合适的 U_p
                E_T.append([[t, p], intersection])
                # E_T.append([[t,p])
                break  # 跳出循环，继续下一个 U_t 的检查

        if not found:  # 如果没有找到任何合适的 U_p
            return False  # 返回 False 表示不满足传交性

    # 如果所有集合都通过了检查，则返回 True 表示满足传交性
    return E_T


def dag_to_dict_format(dag):
    result = {}
    for node in dag.nodes():
        # 获取当前节点的所有邻居节点
        neighbors = {}
        for neighbor in dag.neighbors(node):
            neighbors[neighbor] = 'b'
        result[node] = neighbors
    return result


# # Assuming Generate_lwf is defined as you provided
# graph_dict = Generate_lwf(n=8, edge_density=0.3, p_undirected=0.5)

# # Convert the dictionary to a NetworkX DiGraph object
# G = dict_to_networkx(graph_dict, is_directed=True)


def has_junction_property(clique_list):
    """
    Check if there exists a junction tree over the given list of cliques (sets)
    that satisfies the running intersection property (junction property).

    Returns:
        bool: True if such a tree exists, False otherwise.
    """
    if len(clique_list) <= 1:
        return True

    n = len(clique_list)
    # Step 1: Build complete graph with edge weights = |Ci ∩ Cj|
    G = nx.Graph()
    for i in range(n):
        G.add_node(i, clique=clique_list[i])

    for i in range(n):
        for j in range(i + 1, n):
            w = len(clique_list[i] & clique_list[j])
            G.add_edge(i, j, weight=w)

    # Step 2: Compute maximum spanning tree (MST with max weight)
    T = nx.maximum_spanning_tree(G, weight='weight')

    # Step 3: For every pair of nodes, check if their intersection is contained
    # in all nodes along the unique path between them.
    nodes = list(T.nodes)
    for i, j in itertools.combinations(nodes, 2):
        if i == j:
            continue
        path = nx.shortest_path(T, source=i, target=j)
        inter = clique_list[i] & clique_list[j]
        if not inter:
            continue  # empty intersection always satisfies
        # Check every node on the path contains 'inter'
        for k in path:
            if not inter.issubset(clique_list[k]):
                return False
    return True


def build_hierarchical_junction_tree(clique_list):
    """
    Build a hierarchical junction tree with merge history.

    Each node in the final tree records:
      - 'clique': the (possibly merged) variable set
      - 'sources': list of indices from the original clique_list that were merged into this node

    Returns:
        (final_nodes, T)
        - final_nodes: list of dicts [{'clique': set, 'sources': [int]}, ...]
        - T: NetworkX Graph where each node has attributes 'clique' and 'sources'
    """
    # Initialize: each node is an atomic clique
    nodes = [
        {'clique': set(c), 'sources': [i]}
        for i, c in enumerate(clique_list)
    ]

    while True:
        n = len(nodes)
        if n <= 1:
            T = nx.Graph()
            if nodes:
                T.add_node(0, **nodes[0])
            return nodes, T

        # Step 1: Build complete graph with weights = |Ci ∩ Cj|
        G = nx.Graph()
        for i in range(n):
            G.add_node(i, **nodes[i])
        for i in range(n):
            for j in range(i + 1, n):
                w = len(nodes[i]['clique'] & nodes[j]['clique'])
                G.add_edge(i, j, weight=w)

        # Step 2: Maximum spanning tree
        T = nx.maximum_spanning_tree(G, weight='weight')

        # Step 3: Check RIP and find first violation
        node_indices = list(T.nodes)
        violation_found = False
        merge_pair = None

        for i, j in itertools.combinations(node_indices, 2):
            inter = nodes[i]['clique'] & nodes[j]['clique']
            if not inter:
                continue
            path = nx.shortest_path(T, source=i, target=j)
            for k in path:
                if not inter.issubset(nodes[k]['clique']):
                    pos = path.index(k)
                    if pos == 0:
                        merge_pair = (path[0], path[1])
                    else:
                        merge_pair = (path[pos - 1], path[pos])
                    violation_found = True
                    break
            if violation_found:
                break

        if not violation_found:
            # Success! Return current nodes and tree
            return nodes, T

        # Perform merge
        a, b = merge_pair
        new_clique = nodes[a]['clique'] | nodes[b]['clique']
        new_sources = sorted(set(nodes[a]['sources'] + nodes[b]['sources']))  # dedup & sort

        # Build new node list
        new_nodes = []
        for idx in range(n):
            if idx == a or idx == b:
                continue
            new_nodes.append(nodes[idx])
        new_nodes.append({'clique': new_clique, 'sources': new_sources})
        nodes = new_nodes


# Now you can use NetworkX functions to work with G, including plotting it
#############
if __name__ == "__main__":
    # # CG = {'HYPOVOLEMIA': {'LVEDVOLUME': 'b', 'STROKEVOLUME': 'b'}, 'LVEDVOLUME': {'CVP': 'b', 'PCWP': 'b'}, 'STROKEVOLUME': {'CO': 'b'}, 'CVP': {}, 'PCWP': {}, 'LVFAILURE': {'HISTORY': 'b', 'LVEDVOLUME': 'b', 'STROKEVOLUME': 'b'}, 'HISTORY': {}, 'CO': {'BP': 'b'}, 'ERRLOWOUTPUT': {'HRBP': 'b'}, 'HRBP': {}, 'ERRCAUTER': {'HREKG': 'b', 'HRSAT': 'b'}, 'HREKG': {}, 'HRSAT': {}, 'INSUFFANESTH': {'CATECHOL': 'b'}, 'CATECHOL': {'HR': 'b'}, 'ANAPHYLAXIS': {'TPR': 'b'}, 'TPR': {'CATECHOL': 'b', 'BP': 'b'}, 'BP': {}, 'KINKEDTUBE': {'PRESS': 'b', 'VENTLUNG': 'b'}, 'PRESS': {}, 'VENTLUNG': {'EXPCO2': 'b', 'MINVOL': 'b', 'VENTALV': 'b'}, 'FIO2': {'PVSAT': 'b'}, 'PVSAT': {'SAO2': 'b'}, 'SAO2': {'CATECHOL': 'b'}, 'PULMEMBOLUS': {'PAP': 'b', 'SHUNT': 'b'}, 'PAP': {}, 'SHUNT': {'SAO2': 'b'}, 'INTUBATION': {'MINVOL': 'b', 'SHUNT': 'b', 'PRESS': 'b', 'VENTLUNG': 'b', 'VENTALV': 'b'}, 'MINVOL': {}, 'VENTALV': {'PVSAT': 'b', 'ARTCO2': 'b'}, 'DISCONNECT': {'VENTTUBE': 'b'}, 'VENTTUBE': {'PRESS': 'b', 'VENTLUNG': 'b'}, 'MINVOLSET': {'VENTMACH': 'b'}, 'VENTMACH': {'VENTTUBE': 'b'}, 'EXPCO2': {}, 'ARTCO2': {'EXPCO2': 'b', 'CATECHOL': 'b'}, 'HR': {'HRBP': 'b', 'HREKG': 'b', 'HRSAT': 'b', 'CO': 'b'}}
    # reader = BIFReader("D://ACode_Sci//master//code//decomposition//new test//insurance.bif")
    # # 获取贝叶斯网络
    # model = reader.get_model()
    # # 查看网络中的所有边
    # # print("网络结构：", model.edges())
    # CG = dag_to_dict_format(model)
    # start = time.time()
    # D_G = get_mpd_cliques_fast(model)
    # # D_G = EDC(CG)
    # end = time.time()
    # print(end - start)
    # # G = generate_random_lwf(20,0.3)
    # # G_dict = dag_to_dict_format(G)
    # # H = EDC(G_dict)
    # # H1 = EDC1(G_dict)
    # # print(H)
    # # print(H1)

    # reader = BIFReader("D://ACode_Sci//master//code//decomposition//new test//insurance.bif")
    # # 获取贝叶斯网络
    # model = reader.get_model()
    # # 查看网络中的所有边
    # # print("网络结构：", model.edges())
    # G = dag_to_dict_format(model)
    # H = EDC(G)
    # print(H)
    # atom_ready = [sorted(list(s)) for s in H]
    # print(atom_ready)
    #

    #
    # from pgmpy.utils import get_example_model
    #
    # model = get_example_model('arth150')
    # G = nx.DiGraph()
    # G.add_nodes_from(model.nodes)
    # G.add_edges_from(model.edges)
    # G = dag_to_dict_format(model)
    # H = EDC(G)
    # print(H)
    # atom_ready = [sorted(list(s)) for s in H]
    # print(atom_ready)

    # CG = {
    #     'A': {'F': 'b'},
    #     'B': {'A': 'b','C': 'b'},
    #     'C': {},
    #     'D': {'C': 'b','E': 'b'},
    #     'E': {},
    #     'F': {'C': 'b'}
    # }
    # R = ['A','E']
    # H = CMCSA111_new(CG, R)
    # print(H)
    # # Your data
    # CG = {
    #     'A': {'B': 'b', 'G': 'b'},
    #     'B': {'C': 'b'},
    #     'C': {'D': 'b'},
    #     'E': {'C': 'b'},
    #     'D': {},
    #     'F': {'E': 'b','G': 'b'},
    #     'G': {}
    # }

    # # CG = {'HYPOVOLEMIA': {'LVEDVOLUME': 'b', 'STROKEVOLUME': 'b'}, 'LVEDVOLUME': {'CVP': 'b', 'PCWP': 'b'}, 'STROKEVOLUME': {'CO': 'b'}, 'CVP': {}, 'PCWP': {}, 'LVFAILURE': {'HISTORY': 'b', 'LVEDVOLUME': 'b', 'STROKEVOLUME': 'b'}, 'HISTORY': {}, 'CO': {'BP': 'b'}, 'ERRLOWOUTPUT': {'HRBP': 'b'}, 'HRBP': {}, 'ERRCAUTER': {'HREKG': 'b', 'HRSAT': 'b'}, 'HREKG': {}, 'HRSAT': {}, 'INSUFFANESTH': {'CATECHOL': 'b'}, 'CATECHOL': {'HR': 'b'}, 'ANAPHYLAXIS': {'TPR': 'b'}, 'TPR': {'CATECHOL': 'b', 'BP': 'b'}, 'BP': {}, 'KINKEDTUBE': {'PRESS': 'b', 'VENTLUNG': 'b'}, 'PRESS': {}, 'VENTLUNG': {'EXPCO2': 'b', 'MINVOL': 'b', 'VENTALV': 'b'}, 'FIO2': {'PVSAT': 'b'}, 'PVSAT': {'SAO2': 'b'}, 'SAO2': {'CATECHOL': 'b'}, 'PULMEMBOLUS': {'PAP': 'b', 'SHUNT': 'b'}, 'PAP': {}, 'SHUNT': {'SAO2': 'b'}, 'INTUBATION': {'MINVOL': 'b', 'SHUNT': 'b', 'PRESS': 'b', 'VENTLUNG': 'b', 'VENTALV': 'b'}, 'MINVOL': {}, 'VENTALV': {'PVSAT': 'b', 'ARTCO2': 'b'}, 'DISCONNECT': {'VENTTUBE': 'b'}, 'VENTTUBE': {'PRESS': 'b', 'VENTLUNG': 'b'}, 'MINVOLSET': {'VENTMACH': 'b'}, 'VENTMACH': {'VENTTUBE': 'b'}, 'EXPCO2': {}, 'ARTCO2': {'EXPCO2': 'b', 'CATECHOL': 'b'}, 'HR': {'HRBP': 'b', 'HREKG': 'b', 'HRSAT': 'b', 'CO': 'b'}}
    #
    # D_G = EDC(CG)
    #
    # # Test
    # result = has_junction_property(D_G)
    # print(result)
    # print(D_G)
    # #
    # final_cliques, tree = build_hierarchical_junction_tree(D_G)
    #
    # print("Final cliques (may include merged super-cliques):")
    # for i, c in enumerate(final_cliques):
    #     print(f"Node {i}: {c}")
    #
    # print("\nTree edges:")
    # for u, v in tree.edges():
    #     print(f"{u} -- {v}")
    #
    # # 查看节点属性（团内容）
    # print("\nNode -> Clique mapping:")
    # for node in tree.nodes():
    #     print(f"Node {node}: clique = {tree.nodes[node]['clique']}")
    # separators = []
    # # 1. 超团之间的分离集（来自最终 junction tree 的边）
    # for u, v in tree.edges():
    #     separators.append(tree.nodes[u]['clique'] & tree.nodes[v]['clique'])
    # # 2. 每个超团内部原始团之间的分离集
    # for node in final_cliques:
    #     sources = node['sources']
    #     for i, j in itertools.combinations(sources, 2):
    #         separators.append(D_G[i] & D_G[j])
    # print(separators)

    G = {
        'A': {'B': 'b'},
        'B': {},
        'D': {'E': 'b'},
        'C': {'B': 'b', 'E': 'b', 'D': 'b', 'F': 'b'},
        'E': {'F': 'b'},
        'F': {'G': 'b'},
        'H': {'B': 'b'},
        'I':{'H': 'b'},
        'G': {'H': 'b', 'B': 'b'}
        }
    # G = {
    #
    #
    #     'X': {},
    #     'A': {'X': 'b', 'B': 'b'},
    #     'C':{},
    #     'B': {'C': 'b'},
    #     'D': {'C':'b','Y':'b'},
    #     'Y': {}
    # }
    # R = ['X','Y']
    # # M = ["B","C", "E", "F","G"]
    # # M = [node for node in G if node not in R]
    # # H = ITRSA(G, M)
    # H1 = CMCSA111_new(G, R) #C凸
    # # H = CMCSA_new(G, R)  # T凸
    # print(f"{H1}")
    # M = ['B','A','C','D','E']
    # H2 = uv_DCL(G,'A','E',M)
    # H4= is_set_cremoved(G, M)
    H3=EDC(G)
    # # # print(H)
    # # # print(H1)
    # # # print(H2)
    print(H3)
    # # print(H4)

    # R = ['T', 'L', 'E', 'B', 'S', 'D']
    # H = CMCSA111_new(G, R)
    # H1 = EDC(G)
    # print(f"分解：{H1}")
    #
    # G = {
    #         "AFF": {'CDR': 'b','APA': 'b','ALN': 'b'},
    #         "CDR": {'DET': 'b'},
    #         'SAN': {'AFF': 'b', 'APA': 'b','ALN': 'b', 'AIS': 'b','CDR': 'b'},
    #         'AIS': {'SUS': 'b', 'EGC': 'b'},
    #         'ALN': {'APA': 'b','PER': 'b', 'FTW': 'b','SUS': 'b','DET': 'b'},
    #         'APA': {},
    #         'PER': {'DET': 'b'},
    #         'DET': {},
    #         'SUS': {'FTW': 'b','EGC': 'b','HOS': 'b'},
    #         'FTW': {'EGC': 'b','DET': 'b'},
    #         'HOS': {},
    #         'EGC': {'HOS': 'b'}
    #     }
    # R = ['AFF','HOS']
    # M=[x for x in G if x not in R]
    # H1 = is_set_cremoved(G,M)
    # H = EDC(G)
    # # H1 = EDC(G)
    # print(f"分解：{H,H1}")

    # G ={'X0': {'X1': 'b', 'X2': 'b', 'X3': 'b', 'X8': 'b', 'X9': 'b'},
    #      'X1': {'X2': 'b', 'X6': 'b', 'X7': 'b'},
    #      'X2': {'X4': 'b', 'X8': 'b', 'X9': 'b'},
    #      'X3': {'X9': 'b'},
    #      'X4': {'X9': 'b'},
    #      'X5': {'X8': 'b'},
    #      'X6': {},
    #      'X7': {},
    #      'X8': {},
    #      'X9': {}}
    #
    # R = ['x','y','u','v']
    # H1 = EDC(G)
    # print(f"分解：{H1}")

    # 首先生成随机链图
    # n = 8  # 节点数
    # edge_density = 0.2  # 边密度
    # p_undirected = 0.1 # 双向边概率
    #
    # # 生成随机链图
    # G = Generate_lwf(n, edge_density, p_undirected)
    # plot_result(G, 'j.png')
    #
    # # 随机选择一部分节点作为R集合
    # R = random.sample(list(G.keys()), 2)  # 随机选择3个节点
    # print(R)
    # M = [x for x in G if x not in R]
    # path = uv_DCL(G, R[0], R[1], M)
    # print(f"从 {R[0]} 到 {R[1]} 的最短路径是: {' -> '.join(path)}")

    # print(f"\n随机选择的R集合: {R}")
    #
    # # 使用两种方法计算t凸包
    # H1 = CMCSA(G, R)
    # H2 = CMCSA_new(G, R)

    # 比较结果
    # print(f"\nCMCSA方法得到的t凸包: {H1}")
    # print(f"CMCSA_new方法得到的t凸包: {H2}")
    # print(f"两种方法结果是否一致: {H1 == H2}")

    # G = {
    #     'r1':{'a':'b'},
    #     'b':{'a':'b','c':'b','d':'b'},
    #     'c':{ 'd':'b'},
    #     'a':{'e':'b'},
    #     'e':{'r2':'b'},
    #     'd':{'r2':'b'},
    #     # 'f':{'a':'b','g':'b','d':'b'},
    #     # 'g':{'d':'b'},
    #     'r2':{}
    # }

    # b = time.time()

    # H = [{'v1'}, {'v2', 'v17', 'v13', 'v5', 'v7', 'v10', 'v16', 'v3', 'v4'}, {'v15', 'v2', 'v14', 'v10', 'v3'}, {'v20', 'v8', 'v6', 'v5', 'v10', 'v9', 'v3', 'v4', 'v18'}, {'v10', 'v5', 'v9', 'v19', 'v3', 'v6', 'v12'}, {'v11'}]
    #
    # perm, E_T = check_all_permutations(H)
    # sep_list = separators(perm, E_T)
    # print(H)
    # print(sep_list)
    # print(perm)
    # # # G =  {'Erk': {'Akt': 'b'}, 'Akt': {}, 'Mek': {'Erk': 'b'}, 'PIP3': {'PIP2': 'b'}, 'PIP2': {}, 'PKA': {'Akt': 'b', 'Erk': 'b', 'Jnk': 'b', 'Mek': 'b', 'P38': 'b', 'Raf': 'b'}, 'Jnk': {}, 'P38': {}, 'Raf': {'Mek': 'b'}, 'PKC': {'Jnk': 'b', 'Mek': 'b', 'P38': 'b', 'PKA': 'b', 'Raf': 'b'}, 'Plcg': {'PIP2': 'b', 'PIP3': 'b'}}
    # R = ['r1', 'r2']
    # M = [x for x in G if x not in R]
    # A = CMCSA111_new(G,R)
    # B = CMCSA111(G,R)
    # D = CMCSA(G, R)
    # E= CMCSA_new(G, R)
    # C = is_set_cremoved(G, M)
    # print(A, B, C, D, E)
    # plot_result(G, 'j.png')

    # junction_tree = build_junction_tree(c_decomposition)
    # print(build_junction_tree(c_decomposition))
    # # 绘制图形
    # pos = nx.spring_layout(junction_tree)  # 定义布局
    # nx.draw(junction_tree, pos, with_labels=True, node_size=700, node_color="lightblue", font_size=10, font_weight="bold")
    # labels = nx.get_edge_attributes(junction_tree, 'weight')  # 获取边的权重作为标签
    # nx.draw_networkx_edge_labels(junction_tree, pos, edge_labels=labels)

    # plt.title("Junction Tree Visualization")
    # plt.show()
    # print(LD_HJT(G))

    # n = 100
    # edge_density = 0.3
    # p_undirected = 0
    # G = Generate_lwf(n, edge_density, p_undirected)
    # R = random.sample(list(G.keys()), 5)
    # # print(R)
    # H = CMCSA(G, R)
    # # M =[i for i in G if i not in H]
    # # print(is_set_cremoved(G, M))
    # print(n-len(H))

    # for i in range(100):
    #     lwf = Generate_lwf(n, edge_density, p_undirected)
    #     # print(lwf)
    #
    #     if 'v4' in lwf['v1'].keys() or 'v1' in lwf['v4'].keys():
    #         continue
    # time1 = time.time()
    # for i in range(10):
    #     C = c_Proximal_separator(CG, 'v1', 'v4')
    # print(time.time() - time1)
    # time2 = time.time()
    # for i in range(10):
    #     b = reach_set2(CG, ['v1', 'v4'])
    # print(time.time() - time2)

    # if C != b:
    #     print(f'c={C}, b={b}')
    #     plot_result(lwf, f'{i}.png',title=f'c={C}, b={b}')

    # M = random.sample(list(lwf.keys()), m)
    # MF = ITRSA(lwf, M, re_MF=True)
    # plot_result(lwf, 'j.png', title=f"M={M}, MF = {MF}")
