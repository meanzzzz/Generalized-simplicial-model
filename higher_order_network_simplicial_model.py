"""
Reference:

Rongmei Yang, Fang Zhou, Bo Liu, Linyuan Lü; A generalized simplicial model and its application. Chaos 1 April 2024; 34 (4): 043113."""

import networkx as nx
import random
import numpy as np
import copy
import os
import itertools
def find_clique(G,k):
    '''
    Method of calculating the clique
    Parameters:
        G : network
        k : order of k-clique，1,2,3,4,5,...
    Returns:
        cliquesk : The list of cliques

    '''

    cliques = nx.enumerate_all_cliques(G)
    cliquesk = []
    for clq in cliques:
        if len(clq)<(k+1):
            continue
        if len(clq)==(k+1):
            cliquesk.append(clq)
        if len(clq)>(k+1):
            break
    cliquesk = sorted([tuple(sorted(i)) for i in cliquesk])
    return cliquesk

def remove_2_clique(graph,mk,node2,node3,node4,node5):
    '''
    Calculate the 2-clique should be dismantled
    Parameters:
        graph : network;
        mk: the list of k-clique
        node2/3/4/5 : The selected nodes
    Returns:
        mk_new : the list of k-clique after one rewiring

    '''
    clique_r = []
    # The common neighbors of node2 and node3
    interesction_2_3 = list(set(nx.neighbors(graph, node2)) & set(nx.neighbors(graph, node3)))
    for i in interesction_2_3:
        clique_r.append(tuple(sorted([i,node2,node3])))
    # The common neighbors of node4 and node5
    interesction_4_5 = list(set(nx.neighbors(graph, node4)) & set(nx.neighbors(graph, node5)))
    for i in interesction_4_5:
        clique_r.append(tuple(sorted([i,node4,node5])))
    mk_new=list(set(mk) - set(clique_r))
    return mk_new

def get_satisfied_edge(graph, Edges,node2,node3):
    '''
    To determine which triangle satisfies the condition
    Parameters:
        graph : network;
        Edges: the edge set of network
        node2/3 : The selected nodes
    Returns:
        If satisfied, return edge e[0],e[1]; otherwise,return -1,-1
    '''
    node4_suitable= set(graph.nodes())
    node2_neibor=list(nx.neighbors(graph, node2))
    node4_suitable =  node4_suitable-set(node2_neibor)
    for i in node2_neibor:
        node4_suitable = node4_suitable - set(nx.neighbors(graph, i))
    if len(node4_suitable)==0:
        return -1,-1
    node5_suitable = set(graph.nodes())
    node3_neibor=list(nx.neighbors(graph, node3))
    node5_suitable =  node5_suitable-set(node3_neibor)
    for i in node3_neibor:
        node5_suitable= node5_suitable - set(nx.neighbors(graph, i))
    if len(node5_suitable)==0:
        return -1,-1
    random.shuffle(Edges)
    for e in Edges:
        if (e[0] in node4_suitable) & (e[1] in node5_suitable):
            return e[0],e[1]
    return -1,-1

def get_suitable_node(graph,repick,max_repick):
    '''
    Select the appropriate node
    Parameters:
        graph : network;
        repick: The current number of reselections
        max_repick: The max number of reselections
    Returns:
        str(node1/2/3/4/5):the selected node;
        repick: The current number of reselections
    '''
    while True:
        if repick>max_repick:
            return -1,-1,-1,-1,-1,repick
        repick=repick+1
        node1 = random.choice(list(graph.nodes()))
        if graph.degree(node1)< 2: #degree>2
            continue
        # The neighbors of nodes which degree>=2
        neighbors = [i for i in list(nx.neighbors(graph, str(node1))) if graph.degree(i) >= 2]
        if len(neighbors) < 2:
            continue
        #  no edge between 2 and 3
        combine_2_3 = []
        for i in list(itertools.combinations(neighbors, 2)):
            if graph.has_edge(i[0],i[1]) == False:
                combine_2_3.append(i)
        if len(combine_2_3) < 1:
            continue
        random.shuffle(combine_2_3)
        node4,node5=-1,-1
        for i in range(len(combine_2_3)):
            node2, node3 = combine_2_3[i][0], combine_2_3[i][1]  # node2，node3
            node2_neighbor = []
            for i in list(nx.neighbors(graph, str(node2))):
                if not (set(nx.neighbors(graph, str(node2))) & set(nx.neighbors(graph, i))):
                    if i != node1:
                        node2_neighbor.append(i)
            node3_neighbor = []
            for i in list(nx.neighbors(graph, str(node3))):
                if not (set(nx.neighbors(graph, str(node3))) & set(nx.neighbors(graph, i))):
                    if i != node1:
                        node3_neighbor.append(i)
            # no edge between 4 and 5
            combine_4_5 = []
            for i in node2_neighbor:
                for j in node3_neighbor:
                    if (graph.has_edge(i, j) == False) & (i != j):
                        combine_4_5.append((i, j))
            if len(combine_4_5) < 1:
                continue
            node_4_5=random.choice(combine_4_5)
            node4, node5 = node_4_5[0],node_4_5[1]  # node4，node5
            break
        if (node4==-1) | (node5==-1):
            continue
        return str(node1), str(node2), str(node3), str(node4), str(node5),repick

def get_remove_triangle(graph,mk,repick,max_repick):
    '''
    Select the appropriate triangle to dismantle
    Parameters:
        graph : network;
        mk:the list of k-clique
        repick: The current number of reselections
        max_repick: The max number of reselections
    Returns:
        node1/2/3/4/5:the selected node;
        repick: The current number of reselections
    '''
    Edges = list(graph.edges())
    while True:
        if repick>max_repick:
            return -1,-1,-1,-1,-1,repick
        if len(mk)==0:
            repick=max_repick+1
            continue
        triangle = list(random.choice(mk))
        random.shuffle(triangle)
        repick = repick + 1
        combine_2_3 = list(itertools.combinations(triangle, 2))
        for edge in combine_2_3:
            node2, node3 = edge[0], edge[1]
            node1 = list(set(triangle) - set(edge))[0]
            node4, node5= get_satisfied_edge(graph, Edges,node2,node3)
            if (node4 == -1) | (node5 == -1):
                continue
            return node1,node2,node3,node4, node5,repick

def save_graph(graph,filename):
    '''
    save simplicial model
    Parameters:
        graph : network;
        filename:Saving paths
    '''
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    graph_out = open(filename, 'w')
    edges = graph.edges()
    for j in edges:
        graph_out.write(j[0] + " " + j[1] + "\n")
    graph_out.close()

def generate_or_remove_triangles(graph, iteration_time, operation, filename,k,max_repick=1000):
    '''
    generate or dismante triangles
    Parameters:
        graph : network;
        iteration_time:successful iteration number;
        operation:generate or dismante flags, 0 for generation and 1 for dismantling
        filename:Saving paths;
        k:The number of times the experiment was repeated;
        max_repick: The max number of reselections
    '''
    iterations = 0 #iteration
    count = 0
    repick = 0  #the number of reselection
    if operation==1:
        mk = find_clique(graph, 2)
    while iterations < iteration_time[-1]:
        if repick>max_repick:  # If no more triangles are successfully generated or dismantled, exit
            print('exit, current iterations are'+str(iterations)+',repick number are',str(repick))
            print(iterations, end=' ')
            triangle = nx.triangles(graph)
            print("The number of 2-simplex are:", sum(triangle.values()) / 3)
            # File Saving Path
            path = "results/" + filename + "/" + str(k+10) + "/" + filename + "_" + str(count + 1) + "_" + positive_symbol[operation] + ".txt"
            save_graph(graph, path)
            break
        if  operation==0: #generate 2-simplex
            node1, node2, node3, node4, node5,repick = get_suitable_node(graph,repick,max_repick)
            if (node1==-1) | (node2==-1) | (node3==-1)| (node4==-1)| (node5==-1):
                continue
            graph.remove_edge(node2, node4)
            graph.remove_edge(node3, node5)
            graph.add_edge(node2, node3)
            graph.add_edge(node4, node5)
            if nx.is_connected(graph):
                iterations = iterations + 1
                repick = 0
            else: #back
                graph.add_edge(node2, node4)
                graph.add_edge(node3, node5)
                graph.remove_edge(node2, node3)
                graph.remove_edge(node4, node5)

        if operation==1:  #dismantle 2-simplex
            node1, node2, node3, node4, node5, repick = get_remove_triangle(graph, mk, repick, max_repick)
            if (node1==-1) | (node2==-1) | (node3==-1)| (node4==-1)| (node5==-1):
                continue
            graph.remove_edge(node2, node3)
            graph.remove_edge(node4,node5)
            graph.add_edge(node2, node4)
            graph.add_edge(node3, node5)
            if nx.is_connected(graph):
                iterations = iterations + 1
                repick = 0
                mk=remove_2_clique(graph, mk, node2, node3, node4, node5)
            else:
                graph.add_edge(node2, node3)
                graph.add_edge(node4,node5)
                graph.remove_edge(node2, node4)
                graph.remove_edge(node3, node5)
        if (iterations ==  iteration_time[count]) :
            print('iterations:'+str(iterations),end=' ')
            triangle = nx.triangles(graph)
            print(",The number of 2-simplex are:", sum(triangle.values()) / 3)
            # File Saving Path
            path = "results/" + filename + "/" + str(k+10) + "/" + filename + "_" + str(count+1) + "_"+positive_symbol[operation]+".txt"
            save_graph(graph, path)
            count = count + 1

def Simplicial_Null_Model(dataset, operation ,network_name,experiment_repeats=10):
    '''
    Simplicial_Null_Model
    Parameters:
        dataset : dataset path;
        operation:generate or dismante flags, 0 for generation and 1 for dismantling
        network_name:network name;
        experiment_repeats:The number of times the experiment was repeated;
    '''
    save_interval=50 #Save the model every 50 iterations
    iteration_time = np.linspace(save_interval,save_interval*10, 10, dtype=int)
    G = nx.read_weighted_edgelist(dataset, create_using=nx.Graph)
    #0 is generating 2-simplex，1 is removing 2-simplex
    #repeat is times the model repeatedly generate networks
    if operation==1:
        triangle = nx.triangles(G)
        max_r=1*(sum(triangle.values()) / 3)    #Maximum number of reselections for dismantling 2-simplex
    else:
        max_r=1*nx.number_of_nodes(G)    #Maximum number of reselections for generating 2-simplex
    for i in range(1,experiment_repeats+1):   #repeat times,totally save repeat netwroks
        print('the '+str(i)+'-th simplicial network')
        copy_graph = copy.deepcopy(G)
        generate_or_remove_triangles(copy_graph, list(iteration_time), operation, network_name,k=i,max_repick=max_r)

if __name__ == '__main__':
    positive_symbol=['generate','remove']
    network_name='Email'   #network name
    dataset_path =  network_name + '.txt'  # Path of original network
    Simplicial_Null_Model(dataset_path,0, network_name,experiment_repeats=10)
