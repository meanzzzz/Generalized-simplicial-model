'''
Codes from the paper "A generalized simplicial model and its application"
An algorithm for generating or removing 2-simplex
Copyright (c) Rongmei Yang, Fang Zhou. All rights reserved.
'''

import networkx as nx
import random
import numpy as np
import copy
import os
import itertools
def find_clique(G,k):
    '''
    #Method of calculating the clique
    Parameters
    ----------
    G : network
    k : order of k-clique，1,2,3,4,5,...

    Returns
    -------
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

# To determine if there are triangles
def judege_triangle(graph, edge):
    neighbor1 = list(nx.neighbors(graph, str(edge[0])))
    neighbor2 = list(nx.neighbors(graph, str(edge[1])))
    interesction = list(set(neighbor1) & set(neighbor2))
    if len(interesction) == 0:
        return True # no triangles
    else:
        return False # Exist triangle

# Calculate the 2-clique should be removed
def remove_2_clique(graph,mk,node2,node3,node4,node5):
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

#To determine which triangle satisfies the condition
def get_satisfied_edge(graph, Edges,node2,node3):
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

# Select the appropriate node
def get_suitable_node(graph,repick,max_repick):
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
        return node1, str(node2), str(node3), str(node4), str(node5),repick

def get_remove_triangle(graph,mk,repick,max_repick):
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
            node4, node5= get_satisfied_edge(graph, Edges,node1,node2,node3)
            if (node4 == -1) | (node5 == -1):
                continue
            return node1,node2,node3,node4, node5,repick

#save txt file
def save_graph(graph,filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    graph_out = open(filename, 'w')
    edges = graph.edges()
    for j in edges:
        graph_out.write(j[0] + " " + j[1] + "\n")
    graph_out.close()

#generating or removing triangles
def generate_or_remove_triangles(graph, iteration_time, p1, filename,k,max_repick=1000):
    c_mean=0
    jj = 0 #iteration
    count = 0
    repick = 0  #the number of reselection
    if p1==1:
        mk = find_clique(graph, 2)
    while jj < iteration_time[-1]:
        if repick>max_repick:
            print('quit,iterations is '+str(jj)+',repick num is',str(repick))
            c_mean=jj
            print(jj, end=' ')
            triangle = nx.triangles(graph)
            print("The number of 2-simplex is:", sum(triangle.values()) / 3)
            # File Saving Path
            path = "results/" + filename + "/" + str(k) + "/" + filename + "_" + str(count + 1) + "_" + positive_symbol[p1] + ".txt"
            save_graph(graph, path)
            break
        if  p1==0: #generating 2-simplex
            node1, node2, node3, node4, node5,repick = get_suitable_node(graph,repick,max_repick)
            if (node1==-1) | (node2==-1) | (node3==-1)| (node4==-1)| (node5==-1):
                continue
            graph.remove_edge(node2, node4)
            graph.remove_edge(node3, node5)
            graph.add_edge(node2, node3)
            graph.add_edge(node4, node5)
            if nx.is_connected(graph):
                jj = jj + 1
                repick = 0
            else: #back
                graph.add_edge(node2, node4)
                graph.add_edge(node3, node5)
                graph.remove_edge(node2, node3)
                graph.remove_edge(node4, node5)

        if p1==1:  #removing 2-simplex
            node1, node2, node3, node4, node5, repick = get_remove_triangle(graph, mk, repick, max_repick)
            if (node1==-1) | (node2==-1) | (node3==-1)| (node4==-1)| (node5==-1):
                continue
            graph.remove_edge(node2, node3)
            graph.remove_edge(node4,node5)
            graph.add_edge(node2, node4)
            graph.add_edge(node3, node5)
            if nx.is_connected(graph):
                jj = jj + 1
                repick = 0
                mk=remove_2_clique(graph, mk, node2, node3, node4, node5)
            else:
                graph.add_edge(node2, node3)
                graph.add_edge(node4,node5)
                graph.remove_edge(node2, node4)
                graph.remove_edge(node3, node5)
        if (jj ==  iteration_time[count]) :
            print(jj,end=' ')
            triangle = nx.triangles(graph)
            print("高阶结构数量为:", sum(triangle.values()) / 3)
            # File Saving Path
            path = "results/" + filename + "/" + str(k) + "/" + filename + "_" + str(count+1) + "_"+positive_symbol[p1]+".txt"
            save_graph(graph, path)
            count = count + 1
    return c_mean

def Simplicial_Null_Model(dataset, p1,  file,iteration,repeat):
    #read network
    G = nx.read_weighted_edgelist(dataset, create_using=nx.Graph)
    if p1==1:
        triangle = nx.triangles(G)
        max_r=1*(sum(triangle.values()) / 3)    #Maximum number of reselections
    else:
        max_r=1*nx.number_of_nodes(G)    #Maximum number of reselections
    c_mean=0
    for i in range(1,repeat+1):   #重复10次，保存10个网络
        print(i)
        copy_graph = copy.deepcopy(G)
        c_mean+=generate_or_remove_triangles(copy_graph, list(iteration), p1, file,k=i,max_repick=max_r)
    print('average',c_mean/repeat)

if __name__ == '__main__':
    positive_symbol=['generate','remove']
    network_name='Email'   #网络名称
    k=100 #each k iteration save the model
    iteration_time = np.linspace(k,k*10, 10, dtype=int)
    dataset_path='datasets/'+network_name+'.txt' #Path of original network
    #0 is generating 2-simplex，1 is removing 2-simplex
    #repeat is times the model repeatedly generated
    Simplicial_Null_Model(dataset_path,0, network_name,interation=iteration_time,repeat=10)
