import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import sys


def jaccard_init(dataset_str):
    
    f = open("data/{}/ind.{}.{}".format(dataset_str,dataset_str, graph), 'rb')
    graph = pkl.load(f)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    G = nx.DiGraph()
    nodenum = adj.shape[0]
    inf= pickle.load(open('adj_citeseer.pkl', 'rb'))
    inf= graph                 
    for i in range(len(inf)):
        for j in range(len(inf[i])):
            G.add_edge(i, inf[i][j], weight=1)
    print(G)
    
    for i in range(nodenum):
        print(i)
        for j in range(nodenum):
                rs = nx.astar_path_length(
                        G,
                        i,
                        j,
                    )
            except nx.NetworkXNoPath:
                 rs = 0
            if rs == 0:
                  length = 0
            else:
                # print(rs)
                length = rs
            adj_delta[i][j] = length
            
    a = open("dijkstra_citeseer.pkl", 'wb')
    pickle.dump(adj_delta, a)
    a.close()
    np.set_printoptions(threshold=sys.maxsize)
    e = math.e
    fw = open("dijkstra_citeseer.pkl", 'rb')
    dijkstra = pickle.load(fw)
    Dijkstra = dijkstra.numpy()
    jaccard_all = []
    jaccard_index = []
    
    for i in range(3327):
        index_i = np.where((Dijkstra[i] < 5) & (Dijkstra[i] > 0))#k=3,3-hop perceptive field
        ei = []
        for j in index_i:
            ei.append(e**(-((Dijkstra[i][j]**2)/2)))
        ei = torch.tensor(ei, dtype=torch.float32)
        ei = ei.numpy().tolist()
        jaccard_index.append(index_i[0])
        jaccard_all.append(ei[0])
    fw = open('jaccard_index_citeseer.pkl', 'wb')
    pickle.dump(jaccard_index, fw)
    fw.close()
    fw = open('jaccard_all_citeseer.pkl', 'wb')
    pickle.dump(jaccard_all, fw)
    fw.close() 
    
    return
