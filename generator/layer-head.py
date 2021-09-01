import argparse
import numpy as np
import networkx as nx
import _pickle as pickle

from extract import read_file

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data/AMR/amr_2.0/dev.txt.features.preproc')
    parser.add_argument('--max_attn_path', type=str, default='dev_max_attn.pkl')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_config()
    amrs, _, _, _ = read_file(args.data_path)
    with open(args.max_attn_path, 'rb') as f:
        max_attn = pickle.load(f)

    nvert = 0
    nheads, nlayers = 4, 8
    hl_dist = np.zeros((nheads, nlayers), dtype=np.float)
    for i, amr in enumerate(amrs):
        queue, _, _ = amr.bfs()
        for head in range(nheads):
            for layer in range(nlayers):
                for tgt, src in enumerate(max_attn[i][head, :, layer]):
                    d = nx.shortest_path_length(amr.graph, queue[tgt], queue[src])
                    hl_dist[head, layer] += d
        nvert += max_attn[i].shape[1]
    hl_dist /= nvert
    np.savetxt('dev_attn_max.csv', hl_dist, delimiter=',', fmt='%.4f')
