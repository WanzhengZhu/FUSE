import argparse
from read_data import read_all_text
from utils import clustering_skipgrams, fuse_clusters_all, entity_expansion



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Main.py', description='')
    parser.add_argument('-dataset', default='sample')
    parser.add_argument('-p', default=-10, help='preference for Affinity Propagation')
    args = parser.parse_args()

    all_text = read_all_text('./data/dataset_' + args.dataset + '.txt')
    seeds = ['beaver', 'elk']
    clusters_all = clustering_skipgrams(seeds, all_text, preference=args.p)
    final_sgs_clusters = fuse_clusters_all(clusters_all)
    entity_expansion(final_sgs_clusters, seeds)

