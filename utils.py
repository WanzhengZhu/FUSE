import nltk
import numpy as np
import string
import torch
from collections import defaultdict
from tqdm import tqdm
from scipy.stats import entropy
from sklearn.cross_decomposition import CCA
from sklearn.cluster import AffinityPropagation
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM
from read_data import read_embedding


print('Initialize BERT vocabulary...')
bert_tokenizer = BertTokenizer(vocab_file='data/bert_models/bert-base-uncased-vocab.txt')
print('Initialize BERT model...')
bert_model = BertForMaskedLM.from_pretrained('data/bert_models/bert-base-uncased.tar.gz')
word_embedding = read_embedding('./data/glove.6B.50d.txt')


''' Module 1: Clustering '''
def extract_skip_grams(all_text, seed):
    skip_grams = []
    skip_n = 4
    for i in tqdm(all_text):
        temp = nltk.word_tokenize(i)
        if seed not in temp:
            continue
        temp_index = temp.index(seed)
        for l in range(2, skip_n):
            skip_grams.append(' '.join(temp[max(0, temp_index-l): temp_index]) + ' [MASK] ' + ' '.join(temp[temp_index+1: temp_index+l+1]))
    return skip_grams


def get_sentence_embedding(skip_grams):
    sentence_embedding = []
    for sg in skip_grams:
        temp_embedding = np.array([1e-3 for i in range(50)])
        for i in sg.split():
            if i in nltk.corpus.stopwords.words('english'):
                continue
            if i in word_embedding:
                temp_embedding += np.array(word_embedding[i])
        # if np.linalg.norm(temp_embedding) != 0:
        temp_embedding /= np.linalg.norm(temp_embedding)
        sentence_embedding.append(temp_embedding)
    return sentence_embedding


def clustering(text, text_embeddings, preference):
    print('Start Clustering...')
    clustering_labels = AffinityPropagation(preference=preference).fit_predict(text_embeddings)
    clusters = defaultdict(list)
    for idx, label in enumerate(clustering_labels):
        clusters[label].append(text[idx])
    to_delete = [i for i in clusters if len(clusters[i]) <= 10]
    for i in to_delete:
        del clusters[i]
    clusters = [clusters[x] for x in clusters]
    return clusters


def get_cluster_skip_grams(clusters):
    for i in range(len(clusters)):
        temp_fq = defaultdict(int)
        for w in clusters[i]:
            temp_fq[w] += 1
        temp_fq = [(i, temp_fq[i]) for i in sorted(temp_fq, key=lambda x: temp_fq[x], reverse=True)]
        print(f'Cluster {i} ({len(temp_fq)}): ')
        for j in temp_fq[:10]:
            print('\t', end="")
            print(j)


def clustering_skipgrams(seeds, all_text, preference):
    """ Clustering the skip-grams of a list of words """
    print('[utils.py] Clustering skip-grams...')
    clusters_all = []
    for seed in seeds:
        skip_grams = extract_skip_grams(all_text, seed)
        if skip_grams == []:
            print(seed + ' is not appeared in the corpus.')
            continue
        skip_gram_embeddings = get_sentence_embedding(skip_grams)
        clusters = clustering(skip_grams, skip_gram_embeddings, preference=preference)
        clusters_all.append(clusters)
        get_cluster_skip_grams(clusters)
        print()
    return clusters_all



''' Module 2: Fusing Semantic Facets of Multiple Seeds '''
def fuse_clusters_all(clusters_all):
    """
    Function:
    1) Combine different skipgrams clusters of different seeds;
    2) Denoise skipgrams for each cluster.
    :param clusters_all: A list of (list of skipgram clusters), e.g. [[F1, F2, F3], [F2, F3, F4, F5]],
                            where F1 indicates a semantic facets, represented by a list of skip-grams.
    :return: A list of skipgram clusters, e.g. [F2, F3]
    """

    def evalClusterSim(set_a, set_b):
        """
        :param set_a: set of skip-gram embeddings
        :param set_b: set of skip-gram embeddings
        :return: corr: scalar value quantifying the correlation between set_a and set_b
        """
        def normVec(vecs):
            norm_vecs = []
            for vec in vecs:
                n_vec = np.array(vec) / np.linalg.norm(vec)
                norm_vecs.append(list(n_vec)[:])
            return np.array(norm_vecs)

        model = CCA(n_components=1)
        arr_a = np.transpose(normVec(list(set_a)))  # arr_a: dim, size_a
        arr_b = np.transpose(normVec(list(set_b)))  # arr_b: dim, size_b
        model.fit(arr_a, arr_b)
        sense_vec_a = np.dot(arr_a, model.x_weights_)
        sense_vec_b = np.dot(arr_b, model.y_weights_)
        corr = np.dot(sense_vec_a.reshape((-1,)), sense_vec_b.reshape((-1,)))
        return corr

    def softmax(a):
        a = np.exp(a)
        sum_a = np.sum(a)
        return a / sum_a

    def fuse(sgs_cluster_a, sgs_cluster_b, sgs_embeddings_cluster_a, sgs_embeddings_cluster_b, final_sgs_clusters, final_sgs_embeddings_clusters):
        final_sgs_clusters.append(sgs_cluster_a + sgs_cluster_b)
        final_sgs_embeddings_clusters.append(sgs_embeddings_cluster_a + sgs_embeddings_cluster_b)
        return final_sgs_clusters, final_sgs_embeddings_clusters

    def fuse_common_clusters_of_two(sgs_embeddings_clusters_a, sgs_embeddings_clusters_b, sgs_clusters_a, sgs_clusters_b):
        final_sgs_clusters = []
        final_sgs_embeddings_clusters = []
        divergence = []
        best_index = []
        all_score = []
        for i in range(len(sgs_embeddings_clusters_a)):
            score = []
            for j in range(len(sgs_embeddings_clusters_b)):
                score.append(evalClusterSim(sgs_embeddings_clusters_a[i], sgs_embeddings_clusters_b[j]))
            score = softmax(score)
            all_score.append(score)
            best_index.append(np.argmax(score))
            print('the scores are: ')
            print(score)
            divergence.append(entropy(score, [1/float(len(score))] * len(score)))
            print(divergence[i])
            if divergence[i] > 0.25:  # smaller value means closer to uniform
                final_sgs_clusters, final_sgs_embeddings_clusters = fuse(sgs_clusters_a[i], sgs_clusters_b[best_index[i]], sgs_embeddings_clusters_a[i], sgs_embeddings_clusters_b[best_index[i]], final_sgs_clusters, final_sgs_embeddings_clusters)

        # If there is no cluster.
        if not final_sgs_clusters:
            # Find the best match index
            best_i_value = []
            for i in all_score:
                best_i_value.append(np.max(i))
            i = np.argmax(best_i_value)
            final_sgs_clusters, final_sgs_embeddings_clusters = fuse(sgs_clusters_a[i], sgs_clusters_b[best_index[i]],
                                                                     sgs_embeddings_clusters_a[i],
                                                                     sgs_embeddings_clusters_b[best_index[i]],
                                                                     final_sgs_clusters, final_sgs_embeddings_clusters)
        return final_sgs_clusters, final_sgs_embeddings_clusters


    print('[utils.py] Fusing skip-gram clusters...')
    if len(clusters_all) == 1:
        print('[utils.py] After fusion, there are ' + str(len(clusters_all[0])) + ' skip-gram clusters.')
        return clusters_all[0]
    sgs_embeddings_clusters = [[get_sentence_embedding(i) for i in clusters] for clusters in clusters_all]
    final_sgs_clusters = clusters_all[0]
    final_sgs_embeddings_clusters = sgs_embeddings_clusters[0]
    for num in range(1, len(clusters_all)):
        final_sgs_clusters, final_sgs_embeddings_clusters = fuse_common_clusters_of_two(final_sgs_embeddings_clusters, sgs_embeddings_clusters[num], final_sgs_clusters, clusters_all[num])
    print('[utils.py] After fusion, there are ' + str(len(final_sgs_clusters)) + ' skip-gram clusters.')
    return final_sgs_clusters



''' Module 3: Entity Expansion by Masked Language Model (MLM) '''
def MLM(sgs, seeds):
    def to_bert_input(tokens, bert_tokenizer):
        token_idx = torch.tensor(bert_tokenizer.convert_tokens_to_ids(tokens))
        sep_idx = tokens.index('[SEP]')
        segment_idx = token_idx * 0
        segment_idx[(sep_idx + 1):] = 1
        mask = (token_idx != 0)
        return token_idx.unsqueeze(0), segment_idx.unsqueeze(0), mask.unsqueeze(0)

    def single_MLM(message):
        MLM_k = 20
        tokens = bert_tokenizer.tokenize(message)
        if len(tokens) == 0:
            return []
        if tokens[0] != CLS:
            tokens = [CLS] + tokens
        if tokens[-1] != SEP:
            tokens.append(SEP)
        token_idx, segment_idx, mask = to_bert_input(tokens, bert_tokenizer)
        with torch.no_grad():
            logits = bert_model(token_idx, segment_idx, mask, masked_lm_labels=None)
        logits = logits.squeeze(0)
        probs = torch.softmax(logits, dim=-1)

        for idx, token in enumerate(tokens):
            if token == MASK:
                topk_prob, topk_indices = torch.topk(probs[idx, :], MLM_k)
                topk_tokens = bert_tokenizer.convert_ids_to_tokens(topk_indices.cpu().numpy())

        out = [[topk_tokens[i], float(topk_prob[i])] for i in range(MLM_k)]
        return out

    PAD, MASK, CLS, SEP = '[PAD]', '[MASK]', '[CLS]', '[SEP]'
    MLM_score = defaultdict(float)
    for sgs_i in tqdm(sgs):
        top_words = single_MLM(sgs_i)
        skip = 1
        for seed in seeds:
            if seed in [x[0] for x in top_words]:
                skip = 0
        if skip == 1:
            continue
        for j in top_words:
            if j[0] in string.punctuation:
                    continue
            if j[0] in nltk.corpus.stopwords.words('english'):
                continue
            MLM_score[j[0]] += j[1]
    out = sorted(MLM_score, key=lambda x: MLM_score[x], reverse=True)
    out_tuple = [[x, MLM_score[x]] for x in out]
    return out, out_tuple


def entity_expansion(clusters, seeds):
    """ Get words that fit in the cluster skip-grams well. """
    print('[utils.py] Entity Expansion ...')
    print(seeds)
    for i in range(len(clusters)):
        top_words, top_words_tuple = MLM(clusters[i], seeds)
        print(f'Cluster {i}: ', end="")
        print(top_words[:50])

