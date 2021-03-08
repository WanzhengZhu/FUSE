import nltk
from tqdm import tqdm


def read_all_text(fname):
    print('[read_data.py] Reading data...')
    all_text = []
    num_lines = sum(1 for line in open(fname))
    with open(fname, 'r') as fin:
        for line in tqdm(fin, total=num_lines):
            for i in nltk.sent_tokenize(line):
                all_text.append(i.lower())
    return all_text


def read_embedding(fname):
    word_embedding = {}
    with open(fname, 'r') as fin:
        for line in fin:
            word_embedding[line.split()[0]] = [float(i) for i in line.split()[1:]]
    return word_embedding
