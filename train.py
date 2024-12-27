import logging
import json
from pickle import TRUE
import yaml
import pprint
import logging
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gc
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import networkx as nx

logger = logging.getLogger()
DEBUG_FLAG = 0

# NOTES!
# This can be extended in various ways:
    # Use Other SentenceTransformer models (like RoBERTa, etc)
    # Use ngram keywords in range (x,y) (e.g. each keyword can be 'hubble telescope' instead of single word keywords)
    # Try various multitudes of keywords to determine which one yields better PageRank results (like 5, 10, 20 keywords) GRID SEARCH APPROACH
    # Use only keyword embeddings to obtain similarity between papers -> a LOT simpler, but possibly not as good, needs research
    # Each keyword comes with a similarity to the document itself. How can this information be harnessed?
    # Use Max Sum Distance or Maximal Marginal Relevance for improved keyword extraction


def debug(*text):
    if DEBUG_FLAG:
        print(*text)


def train(params, devices, DEBUG=True):
    if isinstance(devices, list):
        device = devices[0]
    else: device = 'cuda:0'
    global DEBUG_FLAG
    DEBUG_FLAG = DEBUG
    ''' CONFIG '''
    dataset_path = params['dataset_path']
    n_keywords = params['n_keywords']
    ngrams = params['ngrams'] # this is a list of length=2 (min, max) ngrams
    save_path = params['save_path']
    # dataset = ArxivDataset(dataset_path)
    # model name should come from config!
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    keybert = KeyBERT(model=model, )
    paper_to_keywords = dict()
    keyword_to_papers = defaultdict(set)
    id_to_title = dict()
    paper_to_paper = defaultdict(set)
    with open(dataset_path, 'r') as dataset:
        for idx, line in enumerate(dataset):
            entry = json.loads(line)
            id = entry['id']
            title = entry['title']
            id_to_title[id] = title # map id to title

            # extract abstract and clean it
            abstract = entry['abstract'].\
                replace('\n', ' ').\
                replace('\r', '').\
                strip()

            debug(f'{id=}, {title=}, {abstract=}')
            keywords = keybert.extract_keywords(
                abstract,
                top_n=n_keywords,
                keyphrase_ngram_range=(ngrams[0], ngrams[1]),
                stop_words='english', # default
            )
            debug(keywords)
            paper_to_keywords[id] = keywords[0] # extract only the keyword for now, not its similarity to the paper
            for keyword, similarity_to_doc in keywords:
                keyword_to_papers[keyword].add(id)
            if idx == 100:
                break
    for source_paper, keywords in paper_to_keywords.items():
        # """
#        This can be done with list comprehension
        for kw in keywords:
            target_papers = keyword_to_papers[kw]
            for target_paper in target_papers:
                # print('types:', type(source_paper), type(target_paper))
                paper_to_paper[source_paper].add(target_paper)
        # """
        # paper_to_paper[source_paper].add(keyword_to_papers[kw] for kw in keywords)


    # now we have the paper-to-paper mapping
    # Now run pagerank to determine which papers are the best ones
    G = nx.DiGraph()
    for source_paper, target_papers in paper_to_paper.items():
        for target_paper in target_papers:
            # print('type of target paper:', type(target_paper))
            G.add_edge(source_paper, target_paper)

    # or add weighted edges from ?
    # use list comprehension ?? could be faster
    # G.add_edges_from([(source_paper, target_paper) for source_paper, target_papers in paper_to_paper.items() for target_paper in target_papers])

    pagerank_scores = nx.pagerank(G)
    for paper, score in sorted(pagerank_scores.items(), key=lambda x:-x[1]):
        print(f'Paper: {paper}, Title: {id_to_title[paper]}, Score: {score:.6f}')
#
