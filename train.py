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

logger = logging.getLogger()
DEBUG_FLAG = 0

# NOTES!
# This can be extended in various ways:
    # Use ngram keywords in range (x,y) (e.g. each keyword is 'hubble telescope' instead of single word keywords)
    # Try various multitudes of keywords to determine which one yields better PageRank results (like 5, 10, 20 keywords) GRID SEARCH APPROACH
    # Use only keyword embeddings to obtain similarity between papers -> a LOT simpler, but possibly not as good, needs to be researched
    # Each keyword comes with a similarity to the document itself
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
    # dataset = ArxivDataset(dataset_path)
    # model name should come from config!
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    keybert = KeyBERT(model=model, )
    paper_to_keywords = dict()
    keyword_to_papers = defaultdict(list)
    id_to_title = dict()
    paper_to_paper = defaultdict(list)
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
                abstract, top_n=n_keywords,
                keyphrase_ngram_range=(ngrams[0], ngrams[1]),
                stop_words='english',
            )
            print(keywords)
            paper_to_keywords[id] = keywords
            for keyword, _ in keywords:
                keyword_to_papers[keyword].append(id)




            if idx == 100:
                break
