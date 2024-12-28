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
import tqdm

logger = logging.getLogger()
DEBUG_FLAG = 0

# NOTES!
# This can be extended in various ways:
    # Use other SentenceTransformer models (like RoBERTa, etc)
    # Use ngram keywords in range (x,y) (e.g. each keyword can be 'hubble telescope' instead of single word keywords)
    # Try various multitudes of keywords to determine which one yields better PageRank results (like 5, 10, 20 keywords) GRID SEARCH APPROACH
    # Use only keyword embeddings to obtain similarity between papers -> a LOT simpler, but possibly not as good, needs research
    # Each keyword comes with a similarity to the document itself. How can this information be harnessed?
    # Use Max Sum Distance or Maximal Marginal Relevance for improved keyword extraction
    #
    # Use config to test it out. Implement both training and testing using command line
    # Able to search with keywords, paper(s) (extract keywords essentially) or both
    # Tweak to use keywords from whole papers (will need huge dataset and a lot more training)
    # Add in information from the citations to improve PageRank performance
    #
    # Use Client/Server model with requests ?
    # Split program to train and query part
    #
# TODO!:
    # add save/load functionality so that index is not constantly being rebuild from scratch


def debug(*text):
    if DEBUG_FLAG:
        print(*text)


def train(params, devices, DEBUG=True):
    if isinstance(devices, list):
        device = devices[0]
    else: device = 'cuda:0'
    global DEBUG_FLAG
    DEBUG_FLAG = DEBUG
    ''' GRAB DATA FROM CONFIG '''
    # -- TRAIN
    dataset_path = params['dataset_path']
    n_keywords = params['n_keywords']
    ngrams = params['ngrams'] # this is a list of length=2 (min, max) ngrams
    use_mmr = params['use_mmr'] # maximal marginal relevance
    use_msd = params['use_msd'] # maximum sum distance
    # -- INTERMEDIATE
    save_path = params['save_path']
    # -- QUERY

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
        print('Counting lines...')
        num_lines = sum(1 for line in dataset) # count lines
        print('Lines:', num_lines)
    with open(dataset_path, 'r') as dataset:
        for idx, line in enumerate(tqdm.tqdm(dataset)):
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
            # if idx == 100:
            #     break
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
    # use list comprehension ?? could be faster ?
    # G.add_edges_from([(source_paper, target_paper) for source_paper, target_papers in paper_to_paper.items() for target_paper in target_papers])

    pagerank_scores = nx.pagerank(G) # use nonlinearrank
    ranked_papers = sorted(pagerank_scores.items(), key=lambda x:-x[1]) # sort descending
    for paper, score in ranked_papers:
        debug(f'Paper: {paper}, Title: {id_to_title[paper]}, Score: {score:.6f}')

    # print(f'{ranked_papers}')
    paper_rank_index = {paper[0]:idx for idx, paper in enumerate(ranked_papers)} # pos 0 refers to paper_id
    # print(f'{paper_rank_index=}')

    msg = 'Provide keywords, seperated by commas, or hit enter to quit:\n'
    inn = input(msg)
    while inn != '':
        keywords = [kw.strip() for kw in inn.split(',')]
        gather_papers = set()
        for kw in keywords:
            target_papers = keyword_to_papers[kw]
            for target_paper in target_papers:
                gather_papers.add(target_paper)
        # print(f'{gather_papers=}')
        gather_papers = sorted(gather_papers, key=lambda x: paper_rank_index[x]) # this should be done with MaxHeap !
        top_n_papers_query = 5 # this should be configurable
        for i in range(min(top_n_papers_query, len(gather_papers))):
            print(gather_papers[i], id_to_title[gather_papers[i]])



        inn = input(msg)

    print('Done')
