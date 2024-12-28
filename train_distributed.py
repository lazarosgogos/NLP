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
from concurrent.futures import ProcessPoolExecutor, as_completed
import signal
import sys
import os

logger = logging.getLogger()
DEBUG_FLAG = 0

# Global executor for proper cleanup
executor = None

def signal_handler(signum, frame):
    """Handle interrupt signals"""
    print('\nReceived interrupt signal. Cleaning up...')
    if executor:
        print('Shutting down executor...')
        executor.shutdown(wait=False)
        # Force terminate any remaining processes
        for pid in executor._processes:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    sys.exit(1)

def debug(*text):
    if DEBUG_FLAG:
        print(*text)

def process_chunk(chunk_data):
    """Worker function to process a chunk of lines."""
    try:
        lines, n_keywords, ngrams, model_name, device = chunk_data

        # Initialize model for this process
        model = SentenceTransformer(model_name, device=device)
        keybert = KeyBERT(model=model)

        paper_to_keywords = dict()
        keyword_to_papers = defaultdict(set)
        id_to_title = dict()

        for line in lines:
            entry = json.loads(line)
            id = entry['id']
            title = entry['title']
            id_to_title[id] = title

            abstract = entry['abstract'].replace('\n', ' ').replace('\r', '').strip()

            keywords = keybert.extract_keywords(
                abstract,
                top_n=n_keywords,
                keyphrase_ngram_range=(ngrams[0], ngrams[1]),
                stop_words='english',
            )
            paper_to_keywords[id] = keywords[0]
            for keyword, similarity_to_doc in keywords:
                keyword_to_papers[keyword].add(id)

        return paper_to_keywords, keyword_to_papers, id_to_title
    except KeyboardInterrupt:
        # Handle interruption in worker process
        return None

def merge_dicts(dicts):
    """Merge multiple dictionaries."""
    result = defaultdict(set)
    for d in dicts:
        if d is None:  # Skip None results from interrupted processes
            continue
        for k, v in d.items():
            if isinstance(v, set):
                result[k].update(v)
            else:
                result[k] = v
    return dict(result)

def train(params, devices, DEBUG=True):
    global executor  # Make executor global so signal handler can access it

    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if isinstance(devices, list):
        device = devices[0]
    else:
        device = 'cuda:0'

    global DEBUG_FLAG
    DEBUG_FLAG = DEBUG

    # Config parameters
    dataset_path = params['dataset_path']
    n_keywords = params['n_keywords']
    ngrams = params['ngrams']
    use_mmr = params['use_mmr']
    use_msd = params['use_msd']
    save_path = params['save_path']
    num_workers = params.get('num_workers', 4)
    model_name = params.get('model_name', 'all-MiniLM-L6-v2')

    try:
        # Read and split data
        with open(dataset_path, 'r') as dataset:
            print('Counting lines...')
            lines = dataset.readlines()
            num_lines = len(lines)
            print('Lines:', num_lines)

        # Split data into chunks
        chunk_size = num_lines // num_workers
        chunks = [lines[i:i + chunk_size] for i in range(0, num_lines, chunk_size)]

        # Prepare data for workers
        chunk_data = [(chunk, n_keywords, ngrams, model_name, device) for chunk in chunks]

        # Process chunks concurrently
        results = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor_ctx:
            executor = executor_ctx  # Store executor in global variable
            futures = [executor.submit(process_chunk, data) for data in chunk_data]

            try:
                for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
                    try:
                        result = future.result()
                        if result is not None:
                            results.append(result)
                    except Exception as e:
                        print(f"Error processing chunk: {e}")
            except KeyboardInterrupt:
                print("\nInterrupted by user. Cleaning up...")
                executor.shutdown(wait=False)
                raise

        # Merge results
        paper_to_keywords = merge_dicts([res[0] for res in results])
        keyword_to_papers = merge_dicts([res[1] for res in results])
        id_to_title = merge_dicts([res[2] for res in results])

        # Build paper-to-paper mapping
        paper_to_paper = defaultdict(set)
        for source_paper, keywords in paper_to_keywords.items():
            for kw in keywords:
                target_papers = keyword_to_papers[kw]
                for target_paper in target_papers:
                    paper_to_paper[source_paper].add(target_paper)

        # Create graph and compute PageRank
        G = nx.DiGraph()
        for source_paper, target_papers in paper_to_paper.items():
            for target_paper in target_papers:
                G.add_edge(source_paper, target_paper)

        pagerank_scores = nx.pagerank(G)
        ranked_papers = sorted(pagerank_scores.items(), key=lambda x:-x[1])
        for paper, score in ranked_papers:
            debug(f'Paper: {paper}, Title: {id_to_title[paper]}, Score: {score:.6f}')

        paper_rank_index = {paper[0]:idx for idx, paper in enumerate(ranked_papers)}

        # Interactive query section
        msg = 'Provide keywords, seperated by commas, or hit enter to quit:\n'
        while True:
            try:
                inn = input(msg)
                if inn == '':
                    break

                keywords = [kw.strip() for kw in inn.split(',')]
                gather_papers = set()
                for kw in keywords:
                    target_papers = keyword_to_papers[kw]
                    for target_paper in target_papers:
                        gather_papers.add(target_paper)

                gather_papers = sorted(gather_papers, key=lambda x: paper_rank_index[x])
                top_n_papers_query = 5
                for i in range(min(top_n_papers_query, len(gather_papers))):
                    print(gather_papers[i], id_to_title[gather_papers[i]])

            except KeyboardInterrupt:
                print("\nExiting...")
                break

    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        if executor:
            executor.shutdown(wait=False)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        if executor:
            executor.shutdown(wait=False)
        raise
    finally:
        if executor:
            executor.shutdown(wait=False)
        print('Done')
