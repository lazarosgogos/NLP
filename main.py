'''
Authors: Bouzianas Nikoloaos, Fregkos Periklis, Gogos Lazaros
Year: 2024-2025
Task: NLP Project - Scientific Paper Search Engine based on keywords
Data & Web Science - Aristotle University of Thessaloniki
'''
import yaml
import pprint
import logging
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import train
import train_distributed


parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='configs/config.yaml',
)
parser.add_argument(
    '--devices', type=str, nargs='+', default=['cuda:0'],
    help='which devices to use on local machine'
)
parser.add_argument(
    '--debug', action=argparse.BooleanOptionalAction, default=False,
    help='print helpful debug info (true if present, defaults to false)'
)

parser.add_argument(
    '--train', action=argparse.BooleanOptionalAction, default=False,
    help='create new index based on dataset',
)

parser.add_argument(
    '--keywords', type=str, nargs=1,
    help='Query keywords, separated by commas (e.g. "brute force, hubble, space telescope")'
)

parser.add_argument(
    '--papers', type=str, nargs=1,
    help='Query papers, from which keywords will automatically be extracted. \
    Can also be a directory with papers.'
)

def main(fname, devices=None, debug=False): # devices is None for now
    params = yaml.load(open(fname, 'r'), Loader=yaml.FullLoader)
    dataset_path = params['dataset_path']

    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.ERROR)
    logger.info('loaded params from configuration file ..')
    pprint.pprint(params)
    # train_distributed.train(params, devices, debug)
    train.train(params, devices, debug)




if __name__ == '__main__':
    args = parser.parse_args()
    print(args.debug)
    main(args.fname, args.devices, args.debug)
