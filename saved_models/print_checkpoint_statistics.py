'''
'''
import argparse
import os
import torch
import pprint

def parse_args():
    parser = argparse.ArgumentParser()

    return parser.parse_args()


def main():
    args = parse_args()
    checkpoints = sorted(
        [x for x in os.listdir('.') if 'epoch_' in x],
        key=lambda x: float(x.split('_')[1].split('.')[0]))
    print('\t'.join(['ckpt', 'rl?', 'patience', 'validation cider']))
    for ckpt in checkpoints:
        d = torch.load(ckpt)
        val_cider = str(d['val_cider'])
        used_rl = str(d['use_rl'])
        patience = str(d['patience'])
        print('\t'.join([ckpt, used_rl, patience, val_cider]))

    
    
    
if __name__ == '__main__':
    main()
