#!/usr/bin/env python
import os
import torch
import dist as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

from bert import BERTLM
from data import Vocab, DataLoader

import argparse, os
def parse_config():
    parser.add_argument('--embed_dim', type=int)
    parser.add_argument('--ff_embed_dim', type=int)
    parser.add_argument('--num_heads', type=int)
    parser.add_argument('--layers', type=int)
    parser.add_argument('--dropout', type=float)

    parser.add_argument('--train_data', type=str)
    parser.add_argument('--vocab', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--max_len', type=int)

    parser.add_argument('--world_size', type=int)
    parser.add_argument('--gpus', type=int)
    parser.add_argument('--MASTER_ADDR', type=str)
    parser.add_argument('--MASTER_PORT', type=str)
    parser.add_argument('--start_rank', type=str)

    return parser.parse_args()

def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=0)
        param.grad.data /= size

def run(args, local_rank, size):
    """ Distributed Synchronous Example """
    torch.manual_seed(1234)
    vocab = Vocab(args.vocab, min_occur_cnt=5)
    model = BERTLM(vocab, args.embed_dim, args.ff_embed_dim, args.num_heads, args.dropout, args.layers)
    model = model.cuda(loca_rank)
    
    torch.manual_seed(1234+dist.get_rank())
    optimizer = optim.Adam(model.parameters(), lr=1e-4, (0.9, 0.999), weight_decay=0.01)

    train_data = DataLoader(vocab, args.train_data, vocab, args.batch_size, args.max_len)
    for truth, inp, seg, msk, nxt_snt_flag in train_data:

        truth = truth.cuda(local_rank)
        inp = inp.cuda(local_rank)
        seg = seg.cuda(local_rank)
        msk = msk.cuda(local_rank)
        nxt_snt_flag = nxt_snt_flag.cuda(local_rank)


        optimizer.zero_grad()
        loss = model(truth, inp, seg, msk, nxt_snt_flag)
        print loss.item()
        loss.backward()
        average_gradients(model)
        optimizer.step()

def init_processes(args, local_rank, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.MASTER_ADDR
    os.environ['MASTER_PORT'] = args.MASTER_PORT
    dist.init_process_group(backend, rank=args.start_rank+local_rank, world_size=args.world_size)
    fn(args, local_rank, size)

if __name__ == "__main__":
    mp.set_start_method('spawn')
    args = parse_config()

    processes = []
    for rank in range(args.gpus):
        p = mp.Process(target=init_processes, args=(args, rank, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
