import random
import os
import sys
import argparse
import copy
import shutil
import json
from pprint import pprint
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


"""
How To:
Example for running from command line:
python <path_to>/SetToGraph/main_scripts/main_jets.py
"""
# Change working directory to project's main directory, and add it to path - for library and config usages
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_dir)
os.chdir(project_dir)

# Project dependencies
from models.set_to_graph import SetToGraph
from models.set_to_graph_siam import SetToGraphSiam
from models.triplets_model import SetPartitionTri #NameError: name 'SetPartitionTri' is not defined
from dataloaders import jets_loader
from performance_eval.eval_test_jets import eval_jets_on_test_set

DEVICE = 'cpu'#'cuda' #NOT USING CUDA


def parse_args():
    """
    Define and retrieve command line arguements
    :return: argparser instance
    """
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-g', '--gpu', default='0', help='The gpu to run on')
    argparser.add_argument('-e', '--epochs', default=400, type=int, help='The number of epochs to run')
    argparser.add_argument('-l', '--lr', default=0.001, type=float, help='The learning rate')
    argparser.add_argument('-b', '--bs', default=2048, type=int, help='Batch size to use')
    argparser.add_argument('--method', default='lin2', help='Method to transfer from sets to graphs: lin2 for S2G, lin5 for S2G+')
    argparser.add_argument('--res_dir', default='experiments/jets_results', help='Results directory') #For Comfort # '../experiments/jets_results' -> 'experiments/jets_results'
    argparser.add_argument('--baseline', default=None, help='Use a baseline and not set2graph. siam or rnn.')

    argparser.add_argument('--debug_load', dest='debug_load', action='store_true', help='Load only a small subset of the data')
    argparser.add_argument('--save', dest='save', action='store_true', help='Whether to save all to disk')
    argparser.add_argument('--no_save', dest='save', action='store_false')
    argparser.set_defaults(save=True, debug_load=False)

    args = argparser.parse_args()

    assert args.baseline is None or args.baseline in ['siam', 'rnn']

    return args


def calc_metrics(pred_partitions, partitions_as_graph, partitions, accum_info):
    with torch.no_grad():
        B, N = partitions.shape
        C = pred_partitions.max().item() + 1
        pred_partitions = pred_partitions[:, :, np.newaxis]
        pred_onehot = torch.zeros((B, N, C), dtype=torch.float, device=partitions.device)
        pred_onehot.scatter_(2, pred_partitions, 1)
        pred_matrices = torch.matmul(pred_onehot, pred_onehot.transpose(1, 2))

        # calc fscore, precision, recall
        tp = (pred_matrices * partitions_as_graph).sum(dim=(1, 2)) - N  # Don't care about diagonals
        fp = (pred_matrices * (1 - partitions_as_graph)).sum(dim=(1, 2))
        fn = ((1 - pred_matrices) * partitions_as_graph).sum(dim=(1, 2))
        accum_info['recall'] += (tp / (tp + fp + 1e-10)).sum().item()
        accum_info['precision'] += (tp / (tp + fn + 1e-10)).sum().item()
        accum_info['fscore'] += ((2 * tp) / (2 * tp + fp + fn + 1e-10)).sum().item()

        # calc RI
        equiv_pairs = (pred_matrices == partitions_as_graph).float()
        accum_info['accuracy'] += equiv_pairs.mean(dim=(1, 2)).sum().item()
        # ignore pairs of same node
        equiv_pairs[:, torch.arange(N), torch.arange(N)] = torch.zeros((N,), device=DEVICE)  
        ri_results = equiv_pairs.sum(dim=(1, 2)) / (N*(N-1))
        accum_info['ri'] += ri_results.sum().item()

    return accum_info


def infer_clusters(edge_vals):
    '''
    Infer the clusters. Enforce symmetry.
    :param edge_vals: predicted edge score values. shape (B, N, N)
    :return: long tensor shape (B, N) of the clusters.
    '''
    # deployment - infer chosen clusters:
    b, n, _ = edge_vals.shape
    with torch.no_grad():
        pred_matrices = edge_vals + edge_vals.transpose(1, 2)  # to make symmetric
        pred_matrices = pred_matrices.ge(0.).float()  # adj matrix - 0 as threshold
        pred_matrices[:, np.arange(n), np.arange(n)] = 1.  # each node is always connected to itself
        ones_now = pred_matrices.sum()
        ones_before = ones_now - 1
        while ones_now != ones_before:  # get connected components - each node connected to all in its component
            ones_before = ones_now
            pred_matrices = torch.matmul(pred_matrices, pred_matrices)
            pred_matrices = pred_matrices.bool().float()  # remain as 0-1 matrices
            ones_now = pred_matrices.sum()

        clusters = -1 * torch.ones((b, n), device=edge_vals.device)
        tensor_1 = torch.tensor(1., device=edge_vals.device)
        for i in range(n):
            clusters = torch.where(pred_matrices[:, i] == 1, i * tensor_1, clusters)

    return clusters.long()


def get_loss(y_hat, y):
    # No loss on diagonal
    B, N, _ = y_hat.shape
    y_hat[:, torch.arange(N), torch.arange(N)] = torch.finfo(y_hat.dtype).max  # to be "1" after sigmoid

    # calc loss
    loss = F.binary_cross_entropy_with_logits(y_hat, y)  # cross entropy

    y_hat = torch.sigmoid(y_hat)
    tp = (y_hat * y).sum(dim=(1, 2))
    fn = ((1. - y_hat) * y).sum(dim=(1, 2))
    fp = (y_hat * (1. - y)).sum(dim=(1, 2))
    loss = loss - ((2 * tp) / (2 * tp + fp + fn + 1e-10)).sum()  # fscore

    return loss


def train(data, model, optimizer):
    train_info = do_epoch(data, model, optimizer)
    return train_info


def evaluate(data, model):
    val_info = do_epoch(data, model, optimizer=None)
    return val_info


def do_epoch(data, model, optimizer=None):
    if optimizer is not None:
        # train epoch
        model.train()
    else:
        # validation epoch
        model.eval()
    start_time = datetime.now()

    # Iterate over batches
    accum_info = {k: 0.0 for k in ['ri', 'loss', 'insts', 'accuracy', 'fscore', 'precision', 'recall']}
    for sets, partitions, partitions_as_graph in data:
        # One Train step on the current batch
        sets = sets.to(DEVICE, torch.float)
        partitions = partitions.to(DEVICE, torch.long)
        partitions_as_graph = partitions_as_graph.to(DEVICE, torch.float)
        batch_size = sets.shape[0]
        accum_info['insts'] += batch_size

        if isinstance(model, SetPartitionTri): #NameError: name 'SetPartitionTri' is not defined -> from models.triplets_model import SetPartitionTri
            pred_partitions, loss = model(sets, partitions)
        else:
            edge_vals = model(sets).squeeze(1)  # B,N,N
            pred_partitions = infer_clusters(edge_vals)
            loss = get_loss(edge_vals, partitions_as_graph)

        if optimizer is not None:
            # backprop for training epochs only
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # calc ri
        accum_info = calc_metrics(pred_partitions, partitions_as_graph, partitions, accum_info)

        # update results from train_step func
        accum_info['loss'] += loss.item() * batch_size

    num_insts = accum_info.pop('insts')
    accum_info['loss'] /= num_insts
    accum_info['ri'] /= num_insts
    accum_info['accuracy'] /= num_insts
    accum_info['fscore'] /= num_insts
    accum_info['recall'] /= num_insts
    accum_info['precision'] /= num_insts

    accum_info['run_time'] = datetime.now() - start_time
    accum_info['run_time'] = str(accum_info['run_time']).split(".")[0]

    return accum_info


def main():
    start_time = datetime.now()

    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    #torch.cuda.manual_seed(seed) #NOT USING CUDA
    #torch.cuda.manual_seed_all(seed) #NOT USING CUDA
    # torch.backends.cudnn.deterministic = True  # can impact performance
    # torch.backends.cudnn.benchmark = False  # can impact performance

    config = parse_args()

    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # uncomment only for CUDA error debugging
    # os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    #torch.cuda.set_device(int(config.gpu)) #NOT USING CUDA

    pprint(vars(config))
    print(flush=True)

    # Load data
    print('Loading training data...', end='', flush=True)
    train_data = jets_loader.get_data_loader('train', config.bs, config.debug_load)
    print('Loading validation data...', end='', flush=True)
    val_data = jets_loader.get_data_loader('validation', config.bs, config.debug_load)

    # Create model instance
    if config.baseline == 'rnn':
        model = SetToGraph(10,
                           out_features=1,
                           set_fn_feats=[256, 256, 256, 256, 5],
                           method=config.method,
                           hidden_mlp=[256],
                           predict_diagonal=False,
                           attention=True,
                           set_model_type='RNN')
    elif config.baseline == 'siam':
        model = SetToGraphSiam(10, [384, 384, 384, 384, 5], hidden_mlp=[256])
    else:
        assert config.baseline is None
        model = SetToGraph(10,
                           out_features=1,
                           set_fn_feats=[256, 256, 256, 256, 5],
                           method=config.method,
                           hidden_mlp=[256],
                           predict_diagonal=False,
                           attention=True,
                           set_model_type='deepset')
    print('Model:' , model)
    model = model.to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The nubmer of model parameters is {num_params}')

    # Optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.lr)

    # Metrics
    train_loss = np.empty(config.epochs, float)
    train_ri = np.empty(config.epochs, float)
    val_loss = np.empty(config.epochs, float)
    val_ri = np.empty(config.epochs, float)

    best_epoch = -1
    best_val_ri = -1
    best_val_fscore = -1
    best_model = None

    # Training and evaluation process
    for epoch in range(1, config.epochs + 1):
        train_info = train(train_data, model, optimizer)
        print(f"\tTraining - {epoch:4}",
              " loss:{loss:.6f} -- mean_ri:{ri:.4f} -- fscore:{fscore:.4f} -- recall:{recall:.4f}"
              " -- precision:{precision:.4f} -- runtime:{run_time}".format(**train_info), flush=True)
        train_loss[epoch-1], train_ri[epoch-1] = train_info['loss'], train_info['ri']

        val_info = evaluate(val_data, model)
        print(f"\tVal      - {epoch:4}",
              " loss:{loss:.6f} -- mean_ri:{ri:.4f} -- fscore:{fscore:.4f} -- recall:{recall:.4f}"
              " -- precision:{precision:.4f}  -- runtime:{run_time}\n".format(**val_info), flush=True)
        val_loss[epoch-1], val_ri[epoch-1] = val_info['loss'], val_info['ri']

        if val_info['fscore'] > best_val_fscore:
            best_val_fscore = val_info['fscore']
            best_epoch = epoch
            best_model = copy.deepcopy(model)
        
        if best_epoch < epoch - 20:
            print('Early stopping training due to no improvement over the last 20 epochs...')
            break

    del train_data, val_data
    print(f'Best validation F-score: {best_val_fscore:.4f}, best epoch: {best_epoch}.')

    print(f'Training runtime: {str(datetime.now() - start_time).split(".")[0]}')
    print()

    # Saving to disk
    if config.save:
        if not os.path.isdir(config.res_dir):
            os.makedirs(config.res_dir)
        exp_dir = f'jets_{start_time:%Y%m%d_%H%M%S}_0'
        output_dir = os.path.join(config.res_dir, exp_dir)

        i = 0
        while True:
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)  # raises error if dir already exists
                break
            i += 1
            output_dir = output_dir[:-1] + str(i)
            if i > 9:
                print(f'Cannot save results on disk. (tried to save as {output_dir})')
                return

        print(f'Saving all to {output_dir}')
        torch.save(best_model.state_dict(), os.path.join(output_dir, "exp_model.pt"))
        shutil.copyfile(__file__, os.path.join(output_dir, 'code.py'))
        results_dict = {'train_loss': train_loss,
                        'train_ri': train_ri,
                        'val_loss': val_loss,
                        'val_ri': val_ri}
        df = pd.DataFrame(results_dict)
        df.index.name = 'epochs'
        df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
        best_dict = {'best_val_ri': best_val_ri, 'best_epoch': best_epoch}
        best_df = pd.DataFrame(best_dict, index=[0])
        best_df.to_csv(os.path.join(output_dir, "best_val_results.csv"), index=False)
        with open(os.path.join(output_dir, 'used_config.json'), 'w') as fp:
            json.dump(vars(config), fp)

    # print('Loading test data...', end='', flush=True)
    # test_data = jets_loader.get_data_loader('test', config.bs, config.debug_load)
    # test_info = evaluate(test_data, best_model)
    # print(f"\tTest     - {best_epoch:4}",
    #       " loss:{loss:.6f} -- mean_ri:{ri:.4f} -- fscore:{fscore:.4f} -- recall:{recall:.4f} "
    #       "-- precision:{precision:.4f}  -- runtime:{run_time}\n".format(**test_info))

    print(f'Epoch {best_epoch} - evaluating over test set.')
    test_results = eval_jets_on_test_set(best_model)
    print('Test results:')
    print(test_results)
    if config.save:
        test_results.to_csv(os.path.join(output_dir, "test_results.csv"), index=True)

    print(f'Total runtime: {str(datetime.now() - start_time).split(".")[0]}')


if __name__ == '__main__':
    main()
