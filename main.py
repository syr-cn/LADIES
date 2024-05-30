#!/usr/bin/env python
# coding: utf-8


from utils import *
import argparse
import scipy
import multiprocessing as mp
from model import *

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


parser = argparse.ArgumentParser(description='Training GCN on Cora/CiteSeer/PubMed/Reddit Datasets')

'''
    Dataset arguments
'''
parser.add_argument('--save_dir', type=str, default='results/debug')
parser.add_argument('--gnn_type', type=str, default='gcn')
parser.add_argument('--dataset', type=str, default='cora', help='Dataset name: Cora/CiteSeer/PubMed/Reddit')
parser.add_argument('--nhid', type=int, default=256, help='Hidden state dimension')
parser.add_argument('--epoch_num', type=int, default= 100, help='Number of Epoch')
parser.add_argument('--pool_num', type=int, default= 10, help='Number of Pool')
parser.add_argument('--batch_num', type=int, default= 10, help='Maximum Batch Number')
parser.add_argument('--batch_size', type=int, default=512, help='size of output node in a batch')
parser.add_argument('--n_layers', type=int, default=5, help='Number of GCN layers')
parser.add_argument('--num_seeds', type=int, default=5, help='Number of seeds to run')
parser.add_argument('--n_iters', type=int, default=1, help='Number of iteration to run on a batch')
parser.add_argument('--n_stops', type=int, default=200, help='Stop after number of batches that f1 dont increase')
parser.add_argument('--samp_num', type=int, default=64, help='Number of sampled nodes per layer')
parser.add_argument('--sample_method', type=str, default='subgraph', help='Sampled Algorithms: subgraph/fastgcn/full')
parser.add_argument('--cuda', type=int, default=0, help='Avaiable GPU ID')


args = parser.parse_args()

def fastgcn_sampler(batch_nodes, samp_num_list, num_nodes, lap_matrix, depth):
    '''
        Sample a fixed number of nodes per layer. The sampling probability (importance)
                         is pre-computed based on the global degree (lap_matrix)
    '''
    previous_nodes = batch_nodes
    adjs  = []
    #     pre-compute the sampling probability (importance) based on the global degree (lap_matrix)
    pi = np.array(np.sum(lap_matrix.multiply(lap_matrix), axis=0))[0]
    p = pi / np.sum(pi)
    '''
        Sample nodes from top to bottom, based on the pre-computed probability. Then reconstruct the adjacency matrix.
    '''
    for d in range(depth):
        #     row-select the lap_matrix (U) by previously sampled nodes
        U = lap_matrix[previous_nodes , :]
        #     sample the next layer's nodes based on the pre-computed probability (p).
        s_num = np.min([np.sum(p > 0), samp_num_list[d]])
        after_nodes = np.random.choice(num_nodes, s_num, p = p, replace = False)
        #     col-select the lap_matrix (U), and then devided by the sampled probability for 
        #     unbiased-sampling. Finally, conduct row-normalization to avoid value explosion.         
        adj = row_normalize(U[: , after_nodes].multiply(1/p[after_nodes]))
        #     Turn the sampled adjacency matrix into a sparse matrix. If implemented by PyG
        #     This sparse matrix can also provide index and value.
        adjs += [sparse_mx_to_torch_sparse_tensor(row_normalize(adj))]
        #     Turn the sampled nodes as previous_nodes, recursively conduct sampling.
        previous_nodes = after_nodes
    #     Reverse the sampled probability from bottom to top. Only require input how the lastly sampled nodes.
    adjs.reverse()
    return adjs, previous_nodes, batch_nodes

def subgraph_sampler(batch_nodes, samp_num_list, num_nodes, lap_matrix, depth):
    '''
        Sample a fixed number of nodes per layer. The sampling probability (importance)
                         is computed adaptively according to the nodes sampled in the upper layer.
    '''
    previous_nodes = batch_nodes
    adjs  = []
    '''
        Sample nodes from top to bottom, based on the probability computed adaptively (layer-dependent).
    '''
    for d in range(depth):
        #     row-select the lap_matrix (U) by previously sampled nodes
        U = lap_matrix[previous_nodes , :]
        #     Only use the upper layer's neighborhood to calculate the probability.
        pi = np.array(np.sum(U.multiply(U), axis=0))[0]
        p = pi / np.sum(pi)
        s_num = np.min([np.sum(p > 0), samp_num_list[d]])
        #     sample the next layer's nodes based on the adaptively probability (p).
        after_nodes = np.random.choice(num_nodes, s_num, p = p, replace = False)
        #     Add output nodes for self-loop
        after_nodes = np.unique(np.concatenate((after_nodes, batch_nodes)))
        #     col-select the lap_matrix (U), and then devided by the sampled probability for 
        #     unbiased-sampling. Finally, conduct row-normalization to avoid value explosion.      
        adj = U[: , after_nodes].multiply(1/p[after_nodes])
        adjs += [sparse_mx_to_torch_sparse_tensor(row_normalize(adj))]
        #     Turn the sampled nodes as previous_nodes, recursively conduct sampling.
        previous_nodes = after_nodes
    #     Reverse the sampled probability from bottom to top. Only require input how the lastly sampled nodes.
    adjs.reverse()
    return adjs, previous_nodes, batch_nodes
def default_sampler(batch_nodes, samp_num_list, num_nodes, lap_matrix, depth):
    mx = sparse_mx_to_torch_sparse_tensor(lap_matrix)
    return [mx for i in range(depth)], np.arange(num_nodes), batch_nodes
def prepare_data(pool, sampler, process_ids, train_nodes, valid_nodes, samp_num_list, num_nodes, lap_matrix, depth):
    jobs = []
    for _ in process_ids:
        idx = torch.randperm(len(train_nodes))[:args.batch_size]
        batch_nodes = train_nodes[idx]
        p = pool.apply_async(sampler, args=(batch_nodes, samp_num_list, num_nodes, lap_matrix, depth))
        jobs.append(p)
    idx = torch.randperm(len(valid_nodes))[:args.batch_size]
    batch_nodes = valid_nodes[idx]
    p = pool.apply_async(sampler, args=(batch_nodes, samp_num_list * 20, num_nodes, lap_matrix, depth))
    jobs.append(p)
    return jobs
def prepare_data_nonpool(sampler, process_ids, train_nodes, valid_nodes, samp_num_list, num_nodes, lap_matrix, depth):
    samples = []
    for _ in process_ids:
        idx = torch.randperm(len(train_nodes))[:args.batch_size]
        batch_nodes = train_nodes[idx]
        sample = sampler(batch_nodes, samp_num_list, num_nodes, lap_matrix, depth)
        samples.append(sample)
    idx = torch.randperm(len(valid_nodes))[:args.batch_size]
    batch_nodes = valid_nodes[idx]
    sample = sampler(batch_nodes, samp_num_list * 20, num_nodes, lap_matrix, depth)
    samples.append(sample)
    return samples

def package_mxl(mxl, device):
    return [torch.sparse.FloatTensor(mx[0], mx[1], mx[2]).to(device) for mx in mxl]



if args.cuda != -1:
    device = torch.device("cuda:" + str(args.cuda))
else:
    device = torch.device("cpu")
    
    
print(args.dataset, args.sample_method)
edges, labels, feat_data, num_classes, train_nodes, valid_nodes, test_nodes = load_data(args.dataset)

adj_matrix = get_adj(edges, feat_data.shape[0])

lap_matrix = row_normalize(adj_matrix + sp.eye(adj_matrix.shape[0]))
if type(feat_data) == scipy.sparse.lil_matrix:
    feat_data = torch.FloatTensor(feat_data.todense()).to(device) 
else:
    feat_data = torch.FloatTensor(feat_data).to(device)
labels    = torch.LongTensor(labels).to(device) 



if args.sample_method == 'subgraph':
    sampler = subgraph_sampler
elif args.sample_method == 'fastgcn':
    sampler = fastgcn_sampler
elif args.sample_method == 'full':
    sampler = default_sampler

process_ids = np.arange(args.batch_num)
samp_num_list = np.array([args.samp_num, args.samp_num, args.samp_num, args.samp_num, args.samp_num])

all_times = []
all_test_f1s = []
for oiter in range(args.num_seeds):
    seed = np.random.randint(2**32 - 1)
    print(f'Running with seed {seed}.')
    set_random_seed(seed)
    encoder = GNN(gnn_type = args.gnn_type, nfeat = feat_data.shape[1], nhid=args.nhid, layers=args.n_layers, dropout = 0.2).to(device)
    model  = GNNCls(encoder = encoder, num_classes=num_classes, dropout=0.5, inp = feat_data.shape[1])
    model.to(device)

    optimizer = optim.Adam(filter(lambda p : p.requires_grad, model.parameters()))
    best_val = 0
    best_tst = -1
    cnt = 0
    times = []
    res   = []
    print('-' * 10)
    for epoch in np.arange(args.epoch_num):
        model.train()
        train_losses = []
        samples = prepare_data_nonpool(sampler, process_ids, train_nodes, valid_nodes, samp_num_list, len(feat_data), lap_matrix, args.n_layers)
        train_data = samples[:-1]
        valid_data = samples[-1]
        for _iter in range(args.n_iters):
            for adjs, input_nodes, output_nodes in train_data:    
                adjs = package_mxl(adjs, device)
                optimizer.zero_grad()
                model.train()
                t1 = time.time()
                output = model.forward(feat_data[input_nodes], adjs)
                if args.sample_method == 'full':
                    output = output[output_nodes]
                loss_train = F.cross_entropy(output, labels[output_nodes])
                loss_train.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.2)
                optimizer.step()
                times += [time.time() - t1]
                train_losses += [loss_train.detach().tolist()]
                del loss_train
        model.eval()
        with torch.no_grad():
            adjs, input_nodes, output_nodes = valid_data
            adjs = package_mxl(adjs, device)
            output = model.forward(feat_data[input_nodes], adjs)
            if args.sample_method == 'full':
                output = output[output_nodes]
            loss_valid = F.cross_entropy(output, labels[output_nodes]).detach().tolist()
            valid_f1 = f1_score(output.argmax(dim=1).cpu(), labels[output_nodes].cpu(), average='micro')
        print(("Epoch: %d (%.1fs) Train Loss: %.2f    Valid Loss: %.2f Valid F1: %.3f") % (epoch, np.sum(times), np.average(train_losses), loss_valid, valid_f1), flush=True)
        if valid_f1 > best_val + 1e-2:
            best_val = valid_f1
            torch.save(model, os.path.join(args.save_dir, 'best_model.pt'))
            cnt = 0
        else:
            cnt += 1
        if cnt == args.n_stops // args.batch_num:
            break
    best_model = torch.load(os.path.join(args.save_dir, 'best_model.pt'))
    best_model.eval()
    test_f1s = []

    batch_nodes = test_nodes
    adjs, input_nodes, output_nodes = default_sampler(batch_nodes, samp_num_list * 20, len(feat_data), lap_matrix, args.n_layers)
    adjs = package_mxl(adjs, device)
    output = best_model.forward(feat_data[input_nodes], adjs)[output_nodes]
    test_f1s = [f1_score(output.argmax(dim=1).cpu(), labels[output_nodes].cpu(), average='micro')]
    
    print(f'Seed {seed} Iteration: {oiter:d}, Test F1: {np.average(test_f1s):.3f}', flush=True)
    all_test_f1s.append(np.average(test_f1s))
    all_times.append(np.sum(times)/(epoch+1))

avg, std = np.average(all_test_f1s), np.std(all_test_f1s)
print('-' * 100)
print(f'F1 Score: {avg*100:.1f}±{std*100:.1f}')
print(f'F1 Score Average: {avg:.5f}')
print(f'F1 Score Std: {std:.5f}')
time_avg, time_std = np.average(all_times), np.std(all_times)
print('')
print(f'Time/Epoch: {time_avg:.2f}±{time_std:.2f}')
print(f'Time/Epoch Average: {time_avg:.5f}')
print(f'Time/Epoch Std: {time_std:.5f}')
batch_time_avg, batch_time_std = np.average(times), np.std(times)
print('')
print(f'Time/Batch: {batch_time_avg*1000:.2f}±{batch_time_std*1000:.2f}')
print(f'Time/Batch Average: {batch_time_avg*1000:.5f}')
print(f'Time/Batch Std: {batch_time_std*1000:.5f}')