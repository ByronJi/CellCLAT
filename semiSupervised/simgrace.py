import os
import numpy as np
import copy
import pickle
import torch
import torch.optim as optim
import torch.nn as nn
import random
import logging
import copy

import sys



from data.complex import ComplexBatch
from data.data_loading import DataLoader, load_dataset
from torch_geometric.data import DataLoader as PyGDataLoader

from data.datasets import TUDataset

from unsupervised.parser import get_parser
from mp.models import CIN0, SparseCIN, CINpp

from unsupervised.evaluate_embedding import evaluate_embedding


class simclr(nn.Module):
    def __init__(self, dataset, args):
        super(simclr, self).__init__()
        use_coboundaries = args.use_coboundaries.lower() == 'true'
        readout_dims = tuple(sorted(args.readout_dims))
        if args.model == 'cin':
            self.encoder = CIN0(dataset.num_features_in_dim(0),  # num_input_features
                         dataset.num_classes,  # num_classes
                         args.num_layers,  # num_layers
                         args.emb_dim,  # hidden
                         dropout_rate=args.drop_rate,  # dropout rate
                         max_dim=dataset.max_dim,  # max_dim
                         jump_mode=args.jump_mode,  # jump mode
                         nonlinearity=args.nonlinearity,  # nonlinearity
                         readout=args.readout,  # readout
                         )
        elif args.model == 'sparse_cin':
            self.encoder = SparseCIN(dataset.num_features_in_dim(0),  # num_input_features
                              dataset.num_classes,  # num_classes
                              args.num_layers,  # num_layers
                              args.emb_dim,  # hidden
                              dropout_rate=args.drop_rate,  # dropout rate
                              max_dim=dataset.max_dim,  # max_dim
                              jump_mode=args.jump_mode,  # jump mode
                              nonlinearity=args.nonlinearity,  # nonlinearity
                              readout=args.readout,  # readout
                              final_readout=args.final_readout,  # final readout
                              apply_dropout_before=args.drop_position,  # where to apply dropout
                              use_coboundaries=use_coboundaries,  # whether to use coboundaries in up-msg
                              graph_norm=args.graph_norm,  # normalization layer
                              readout_dims=readout_dims  # readout_dims (0,1,2)
                              )
        elif args.model == 'cin++':
            self.encoder = CINpp(dataset.num_features_in_dim(0),  # num_input_features
                          dataset.num_classes,  # num_classes
                          args.num_layers,  # num_layers
                          args.emb_dim,  # hidden
                          dropout_rate=args.drop_rate,  # dropout rate
                          max_dim=dataset.max_dim,  # max_dim
                          jump_mode=args.jump_mode,  # jump mode
                          nonlinearity=args.nonlinearity,  # nonlinearity
                          readout=args.readout,  # readout
                          final_readout=args.final_readout,  # final readout
                          apply_dropout_before=args.drop_position,  # where to apply dropout
                          use_coboundaries=use_coboundaries,  # whether to use coboundaries in up-msg
                          graph_norm=args.graph_norm,  # normalization layer
                          readout_dims=readout_dims  # readout_dims (0,1,2)
                          )
        # test projection head
        # self.embedding_dim = args.max_dim * args.emb_dim
        # self.proj_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
        #                                nn.Linear(self.embedding_dim, self.embedding_dim))

    def forward(self, complexBatch):
        x = self.encoder(complexBatch)
        # test projection
        # x = self.proj_head(x)
        return x


    @staticmethod
    def simclr_loss(x, x_aug):
        T = 0.2
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss

# perturb the parameters of the encoder
def gen_ran_output(data, model, vice_model, args):
    for (adv_name,adv_param), (name,param) in zip(vice_model.named_parameters(), model.named_parameters()):
        if name.split('.')[0] == 'proj_head':
            adv_param.data = param.data
        else:
            adv_param.data = param.data + args.eta * torch.normal(0,torch.ones_like(param.data)*param.data.std()).to(device)
    z2 = vice_model(data)
    return z2

def get_embeddings(model, loader, device):
    ret = []
    y = []
    with torch.no_grad():
        for data in loader:
            data.to(device)

            x = model.forward(data)

            ret.append(x.cpu().numpy())
            y.append(data.y.cpu().numpy())
    ret = np.concatenate(ret, 0)
    y = np.concatenate(y, 0)
    return ret, y

if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()
    """The common training and evaluation script used by all the experiments."""
    # set device
    device = torch.device(
        "cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")


    print("==========================================================")
    print("Using device", str(device))
    print(f"Seed: {args.seed}")
    print("======================== Args ===========================")
    print(args)
    print("===================================================")

    # Set the seed for everything
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Create results folder
    result_folder = os.path.join(
        args.result_folder, f'{args.dataset}-{args.exp_name}', f'seed-{args.seed}')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    filename = os.path.join(result_folder, 'results.txt')

    # Data loading
    dataset = load_dataset(args.dataset, max_dim=args.max_dim,
                           init_method=args.init_method, emb_dim=args.emb_dim,
                           flow_points=args.flow_points, flow_classes=args.flow_classes,
                           max_ring_size=args.max_ring_size,
                           use_edge_features=args.use_edge_features,
                           include_down_adj=args.include_down_adj,
                           simple_features=args.simple_features, n_jobs=args.preproc_jobs,
                           train_orient=args.train_orient, test_orient=args.test_orient)

    dataset_eval = load_dataset(args.dataset, max_dim=args.max_dim,
                           init_method=args.init_method, emb_dim=args.emb_dim,
                           flow_points=args.flow_points, flow_classes=args.flow_classes,
                           max_ring_size=args.max_ring_size,
                           use_edge_features=args.use_edge_features,
                           include_down_adj=args.include_down_adj,
                           simple_features=args.simple_features, n_jobs=args.preproc_jobs,
                           train_orient=args.train_orient, test_orient=args.test_orient)

    # Instantiate data loaders
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, max_dim=dataset.max_dim)
    dataloader_eval = DataLoader(dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, max_dim=dataset.max_dim)

    # Use coboundaries?
    use_coboundaries = args.use_coboundaries.lower() == 'true'

    # Readout dimensions
    readout_dims = tuple(sorted(args.readout_dims))

    model = simclr(dataset, args).to(device)
    vice_model = simclr(dataset, args).to(device)

    print("============= Model Parameters =================")
    trainable_params = 0
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.size())
            trainable_params += param.numel()
        total_params += param.numel()
    print("============= Params stats ==================")
    print(f"Trainable params: {trainable_params}")
    print(f"Total params    : {total_params}")

    # instantiate optimiser
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # (!) start training/evaluation
    best_val_epoch = 0
    valid_curve = []
    test_curve = []
    train_curve = []
    train_loss_curve = []
    params = []
    accuracies = {'val': [], 'test': []}

    print('Training...')

    for epoch in range(1, args.epochs + 1):

        # perform one epoch
        print("=====Epoch {}".format(epoch))


        loss_all = 0

        model.train()
        num_skips = 0
        for batch in dataloader:
            optimizer.zero_grad()
            batch1 = batch.to(device)
            batch2 = copy.deepcopy(batch).to(device)

            x1 = model(batch1)
            x2 = gen_ran_output(batch2, model, vice_model, args)
            loss_aug = simclr.simclr_loss(x1, x2)
            loss = loss_aug
            loss_all += loss.item()
            loss.backward()
            optimizer.step()
        print('Epoch {}, Loss {}'.format(epoch, loss_all / len(dataloader)))

        if epoch % 5 == 0:
            model.eval()
            emb, y = get_embeddings(model.encoder, dataloader_eval, device)


            acc_val, acc = evaluate_embedding(emb, y)
            accuracies['val'].append(acc_val)
            accuracies['test'].append(acc)
            logging.info('\t%.4f' % acc)
    logging.info('Final:')
    logging.info('\t%.4f' % (accuracies['test'][-1]))

    msg = (
        f'========Params========\n'
        f'args: {args}\n'
        f'========Result=======\n'
        f'Dataset: {args.dataset}\n'
        f"accuracies[val]: {accuracies['val']}\n"
        f"accuracies[test]: {accuracies['test']}\n"
        f"Test max:\n{max(accuracies['test'])}\n"
    )
    print(msg)

    with open(filename, 'a') as f:
        f.write(msg)


