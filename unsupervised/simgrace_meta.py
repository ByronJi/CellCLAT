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
import math

import sys

from torch.autograd import Variable

from data.complex import ComplexBatch
from data.data_loading import DataLoader, load_dataset

from data.datasets import TUDataset

from unsupervised.parser import get_parser
from mp.models import CIN0, SparseCIN, CINpp

from unsupervised.evaluate_embedding import evaluate_embedding
from unsupervised.mask_generator import MaskSimclr, FeatureMask, SampleSelector


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])

class MetaMask(nn.Module):
    def __init__(self, dataset, args, second_order):
        super(MetaMask, self).__init__()
        self.mask_simclr = MaskSimclr(dataset, args)
        self.mask_simclr_ = copy.deepcopy(self.mask_simclr)
        self.vice_model = MaskSimclr(dataset, args)
        # self.cell_mask_model = FeatureMask(96)
        self.cell_mask_model = SampleSelector(input_dim=96)
        self.second_order = second_order
        self.args = args

    def unrolled_backward(self, complex_data, model_optim, mask_optim, eta):
        #  compute unrolled multi-task network theta_1^+ (virtual step)
        # print("compute trial weights")
        data1 = copy.deepcopy(complex_data)
        data2 = copy.deepcopy(complex_data)
        x1 = self.mask_simclr.cell_mask_forward(complexBatch=data1, cell_mask_model=self.cell_mask_model)
        x2 = gen_ran_output(data2, self.mask_simclr, self.vice_model, self.cell_mask_model, self.args)
        loss = MaskSimclr.simclr_loss(x1, x2)

        model_optim.zero_grad()
        mask_optim.zero_grad()
        # calculate a trial step
        loss.backward()
        # copy the gradients
        gradients = copy.deepcopy(
            [v.grad.data if v.grad is not None else None for v in self.mask_simclr.parameters()]
        )

        model_optim.zero_grad()
        mask_optim.zero_grad()

        with torch.no_grad():
            for weight, weight_, d_p in zip(self.mask_simclr.parameters(),
                                            self.mask_simclr_.parameters(),
                                            gradients):
                if d_p is None:
                    weight_.copy_(weight)
                    continue

            d_p = -d_p
            g = model_optim.param_groups[0]
            state = model_optim.state[weight]
            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']
            step_t = state['step']
            step_t += 1

            if g['weight_decay'] != 0:
                d_p = d_p.add(weight, alpha=g['weight_decay'])
            beta1, beta2 = g['betas']
            beta2 = g['betas'][1]
            # Decay the first and second moment running average coefficient
            exp_avg.mul_(g['betas'][0]).add_(d_p, alpha = 1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(d_p, d_p.conj(), value=1 - beta2)

            step = step_t

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            step_size = g['lr'] / bias_correction1

            bias_correction2_sqrt = math.sqrt(bias_correction2)
            denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(g['eps'])

            weight.addcdiv_(exp_avg, denom, value=-step_size)
            weight_ = copy.deepcopy(weight)
            weight_.grad = None

        # print("second")
        data3 = copy.deepcopy(complex_data)
        data4 = copy.deepcopy(complex_data)
        x1 = self.mask_simclr_.cell_mask_forward(complexBatch=data3, cell_mask_model=self.cell_mask_model)
        x2 = gen_ran_output(data4, self.mask_simclr_, self.vice_model, self.cell_mask_model, self.args)
        loss = MaskSimclr.simclr_loss(x1, x2)

        mask_optim.zero_grad()
        loss.backward()

        # dalpha = [v.grad for v in self.cell_mask_model.mask_encoder.parameters()]
        dalpha = [v.grad for v in self.cell_mask_model.mlp.parameters()]

        if self.second_order:
            vector = [v.grad.data if v.grad is not None else None for v in self.mask_simclr_.parameters()]

            implicit_grads = self._hessian_vector_product(vector, complex_data)

            for g, ig in zip(dalpha, implicit_grads):
                g.data.sub_(ig.data, alpha=eta)

        for v, g in zip(self.cell_mask_model.parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    def _hessian_vector_product(self, gradients, complex_data, r=1e-2):
        with torch.no_grad():
            for weight, weight_ in zip(self.mask_simclr.parameters(), self.mask_simclr_.parameters()):
                weight_.copy_(weight)
                weight_.grad = None

        valid_grad = []
        for grad in gradients:
            if grad is not None:
                valid_grad.append(grad)
        R = r / _concat(valid_grad).norm()
        for p, v in zip(self.mask_simclr_.parameters(), gradients):
            if v is not None:
                p.data.add_(v, alpha=R)

        # print("third")
        data1 = copy.deepcopy(complex_data)
        data2 = copy.deepcopy(complex_data)
        x1 = self.mask_simclr_.cell_mask_forward(complexBatch=data1, cell_mask_model=self.cell_mask_model)
        x2 = gen_ran_output(data2, self.mask_simclr_, self.vice_model, self.cell_mask_model, self.args)
        loss = self.mask_simclr.simclr_loss(x1, x2)

        # grads_p = torch.autograd.grad(loss, self.cell_mask_model.mask_encoder.parameters())
        grads_p = torch.autograd.grad(loss, self.cell_mask_model.mlp.parameters())

        for p, v in zip(self.mask_simclr_.parameters(), gradients):
            if v is not None:
                p.data.sub_(v, alpha=2 * R)

        # print("fourth")
        data3 = copy.deepcopy(complex_data)
        data4 = copy.deepcopy(complex_data)
        x1 = self.mask_simclr_.cell_mask_forward(complexBatch=data3, cell_mask_model=self.cell_mask_model)
        x2 = gen_ran_output(data4, self.mask_simclr_, self.vice_model, self.cell_mask_model, self.args)
        loss = self.mask_simclr.simclr_loss(x1, x2)

        # grads_n = torch.autograd.grad(loss, self.cell_mask_model.mask_encoder.parameters())
        grads_n = torch.autograd.grad(loss, self.cell_mask_model.mlp.parameters())

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]

    def per_optimizer_step(self,
                           optimizer_a=None,
                           optimizer_b=None,
                           loss=None):

        # update params
        if loss is not None:
            optimizer_a.zero_grad()
            if optimizer_b is not None:
                optimizer_b.zero_grad()
            loss.backward()

        optimizer_a.step()
        optimizer_a.zero_grad()
        if optimizer_b is not None:
            optimizer_b.step()
            optimizer_b.zero_grad()


# perturb the parameters of the encoder
def gen_ran_output(data, model, vice_model, cell_mask_model, args):
    for (adv_name,adv_param), (name,param) in zip(vice_model.named_parameters(), model.named_parameters()):
        if name.split('.')[0] == 'proj_head':
            adv_param.data = param.data
        else:
            adv_param.data = param.data + args.eta * torch.normal(0,torch.ones_like(param.data)*param.data.std()).to(device)
    z2 = vice_model.cell_mask_forward(complexBatch=data, cell_mask_model=cell_mask_model)
    return z2


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

    args.no_second_order = False
    second_order = not args.no_second_order

    model = MetaMask(dataset, args, second_order).to(device)


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
    optimizer = optim.Adam(model.mask_simclr.parameters(), lr=args.lr)
    mask_optim = optim.SGD(model.cell_mask_model.parameters(), lr=0.01)
    mask_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(mask_optim, args.epochs)

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
            batch = batch.to(device)
            data1 = copy.deepcopy(batch).to(device)
            data2 = copy.deepcopy(batch).to(device)


            x1 = model.mask_simclr.cell_mask_forward(complexBatch=data1, cell_mask_model=model.cell_mask_model)
            x2 = gen_ran_output(data2, model.mask_simclr, model.vice_model, model.cell_mask_model, model.args)
            loss = model.mask_simclr.simclr_loss(x1, x2)
            model.per_optimizer_step(optimizer_a=optimizer, optimizer_b=None, loss=loss)
            # print("unrolled_backward phase")

            model.unrolled_backward(complex_data=batch, model_optim=optimizer, mask_optim=mask_optim, eta=optimizer.param_groups[0]['lr'])
            model.per_optimizer_step(mask_optim)

            loss_all += loss.item()
            mask_scheduler.step()

        print('Epoch {}, Loss {}'.format(epoch, loss_all / len(dataloader)))

        if epoch % 5 == 0:
            model.eval()
            emb, y = model.mask_simclr.encoder.get_mask_embeddings(dataloader_eval, device, model.cell_mask_model)
            # np.save(str(epoch)+"-cwn-ssl-gumbel_emb.npy", emb)
            # np.save(str(epoch)+"-cwn-ssl-gumbel_y.npy", y)
            acc_val, acc = evaluate_embedding(emb, y)
            print(f"Epoch {epoch}: val_acc={acc_val}  test_acc={acc}")
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
