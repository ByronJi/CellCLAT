import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from mp.models import CIN0, SparseCIN, CINpp

class FeatureMask(nn.Module):
    def __init__(self, feature_out_dim):
        super(FeatureMask, self).__init__()
        self.mask_encoder = nn.Linear(feature_out_dim, feature_out_dim)
        # self.mask_encoder.weight.fill_(1) # init weights of Linear to 1
        init.constant_(self.mask_encoder.weight, 1)

    def forward(self, x):
        return torch.sigmoid(self.mask_encoder(x))


class SampleSelector(nn.Module):
    def __init__(self, input_dim=96):
        super(SampleSelector, self).__init__()
        self.mlp = nn.Linear(input_dim, 2)  # 选择保留或丢弃的 logits

    def gumbel_softmax(self, logits, tau=1.0, hard=True):
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
        y = F.softmax((logits + gumbel_noise) / tau, dim=-1)
        if hard:
            y_hard = torch.zeros_like(y).scatter_(-1, y.argmax(dim=-1, keepdim=True), 1.0)
            y = (y_hard - y).detach() + y  # STE 使梯度可传递
        return y

    def forward(self, x):
        logits = self.mlp(x)
        mask = self.gumbel_softmax(logits, tau=0.5, hard=True)
        binary_mask = mask[:, 1]  # 取出保留的概率
        masked_x = x * binary_mask.unsqueeze(-1)  # 应用 mask
        # # 统计被 mask 掉的样本数
        # num_masked = (binary_mask == 0).sum().item()
        # num_not_masked = (binary_mask == 1).sum().item()
        # print(f"num_masked: {num_masked:d}, num_not_masked: {num_not_masked:d}")
        return masked_x
        # return masked_x, binary_mask


class MaskSimclr(nn.Module):
    def __init__(self, dataset, args):
        super(MaskSimclr, self).__init__()
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
        self.embedding_dim = args.max_dim * args.emb_dim
        self.proj_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                        nn.Linear(self.embedding_dim, self.embedding_dim))
        # self.proj_head = nn.Linear(self.embedding_dim, self.embedding_dim)

    def forward(self, complexBatch):
        x = self.encoder(complexBatch)
        # test projection
        x = self.proj_head(x)
        return x

    def cell_mask_forward(self, complexBatch, cell_mask_model):
        x = self.encoder.cell_mask_forward(data=complexBatch, mask_model=cell_mask_model)
        # projection
        x = self.proj_head(x)
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
