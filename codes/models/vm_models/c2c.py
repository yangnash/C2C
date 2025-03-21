import torch
import torch.nn as nn
import torch.nn.functional as F
from .word_embedding import load_word_embeddings
from itertools import product
from models.vm_models.get_extractor import get_video_extractor
import numpy as np


class MLP_ST(nn.Module):
    '''
    Baseclass to create a simple MLP
    Inputs
        inp_dim: Int, Input dimension
        out-dim: Int, Output dimension
        num_layer: Number of hidden layers
        relu: Bool, Use non linear function at output
        bias: Bool, Use bias
    '''

    def __init__(self, inp_dim, out_dim, num_layers=1, relu=True, bias=True, dropout=False, norm=False, layers=[]):
        super(MLP_ST, self).__init__()
        mod = []
        incoming = inp_dim
        for layer_ind in range(num_layers - 1):
            if len(layers) == 0:
                outgoing = incoming
            else:
                outgoing = layers[layer_ind]
            # mod.append(nn.Linear(incoming, outgoing, bias=bias))
            mod.append(nn.Conv1d(incoming, outgoing, kernel_size=3, bias=bias, padding=1))

            incoming = outgoing
            if norm:
                mod.append(nn.LayerNorm(outgoing))
                # mod.append(nn.BatchNorm1d(outgoing))
            mod.append(nn.ReLU(inplace=True))
            # mod.append(nn.LeakyReLU(inplace=True, negative_slope=0.2))
            if dropout:
                mod.append(nn.Dropout(p=0.5))

        # mod.append(nn.Linear(incoming, out_dim, bias=bias))
        mod.append(nn.Conv1d(incoming, out_dim, kernel_size=3, bias=bias, padding=1))

        if relu:
            mod.append(nn.ReLU(inplace=True))
            # mod.append(nn.LeakyReLU(inplace=True, negative_slope=0.2))
        self.mod = nn.Sequential(*mod)

    def forward(self, x):
        for o in self.mod:
            if isinstance(o, nn.LayerNorm):
                x = x.transpose(1, 2)
                x = o(x)
                x = x.transpose(1, 2)
            else:
                x = o(x)
        return x


class MLP(nn.Module):
    '''
    Baseclass to create a simple MLP
    Inputs
        inp_dim: Int, Input dimension
        out-dim: Int, Output dimension
        num_layer: Number of hidden layers
        relu: Bool, Use non linear function at output
        bias: Bool, Use bias
    '''

    def __init__(self, inp_dim, out_dim, num_layers=1, relu=True, bias=True, dropout=False, norm=False, layers=[]):
        super(MLP, self).__init__()
        mod = []
        incoming = inp_dim
        for layer_ind in range(num_layers - 1):
            if len(layers) == 0:
                outgoing = incoming
            else:
                outgoing = layers[layer_ind]
            mod.append(nn.Linear(incoming, outgoing, bias=bias))

            incoming = outgoing
            if norm:
                mod.append(nn.LayerNorm(outgoing))
                # mod.append(nn.BatchNorm1d(outgoing))
            mod.append(nn.ReLU(inplace=True))
            # mod.append(nn.LeakyReLU(inplace=True, negative_slope=0.2))
            if dropout:
                mod.append(nn.Dropout(p=0.5))

        mod.append(nn.Linear(incoming, out_dim, bias=bias))

        if relu:
            mod.append(nn.ReLU(inplace=True))
            # mod.append(nn.LeakyReLU(inplace=True, negative_slope=0.2))
        self.mod = nn.Sequential(*mod)

    def forward(self, x):
        return self.mod(x)


class C2C(nn.Module):

    def __init__(self, dset, cfg):
        super(C2C, self).__init__()
        self.video_encoder = get_video_extractor(cfg)
        self.cfg = cfg
        self.dset = dset

        def get_all_ids(relevant_pairs):
            # Precompute validation pairs
            attrs, objs = zip(*relevant_pairs)
            attrs = [dset.attr2idx[attr] for attr in attrs]
            objs = [dset.obj2idx[obj] for obj in objs]
            pairs = [a for a in range(len(relevant_pairs))]
            attrs = torch.Tensor(attrs)
            objs = torch.Tensor(objs)
            pairs = torch.Tensor(pairs)
            return attrs, objs, pairs

        # Validation
        val_attrs, val_objs, val_pairs = get_all_ids(self.dset.pairs)
        self.register_buffer('val_attrs', val_attrs)
        self.register_buffer('val_objs', val_objs)
        self.register_buffer('val_pairs', val_pairs)

        # for indivual projections
        uniq_attrs, uniq_objs = torch.arange(len(self.dset.attrs)), \
                                torch.arange(len(self.dset.objs))
        self.register_buffer('uniq_attrs', uniq_attrs)
        self.register_buffer('uniq_objs', uniq_objs)
        self.factor = 2

        self.scale = cfg.cosine_scale

        self.train_forward = self.train_forward_closed

        # Precompute training compositions
        if cfg.train_only:
            train_attrs, train_objs, train_pairs = get_all_ids(self.dset.train_pairs)
        else:
            train_attrs, train_objs, train_pairs = val_attrs, val_objs, val_pairs

        self.register_buffer('train_attrs', train_attrs)
        self.register_buffer('train_objs', train_objs)
        self.register_buffer('train_pairs', train_pairs)

        try:
            self.fc_emb = cfg.fc_emb.split(',')

        except:
            self.fc_emb = [cfg.fc_emb]
        layers = []
        for a in self.fc_emb:
            a = int(a)
            layers.append(a)

        input_dim = cfg.emb_dim
        self.attr_embedder = nn.Embedding(len(dset.attrs), input_dim)
        self.obj_embedder = nn.Embedding(len(dset.objs), input_dim)

        # init with word embeddings
        if cfg.emb_init:
            pretrained_weight = load_word_embeddings(cfg.emb_init, dset.attrs)
            self.attr_embedder.weight.data.copy_(pretrained_weight)
            pretrained_weight = load_word_embeddings(cfg.emb_init, dset.objs)
            self.obj_embedder.weight.data.copy_(pretrained_weight)

        # static inputs
        if cfg.static_inp:
            for param in self.attr_embedder.parameters():
                param.requires_grad = False
            for param in self.obj_embedder.parameters():
                param.requires_grad = False

        # Composition MLP
        self.o_projection1 = nn.Linear(input_dim, cfg.emb_dim)
        self.v_projection1 = nn.Linear(input_dim, cfg.emb_dim)

        self.OE1 = MLP(cfg.feat_dim, int(cfg.emb_dim), relu=cfg.relu, num_layers=cfg.nlayers,
                       dropout=False,
                       norm=self.cfg.norm, layers=layers)

       

        self.VE1 = MLP_ST(cfg.feat_dim, int(cfg.emb_dim), relu=cfg.relu, num_layers=cfg.nlayers,
                          dropout=False,
                          norm=self.cfg.norm, layers=layers)



    def freeze_representations(self):
        print('Freezing representations')
        for param in self.video_embedder.parameters():
            param.requires_grad = False
        for param in self.attr_embedder.parameters():
            param.requires_grad = False
        for param in self.obj_embedder.parameters():
            param.requires_grad = False

   

    def val_forward_closed(self, x, pairs, visual=False):
        vid_feat = self.video_encoder(x)  # b,l,t
        #vid_feat = self.video_encoder(x.permute(0,2,1,3,4)).squeeze(-1).squeeze(-1)
        if len(vid_feat.shape) == 5:
            vid_feat = vid_feat.mean(-1).mean(-1)
        o_feat = self.OE1(vid_feat.mean(dim=-1))
        o_feat_normed = F.normalize(o_feat, dim=1)
        v_feat = self.VE1(vid_feat)
        v_feat = v_feat.mean(dim=-1)  # b,c
        v_feat_normed = F.normalize(v_feat, dim=1)

        all_verbs, all_objs = self.attr_embedder(self.uniq_attrs), self.obj_embedder(self.uniq_objs)

        v_emb = self.v_projection1(all_verbs)
        v_emb_normed = F.normalize(v_emb, dim=1)
        o_emb = self.o_projection1(all_objs)  # n,c
        o_emb_normed = F.normalize(o_emb, dim=1)

        p_v = torch.matmul(v_feat_normed, v_emb_normed.permute(1, 0)) * 0.5 + 0.5  # b,nv
        p_o = torch.matmul(o_feat_normed, o_emb_normed.permute(1, 0)) * 0.5 + 0.5  # b,no

        verb_ids, obj_ids = pairs[:, 0], pairs[:, 1]
        f=torch.einsum('ij,ik->ijk', p_v, p_o)
       
        pair_pred=f[:, verb_ids, obj_ids]

        return pair_pred

    def train_forward_closed(self, x):
        vid_feat = self.video_encoder(x)  # b,l,t
        #vid_feat = self.video_encoder(x.permute(0,2,1,3,4)).squeeze(-1).squeeze(-1)
        # if len(vid_feat.shape) == 5:
        #     vid_feat = vid_feat.mean(-1).mean(-1)
        # independent learning
        o_feat = self.OE1(vid_feat.mean(dim=-1))  # b,c
        o_feat_normed = F.normalize(o_feat, dim=1)
        v_feat_t = self.VE1(vid_feat)  # b,c,t
        v_feat = v_feat_t.mean(dim=-1)  # b,c
        v_feat_normed = F.normalize(v_feat, dim=1)

        all_verbs, all_objs = self.attr_embedder(self.uniq_attrs), self.obj_embedder(self.uniq_objs)

        v_emb = self.v_projection1(all_verbs)
        v_emb_normed = F.normalize(v_emb, dim=1)
        o_emb = self.o_projection1(all_objs)  # n,c
        o_emb_normed = F.normalize(o_emb, dim=1)

        p_v = torch.matmul(v_feat_normed, v_emb_normed.permute(1, 0)) * 0.5 + 0.5  # b,nv
        p_o = torch.matmul(o_feat_normed, o_emb_normed.permute(1, 0)) * 0.5 + 0.5  # b,no
        f=torch.einsum('ij,ik->ijk', p_v, p_o)
        return f
      

    def forward(self, x, pair=None):
        if self.training:
            pred = self.train_forward_closed(x)
        else:
            pred = self.val_forward_closed(x, pair)
        return pred
