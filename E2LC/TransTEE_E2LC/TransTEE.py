import torch
import torch.nn as nn
import torch.distributions as dist
from utils.transformers import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder
from utils.utils import get_initialiser
from utils.mlp import MLP
from utils.trans_ci import TransformerModel, Embeddings
import numpy as np
# replace the feature extractor of x by self-attention
# 0.015


class Linear(nn.Module):
    def __init__(self, ind, outd, act='relu', isbias=1):
        super(Linear, self).__init__()
        # ind does NOT include the extra concat treatment
        self.ind = ind
        self.outd = outd
        self.isbias = isbias

        self.weight = nn.Parameter(torch.rand(self.ind, self.outd), requires_grad=True)

        if self.isbias:
            self.bias = nn.Parameter(torch.rand(self.outd), requires_grad=True)
        else:
            self.bias = None

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            self.act = None

    def forward(self, x):
        # x: batch_size * (treatment, other feature)

        out = torch.matmul(x, self.weight)

        if self.isbias:
            out = out + self.bias

        if self.act is not None:
            out = self.act(out)

        return out


class auxiliary_model(nn.Module):
    def __init__(self, params, ts, num_heads=2, att_layers=1, dropout=0.0, \
                 init_range_f=0.1, init_range_t=0.1):
        super(auxiliary_model, self).__init__()
        
        num_features = params['num_features'] + params['t_grid']
        num_treatments = params['num_treatments']
        self.dx = params['num_features']

        h_dim = params['h_dim']
        self.t_grid = params['t_grid']
        self.s = params['s']
        self.ts = ts
        #self.h_inv_eqv_dim = params['h_inv_eqv_dim']
        self.batch_size = params['batch_size']
        #self.alpha = params['alpha']
        #self.num_dosage_samples = params['num_dosage_samples']
        
        self.cov_dim = int(params['cov_dim'] * params['dz'])
        self.linear1 = nn.Linear(num_features, self.cov_dim)

        self.feature_weight = Embeddings(h_dim, initrange=init_range_f)
        self.treat_emb = Embeddings(h_dim, act='id', initrange=init_range_t)
        self.dosage_emb = Embeddings(h_dim, act='id', initrange=init_range_t)
        self.linear2 = MLP(
            dim_input=h_dim * 2,
            dim_hidden=h_dim,
            dim_output=h_dim,
        )

        encoder_layers = TransformerEncoderLayer(h_dim, nhead=num_heads, dim_feedforward=h_dim, dropout=dropout, num_cov=self.cov_dim)
        self.encoder = TransformerEncoder(encoder_layers, att_layers)

        decoder_layers = TransformerDecoderLayer(h_dim, nhead=num_heads, dim_feedforward=h_dim, dropout=dropout,num_t=1)
        self.decoder = TransformerDecoder(decoder_layers, att_layers)

        self.Q = MLP(
            dim_input=h_dim,
            dim_hidden=h_dim,
            dim_output=1,
            is_output_activation=False,
        )


    def get_loss(self, xy, y_f, t_f, d_f, s):
        # y_cf: i: s, j: bs
        # example y_cf: torch.tensor([[1.1,1.2],[2.1,2.2],[3.1,3.2],[1.3,1.4],
        # [2.3,2.4],[3.3,3.4],[1.5,1.6],[2.5,2.6],[3.5,3.6],[1.7,1.8],
        # [2.7,2.8],[3.7,3.8]]), bs = 3, s = 4, t_grid = 2
        bs = int(xy.shape[0] / s)
        hidden = self.feature_weight(self.linear1(xy))
        memory = self.encoder(hidden)
        memory_d2 = memory.shape[1]
        memory_d3 = memory.shape[2]
        memory1 = memory.reshape(bs*s, memory_d2*memory_d3).repeat(1, self.t_grid).\
                 reshape(bs*s*self.t_grid, memory_d2, memory_d3)
        ds = self.ts.repeat(bs*s, 1)
        t = torch.ones_like(ds).cuda().detach().float()
        tgt = torch.cat([self.treat_emb(t), self.dosage_emb(ds)], dim=-1)
        tgt = self.linear2(tgt)
        if len(tgt.shape) < 3:
            tgt = tgt.unsqueeze(1)
        out = self.decoder(tgt.permute(1, 0, 2), memory1.permute(1, 0, 2))
        if out.shape[0] != 1:
            out = torch.mean(out, dim=1)
        pre_y_cf = self.Q(out.squeeze(0)).reshape(bs*s, self.t_grid)
        y_cf = xy[:,self.dx:]
        loss_y = torch.mean((y_cf - pre_y_cf)**2)

        tgt = torch.cat([self.treat_emb(t_f), self.dosage_emb(d_f)], dim=-1)
        tgt = self.linear2(tgt)
        if len(tgt.shape) < 3:
            tgt = tgt.unsqueeze(1)
        out = self.decoder(tgt.permute(1, 0, 2), memory.permute(1, 0, 2))
        if out.shape[0] != 1:
            out = torch.mean(out, dim=1)
        pre_y_f = self.Q(out.squeeze(0))
        loss_yf = torch.mean((pre_y_f - y_f)**2)
        
        return  0.01 * loss_y + loss_yf






# Repalce dynamic-Q by feature embeddings, it works well
class main_model(nn.Module):
    def __init__(self, params, num_heads=2, att_layers=1, dropout=0.0, \
                 init_range_f=0.1, init_range_t=0.1):
        super(main_model, self).__init__()
        """
        cfg_density: cfg for the density estimator; [(ind1, outd1, isbias1), 'act', ....]; the cfg for density estimator head is not included
        num_grid: how many grid used for the density estimator head
        """

        # cfg/cfg_density = [(ind1, outd1, isbias1, activation),....]
        num_features = params['num_features']
        num_treatments = params['num_treatments']
        self.dx = num_features

        h_dim = params['h_dim']
        self.t_grid = params['t_grid']
        self.s = params['s']
        self.ts = torch.linspace(0, 1, self.t_grid).view(self.t_grid, 1).cuda().detach().float()
        #self.h_inv_eqv_dim = params['h_inv_eqv_dim']
        self.batch_size = params['batch_size']
        #self.alpha = params['alpha']
        #self.num_dosage_samples = params['num_dosage_samples']
        
        self.linear1 = nn.Linear(num_features, params['cov_dim'])

        self.feature_weight = Embeddings(h_dim, initrange=init_range_f)
        self.treat_emb = Embeddings(h_dim, act='id', initrange=init_range_t)
        self.dosage_emb = Embeddings(h_dim, act='id', initrange=init_range_t)
        self.linear2 = MLP(
            dim_input=h_dim * 2,
            dim_hidden=h_dim,
            dim_output=h_dim,
        )

        encoder_layers = TransformerEncoderLayer(h_dim, nhead=num_heads, dim_feedforward=h_dim, dropout=dropout, num_cov=params['cov_dim'])
        self.encoder = TransformerEncoder(encoder_layers, att_layers)

        decoder_layers = TransformerDecoderLayer(h_dim, nhead=num_heads, dim_feedforward=h_dim, dropout=dropout,num_t=1)
        self.decoder = TransformerDecoder(decoder_layers, att_layers)

        self.Q = MLP(
            dim_input=h_dim,
            dim_hidden=h_dim,
            dim_output=1,
            is_output_activation=False,
        )

    def forward(self, x, t, d):
        hidden = self.feature_weight(self.linear1(x))
        memory = self.encoder(hidden)

        t = t.view(t.shape[0], 1)
        d = d.view(d.shape[0], 1)
        tgt = torch.cat([self.treat_emb(t), self.dosage_emb(d)], dim=-1)
        tgt = self.linear2(tgt)
        if len(tgt.shape) < 3:
            tgt = tgt.unsqueeze(1)
        out = self.decoder(tgt.permute(1, 0, 2), memory.permute(1, 0, 2))
        if out.shape[0] != 1:
            out = torch.mean(out, dim=1)
        Q = self.Q(out.squeeze(0))
        return torch.mean(hidden, dim=1).squeeze(), Q


    def get_loss(self, x, t, d, y, requires_sample=0):
        t = t.view(t.shape[0], 1)
        d = d.view(d.shape[0], 1)
        hidden = self.feature_weight(self.linear1(x))
        memory = self.encoder(hidden)
        tgt = torch.cat([self.treat_emb(t), self.dosage_emb(d)], dim=-1)
        tgt = self.linear2(tgt)
        if len(tgt.shape) < 3:
            tgt = tgt.unsqueeze(1)
        out = self.decoder(tgt.permute(1, 0, 2), memory.permute(1, 0, 2))
        if out.shape[0] != 1:
            out = torch.mean(out, dim=1)
        y_mu = self.Q(out.squeeze(0))
        y_std = torch.ones_like(y_mu)/2
        y_dist = dist.Normal(y_mu, y_std)
        loss = - torch.mean(y_dist.log_prob(y))
        if requires_sample:
            hidden = self.feature_weight(self.linear1(x))
            memory = self.encoder(hidden)
            bs = x.shape[0]
            memory_d2 = memory.shape[1]
            memory_d3 = memory.shape[2]
            memory = memory.repeat(self.t_grid, 1, 1).view(bs*self.t_grid, \
                                                           memory_d2, memory_d3)

            
            d1 = self.ts.repeat(1, bs).view(bs*self.t_grid, 1)
            t1 = torch.ones_like(d1).cuda().detach().float()
            tgt = torch.cat([self.treat_emb(t1), self.dosage_emb(d1)], dim=-1)
            tgt = self.linear2(tgt)
            if len(tgt.shape) < 3:
                tgt = tgt.unsqueeze(1)
            out = self.decoder(tgt.permute(1, 0, 2), memory.permute(1, 0, 2))
            if out.shape[0] != 1:
                out = torch.mean(out, dim=1)
            y_mu = self.Q(out.squeeze(0))
            y_std = torch.ones_like(y_mu)/2
            y_dist = dist.Normal(y_mu, y_std)
            y_s = y_dist.rsample(sample_shape=[self.s]).reshape(self.s,self.t_grid,bs).\
                  transpose(1,2).reshape(bs*self.s,self.t_grid)
            xy = torch.cat((x.repeat(self.s, 1).reshape(bs*self.s, self.dx), y_s), 1)
            y_f = y.repeat(self.s, 1).reshape(self.s*bs, 1)
            t_f = t.repeat(self.s, 1).reshape(self.s*bs, 1)
            d_f = d.repeat(self.s, 1).reshape(self.s*bs, 1)
            return loss, xy, y_f, t_f, d_f
            
        else:
            return loss, loss.detach().item()

    @torch.no_grad()
    def get_predict(self,x,d):
        bs = len(d)
        x = x.repeat(bs, 1)
        t = torch.ones_like(d).cuda()
        y_mu = self.forward(x,t,d)[1]
        return y_mu
        

    @torch.no_grad()
    def impute_y(self, x, t, d, y):
        t = t.view(t.shape[0], 1)
        d = d.view(d.shape[0], 1)
        bs = x.shape[0]
        hidden = self.feature_weight(self.linear1(x))
        memory = self.encoder(hidden)
        memory_d2 = memory.shape[1]
        memory_d3 = memory.shape[2]
        memory = memory.repeat(self.t_grid, 1, 1).view(bs*self.t_grid, \
                                memory_d2, memory_d3)
        d1 = self.ts.repeat(1, bs).view(bs*self.t_grid, 1)
        t1 = torch.ones_like(d1).cuda().detach().float()
        tgt = torch.cat([self.treat_emb(t1), self.dosage_emb(d1)], dim=-1)
        tgt = self.linear2(tgt)
        if len(tgt.shape) < 3:
            tgt = tgt.unsqueeze(1)
        out = self.decoder(tgt.permute(1, 0, 2), memory.permute(1, 0, 2))
        if out.shape[0] != 1:
            out = torch.mean(out, dim=1)
        y_mu = self.Q(out.squeeze(0)).reshape(1,self.t_grid,bs).\
               transpose(1,2).reshape(bs,self.t_grid)
        xy = torch.cat((x, y_mu), 1)
        return xy.detach(), y, t, d

    


class DA_model(nn.Module):
    def __init__(self, params, num_heads=2, att_layers=1, dropout=0.0, \
                 init_range_f=0.1, init_range_t=0.1):
        super(DA_model, self).__init__()
        self.main_model = main_model(params)
        ts = self.main_model.ts
        self.auxiliary_model = auxiliary_model(params, ts)
        
        



    def get_loss(self, x, t, d, y):
        main_loss, xy, y_f, t_f, d_f = self.main_model.\
                               get_loss(x, t, d, y, requires_sample=True)
        aux_loss = self.auxiliary_model.get_loss(xy, y_f, t_f, d_f, self.main_model.s)
        return aux_loss + main_loss

    def get_pretrain_aux_loss(self, x, t, d, y):
        xy, y_f, t_f, d_f = self.main_model.impute_y(x, t, d, y)
        aux_loss = self.auxiliary_model.get_loss(xy, y_f, t_f, d_f, 1)
        return aux_loss
 
    @torch.no_grad()
    def get_predict(self, x, t):
        
        return self.main_model.get_predict(x, t)

