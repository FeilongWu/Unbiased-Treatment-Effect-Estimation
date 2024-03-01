import torch
import torch.nn as nn
import torch.distributions as dist
from torch.autograd import Function



eps = 1e-03
eps1 = 1e-05

def find_act(act):
    # act: function name
    if act.lower() == 'relu':
        actv = nn.ReLU()
    elif act.lower() == 'sigmoid':
        actv = nn.Sigmoid()
    elif act.lower() == 'softplus':
        actv = nn.Softplus()
    elif act.lower() == 'elu':
        actv = nn.ELU()
    return actv


class Truncated_power():
    def __init__(self, degree, knots):
        """
        This class construct the truncated power basis; the data is assumed in [0,1]
        :param degree: int, the degree of truncated basis
        :param knots: list, the knots of the spline basis; two end points (0,1) should not be included
        """
        self.degree = degree
        self.knots = knots
        self.num_of_basis = self.degree + 1 + len(self.knots)
        self.relu = nn.ReLU(inplace=True)

        if self.degree == 0:
            print('Degree should not set to be 0!')
            raise ValueError

        if not isinstance(self.degree, int):
            print('Degree should be int')
            raise ValueError

    def forward(self, x):
        """
        :param x: torch.tensor, batch_size * 1
        :return: the value of each basis given x; batch_size * self.num_of_basis
        """
        x = x.squeeze()
        out = torch.zeros(x.shape[0], self.num_of_basis)
        for _ in range(self.num_of_basis):
            if _ <= self.degree:
                if _ == 0:
                    out[:, _] = 1.
                else:
                    out[:, _] = x**_
            else:
                if self.degree == 1:
                    out[:, _] = (self.relu(x - self.knots[_ - self.degree]))
                else:
                    out[:, _] = (self.relu(x - self.knots[_ - self.degree - 1])) ** self.degree

        return out # bs, num_of_basis



        


def comp_grid(y, num_grid):

    # L gives the lower index
    # U gives the upper index
    # inter gives the distance to the lower int

    U = torch.ceil(y * num_grid)
    inter = 1 - (U - y * num_grid)
    L = U - 1
    L += (L < 0).int()

    return L.int().tolist(), U.int().tolist(), inter


class Dynamic_FC(nn.Module):
    def __init__(self, ind, outd, degree, knots, act='relu', isbias=1, islastlayer=0):
        super(Dynamic_FC, self).__init__()
        self.ind = ind
        self.outd = outd
        self.degree = degree
        self.knots = knots

        self.islastlayer = islastlayer

        self.isbias = isbias

        self.spb = Truncated_power(degree, knots)
        self.d = self.spb.num_of_basis # num of basis

        self.weight = nn.Parameter(torch.rand(self.ind, self.outd, self.d), requires_grad=True)

        if self.isbias:
            self.bias = nn.Parameter(torch.rand(self.outd, self.d), requires_grad=True)
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
        x_feature = x[:, 1:]
        x_treat = x[:, 0]

        x_feature_weight = torch.matmul(self.weight.T, x_feature.T).T # bs, outd, d

        x_treat_basis = self.spb.forward(x_treat) # bs, d
        x_treat_basis_ = torch.unsqueeze(x_treat_basis, 1)

        # x_feature_weight * x_treat_basis; bs, outd, d
        out = torch.sum(x_feature_weight * x_treat_basis_, dim=2) # bs, outd

        if self.isbias:
            out_bias = torch.matmul(self.bias, x_treat_basis.T).T
            out = out + out_bias

        if self.act is not None:
            out = self.act(out)

        # concat the treatment for intermediate layer
        if not self.islastlayer:
            out = torch.cat((torch.unsqueeze(x_treat, 1), out), 1)

        return out





class auxiliary_model(nn.Module):
    def __init__(self, dx, encode, cfg, degree, knots, ts=torch.linspace(0,1,10), \
                 act='relu', sample_w=1):
        super(auxiliary_model, self).__init__()
        
        self.dx = dx
        self.dz = encode[-1][1]
        self.sample_w = sample_w
        self.t_grid = len(ts)
        self.ts = ts # num_ts by 1
        self.degree = degree
        self.knots = knots
        n_layer = len(encode)
        encoder, decode_x = [], []
        for i in encode: # encode
           encoder.append(nn.Linear(i[0], i[1]))
           encoder.append(find_act(act))
        del encoder[-1]

        self.encoder = nn.Sequential(*encoder)



        blocks = []
        for layer_idx, layer_cfg in enumerate(cfg):
            if layer_idx == len(cfg)-1: # last layer
                last_layer = Dynamic_FC(layer_cfg[0], layer_cfg[1], self.degree, self.knots, act=layer_cfg[3], isbias=layer_cfg[2], islastlayer=1)
            else:
                blocks.append(
                    Dynamic_FC(layer_cfg[0], layer_cfg[1], self.degree, self.knots, act=layer_cfg[3], isbias=layer_cfg[2], islastlayer=0))
        blocks.append(last_layer)


        self.Q = nn.Sequential(*blocks)


    def get_loss(self, xy, y_f, t_f, s):
        # y_cf: i: s, j: bs
        # example y_cf: torch.tensor([[1.1,1.2],[2.1,2.2],[3.1,3.2],[1.3,1.4],
        # [2.3,2.4],[3.3,3.4],[1.5,1.6],[2.5,2.6],[3.5,3.6],[1.7,1.8],
        # [2.7,2.8],[3.7,3.8]]), bs = 3, s = 4, t_grid = 2
        bs = int(xy.shape[0] / s)
        z = self.encoder(xy)
        ts = self.ts.repeat(bs*s, 1) #  i: s, j: bs, k: t_grid
        z1 = z.repeat(1, self.t_grid).reshape(bs*s*self.t_grid, self.dz)
        t_hidden = torch.cat((ts, z1), 1)
        pre_y_cf = self.Q(t_hidden).reshape(bs*s, self.t_grid)
        y_cf = xy[:,self.dx:]
        x = xy[:,:self.dx]
        loss_y = torch.mean(((y_cf - pre_y_cf)**2) * self.sample_w.repeat(bs*s, 1))
        
        t_hidden = torch.cat((t_f, z), 1)
        pre_y_f = self.Q(t_hidden)
        loss_yf = torch.mean((pre_y_f - y_f)**2)
        return 0.01 * loss_y + loss_yf




    def _initialize_weights(self):
        # TODO: maybe add more distribution for initialization
        for m in self.modules():
            if isinstance(m, Dynamic_FC):
                m.weight.data.normal_(0, 1.)
                if m.isbias:
                    m.bias.data.zero_()
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
        
        



class main_model(nn.Module):
    def __init__(self, cfg_density, num_grid, cfg, degree, knots,\
                 t_grid=30, s=20, ts=torch.linspace(0, 1, 10).view(10, 1)):
        super(main_model, self).__init__()
        """
        cfg_density: cfg for the density estimator; [(ind1, outd1, isbias1), 'act', ....]; the cfg for density estimator head is not included
        num_grid: how many grid used for the density estimator head
        """

        # cfg/cfg_density = [(ind1, outd1, isbias1, activation),....]
        self.cfg_density = cfg_density
        self.num_grid = num_grid

        self.cfg = cfg
        self.degree = degree
        self.knots = knots
        self.dz = cfg_density[-1][1]
        self.t_grid = t_grid
        self.s = s # sample size of y
        self.dx = cfg_density[0][0]
        self.uniform = dist.Uniform(torch.tensor(0.), torch.tensor(1.))
        self.ts = ts
        

        # construct the density estimator
        density_blocks = []
        density_hidden_dim = -1
        for layer_idx, layer_cfg in enumerate(cfg_density):
            # fc layer
            if layer_idx == 0:
                # weight connected to feature
                self.feature_weight = nn.Linear(in_features=layer_cfg[0], out_features=layer_cfg[1], bias=layer_cfg[2])
                density_blocks.append(self.feature_weight)
            else:
                density_blocks.append(nn.Linear(in_features=layer_cfg[0], out_features=layer_cfg[1], bias=layer_cfg[2]))
            density_hidden_dim = layer_cfg[1]
            if layer_cfg[3] == 'relu':
                density_blocks.append(nn.ReLU(inplace=True))
            elif layer_cfg[3] == 'tanh':
                density_blocks.append(nn.Tanh())
            elif layer_cfg[3] == 'sigmoid':
                density_blocks.append(nn.Sigmoid())
            else:
                print('No activation')

        self.hidden_features = nn.Sequential(*density_blocks)



        # construct the dynamics network
        blocks = []
        for layer_idx, layer_cfg in enumerate(cfg):
            if layer_idx == len(cfg)-1: # last layer
                last_layer = Dynamic_FC(layer_cfg[0], layer_cfg[1], self.degree, self.knots, act=layer_cfg[3], isbias=layer_cfg[2], islastlayer=1)
            else:
                blocks.append(
                    Dynamic_FC(layer_cfg[0], layer_cfg[1], self.degree, self.knots, act=layer_cfg[3], isbias=layer_cfg[2], islastlayer=0))
        blocks.append(last_layer)


        self.Q = nn.Sequential(*blocks)

        


    def get_loss(self, x, t, y, requires_sample=0):
        y = y.unsqueeze(-1)
        t = t.unsqueeze(-1)
        z = self.hidden_features(x)
        t_hidden = torch.cat((t, z), 1)
        if requires_sample:
            y_mu = self.Q(t_hidden)
            y_std = torch.ones_like(y_mu)/2
            y_dist = dist.Normal(y_mu, y_std)
            loss = - torch.mean(y_dist.log_prob(y))
            bs = x.shape[0]
            z = z.repeat(self.t_grid, 1).view(bs*self.t_grid, self.dz)
            t1 = self.ts.repeat(1, bs).view(bs*self.t_grid, 1)
            t_hidden = torch.cat((t1, z), 1)
            y_mu = self.Q(t_hidden)
            y_std = torch.ones_like(y_mu)/2
            y_dist = dist.Normal(y_mu, y_std)
            y_s = y_dist.rsample(sample_shape=[self.s]).reshape(self.s,self.t_grid,bs).\
                  transpose(1,2).reshape(bs*self.s,self.t_grid) # [[y0.1,..,y0.9]_ij,]
                  # i: s, j: bs
            xy = torch.cat((x.repeat(self.s, 1).reshape(bs*self.s, self.dx), y_s), 1)
            y_f = y.repeat(self.s, 1).reshape(self.s*bs, 1)
            t_f = t.repeat(self.s, 1).reshape(self.s*bs, 1)
            return loss, xy, y_f, t_f
        else:
            y_mu = self.Q(t_hidden)
            y_std = torch.ones_like(y_mu)/2
            y_dist = dist.Normal(y_mu, y_std)
            loss = - torch.mean(y_dist.log_prob(y))
            return loss, loss.detach().item()


    @torch.no_grad()
    def impute_y(self, x, t, y):
        y = y.unsqueeze(-1)
        t = t.unsqueeze(-1)
        bs = x.shape[0]
        z = self.hidden_features(x)
        z = z.repeat(self.t_grid, 1).view(bs*self.t_grid, self.dz)
        t1 = self.ts.repeat(1, bs).view(bs*self.t_grid, 1)
        t_hidden = torch.cat((t1, z), 1)
        y_mu = self.Q(t_hidden).reshape(1,self.t_grid,bs).\
               transpose(1,2).reshape(bs*1,self.t_grid)

        xy = torch.cat((x, y_mu), 1)
        return xy.detach(), y, t

    

        
    @torch.no_grad()
    def get_predict(self, x, t):
        n = len(t)
        z_mu = self.hidden_features(x).repeat(n, 1)
        t_hidden = torch.cat((torch.unsqueeze(t, 1), z_mu), 1)
        y = self.Q(t_hidden).flatten()
        return y
        

    def _initialize_weights(self):
        # TODO: maybe add more distribution for initialization
        for m in self.modules():
            if isinstance(m, Dynamic_FC):
                m.weight.data.normal_(0, 1.)
                if m.isbias:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


    @torch.no_grad()
    def sample_z(self, x, sz=20):
        bs = x.shape[0]
        z_mu = self.hidden_features(x)
        z_std = z_mu / self.factor + eps
        z_dist = dist.Normal(z_mu, z_std)
        zs = z_dist.sample(sample_shape=[sz]).transpose(0, 1).reshape(bs, self.dz*sz)
        return zs
        



     

    
        




class DA_model(nn.Module):
    def __init__(self, cfg_density, num_grid, cfg, cfg_aux, degree, knots,\
                 dx, encode, ts, sample_w, act='relu', \
                 t_grid=30, s=20):
        super(DA_model, self).__init__()
        self.main_model = main_model(cfg_density, num_grid, cfg, degree, knots,\
                 t_grid=t_grid, s=s, ts=ts)
        self.auxiliary_model = auxiliary_model(dx, encode, cfg_aux, degree, \
                                               knots, ts=ts, act=act, \
                                               sample_w=sample_w)
        
        


    def _initialize_weights(self):
        self.main_model._initialize_weights()
        #self.auxiliary_model._initialize_weights()

    def get_loss(self, x, t, y):
        main_loss, xy, y_f, t_f = self.main_model.\
                               get_loss(x, t, y, requires_sample=True)
        aux_loss = self.auxiliary_model.get_loss(xy, y_f, t_f, self.main_model.s)
        return aux_loss + main_loss

    def get_pretrain_aux_loss(self, x, t, y):
        xy, y_f, t_f = self.main_model.impute_y(x, t, y)
        aux_loss = self.auxiliary_model.get_loss(xy, y_f, t_f, 1)
        return aux_loss
 
    @torch.no_grad()
    def get_predict(self, x, t):
        
        return self.main_model.get_predict(x, t)








