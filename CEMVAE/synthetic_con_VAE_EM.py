import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

eps = 1e-7
bound = 1e05

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

class px(nn.Module):
    def __init__(self, dz, nf, h, dl, actv):
        super(px, self).__init__()
        net = [nn.Linear(dz, dl), actv]
        for i in range(h-1):
            net.append(nn.Linear(dl, dl))
            net.append(actv)
        self.NN = nn.Sequential(*net)
        self.loc = nn.Linear(dl, nf)
        self.scale = nn.Sequential(*[nn.Linear(dl, nf), nn.Softplus()]) 

    def forward(self, z):
        rep_x = self.NN(z)
        loc = self.loc(rep_x)
        scale = self.scale(rep_x) + eps
        return loc.clamp(min=-bound,max=bound), scale.clamp(min=-bound,max=bound)

class pxdis(nn.Module):
    def __init__(self, dz, nf, h, dl, actv):
        super(pxdis, self).__init__()
        net = [nn.Linear(dz, dl), actv]
        for i in range(h):
            net.append(nn.Linear(dl, dl))
            net.append(actv)
        net.append(nn.Linear(dl, nf))
        net.append(nn.Sigmoid())
        self.NN = nn.Sequential(*net)
    def forward(self, z):
        loc_t = self.NN(z)
        return loc_t
    
class pt(nn.Module):
    def __init__(self, dz, out, h, dl, actv):
        super(pt, self).__init__()
        net = [nn.Linear(dz, dl), actv]
        for i in range(h):
            net.append(nn.Linear(dl, dl))
            net.append(actv)
        net.append(nn.Linear(dl, out))
        net.append(nn.Sigmoid())
        self.NN = nn.Sequential(*net)

    def forward(self, z):
        loc_t = self.NN(z)
        return loc_t.clamp(min=-bound,max=bound)

class py0(nn.Module):
    def __init__(self, dz, out, h, dl, actv):
        super(py0, self).__init__()
        net = [nn.Linear(dz, dl), actv]
        for i in range(h-1):
            net.append(nn.Linear(dl, dl))
            net.append(actv)
        self.NN = nn.Sequential(*net)
        self.loc = nn.Linear(dl, 1)
        self.scale = nn.Sequential(*[nn.Linear(dl, 1), nn.Sigmoid()]) 

    def forward(self, z):
        rep_z = self.NN(z)
        loc_y = self.loc(rep_z)
        scale = self.scale(rep_z)
        return  torch.cat((loc_y, scale), 1)

class py1(nn.Module):
    def __init__(self, dz, out, h, dl, actv):
        super(py1, self).__init__()
        net = [nn.Linear(dz, dl), actv]
        for i in range(h-1):
            net.append(nn.Linear(dl, dl))
            net.append(actv)
        self.NN = nn.Sequential(*net)
        self.loc = nn.Linear(dl, 1)
        self.scale = nn.Sequential(*[nn.Linear(dl, 1), nn.Sigmoid()]) 

    def forward(self, z):
        rep_z = self.NN(z)
        loc_y = self.loc(rep_z)
        scale = self.scale(rep_z)
        return  torch.cat((loc_y, scale), 1)


class rep_xy(nn.Module):
    def __init__(self, dxy, out, h, dl, actv):
        super(rep_xy, self).__init__()
        net = [nn.Linear(dxy, dl), actv]
        for i in range(h):
            net.append(nn.Linear(dl, dl))
            net.append(actv)
        self.NN = nn.Sequential(*net)

    def forward(self, xy):
        return self.NN(xy).clamp(min=-bound,max=bound)

class qz0(nn.Module):
    def __init__(self, insize, dz, h, dl, actv):
        super(qz0, self).__init__()
        net = [nn.Linear(insize, dl), actv]
        for i in range(h-1):
            net.append(nn.Linear(dl, dl))
            net.append(actv)
        self.NN = nn.Sequential(*net)
        self.loc = nn.Linear(dl, dz)
        self.scale = nn.Sequential(*[nn.Linear(dl, dz), nn.Softplus()])

    def forward(self, rep_xy):
        rep = self.NN(rep_xy)
        loc = self.loc(rep).clamp(min=-bound,max=bound)
        scale = self.scale(rep).clamp(max=bound) + eps
        return torch.cat((loc, scale), 1)

class qz1(nn.Module):
    def __init__(self, insize, dz, h, dl, actv):
        super(qz1, self).__init__()
        net = [nn.Linear(insize, dl), actv]
        for i in range(h-1):
            net.append(nn.Linear(dl, dl))
            net.append(actv)
        self.NN = nn.Sequential(*net)
        self.loc = nn.Linear(dl, dz) 
        self.scale = nn.Sequential(*[nn.Linear(dl, dz), nn.Softplus()])

    def forward(self, rep_xy):
        rep = self.NN(rep_xy)
        loc = self.loc(rep).clamp(min=-bound,max=bound)
        scale = self.scale(rep).clamp(max=bound) + eps
        return torch.cat((loc, scale), 1)
        
class rep_x(nn.Module):
    def __init__(self, dx, out, h, dl, actv):
        super(rep_x, self).__init__()
        net = [nn.Linear(dx, dl), actv]
        for i in range(h):
            net.append(nn.Linear(dl, dl))
            net.append(actv)
        self.NN = nn.Sequential(*net)

    def forward(self, x):
        return self.NN(x).clamp(min=-bound,max=bound)

class qt(nn.Module):
    def __init__(self, nf, dt, h, dl, actv):
        super(qt, self).__init__()
        net = [nn.Linear(nf, dl), actv]
        for i in range(h):
            net.append(nn.Linear(dl, dl))
            net.append(actv)
        net.append(nn.Linear(dl, dt))
        net.append(nn.Sigmoid())
        self.NN = nn.Sequential(*net)

    def forward(self, x):
        t_loc = self.NN(x)
        return t_loc.clamp(min=-bound,max=bound)

class qy0(nn.Module):
    def __init__(self, insize, dy, h, dl, actv):
        super(qy0, self).__init__()
        net = [nn.Linear(insize, dl), actv]
        for i in range(h):
            net.append(nn.Linear(dl, dl))
            net.append(actv)
        net.append(nn.Linear(dl, dy))
        net.append(nn.Sigmoid())
        self.NN = nn.Sequential(*net)

    def forward(self, rep_x):
        y_loc = self.NN(rep_x)
        return y_loc.clamp(min=-bound,max=bound)

class qy1(nn.Module):
    def __init__(self, insize, dy, h, dl, actv):
        super(qy1, self).__init__()
        net = [nn.Linear(insize, dl), actv]
        for i in range(h):
            net.append(nn.Linear(dl, dl))
            net.append(actv)
        net.append(nn.Linear(dl, dy))
        net.append(nn.Sigmoid())
        self.NN = nn.Sequential(*net)

    def forward(self, rep_x):
        y_loc = self.NN(rep_x)
        return y_loc.clamp(min=-bound,max=bound)






    
class VAE_EM(nn.Module):
    def __init__(self, xcon, xdis, dz, device, h=3, dl=100, act='elu', alpha=1.0):
        # h: num of hidden layers
        # nf: dim of features
        # dz: dim of z
        # h: # of hidden layers
        # dl: size of hidden layer

        super(VAE_EM, self).__init__()
        self.device = device
        self.dz = dz
        self.xcon = xcon
        self.xdis = xdis
        self.nf = xcon + xdis
        actv = find_act(act)
        self.pxcon = px(dz, xcon, h, dl, actv)
        self.pxdis = pxdis(dz, xdis, h, dl, actv)
        self.py0 = py0(dz, 1, h, dl, actv)
        self.py1 = py1(dz, 1, h, dl, actv)
        self.qy0 = py0(xcon+xdis, 1, h, dl, actv)
        self.qy1 = py1(xcon+xdis, 1, h, dl, actv)
        self.qz = qz0(xcon+xdis+2, dz, h-1, dl, actv)
        self.pz = dist.Normal(torch.zeros(dz).to(device), torch.ones(dz).to(device))
        self.alpha = alpha
        

    #@torch.no_grad()
    def sample_counterfactuals(self, x, t, sample=1):
        dt = t.shape[0]
        qy_loc, qy_scale = torch.chunk(self.qy1(x) * t + self.qy0(x) * (1 - t), 2, dim=1)
        y_dist = dist.Normal(qy_loc.flatten(), qy_scale.flatten())
        #return y_dist.sample(sample_shape=[sample]).transpose(0,1).flatten().view(dt*sample, 1)
        return y_dist.rsample(sample_shape=[sample]).transpose(0,1).flatten().view(dt*sample, 1)

    def getloss(self, xcon, xdis, y, t, sample=50):
        # x, y, t shape: [batch size, column]
        dt = t.shape[0]
        x = torch.cat((xcon, xdis), 1)
        qy_loc, qy_scale = torch.chunk(self.qy1(x) * t + self.qy0(x) * (1 - t), 2, dim=1)
        auxiliary = torch.sum(dist.Normal(qy_loc, qy_scale).log_prob(y))
        
        # variational approximation
        yc = self.sample_counterfactuals(x, 1-t, sample=sample)
        x = x.repeat(1, sample).view(dt*sample, self.nf)
        t = t.repeat(1, sample).view(dt*sample, 1)
        y = y.repeat(1, sample).view(dt*sample, 1)
        y0 = y * (1 - t) + yc * t
        y1 = y * t + yc * (1 - t)
        xy0y1 = torch.cat((x,y0,y1), 1).float()
        qz_loc, qz_scale = torch.chunk(self.qz(xy0y1), 2, dim=1)
        qz_dist = dist.Normal(qz_loc, qz_scale)
        qz_s = qz_dist.rsample().float()

        # reconstruct distributions
        pxcon_loc, pxcon_scale = self.pxcon(qz_s)
        pxdis_loc = self.pxdis(qz_s)
        y0_loc, y0_scale = torch.chunk(self.py0(qz_s), 2, dim=1)
        y1_loc, y1_scale = torch.chunk(self.py1(qz_s), 2, dim=1)

        #loss
        logpxcon = dist.Normal(pxcon_loc, pxcon_scale).log_prob(x[:,0:self.xcon])
        logpxdis = dist.Bernoulli(pxdis_loc).log_prob(x[:, self.xcon:])
        logpy0 = dist.Normal(y0_loc, y0_scale).log_prob(y0)
        logpy1 = dist.Normal(y1_loc, y1_scale).log_prob(y1)
        density = torch.exp(logpy1 * (1 -t) + logpy0 * t).detach().view(dt, sample)
        tot = torch.sum(density, 1).unsqueeze(-1)
        density = (density / tot).view(dt*sample, 1)
        logpz = self.pz.log_prob(qz_s)
        logqz = qz_dist.log_prob(qz_s)
        
        lb = torch.sum(density * logpxcon) + torch.sum(density * logpxdis) \
             + torch.sum(density * logpz) - torch.sum(density * logqz) + \
             torch.sum(density * logpy0) + torch.sum(density * logpy1)



        return - (lb + self.alpha * auxiliary) / dt



    #### to be removed #####
    def get_aux(self, xcon, xdis, y, t, sample=50):
        # x, y, t shape: [batch size, column]
        dt = t.shape[0]
        x = torch.cat((xcon, xdis), 1)
        qy_loc, qy_scale = torch.chunk(self.qy1(x) * t + self.qy0(x) * (1 - t), 2, dim=1)
        auxiliary = torch.sum(dist.Normal(qy_loc, qy_scale).log_prob(y))


        return - (auxiliary) / dt

    #### to be removed #####
    def getloss_lb(self, xcon, xdis, y, t, sample=50):
        # x, y, t shape: [batch size, column]
        dt = t.shape[0]
        x = torch.cat((xcon, xdis), 1)
        qy_loc, qy_scale = torch.chunk(self.qy1(x) * t + self.qy0(x) * (1 - t), 2, dim=1)
        
        # variational approximation
        yc = self.sample_counterfactuals(x, 1-t, sample=sample)
        x = x.repeat(1, sample).view(dt*sample, self.nf)
        t = t.repeat(1, sample).view(dt*sample, 1)
        y = y.repeat(1, sample).view(dt*sample, 1)
        y0 = y * (1 - t) + yc * t
        y1 = y * t + yc * (1 - t)
        xy0y1 = torch.cat((x,y0,y1), 1).float()
        qz_loc, qz_scale = torch.chunk(self.qz(xy0y1), 2, dim=1)
        qz_dist = dist.Normal(qz_loc, qz_scale)
        qz_s = qz_dist.rsample().float()

        # reconstruct distributions
        pxcon_loc, pxcon_scale = self.pxcon(qz_s)
        pxdis_loc = self.pxdis(qz_s)
        y0_loc, y0_scale = torch.chunk(self.py0(qz_s), 2, dim=1)
        y1_loc, y1_scale = torch.chunk(self.py1(qz_s), 2, dim=1)

        #loss
        logpxcon = dist.Normal(pxcon_loc, pxcon_scale).log_prob(x[:,0:self.xcon])
        logpxdis = dist.Bernoulli(pxdis_loc).log_prob(x[:, self.xcon:])
        logpy0 = dist.Normal(y0_loc, y0_scale).log_prob(y0)
        logpy1 = dist.Normal(y1_loc, y1_scale).log_prob(y1)
        density = torch.exp(logpy1 * (1 -t) + logpy0 * t).detach().view(dt, sample)
        tot = torch.sum(density, 1).unsqueeze(-1)
        density = (density / tot).view(dt*sample, 1)
        logpz = self.pz.log_prob(qz_s)
        logqz = qz_dist.log_prob(qz_s)
        
        lb = torch.sum(density * logpxcon) + torch.sum(density * logpxdis) \
             + torch.sum(density * logpz) - torch.sum(density * logqz) + \
             torch.sum(density * logpy0) + torch.sum(density * logpy1)



        return - (lb) / dt



    @torch.no_grad()
    def lower_bound_aux(self, xcon, xdis, y, t):
        dt = t.shape[0]
        x = torch.cat((xcon, xdis), 1)
        qy_loc, qy_scale = torch.chunk(self.qy1(x) * t + self.qy0(x) * (1 - t), 2, dim=1)
        #auxiliary = torch.sum(dist.Normal(qy_loc, qy_scale).log_prob(y))
        auxiliary = torch.sum((qy_loc - y)**2)

        #return  (auxiliary) 
        return  -(auxiliary)



    


    def y_mean(self, z, t):
        loc, scale = torch.chunk(self.py0(z) * (1-t) + self.py1(z) * t, 2, dim=1)
        return loc, scale

    @torch.no_grad()
    def lower_bound(self, xcon, xdis, y, t):
        # x, y, t shape: [batch size, column]
        
        # variational approximation
        x = torch.cat((xcon, xdis), 1)
        yc, _ = torch.chunk(self.qy0(x) * t + self.qy1(x) * (1 - t), 2, dim=1)
        y0 = y * (1 - t) + yc * t
        y1 = y * t + yc * (1 - t)
        xy0y1 = torch.cat((x,y0,y1), 1).float()
        qz_loc, qz_scale = torch.chunk(self.qz(xy0y1), 2, dim=1)
        qz_dist = dist.Normal(qz_loc, qz_scale)
        qz_s = qz_dist.rsample().float()

        # reconstruct distributions
        pxcon_loc, pxcon_scale = self.pxcon(qz_s)
        pxdis_loc = self.pxdis(qz_s)
        y0_loc, y0_scale = torch.chunk(self.py0(qz_s), 2, dim=1)
        y1_loc, y1_scale = torch.chunk(self.py1(qz_s), 2, dim=1)

        #loss
        logpxcon = torch.sum(dist.Normal(pxcon_loc, pxcon_scale).log_prob(xcon))
        logpxdis = torch.sum(dist.Bernoulli(pxdis_loc).log_prob(xdis))
        logpy0 = dist.Normal(y0_loc, y0_scale).log_prob(y0)
        logpy1 = dist.Normal(y1_loc, y1_scale).log_prob(y1)
        logpz = torch.sum(self.pz.log_prob(qz_s))
        logqz = torch.sum(qz_dist.log_prob(qz_s))
        logpy = torch.sum(t * logpy1 + (1 - t) * logpy0)
        
        lb = logpxcon + logpxdis + logpy + logpz - logqz



        return lb
        


    @torch.no_grad()
    def predict(self, xcon, xdis, sample = 250):
        dt = xcon.shape[0]
        x = torch.cat((xcon, xdis), 1)
        y0_loc, y0_scale = torch.chunk(self.qy0(x), 2, dim=1)
        y1_loc, y1_scale = torch.chunk(self.qy1(x), 2, dim=1)
        y0_dist = dist.Normal(y0_loc, y0_scale)
        y1_dist = dist.Normal(y1_loc, y1_scale)
        y0 = y0_dist.sample(sample_shape=[sample]).view(sample*dt, 1)
        y1 = y1_dist.sample(sample_shape=[sample]).view(sample*dt, 1)
        xy0y1 = torch.cat((x.repeat(sample, 1),y0,y1), 1).float()
        qz_loc, qz_scale = torch.chunk(self.qz(xy0y1), 2, dim=1)
        qz_dist = dist.Normal(qz_loc, qz_scale)
        qz_samples = qz_dist.sample().view(sample*dt, self.dz)
        py0_loc, _ = torch.chunk(self.py0(qz_samples), 2, dim=1)
        py1_loc, _ = torch.chunk(self.py1(qz_samples), 2, dim=1)
        y0 = torch.mean(py0_loc.view(sample, dt, 1), dim=0)
        y1 = torch.mean(py1_loc.view(sample, dt, 1), dim=0)
        
        return y0.flatten(), y1.flatten()
        

        


class target_classifer(nn.Module):
    def __init__(self, nf, h, dl, actv, device, missing_target=False):
        super(qt, self).__init__()
        self.device = device
        self.nf = nf
        if missing_target == False:
            net = [nn.Linear(nf, dl), actv]
            for i in range(h):
                net.append(nn.Linear(dl, dl))
                net.append(actv)
            net.append(nn.Linear(dl, 1))
            net.append(nn.Sigmoid())
            self.NN = nn.Sequential(*net)
            self.get_weight = self.classify
            self.mean_p = 1
        else:
            self.get_weight = self.ones

    @torch.no_grad()
    def mean_prob(self):
        mu = torch.zeros(self.nf).to(self.device)
        mean = self.NN(mu)
        self.mean_p = mean.item()

    @torch.no_grad()
    def create_onehot(self, batch_size, y):
        d = torch.zeros([batch_size, 1]) + y
        return d.to(self.device)
    
    def get_loss(self, x, y):
        dt = x.shape[0]
        y = self.create_onehot(dt, y)
        y_loc = self.forward(x)
        y_dist = dist.Bernoulli(y_loc)
        return - torch.sum(y_dist.log_prob(y))
        

    def forward(self, x):
        t_loc = self.NN(x)
        return t_loc

    @torch.no_grad()
    def get_weight(self):
        pass

    @torch.no_grad()
    def ones(self, x):
        dt = x.shape[0]
        return torch.ones([dt, 1]).to(self.device)

    @torch.no_grad()
    def classify(self, x):
        dt = x.shape[0]
        p = self.NN(x) / self.mean_p

    
