import torch
from data.dataset import get_iter
import numpy as np
import random
from utils.model_helper import IPM_loss

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def early_stop(epoch, best_epoch, tol=17):
    if epoch - best_epoch > tol:
        return True
    else:
        return False

def rwt_regression_loss(w, y, y_pre):
    #y_pre, w = y_pre.to('cpu'), w.to('cpu')

    return ((y_pre.squeeze() - y.squeeze())**2 * w.squeeze()).mean()

def train(model, dataloader, args, k=5):
    model.train()
    epochs = args.n_epochs
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate, weight_decay=args.weight_decay
        )

    best_loss = np.inf
    
    for epoch in range(epochs):
        total_loss = []
        mmds = []
        cum_loss = 0
        for batch in dataloader:
            x=batch['x'].float().to(args.device)
            t=batch['t'].float().to(args.device)
            if args.scale:
                y = batch['y'].detach().cpu().numpy()
                y = args.scaler.transform(y.reshape(-1, 1))
                y = torch.from_numpy(y).float().to(args.device)
            else:
                y = batch['y'].float().to(args.device)
            optimizer.zero_grad()
            y_pre, w, _ = model(x, t)
            loss = rwt_regression_loss(w, y, y_pre) 
            
            #total_loss.append(loss.data)
            
            
            mmd = IPM_loss(x, t, w, k=k) 
            mmds.append(mmd.data)
            loss = loss + mmd
            
            loss.backward()
            optimizer.step()
            cum_loss += loss.item()
        if cum_loss < best_loss:
            best_loss = cum_loss
            best_epoch = epoch
            torch.save(model.state_dict(), './saved.pt')
        if early_stop(epoch, best_epoch, tol=17):
            break
    model.load_state_dict(torch.load('./saved.pt'))
    return model

        #total_loss = torch.mean(total_loss)
        
        
