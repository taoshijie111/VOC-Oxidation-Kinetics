from matplotlib import pyplot as plt
from sklearn.metrics import  r2_score
from datetime import datetime
import os.path
import numpy as np
import torch.optim as optim
import random
import torch


class plotUtils:
    @classmethod
    def plot_test_pred(cls, test_y,pred_y,rmse_test,react, log_dir, fold=None):

        ax = plt.gca()
        plt.plot(test_y, test_y, color='red', linewidth=3)
        plt.scatter(test_y,  pred_y, color='black', label='pred')
        plt.text(0.6,0.45,"R2 = " + str(round(r2_score(test_y,  pred_y),3)),fontsize=16, color ='blue',transform=ax.transAxes)
        plt.text(0.6,0.4,"RMSE = " + str(round(rmse_test,3)),fontsize=16,color = 'blue',transform=ax.transAxes)
        plt.xlabel("logK-EXP",fontsize=18) 
        plt.ylabel("logK-GNN",fontsize=18)  
        plt.legend(['y=x','pred_test'],fontsize=18)
        plt.title('testset')
        if fold is not None:
            plt.savefig(os.path.join(log_dir, f'test_{react}_{fold}.jpg'))
        else:
            plt.savefig(os.path.join(log_dir, f'test_{react}.jpg'))
        # plt.show()
        plt.close()

    @classmethod
    def plot_loss(cls,train_epochs,train_loss_list,val_loss_list,react, log_dir, fold=None):
        if fold is not None:
            np.savetxt(os.path.join(log_dir, f'loss_data_{react}_{fold}.dat'), np.column_stack((train_loss_list, val_loss_list)), header='train_loss val_loss', comments='', fmt='%0.4f')
        else:
            np.savetxt(os.path.join(log_dir, f'loss_data_{react}.dat'), np.column_stack((train_loss_list, val_loss_list)), header='train_loss val_loss', comments='', fmt='%0.4f')
        x1 = range(0, train_epochs)
        x2 = range(0, train_epochs)
        y1 = train_loss_list
        y2 = val_loss_list
        plt.plot(x1, y1, color='black', linewidth=3)
        plt.plot(x2, y2, color='red', linewidth=3)
        plt.ylim(0, 2) 
        plt.legend(['train_loss','validation_loss'],fontsize=18)
        plt.title('train_validation_loss')
        plt.ylabel('Loss',fontsize=18)
        plt.xlabel('Epoches',fontsize=18)
        # plt.show()
        if fold is not None:
            plt.savefig(os.path.join(log_dir, f'loss_{react}_{fold}.jpg'))
        else:
            plt.savefig(os.path.join(log_dir, f'loss_{react}.jpg'))
        plt.close()


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_steps):
        if isinstance(warmup, float):
            self.warmup = int(warmup * max_steps)
        elif isinstance(warmup, int):
            self.warmup = warmup
        self.max_num_iters = max_steps
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(step=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, step):
        lr_factor = 0.5 * (1 + np.cos(np.pi * step / self.max_num_iters))
        if step <= self.warmup:
            lr_factor *= step * 1.0 / self.warmup
        return lr_factor

def add_time_to_log_dir(name, prefix=None):
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = f'{prefix}_{name}_{current_time}'
    return log_dir

def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EnsembleModel(torch.nn.Module):

    def __init__(self, models, weights=None):
        super(EnsembleModel, self).__init__()
        self.models = torch.nn.ModuleList(models)
        self.weights = weights
        if weights is not None:
            assert abs(sum(weights) - 1.0) < 1e-5, "weights must sum to 1"

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        outputs = torch.stack(outputs, dim=0)
        if self.weights is not None:
            outputs = torch.sum(outputs.permute(1, 0, 2) * torch.tensor(self.weights).reshape(-1, 1).to(outputs.device), dim=1)
            return outputs
        else:
            outputs = torch.mean(outputs, dim=0)
            return outputs
