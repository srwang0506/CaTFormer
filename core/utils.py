import csv
import torch
import random
import numpy as np
import os

def set_random_seed(seed, deterministic=False, use_rank_shift=False):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
        rank_shift (bool): Whether to add rank number to the random seed to
            have different random seed in different threads. Default: False.
    """
    #if use_rank_shift:
    #    rank, _ = mmcv.runner.get_dist_info()
    #    seed += rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    #if deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
        
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()

    return n_correct_elems / batch_size

def my_collate(batch):
    (data,label) = zip(*batch)
    timesteps = [inputs.size(1)//16 for inputs in data]# get the time steps for item in batch
    data_chunk = []
    for i in range(len(timesteps)):
        data_chunk.extend(torch.chunk(data[i], timesteps[i], dim=1))
    data_chunk = torch.stack(data_chunk)
    target = torch.LongTensor(label)
    return data_chunk, target, timesteps


def init_log(output_dir):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y%m%d-%H:%M:%S',
                        filename=os.path.join(output_dir, 'log.log'),
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    return logging

def get_mean(norm_value=255, dataset='activitynet'):
    assert dataset in ['activitynet', 'kinetics']

    if dataset == 'activitynet':
        return [
            114.7748 / norm_value, 107.7354 / norm_value, 99.4750 / norm_value
        ]
    elif dataset == 'kinetics':
        # Kinetics (10 videos for each class)
        return [
            110.63666788 / norm_value, 103.16065604 / norm_value,
            96.29023126 / norm_value
        ]


def get_std(norm_value=255):
    # Kinetics (10 videos for each class)
    return [
        38.7568578 / norm_value, 37.88248729 / norm_value,
        40.02898126 / norm_value
    ]

def causal_effect(attn, q, kv):
    obs, _ = attn(q, kv, kv)                       
    cf_kv = kv.mean(0, keepdim=True).detach().expand_as(kv)
    cf, _ = attn(q, cf_kv, cf_kv)                    
    delta = obs - cf                                 

    baseline = kv.mean(0, keepdim=True)              
    proj = (delta * baseline).sum(-1, keepdim=True) / (baseline.norm(dim=-1, keepdim=True)**2 + 1e-6)
    delta = delta - proj * baseline                  
    return delta
        
        
def apply_gate(delta, residual):
    gate = torch.sigmoid(self.attfc(delta[-1]))      
    return residual[-1] + gate * delta[-1]

class RMSNorm(nn.Module):
    """Root-Mean-Square Normalization (parameter-efficient LayerNorm)."""
    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x * rms * self.scale


class PositionalEncoding(nn.Module):
    """Standard sine/cosine positional encoding."""
    def __init__(self, d_model: int, max_len: int = 5e3):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-torch.log(torch.tensor(1e4)) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(1)) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add PE to [T,B,D] tensor."""
        return x + self.pe[: x.size(0)]
    
