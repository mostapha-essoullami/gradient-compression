from inspect import signature
from collections import defaultdict, namedtuple
import json
import os
import time
import numpy as np
from functools import singledispatch
import torch.distributed as dist
import torch
from torch import nn
import math
from compressor import *
#####################
# utils
#####################
torch.manual_seed(42)
torch.cuda.manual_seed(42)

class Timer:
    def __init__(self, synch=None):
        self.synch = synch or (lambda: None)
        self.synch()
        self.times = [time.time()]
        self.total_time = 0.0

    def __call__(self, include_in_total=True):
        self.synch()
        self.times.append(time.time())
        delta_t = self.times[-1] - self.times[-2]
        if include_in_total:
            self.total_time += delta_t
        return delta_t


localtime = lambda: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


class TableLogger:
    def append(self, output):
        if not hasattr(self, "keys"):
            self.keys = output.keys()
            print(*(f"{k:>12s}" for k in self.keys))
        filtered = [output[k] for k in self.keys]
        print(*(f"{v:12.4f}" if isinstance(v, float) else f"{v:12}" for v in filtered))


#####################
## data preprocessing
#####################

cifar10_mean = (
    0.4914,
    0.4822,
    0.4465,
)  # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (
    0.2471,
    0.2435,
    0.2616,
)  # equals np.std(train_set.train_data, axis=(0,1,2))/255


def normalise(x, mean=cifar10_mean, std=cifar10_std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean * 255
    x *= 1.0 / (255 * std)
    return x


def pad(x, border=4):
    return np.pad(
        x, [(0, 0), (border, border), (border, border), (0, 0)], mode="reflect"
    )


def transpose(x, source="NHWC", target="NCHW"):
    return x.transpose([source.index(d) for d in target])


#####################
## data augmentation
#####################


class Crop(namedtuple("Crop", ("h", "w"))):
    def __call__(self, x, x0, y0):
        return x[:, y0 : y0 + self.h, x0 : x0 + self.w]

    def options(self, x_shape):
        C, H, W = x_shape
        return {"x0": range(W + 1 - self.w), "y0": range(H + 1 - self.h)}

    def output_shape(self, x_shape):
        C, H, W = x_shape
        return (C, self.h, self.w)


class FlipLR(namedtuple("FlipLR", ())):
    def __call__(self, x, choice):
        return x[:, :, ::-1].copy() if choice else x

    def options(self, x_shape):
        return {"choice": [True, False]}


class Cutout(namedtuple("Cutout", ("h", "w"))):
    def __call__(self, x, x0, y0):
        x = x.copy()
        x[:, y0 : y0 + self.h, x0 : x0 + self.w].fill(0.0)
        return x

    def options(self, x_shape):
        C, H, W = x_shape
        return {"x0": range(W + 1 - self.w), "y0": range(H + 1 - self.h)}


class Transform:
    def __init__(self, dataset, transforms):
        self.dataset, self.transforms = dataset, transforms
        self.choices = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, labels = self.dataset[index]
        for choices, f in zip(self.choices, self.transforms):
            args = {k: v[index] for (k, v) in choices.items()}
            data = f(data, **args)
        return data, labels

    def set_random_choices(self):
        self.choices = []
        x_shape = self.dataset[0][0].shape
        N = len(self)
        for t in self.transforms:
            options = t.options(x_shape)
            x_shape = t.output_shape(x_shape) if hasattr(t, "output_shape") else x_shape
            self.choices.append(
                {k: np.random.choice(v, size=N) for (k, v) in options.items()}
            )


#####################
## dict utils
#####################

union = lambda *dicts: {k: v for d in dicts for (k, v) in d.items()}


def path_iter(nested_dict, pfx=()):
    for name, val in nested_dict.items():
        if isinstance(val, dict):
            yield from path_iter(val, (*pfx, name))
        else:
            yield ((*pfx, name), val)


#####################
## graph building
#####################

sep = "_"
RelativePath = namedtuple("RelativePath", ("parts"))
rel_path = lambda *parts: RelativePath(parts)


def build_graph(net):
    net = dict(path_iter(net))
    default_inputs = [[("input",)]] + [[k] for k in net.keys()]
    with_default_inputs = lambda vals: (
        val if isinstance(val, tuple) else (val, default_inputs[idx])
        for idx, val in enumerate(vals)
    )
    parts = (
        lambda path, pfx: tuple(pfx) + path.parts
        if isinstance(path, RelativePath)
        else (path,)
        if isinstance(path, str)
        else path
    )
    return {
        sep.join((*pfx, name)): (val, [sep.join(parts(x, pfx)) for x in inputs])
        for (*pfx, name), (val, inputs) in zip(
            net.keys(), with_default_inputs(net.values())
        )
    }


#####################
## training utils
#####################


@singledispatch
def cat(*xs):
    raise NotImplementedError


@singledispatch
def to_numpy(x):
    raise NotImplementedError


class PiecewiseLinear(namedtuple("PiecewiseLinear", ("knots", "vals"))):
    def __call__(self, t):
        return np.interp([t], self.knots, self.vals)[0]


class StatsLogger:
    def __init__(self, keys):
        self._stats = {k: [] for k in keys}

    def append(self, output):
        for k, v in self._stats.items():
            v.append(output[k].detach())

    def stats(self, key):
        return cat(*self._stats[key])

    def mean(self, key):
        return np.mean(to_numpy(self.stats(key)), dtype=float)

error_feedback={}
transfer_element={}
def layerwise_compressed_comm(
    model, world_size, method=None, K=None, V=None, qstates=None, extras="", anything=None
):
    model_size = len(list(model.parameters()))
    aggressive=False
   
    for index, layer in enumerate(model.parameters()):
        flatten_grad = layer.grad.data.view(-1)
        if index not in transfer_element:
            transfer_element[index]= [0 for i in range(flatten_grad.numel())]
        if method == "Topk" and K:
            # Top K
            flatten_grad_abs = flatten_grad.abs()
            thres, _ = flatten_grad_abs.kthvalue(
                math.ceil(flatten_grad.numel() * (1 - K))
            )  # send >= thres
            compress_grad = flatten_grad.clone()
            compress_grad[flatten_grad_abs < thres] = 0
        elif method == "Randomk" and K:
            # Random K
            mask = torch.randperm(flatten_grad.numel(), device=layer.device).lt(
                flatten_grad.numel() * K
            )
            compress_grad = flatten_grad.clone()
            compress_grad *= mask.float()

        elif method == "Randomk_layer":
            # Random K

            compression = []
            if extras[1] != "":
                compression = [int(i) for i in extras[1]]
            K_thres = [1, 0.01, 0.001]
            if flatten_grad.numel() < 10000:
                take = compression[0]
            else:
                take = compression[1]
            K = K_thres[take]
            if K != 1:
                mask = torch.randperm(flatten_grad.numel(), device=layer.device).lt(
                    flatten_grad.numel() * K
                )
                compress_grad = flatten_grad.clone()
                compress_grad *= mask.float()
            else:
                compress_grad = flatten_grad.clone()

        elif method == "Randomk_level":
            # Random K

            compression = []
            if extras[1] != "":
                compression = [int(i) for i in extras[1]]
            K_thres = [1, 0.01, 0.001]
            if index < model_size / 2:
                take = compression[0]
            else:
                take = compression[1]
            K = K_thres[take]
            if K != 1:
                mask = torch.randperm(flatten_grad.numel(), device=layer.device).lt(
                    flatten_grad.numel() * K
                )
                compress_grad = flatten_grad.clone()
                compress_grad *= mask.float()
            else:
                compress_grad = flatten_grad.clone()

        elif method == "Topk_layer_adaptative":
            # Top K
            compress_grad=topk_layer_adaptative(flatten_grad, extras, index, error_feedback,anything)
  
        elif method == "Topk_level_adaptative":
            compress_grad=topk_level_adaptative(flatten_grad, extras, index, error_feedback,model_size, anything)
            


        elif method == "Randomk_layer_adaptative":
            # Random K
            current_epoch = extras[0]
            if current_epoch<15:
                compression = [int(i) for i in extras[1][:2]]
            else:
                compression = [int(i) for i in extras[1][2:]]
            K_thres = [1, 0.01, 0.001]
            if flatten_grad.numel() < 10000:
                take = compression[0]
            else:
                take = compression[1]
            K = K_thres[take]
            if K != 1:
                mask = torch.randperm(flatten_grad.numel(), device=layer.device).lt(
                    flatten_grad.numel() * K
                )
                compress_grad = flatten_grad.clone()
                compress_grad *= mask.float()
            else:
                compress_grad = flatten_grad.clone()

        elif method == "Randomk_level_adaptative":
            # Random K
            current_epoch = extras[0]
            if current_epoch<15:
                compression = [int(i) for i in extras[1][:2]]
            else:
                compression = [int(i) for i in extras[1][2:]]
            K_thres = [1, 0.01, 0.001]
            if index < model_size / 2:
                take = compression[0]
            else:
                take = compression[1]
            K = K_thres[take]
            if K != 1:
                mask = torch.randperm(flatten_grad.numel(), device=layer.device).lt(
                    flatten_grad.numel() * K
                )
                compress_grad = flatten_grad.clone()
                compress_grad *= mask.float()
            else:
                compress_grad = flatten_grad.clone()

        elif method == 'Topk-level':
            compress_grad = topk_level(flatten_grad, extras, index, error_feedback, model_size, anything)
        elif method == 'Topk-layer':
            compress_grad = topk_layer(flatten_grad, extras, index, error_feedback, anything)
        elif method == "Topk-adaptative":
            compress_grad = topk_adaptative(flatten_grad, extras, index, error_feedback, model_size, anything)

        elif method == "Randomk_adaptive" and extras:
            # Random K
            current_epoch = extras[0]
            policy = [int(i) for i in extras[1]]
            compresions = [1, 0.1, 0.01, 0.001]
            K = compresions[policy[current_epoch // 10]]
            if K != 1:
                mask = torch.randperm(
                    flatten_grad.numel(), device=flatten_grad.device
                ).lt(flatten_grad.numel() * K)
                compress_grad = flatten_grad.clone()
                compress_grad *= mask.float()
            else:
                compress_grad = flatten_grad.clone()
        
        elif method == "Topk_level_batchalternative":
            current_epoch = extras[0]

            K_thres = [1, 0.01, 0.001]
            if aggressive:
                take = 2
            else:
                take = 1
            K = K_thres[take]
            if K != 1:
                flatten_grad_abs = flatten_grad.abs()
                thres, _ = flatten_grad_abs.kthvalue(
                    math.ceil(flatten_grad.numel() * (1 - K))
                )  # send >= thres
                compress_grad = flatten_grad.clone()
                compress_grad[flatten_grad_abs < thres] = 0
            else:
                compress_grad = flatten_grad.clone()
        elif method == "Topk_level_epochalternative":
            current_epoch = extras[0]+1

            K_thres = [1, 0.01, 0.001]

            take = 1+(current_epoch%2)
            K = K_thres[take]
            if K != 1:
                flatten_grad_abs = flatten_grad.abs()
                thres, _ = flatten_grad_abs.kthvalue(
                    math.ceil(flatten_grad.numel() * (1 - K))
                )  # send >= thres
                compress_grad = flatten_grad.clone()
                compress_grad[flatten_grad_abs < thres] = 0
            else:
                compress_grad = flatten_grad.clone()

        elif method == "Threshold_level_adaptative":
            compress_grad=threshold_level_adaptative(flatten_grad, extras, index, error_feedback, model_size)
            # transfer_element[extras[0]][-1].append(len(torch.nonzero(compress_grad)))
        elif method == "Threshold_layer_adaptative":
            compress_grad=threshold_layer_adaptative(flatten_grad, extras, index, error_feedback)
            # transfer_element[extras[0]][-1].append(len(torch.nonzero(compress_grad)))

        else:
            raise IOError("compress method not found")

        # non_zero_idx=compress_grad.nonzero().flatten().tolist()
        # for idx in non_zero_idx:
        #     transfer_element[index][idx]+=1
        # Perform All reduce
        dist.all_reduce(compress_grad)
        # if anything['memory']==1:
        #     error_feedback[index]=flatten_grad-compress_grad
        # average
        if compress_grad.numel() > 0:
            compress_grad /= float(world_size)
            flatten_grad.copy_(compress_grad)
        else:
            flatten_grad.zero_()
    aggressive=not aggressive


def entiremodel_compressed_comm(
    model, world_size, method=None, K=None, V=None, qstates=None, extras=None, anything=None
):
    # concat model grads into one flattened vector
    vec = []
    for param in model.parameters():
        # Ensure the parameters are located in the same device
        # param_device = _check_param_device(param, param_device)

        vec.append(param.grad.data.view(-1))

    flatten_grad = torch.cat(vec)

    if method == "Topk" and K:
        # Top K
        if K!=1:
            flatten_grad_abs = flatten_grad.abs()
            vals, _ = flatten_grad_abs.kthvalue(math.ceil(flatten_grad.numel() * (1 - K)))
            compress_grad = flatten_grad.clone()
            compress_grad[flatten_grad_abs < vals] = 0
        else:
            compress_grad = flatten_grad.clone()
    elif method == "Randomk" and K:
        # Random K
        mask = torch.randperm(flatten_grad.numel(), device=flatten_grad.device).lt(
            flatten_grad.numel() * K
        )
        compress_grad = flatten_grad.clone()
        compress_grad *= mask.float()
    elif method == "Thresholdv" and V:
        # Threshold V
        flatten_grad_abs = flatten_grad.abs()
        compress_grad = flatten_grad.clone()
        compress_grad[flatten_grad_abs < V] = 0
    elif method == "AdaptiveThreshold":
        # Adaptive Threshold
        H = flatten_grad * 2
        G_max = flatten_grad.abs().max()
        compress_grad = flatten_grad.clone()
        compress_grad[H.abs() < G_max] = 0
    elif method == "TernGrad":
        # TernGrad: Ternarized Gradient
        flatten_grad_abs = flatten_grad.abs()
        maxval = flatten_grad_abs.max()
        prob = flatten_grad_abs.div(maxval)  # [0, 1]
        binaryrand = (
            torch.rand(flatten_grad.shape, dtype=flatten_grad, device=flatten_grad)
            .lt_(prob)
            .float()
        )
        compress_grad = torch.mul(flatten_grad.sign() * maxval, binaryrand)
    elif method == "RandomDithering" and qstates:
        # Random Dithering - where QSGD is default which sets qstates=255
        norm = torch.norm(flatten_grad)  # 2-norm
        floor = torch.floor(
            flatten_grad.abs().div(norm) * qstates
            + torch.zeros(
                flatten_grad.shape, dtype=flatten_grad.dtype, device=flatten_grad.device
            ).uniform_(0, 1)
        )
        compress_grad = torch.mul(flatten_grad.sign() * norm, floor / qstates)
        compress_grad = torch.where(
            torch.isinf(compress_grad), torch.zeros_like(compress_grad), compress_grad
        )

    elif method == "Topk_adaptive" and extras:
        # Top K
        current_epoch = extras[0]
        policy = [int(i) for i in extras[1]]
        compresions = [1, 0.1, 0.01, 0.001]
        K = compresions[policy[current_epoch // 10]]
        if K != 1:
            flatten_grad_abs = flatten_grad.abs()
            vals, _ = flatten_grad_abs.kthvalue(
                math.ceil(flatten_grad.numel() * (1 - K))
            )
            compress_grad = flatten_grad.clone()
            compress_grad[flatten_grad_abs < vals] = 0
        else:
            compress_grad = flatten_grad.clone()
    elif method == "Randomk_adaptive" and extras:
        # Random K
        current_epoch = extras[0]
        policy = [int(i) for i in extras[1]]
        compresions = [1, 0.1, 0.01, 0.001]
        K = compresions[policy[current_epoch // 10]]
        if K != 1:
            mask = torch.randperm(flatten_grad.numel(), device=flatten_grad.device).lt(
                flatten_grad.numel() * K
            )
            compress_grad = flatten_grad.clone()
            compress_grad *= mask.float()
        else:
            compress_grad = flatten_grad.clone()
    else:
        compress_grad = flatten_grad.clone()

    # Perform All reduce (sum)
    dist.all_reduce(compress_grad)

    # average gradients
    if compress_grad.numel() > 0:
        compress_grad /= float(world_size)
        flatten_grad.copy_(compress_grad)
    else:
        flatten_grad.zero_()

    # Restore the gradient data into each indiviual layer of the model
    # Flag for the device where the parameter is located
    param_device = None
    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in model.parameters():
        # Ensure the parameters are located in the same device
        # param_device = _check_param_device(param, param_device)

        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        param.grad.data = (
            flatten_grad[pointer : pointer + num_param].view_as(param).data
        )
        # Increment the pointer
        pointer += num_param

criterion = nn.CrossEntropyLoss(reduction='none')
all_grads=[]
gradient_norms=[]
grad_infos={'mean':[], "median": [], 'std':[], 'mean-median': []}
abs_grad_infos={'mean':[], "median": [], 'std':[], 'mean-median': [], 'count_ls_mean': [], 
                'count_ls_0.05':[], 'count_ls_0.005':[], 'count_ls_first_mean':[], 'count_ls_first_median':[],
                 'per_iter_grad':[], 'max':[], 'count_ls_max/100':[], 'grad_norm':[],
                'grad_sum':[], '75_percentile':[],  'per_epoch_test_loss':[], 'per_epoch_test_acc':[],
                # 'learning_rate':[],'per_iter_median':[]
                }
first_median=-1
first_mean=-1
per_iter= 3200 #100*20

curr_iter=0
def run_batches(
    model,
    batches,
    training,
    world_size,
    optimizer_step=None,
    stats=None,
    compress=None,
    method=None,
    K=None,
    V=None,
    qstates=None,
    epoch=None,
    extras="",
    anything=None
):
    global first_mean, first_median, curr_iter
    stats = stats or StatsLogger(("loss", "correct"))
    model.train(training)
    for X, labels in batches:
        X=X.to(device)
        labels= labels.to(device)
        output = model(X)
        loss = criterion(output, labels)
        pred = output.data.max(1, keepdim=True)[1]
        correct = pred.eq(labels.data.view_as(pred))
        resul={'loss' : loss, 'correct': correct}
        stats.append(resul)
        if training:
            loss.sum().backward()
    # for batch in batches:        
    #     output = model(batch)
    #     stats.append(output)
    #     if training:
    #         output["loss"].sum().backward()
            # vec=[]
            # for param in model.parameters():
            #     vec.append(param.grad.data.view(-1))
            # flatten_grad = torch.cat(vec)
            # flatten_grad_abs = flatten_grad.abs()

            # gd_mean=flatten_grad.mean()
            # gd_median=flatten_grad.median()
            # gd_std=flatten_grad.std()
            # abs_gd_mean=flatten_grad_abs.mean()
            # abs_gd_median=flatten_grad_abs.median()
            # abs_gd_std=flatten_grad_abs.std()
            # abs_gd_max=flatten_grad_abs.max()

            # if first_median==-1:
            #     first_median=abs_gd_median
            #     first_mean=abs_gd_mean
            #     abs_grad_infos['first_mean']=first_mean.item()
            #     abs_grad_infos['first_median']=first_median.item()
            
            # grad_infos['mean'].append(gd_mean.item())
            # grad_infos['median'].append(gd_median.item())
            # grad_infos['std'].append(gd_std.item())
            # grad_infos['mean-median'].append((gd_mean-gd_median).item())

            # abs_grad_infos['mean'].append(abs_gd_mean.item())
            # abs_grad_infos['median'].append(abs_gd_median.item())
            # abs_grad_infos['std'].append(abs_gd_std.item())
            # abs_grad_infos['mean-median'].append((abs_gd_mean-abs_gd_median).item())
            
            # abs_grad_infos['count_ls_mean'].append(torch.sum(flatten_grad_abs <= abs_gd_mean).item())
            # abs_grad_infos['count_ls_0.05'].append(torch.sum(flatten_grad_abs <= 0.05).item())
            # abs_grad_infos['count_ls_0.005'].append(torch.sum(flatten_grad_abs <= 0.005).item())
            # abs_grad_infos['count_ls_first_mean'].append(torch.sum(flatten_grad_abs <= first_mean).item())
            # abs_grad_infos['count_ls_first_median'].append(torch.sum(flatten_grad_abs <= first_median).item())
            # abs_grad_infos['max'].append(abs_gd_max.item())
            # abs_grad_infos['count_ls_max/100'].append(torch.sum(flatten_grad_abs <= abs_gd_max/100).item())
            # abs_grad_infos['grad_norm'].append(torch.norm(flatten_grad_abs).item())
            # abs_grad_infos['grad_sum'].append(torch.sum(flatten_grad_abs).item())
            # abs_grad_infos['75_percentile'].append(torch.quantile(flatten_grad_abs, 0.75).item())


            # if curr_iter%per_iter==0:
            #     print(curr_iter)
            #     # abs_grad_infos['per_iter_median'].append(abs_gd_median.item())
            #     abs_grad_infos['per_iter_grad'].append(flatten_grad.tolist())
            # curr_iter+=1

            # abs_grad_infos['learning_rate'].append(anything['optimizer'].param_values()["lr"])
            # current_gradients = torch.cat([param.grad.flatten() for param in model.parameters()])

            # all_grads.append([torch.norm(param.grad).item() for name, param in model.named_parameters()])
            # output["loss"].sum().backward()
            if compress == "layerwise":
                layerwise_compressed_comm(
                    model, world_size, method, K, V, qstates, extras=extras, anything=anything
                )
            elif compress == "entiremodel":
                entiremodel_compressed_comm(
                    model, world_size, method, K, V, qstates, extras=extras, anything=anything
                )
            else:
                # no compression, communicate the gradients layer-by-layer
                for layer in model.parameters():
                    dist.all_reduce(layer.grad.data)
                    layer.grad.data /= world_size
            # previous_gradients = torch.cat([prev_grad.grad.flatten() for prev_grad in model.parameters()])
            # gradient_diff_norm = torch.norm(current_gradients - previous_gradients)/torch.norm(current_gradients)
            # gradient_norms.append(1-(gradient_diff_norm.item())**2)
            optimizer_step()
            model.zero_grad()
    return stats


def train_epoch(
    model,
    train_batches,
    test_batches,
    optimizer_step,
    timer,
    world_size,
    test_time_in_total=True,
    compress=None,
    method=None,
    K=None,
    V=None,
    qstates=None,
    extras="",
    anything=None
):
    train_stats, train_time = (
        run_batches(
            model,
            train_batches,
            True,
            world_size,
            optimizer_step,
            compress=compress,
            method=method,
            K=K,
            V=V,
            qstates=qstates,
            extras=extras,
            anything=anything
        ),
        timer(),
    )
    test_stats, test_time = run_batches(
        model, test_batches, False, world_size, extras=extras, anything=anything
    ), timer(test_time_in_total)
    abs_grad_infos['per_epoch_test_acc'].append(test_stats.mean("correct"))
    abs_grad_infos['per_epoch_test_loss'].append(test_stats.mean("loss"))
    return {
        "train_time": train_time,
        "train_loss": train_stats.mean("loss"),
        "train_acc": train_stats.mean("correct"),
        "test_time": test_time,
        "test_loss": test_stats.mean("loss"),
        "test_acc": test_stats.mean("correct"),
        "total_time": timer.total_time,
    }


def get_test_nb(file_suffix, json_dir='json_output'):
    test=0
    for file in os.listdir(json_dir):
        if file_suffix in file:
            test+=1
    return str(test)

device = None
def train(
    model,
    optimizer,
    train_batches,
    test_batches,
    epochs,
    master_address,
    world_size,
    rank,
    loggers=(),
    test_time_in_total=True,
    timer=None,
    compress=None,
    method=None,
    K=None,
    V=None,
    qstates=None,
    extras="",
    anything=None
):
    global device
    device = anything['device']
    dist.init_process_group(
        backend="gloo", init_method=master_address, world_size=world_size, rank=rank
    )
    timer = timer or Timer()
    for epoch in range(epochs):
        epoch_stats = train_epoch(
            model,
            train_batches,
            test_batches,
            optimizer.step,
            timer,
            world_size,
            test_time_in_total=test_time_in_total,
            compress=compress,
            method=method,
            K=K,
            V=V,
            qstates=qstates,
            extras=(epoch, extras),
            anything=anything
        )
        summary = union(
            {
                "epoch": epoch + 1,
                "lr": optimizer.param_values()["lr"] * train_batches.batch_size,
            },
            epoch_stats,
        )
        for logger in loggers:
            logger.append(summary)
            
    # torch.save(all_grads, 'all_gradients.pth')
    save_path='/home/mostapha.essoullami/lustre/opt_for_ml-um6p-sccs-en-tifcyqktztk/users/mostapha.essoullami/tests/gradient_infos/cifar100/grad+learningrate'
    
    # with open(save_path+'/test'+str(int(K))+'_nocomp_abs-grad_'+str(optimizer.param_values()["lr"])+'-learning_rates_100ep_res18_per900iter'+'.json', 'w') as json_file:
    #             json.dump(abs_grad_infos, json_file)
    # with open(save_path+'/extra-test'+str(int(K))+'_'+anything['extras']+'_nocomp_no-abs-grad_100ep_'+str(K)+'_node'+str(anything['gpu']-1)+'.json', 'w') as json_file:
    #             json.dump(grad_infos, json_file)
    # # if rank ==0:
    #     feedback='feedback' if anything['memory']==1 else 'nomemory'
    #     network= anything['network']
    #     json_file = network+'_'+ method+'_'+extras+'_'+feedback+'.json'
    #     sv_dir='saved_idx_'+method
    #     if not os.path.exists(sv_dir):
    #         os.mkdir(sv_dir)
    #     test= get_test_nb(json_file, sv_dir)
    #     json_file_path=sv_dir+'/test'+test+'_'+json_file
    # # Write the list to the JSON file

    #     with open(json_file_path, 'w') as json_file:
    #         json.dump([transfer_element], json_file)
      
    return summary


#####################
## network visualisation (requires pydot)
#####################
class ColorMap(dict):
    palette = (
        "bebada,ffffb3,fb8072,8dd3c7,80b1d3,fdb462,b3de69,fccde5,bc80bd,ccebc5,ffed6f,1f78b4,33a02c,e31a1c,ff7f00,"
        "4dddf8,e66493,b07b87,4e90e3,dea05e,d0c281,f0e189,e9e8b1,e0eb71,bbd2a4,6ed641,57eb9c,3ca4d4,92d5e7,b15928"
    ).split(",")

    def __missing__(self, key):
        self[key] = self.palette[len(self) % len(self.palette)]
        return self[key]


def make_pydot(nodes, edges, direction="LR", sep=sep, **kwargs):
    import pydot

    parent = lambda path: path[:-1]
    stub = lambda path: path[-1]

    class Subgraphs(dict):
        def __missing__(self, path):
            subgraph = pydot.Cluster(
                sep.join(path),
                label=stub(path),
                style="rounded, filled",
                fillcolor="#77777744",
            )
            self[parent(path)].add_subgraph(subgraph)
            return subgraph

    subgraphs = Subgraphs()
    subgraphs[()] = g = pydot.Dot(rankdir=direction, directed=True, **kwargs)
    g.set_node_defaults(shape="box", style="rounded, filled", fillcolor="#ffffff")
    for node, attr in nodes:
        path = tuple(node.split(sep))
        subgraphs[parent(path)].add_node(
            pydot.Node(name=node, label=stub(path), **attr)
        )
    for src, dst, attr in edges:
        g.add_edge(pydot.Edge(src, dst, **attr))
    return g


get_params = lambda mod: {
    p.name: getattr(mod, p.name, "?") for p in signature(type(mod)).parameters.values()
}


class DotGraph:
    colors = ColorMap()

    def __init__(self, net, size=15, direction="LR"):
        graph = build_graph(net)
        self.nodes = [
            (
                k,
                {
                    "tooltip": "%s %.1000r" % (type(n).__name__, get_params(n)),
                    "fillcolor": "#" + self.colors[type(n)],
                },
            )
            for k, (n, i) in graph.items()
        ]
        self.edges = [(src, k, {}) for (k, (n, i)) in graph.items() for src in i]
        self.size, self.direction = size, direction

    def dot_graph(self, **kwargs):
        return make_pydot(
            self.nodes, self.edges, size=self.size, direction=self.direction, **kwargs
        )

    def svg(self, **kwargs):
        return self.dot_graph(**kwargs).create(format="svg").decode("utf-8")

    try:
        import pydot

        def _repr_svg_(self):
            return self.svg()

    except ImportError:

        def __repr__(self):
            return "pydot is needed for network visualisation"


walk = lambda dict_, key: walk(dict_, dict_[key]) if key in dict_ else key


def remove_by_type(net, node_type):
    # remove identity nodes for more compact visualisations
    graph = build_graph(net)
    remap = {k: i[0] for k, (v, i) in graph.items() if isinstance(v, node_type)}
    return {
        k: (v, [walk(remap, x) for x in i])
        for k, (v, i) in graph.items()
        if not isinstance(v, node_type)
    }
