import math
import torch
# error feedback 1->6.5 ; 5->0.09
V_min = 1.5 #--6.5%
V_max= 4.5 # -- 0.09%
def threshold_layer_adaptative(flatten_grad, extras,index, error_feedback):
    current_epoch = extras[0]
    if current_epoch<11:
        compression = [int(i) for i in extras[1][:2]]
    else:
        compression = [int(i) for i in extras[1][2:]]
    
    V_thres = [0, V_min, V_max]
    if flatten_grad.numel() < 10000:
        take = compression[0]
    else:
        take = compression[1]
    V = V_thres[take]
    if V!=0:
        flatten_grad_abs = flatten_grad.abs()
        # if index in error_feedback :
        #     flatten_grad_abs +=error_feedback[index].abs()
        compress_grad = flatten_grad.clone()
        compress_grad[flatten_grad_abs < V] = 0
    else:
        compress_grad = flatten_grad.clone()
        # if index in error_feedback:
        #     compress_grad-=error_feedback[index]
    return compress_grad

def threshold_level_adaptative(flatten_grad, extras,index, error_feedback, model_size):
    current_epoch = extras[0]
    if current_epoch<11:
        compression = [int(i) for i in extras[1][:2]]
    else:
        compression = [int(i) for i in extras[1][2:]]
    
    V_thres = [0, V_min, V_max]
    if index < model_size / 2:
        take = compression[0]
    else:
        take = compression[1]
    V = V_thres[take]
    if V!=0:
        
        flatten_grad_abs = flatten_grad.abs()
        # if index in error_feedback :
        #     flatten_grad_abs +=error_feedback[index].abs()
        compress_grad = flatten_grad.clone()
        compress_grad[flatten_grad_abs < V] = 0
    else:
        compress_grad = flatten_grad.clone()
        # if index in error_feedback:
        #     compress_grad-=error_feedback[index]
    return compress_grad

def topk_layer_adaptative(flatten_grad, extras,index, error_feedback, anything):
    current_epoch = extras[0]
    if current_epoch<15:
        compression = [int(i) for i in extras[1][:2]]
        # K_thres= [1 , anything['k_min0'], anything['k_max0']]
    else:
        compression = [int(i) for i in extras[1][2:]]
        # K_thres= [1 , anything['k_min1'], anything['k_max1']]
    
    if flatten_grad.numel() < 10000:
        take = compression[0]
    else:
        take = compression[1]
    K_thres= [1 , 0.01, 0.001]
    K = K_thres[take]
    if K != 1:
        flatten_grad_abs = flatten_grad.abs()
        if index in error_feedback and anything['memory'] ==1:
            flatten_grad_abs +=error_feedback[index].abs()
        thres, _ = flatten_grad_abs.kthvalue(
            math.ceil(flatten_grad.numel() * (1 - K))
        )  # send >= thres
        compress_grad = flatten_grad.clone()
        compress_grad[flatten_grad_abs < thres] = 0
    else:
        compress_grad = flatten_grad.clone()
    return compress_grad

def topk_level_adaptative(flatten_grad, extras,index, error_feedback, model_size, anything):
    current_epoch = extras[0]
    # print(extras)
    if current_epoch<15:
        compression = [int(i) for i in extras[1][:2]]
        # K_thres= [1 , anything['k_min0'], anything['k_max0']]
    else:
        compression = [int(i) for i in extras[1][2:]]
        # K_thres= [1 , anything['k_min1'], anything['k_max1']]

    if index < model_size / 2:
        take = compression[0]
    else:
        take = compression[1]
    K_thres= [1 , 0.1, 0.01]
    K = K_thres[take]
    if K != 1:
        flatten_grad_abs = flatten_grad.abs()
        # if index in error_feedback and anything['memory'] ==1:
        #     flatten_grad_abs +=error_feedback[index].abs()
        thres, _ = flatten_grad_abs.kthvalue(
            math.ceil(flatten_grad.numel() * (1 - K))
        )  # send >= thres
        compress_grad = flatten_grad.clone()
        compress_grad[flatten_grad_abs < thres] = 0
    else:
        compress_grad = flatten_grad.clone()
    return compress_grad




def topk_layer(flatten_grad, extras,index, error_feedback, anything):
    current_epoch = extras[0]
    compression = [int(i) for i in extras[1][:2]]
    K_thres= [1 , 0.01, 0.001]
    
    if flatten_grad.numel() < 10000:
        take = compression[0]
        # K= anything['k_min0']
    else:
        take = compression[1]
        # K= anything['k_max0']
    K = K_thres[take]
    if K != 1:
        flatten_grad_abs = flatten_grad.abs()
        # if index in error_feedback and anything['memory'] ==1:
        #     flatten_grad_abs +=error_feedback[index].abs()
        thres, _ = flatten_grad_abs.kthvalue(
            math.ceil(flatten_grad.numel() * (1 - K))
        )  # send >= thres
        compress_grad = flatten_grad.clone()
        compress_grad[flatten_grad_abs < thres] = 0
    else:
        compress_grad = flatten_grad.clone()
    return compress_grad

def topk_level(flatten_grad, extras,index, error_feedback, model_size, anything):
    current_epoch = extras[0]
    compression = [int(i) for i in extras[1][:2]]
    K_thres= [1 , 0.01, 0.001]

    if index < model_size / 2:
        take = compression[0]
        # K= anything['k_min0']
    else:
        take = compression[1]
        # K= anything['k_max0']
    K = K_thres[take]
    if K != 1:
        flatten_grad_abs = flatten_grad.abs()
        # if index in error_feedback and anything['memory'] ==1:
        #     flatten_grad_abs +=error_feedback[index].abs()
        thres, _ = flatten_grad_abs.kthvalue(
            math.ceil(flatten_grad.numel() * (1 - K))
        )  # send >= thres
        compress_grad = flatten_grad.clone()
        compress_grad[flatten_grad_abs < thres] = 0
    else:
        compress_grad = flatten_grad.clone()
    return compress_grad


def topk_adaptative(flatten_grad, extras,index, error_feedback, model_size, anything):
    current_epoch = extras[0]
    compression = [int(i) for i in extras[1][:2]]
    K_thres= [1 , 0.01, 0.001]
    if current_epoch <= 45:
        take = compression[0]
        # K= anything['k_min0']
    else:
        take = compression[1]
        # K= anything['k_max0']
    K = K_thres[take]
    if K != 1:
        flatten_grad_abs = flatten_grad.abs()
        # if index in error_feedback and anything['memory'] ==1:
        #     flatten_grad_abs +=error_feedback[index].abs()
        thres, _ = flatten_grad_abs.kthvalue(
            math.ceil(flatten_grad.numel() * (1 - K))
        )  # send >= thres
        compress_grad = flatten_grad.clone()
        compress_grad[flatten_grad_abs < thres] = 0
    else:
        compress_grad = flatten_grad.clone()
    return compress_grad