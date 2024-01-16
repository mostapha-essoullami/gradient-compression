from time import sleep
import googlenet, senet
from core import *
from torch_backend import *
import argparse
import os.path
from vgg16 import *
from alexnet import *
import resnet
import resnet_img
from device_controller import Device_Singleton
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--log_dir', type=str, default='.')
parser.add_argument('--master_address', '-m', type=str, default='tcp://localhost:2221')
parser.add_argument('--rank', '-r', type=int, default=0)
parser.add_argument('--world_size', '-w', type=int, default=1)
parser.add_argument('--network', '-n', type=str, default='Resnet18')
parser.add_argument('--compress', '-c', type=str, default='layerwise')
parser.add_argument('--method', type=str, default='Topk-layer')
parser.add_argument('--ratio', '-K', type=float, default=0)
parser.add_argument('--threshold', '-V', type=float, default=0.001)
parser.add_argument('--qstates', '-Q', type=int, default=0)
parser.add_argument('--momentum', type=float, default=0.0)
parser.add_argument('--extras', type=str, default="12")
parser.add_argument('--memory', type=int, default=0)
parser.add_argument('--gpu', type=int, default=0)
#Network definition
def conv_bn(c_in, c_out, bn_weight_init=1.0, **kw):
    return {
        'conv': nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False),
        'bn': batch_norm(c_out, bn_weight_init=bn_weight_init, **kw),
        'relu': nn.ReLU(True)
    }

def conv_bn_stride(c_in, c_out, bn_weight_init=1.0, **kw):
    return {
        'conv': nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1, bias=False),
        'bn': batch_norm(c_out, bn_weight_init=bn_weight_init, **kw),
        'relu': nn.ReLU(True)
    }

def residual(c, **kw):
    return {
        'in': Identity(),
        'res1': conv_bn(c, c, **kw),
        'res2': conv_bn(c, c, **kw),
        'add': (Add(), [rel_path('in'), rel_path('res2', 'relu')]),
    }

def basic_resnet9(channels, weight,  pool, **kw):
    return {
        'prep': conv_bn(3, channels['prep'], **kw),
        'layer1': dict(conv_bn(channels['prep'], channels['layer1'], **kw), pool=pool),
        'layer2': dict(conv_bn(channels['layer1'], channels['layer2'], **kw), pool=pool),
        'layer3': dict(conv_bn(channels['layer2'], channels['layer3'], **kw), pool=pool),
        'pool': nn.MaxPool2d(4),
        'flatten': Flatten(),
        'linear': nn.Linear(channels['layer3'], 10, bias=False),
        'classifier': Mul(weight),
    }

def basic_alexnet(channels, weight,  pool, **kw):
    return {
        'prep': dict(conv_bn_stride(3, channels['prep'], **kw), pool=pool),
        'layer1': dict(conv_bn(channels['prep'], channels['layer1'], **kw), pool=pool),
        'layer2': conv_bn(channels['layer1'], channels['layer2'], **kw),
        'layer3': conv_bn(channels['layer2'], channels['layer3'], **kw),
        'layer4': dict(conv_bn(channels['layer3'], channels['layer4'], **kw), pool=pool),
        'pool': nn.MaxPool2d(2),
        'flatten': Flatten(),
        'linear': nn.Linear(channels['layer4'], 10, bias=False),
        'classifier': Mul(weight),
    }

def resnet9(channels=None, weight=0.125, pool=nn.MaxPool2d(2), extra_layers=(), res_layers=('layer1', 'layer3'), **kw):
    channels = channels or {'prep': 64, 'layer1': 128, 'layer2': 256, 'layer3': 512}
    n = basic_resnet9(channels, weight, pool, **kw)
    for layer in res_layers:
        n[layer]['residual'] = residual(channels[layer], **kw)
    for layer in extra_layers:
        n[layer]['extra'] = conv_bn(channels[layer], channels[layer], **kw)
    return n

def alexnet(channels=None, weight=0.125, pool=nn.MaxPool2d(2), extra_layers=(), **kw):
    channels = channels or {'prep': 64, 'layer1': 192, 'layer2': 384, 'layer3': 256, 'layer4': 256}
    n = basic_alexnet(channels, weight, pool, **kw)
    return n

losses = {
    'loss':  (nn.CrossEntropyLoss(reduction='none'), [('classifier',), ('target',)]),
    'correct': (Correct(), [('classifier',), ('target',)]),
}

class TSVLogger():
    def __init__(self):
        self.log = ['epoch\thours\ttop1Accuracy']
    def append(self, output):
        epoch, hours, acc = output['epoch'], output['total_time']/3600, output['test_acc']*100
        self.log.append(f'{epoch}\t{hours:.8f}\t{acc:.2f}')
    def __str__(self):
        return '\n'.join(self.log)

def main():

    # torch.manual_seed(42)
    # torch.cuda.manual_seed(42)
    args = parser.parse_args()
    
    # update_device(args.rank+1)
    update_device(args.gpu)
    memory= args.memory
    device = get_device()
    anything={ "memory": memory,'network': args.network, 'device': device, "gpu": args.gpu, 'extras': args.extras}
    
    print(device)
    print(args)
    batch_size = 256*2 #412 #512*2
    
    print('Starting timer',device)
    tt= lambda : torch.cuda.synchronize(device=device)
    timer = Timer(synch=torch.cuda.synchronize)
    timer = Timer() 
    print('Downloading datasets')
    # dataset = cifar10(args.data_dir)    
    # train_transforms = [Crop(32, 32), FlipLR(), Cutout(8, 8)]

    # print('Preprocessing training data')
    # train_set = list(zip(transpose(normalise(pad(dataset['train']['data'], 4))), dataset['train']['labels']))
    # print(f'Finished in {timer():.2} seconds')
    # print('Preprocessing test data')
    # test_set = list(zip(transpose(normalise(dataset['test']['data'])), dataset['test']['labels']))
    # print(f'Finished in {timer():.2} seconds')
    # train_batches = Batches(Transform(train_set, train_transforms), batch_size, shuffle=True, set_random_choices=True, drop_last=True)
    # test_batches = Batches(test_set, batch_size, shuffle=False, drop_last=False)
    train_batches, test_batches= cifar100(args.data_dir, batch_size)
    # train_batches, test_batches =imagenet(args.data_dir, batch_size)
    print('train',len(train_batches), 'test', len(test_batches))
    epochs = 100 #30
    lr_schedule = PiecewiseLinear([0, 5, epochs], [0, 0.4, 0])
    # train_transforms = [Crop(32, 32), FlipLR(), Cutout(8, 8)]
    print(device)
    #choose the neural network
    if args.network == 'Resnet9':
        model = torch.load(args.network+'.pth').to(device)
        # model = Network(union(resnet9(), losses)).to(device)
    elif args.network == 'Resnet18':
        # model= resnet.resnet20()
        model=torchvision.models.resnet18()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 100) 
        model = model.to(device)
    elif args.network == 'Googlenet':
        # model= resnet.resnet20()
        # model=googlenet.GoogLeNet()
        # model=senet.SENet18()
        # print(model)
        model=torchvision.models.efficientnet_v2_l()
        # print(model)
        model.classifier = nn.Sequential(nn.Dropout(p=0.4, inplace=True),
                  nn.Linear(in_features=model.classifier[1].in_features, out_features=100, bias=True))

        # num_ftrs = model.fc.in_features
        # model.fc = nn.Linear(num_ftrs, 100) 
        # print(model)
        model = model.to(device)
    elif args.network == 'Resnet50':
        model=torchvision.models.resnet50()
        # model=torchvision.models.resnext50_32x4d()
        # num_ftrs = model.fc.in_features
        # model.fc = nn.Linear(num_ftrs, 200) 
        # model = resnet_img.resnet50()
        model = model.to(device)
    elif args.network == 'Alexnet':
        model = torch.load(args.network+'.pth').to(device)
        # model = Network(union(alexnet(), losses)).to(device)
    elif args.network == 'Alexnet1':
        model = AlexNet().to(device)
    elif args.network == 'vgg16':
        model =vgg16model.to(device)
    # torch.save(model, args.network+'.pth' )
    # return True
    small=0
    total_params=0
    l=[]
    # print(model)
    for name, param in model.named_parameters():
        if param.requires_grad:
            layer_params = param.numel()
            l.append(layer_params)
            if layer_params <10000:
                small+=layer_params
            total_params += layer_params
            print(f"Layer: {name}, Parameters: {layer_params}")

    print(f"Total Parameters: {total_params}")
    print('small',small,'large',total_params-small)
    print('first',sum(l[:len(l)//2]),'last',sum(l[len(l)//2:]))
    print('Warming up cudnn on random inputs on device', next(model.parameters()).device)
    print('Warming up cudnn on random inputs on device', next(model.parameters()).device)

    # for size in [batch_size, len(dataset['test']['labels']) % batch_size]:
    #     warmup_cudnn(model, size)



    TSV = TSVLogger()
    # train_batches = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
    #                                             pin_memory=True, shuffle=True
    #     )
    # test_batches = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
    #                                             pin_memory=True, shuffle=False
    #     )
    # train_batches = Batches(Transform(train_set, train_transforms), batch_size, shuffle=True, set_random_choices=True, drop_last=True)
    # test_batches = Batches(test_set, batch_size, shuffle=False, drop_last=False)

    lr = lambda step: lr_schedule(step/len(train_batches))/batch_size

    #use Nestrov's momentum SGD or vanillia SGD
    if args.momentum > 0:
        opt = SGD(trainable_params(model), lr=lr, momentum=args.momentum, weight_decay=5e-4*batch_size, nesterov=True)
    else:
        opt = SGD(trainable_params(model), lr=lr, weight_decay=5e-4*batch_size)
        # opt = SGD(trainable_params(model), lr=lr, momentum=0.5, weight_decay=5e-4*batch_size)
        # opt = SGD(trainable_params(model), lr=0.0005, momentum=0.5, weight_decay=5e-4*batch_size)
    anything['optimizer']=opt
    train(model, opt, train_batches, test_batches, epochs, args.master_address, args.world_size, args.rank, loggers=(TableLogger(), TSV), timer=timer, test_time_in_total=False, compress=args.compress, method=args.method, K=args.ratio, V=args.threshold, qstates=args.qstates, extras= args.extras, anything=anything)

    results_file='_'.join([args.network, str(args.compress), str(args.method), 'epoch-'+str(epochs), str(args.extras)])+'.logs'
    with open(os.path.join(os.path.expanduser(args.log_dir), results_file), 'a') as f:
        f.write(str(TSV))
    # feedback='feedback' if anything['memory']==1 else 'nomemory'
    # sv_dir='save_'+args.method
    # if not os.path.exists(sv_dir):
    #     os.mkdir(sv_dir)
    # save_path= 'sv_dir'+'/'+anything['network']+'_'+ args.method+'_'+anything['extras']+'_'+feedback+'.pth'
    # torch.save(model, save_path )

main()
