# python classify/wll.py --model downmodel/yolov5s-cls.pt --data cifar100 --epochs 5 --img 224 --batch 8
import os
import sys
from pathlib import Path
import torch
import torch.distributed as dist
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
from torch.cuda import amp #可能会导致精度损失
from sklearn.metrics import accuracy_score
# 我添加的
import types
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import math
# 
from copy import deepcopy
import pkg_resources as pkg

####################### 数据集一 ##################################

base_dir = r'../datasets/me/4weather'  #data目录 

# 1、获取数据

# 目录
base_dir = r'../datasets/me/4weather'  #data目录 
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

#增强方法转换，下面调用此方法，先定义数据增强方法
transform = transforms.Compose([
                  transforms.Resize((224, 224)),            #所有图片转统一大小，大一点会更好，保存模型信息更多
                  transforms.ToTensor(),                    #1、转成Tensor格式，2、转换0到1之间,归一化，3、会改变图片的channel放在第一维度上：【batch， channel， hight， width】，所以不能用其它的方式如：plt加载图片
                  transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                       std=[0.5, 0.5, 0.5]) #标准化，不是必须的。 mean图片均值，std方差，要了解数据集才行
])
# 训练数据-dataset
trainDataset = torchvision.datasets.ImageFolder( # 从图片文件夹中加载数据-返回dataset格式
               train_dir,
               transform=transform
)
# 验证数据-dataset
validDataset = torchvision.datasets.ImageFolder( # 从图片文件夹中加载数据-返回dataset格式
               test_dir,
               transform=transform
)
# 查看分类
trainDataset.classes #['cloudy', 'rain', 'shine', 'sunrise'] #查看文件夹，也卞是分类
trainDataset.class_to_idx #{'cloudy': 0, 'rain': 1, 'shine': 2, 'sunrise': 3} #类别编号
len(trainDataset), len(validDataset)#(900, 225)

#dataLoader - 数据切片进行自动分页
batch = 8
trainRow = torch.utils.data.DataLoader(trainDataset, batch_size=batch, shuffle=True)
validRow = torch.utils.data.DataLoader(validDataset, batch_size=batch) #验证测试数据集可以设置大些，因为不做反向传播，占用内存显存比较小

imgs, labels = next(iter(trainRow))
# print('imgs:',imgs.shape)   #imgs: torch.Size([8, 3, 224, 224])   torch.Size([8, 3, 224, 224])
# print('labels:',labels.shape) #torch.Size([8])                    torch.Size([8])
# exit()

trainloader = trainRow
testloader = validRow

################ 更新模型分类数 reshape_classifier_output #######################

class Classify(nn.Module):
    # YOLOv5 classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self,
                 c1,
                 c2,
                 k=1,
                 s=1,
                 p=None,
                 g=1,
                 dropout_p=0.0):  # ch_in, ch_out, kernel, stride, padding, groups, dropout probability
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, autopad(k, p), g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=dropout_p, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, 1)
        return self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))

#
def reshape_classifier_output(model, n=1000):
    # 如果需要，将TorchVision分类模型更新为类计数“n” / Update a TorchVision classification model to class count 'n' if required   
    name, m = list((model.model if hasattr(model, 'model') else model).named_children())[-1]  # last module
    
    if isinstance(m, Classify):  # YOLOv5 Classify() head  / yolo5改分类
        if m.linear.out_features != n:
            m.linear = nn.Linear(m.linear.in_features, n)
    
    elif isinstance(m, nn.Linear):  # ResNet, EfficientNet  / ResNet 改分类
        if m.out_features != n:
            setattr(model, name, nn.Linear(m.in_features, n))

    elif isinstance(m, nn.Sequential): #其它方式改分类数
        types = [type(x) for x in m]
        if nn.Linear in types:
            i = types.index(nn.Linear)  # nn.Linear index
            if m[i].out_features != n:
                m[i] = nn.Linear(m[i].in_features, n)
        elif nn.Conv2d in types:
            i = types.index(nn.Conv2d)  # nn.Conv2d index
            if m[i].out_channels != n:
                m[i] = nn.Conv2d(m[i].in_channels, n, m[i].kernel_size, m[i].stride, bias=m[i].bias is not None)


###############  ModelEMA 使模型滑动平均 ########################
'''
EMA，指数移动平均，常用于更新模型参数、梯度等。
EMA的优点是能提升模型的鲁棒性（融合了之前的模型权重信息）
其实EMA就是把每一次梯度下降更新后的权重值和前一次的权重值进行了一种“联系”，这种联系 让我们的模型更新还需要看上一次更新的脸色，没那么“随意”。
说明：
    https://blog.csdn.net/niuxuerui11/article/details/129123868 
    https://zhuanlan.zhihu.com/p/360842167
    https://blog.csdn.net/qq_38964360/article/details/131482442
'''

def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model

def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)
#
class ModelEMA:
    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        self.updates += 1
        d = self.decay(self.updates)

        msd = de_parallel(model).state_dict()  # model state_dict
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:  # true for FP16 and FP32
                v *= d
                v += (1 - d) * msd[k].detach()
        # assert v.dtype == msd[k].dtype == torch.float32, f'{k}: EMA {v.dtype} and model {msd[k].dtype} must be FP32'

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)


################## 模型优化器 smart_optimizer #####################
    
def smart_optimizer(model, name='Adam', lr=0.001, momentum=0.9, decay=1e-5):
    # YOLOv5 3-param group optimizer: 0) weights with decay, 1) weights no decay, 2) biases no decay
    g = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
    for v in model.modules():
        for p_name, p in v.named_parameters(recurse=0):
            if p_name == 'bias':  # bias (no decay)
                g[2].append(p)
            elif p_name == 'weight' and isinstance(v, bn):  # weight (no decay)
                g[1].append(p)
            else:
                g[0].append(p)  # weight (with decay)

    if name == 'Adam':
        optimizer = torch.optim.Adam(g[2], lr=lr, betas=(momentum, 0.999))  # adjust beta1 to momentum
    elif name == 'AdamW':
        optimizer = torch.optim.AdamW(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    elif name == 'RMSProp':
        optimizer = torch.optim.RMSprop(g[2], lr=lr, momentum=momentum)
    elif name == 'SGD':
        optimizer = torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
    else:
        raise NotImplementedError(f'Optimizer {name} not implemented.')

    optimizer.add_param_group({'params': g[0], 'weight_decay': decay})  # add g0 with weight_decay
    optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})  # add g1 (BatchNorm2d weights)
    
    print(f"{'optimizer:'} {type(optimizer).__name__}(lr={lr}) with parameter groups "
                f'{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias')
    return optimizer

############### 加载模型 ########################


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output

def attempt_load(weights, device=None, inplace=True, fuse=True):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    from myModel import Detect, Model

    # # 
    # import types
    # opt = types.SimpleNamespace()
    # opt.batch_size=8
    # opt.batch_size=1
    # opt.cfg='modelXml/yolov5s.yaml'
    # opt.device=''
    # opt.line_profile=False
    # opt.profile=False
    # opt.test=False
    # device = 'cpu'#select_device(opt.device)    
    # model = Model(opt.cfg).to(device)

    # # print(model)
    # return model    

    model = Ensemble()    
    for w in weights if isinstance(weights, list) else [weights]:
        
        # print(w) #downmodel/yolov5s-cls.pt  
        # w = Path(str(w).strip().replace("'", ''))#我添加的
        # print('--------',w) #downmodel/yolov5s-cls.pt
        ckpt = torch.load(w, map_location='cpu')  # cpu 方式加载 load - 取消自动加载： torch.load(attempt_download(w), map_location='cpu') 
        # print('--------',ckpt)
       
        ckpt = (ckpt.get('ema') or ckpt['model']).to(device).float()  # FP32 model - 使模型滑动平均

        # Model compatibility updates
        if not hasattr(ckpt, 'stride'):
            ckpt.stride = torch.tensor([32.])
        if hasattr(ckpt, 'names') and isinstance(ckpt.names, (list, tuple)):
            ckpt.names = dict(enumerate(ckpt.names))  # convert to dict

        model.append(ckpt.fuse().eval() if fuse and hasattr(ckpt, 'fuse') else ckpt.eval())  # model in eval mode

    # Module updates
    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
            m.inplace = inplace
            if t is Detect and not isinstance(m.anchor_grid, list):
                delattr(m, 'anchor_grid')
                setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
        elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model
    if len(model) == 1:
        return model[-1]

    # Return detection ensemble
    print(f'Ensemble created with {weights}\n')
    for k in 'names', 'nc', 'yaml':
        setattr(model, k, getattr(model[0], k))
    model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
    assert all(model[0].nc == m.nc for m in model), f'Models have different class counts: {[m.nc for m in model]}'
    return model

######################### 训练 ######################################

RANK = int(os.getenv('RANK', -1))

def train(opt, device):
       
    cuda = device.type != 'cpu' #False    
    
    # 方式一、Model    
    model = attempt_load(opt.model, device='cpu', fuse=False)

    # 方式二、分类器通用调用接口:https://blog.csdn.net/weixin_45209433/article/details/112478806
    # model = torchvision.models.__dict__[opt.model](weights='IMAGENET1K_V1' if opt.pretrained else None)

    reshape_classifier_output(model, opt.nc)  # 修改类别数 / update class count    

    for m in model.modules():
        if not opt.pretrained and hasattr(m, 'reset_parameters'):
            m.reset_parameters()
        if isinstance(m, torch.nn.Dropout) and opt.dropout is not None:
            m.p = opt.dropout  # set dropout
    for p in model.parameters():
        p.requires_grad = True  # for training
    
    model = model.to(device)
    # model.names = trainloader.dataset.classes  # attach class names  
    # # print('___',trainloader.dataset.classes ) #我的 ['cloudy', 'rain', 'shine', 'sunrise']    
    # model.transforms = transform #我禁用了：testloader.dataset.torch_transforms  # attach inference transforms
   

    # 预训练    
    epoches=60

    # 定义交叉熵损失函数和优化器
    Loss = []
    # loss_fn = return nn.CrossEntropyLoss(label_smoothing=opt.label_smoothing)    # nn.CrossEntropyLoss()
    # optimizer = smart_optimizer(model, opt.optimizer, opt.lr0, momentum=0.9, decay=opt.decay) #torch.optim.Adam(model.parameters(), lr=opt.lr0, betas=(0.9, 0.999)) 
    # scaler = amp.GradScaler(enabled=cuda)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = smart_optimizer(model, opt.optimizer, opt.lr0, momentum=0.9, decay=opt.decay)

    # 步长
    lrf = 0.01  # final lr (fraction of lr0)   
    lf = lambda x: (1 - x / epoches) * (1 - lrf) + lrf  # linear
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    
    # EMA
    ema = ModelEMA(model) if RANK in {-1, 0} else None
    

    # 模型训练
    for epoch in range(epoches):
        train_loss = []  
        predict = []
        image_label = []
        
        for x, y in trainloader:

            # print(x.shape) #torch.Size([8, 3, 224, 224])
            # print(y.shape) #torch.Size([8])
            
            y_pred = model(x.to(device))            #预测结果
            # print(y_pred[0].shape)#torch.Size([8, 3, 28, 28, 85])  这样才对： torch.Size([1000])
            # exit()

            loss = loss_fn(y_pred, y.to(device))    #预测值，真实值        
            optimizer.zero_grad()                   #默认梯度为0
            loss.backward()                         #损失反向传播放        
            optimizer.step()                        #优化模型
            
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients

            # optimizer.zero_grad()                   #默认梯度为0
            # scaler.scale(loss).backward()           #损失反向传播放             
            # scaler.unscale_(optimizer)              # unscale gradients            
            # scaler.step(optimizer)                  #优化模型
            # scaler.update()            
            
            if ema:
                # print('---------------- ema ------------------')
                ema.update(model)

            
            # 损失
            train_loss.append(loss.detach().cpu().item())
            # tloss = (tloss * i + loss.item()) / (i + 1)  # update mean losses
            # 
            
            # 网络预测
            predict.extend(y_pred.argmax(dim=1).detach().cpu().tolist())
            image_label.extend(y.detach().cpu().tolist())
        
        assert len(predict) == len(image_label)
        acc = accuracy_score(image_label, predict)


        # 每个Epoch损失平均值
        Loss.append(np.mean(np.array(train_loss)))
        print('Epoch:', epoch, 'Model Loss:', Loss[-1], 'Accuracy: ', acc)
        

        # 步长
        # scheduler.step()

        # # 模型保存
        # torch.save(model.state_dict(), './output.pth', _use_new_zipfile_serialization=False)

    import matplotlib.pyplot as plt
    plt.title('The Train Loss of Vision Transformer')
    plt.plot(range(len(Loss)), Loss, label='Loss')
    plt.legend()
    plt.show()

    

# 
opt = types.SimpleNamespace()
opt.batch_size=8
opt.cache=None
opt.cutoff=None
opt.data='cifar100'
opt.decay=5e-05
opt.device=''
opt.dropout=None
opt.epochs=5
opt.exist_ok=False
opt.imgsz=224
opt.label_smoothing=0.1
opt.local_rank=-1
opt.lr0=0.001
opt.model='downmodel/yolov5s-cls.pt'
opt.name='exp'
opt.nosave=False
opt.optimizer='Adam'
opt.pretrained=True
opt.project='runs/train-cls'
opt.seed=0
opt.verbose=False
opt.workers=8
opt.nc = 4  # 分类数 / number of classes
opt.save_dir='aa.pt'


# Train
device = torch.device('cpu')
train(opt, device)

