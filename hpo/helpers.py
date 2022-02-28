import json
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import models, transforms
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from efficientnet_pytorch import EfficientNet
from PIL import Image
from trivialaugment import aug_lib

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

ARCH = ['resnet18', 'resnet34', 'resnet50', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4'] 

def initialize_model(architecture, num_classes, pretrained = True):
    model = None
    
    if architecture == 'resnet18':
        model = models.resnet18(pretrained = pretrained)
        model.fc = nn.Linear(512, num_classes)
    elif architecture == 'resnet34':
        model = models.resnet34(pretrained = pretrained)
        model.fc = nn.Linear(512, num_classes)
    elif architecture == 'resnet50':
        model = models.resnet50(pretrained = pretrained)
        model.fc = nn.Linear(2048, num_classes)
    elif architecture == 'wide_resnet50_2':
        model = models.wide_resnet50_2(pretrained=pretrained)
        model.fc = nn.Linear(2048, num_classes)
    elif architecture == 'resnext50_32x4d':
        model = models.resnext50_32x4d(pretrained=pretrained)
        model.fc = nn.Linear(2048, num_classes)
    elif architecture == 'densenet121':
        model = models.densenet121(pretrained=pretrained)
        model.classifier = nn.Linear(1024, num_classes)
    elif architecture == 'densenet161':
        model = models.densenet161(pretrained=pretrained)
        model.classifier = nn.Linear(2208, num_classes)
    elif architecture == 'densenet169':
        model = models.densenet169(pretrained=pretrained)
        model.classifier = nn.Linear(1664, num_classes)
    elif architecture == 'densenet201':
        model = models.densenet201(pretrained=pretrained)
        model.classifier = nn.Linear(1920, num_classes)
    elif architecture == 'mnasnet':
        model = models.mnasnet1_0(pretrained=pretrained)
        model.classifier[1] = nn.Linear(1280, num_classes)
    elif architecture == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(pretrained = pretrained)
        model.classifier[3] = nn.Linear(1280, num_classes)
    elif architecture == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(pretrained = pretrained)
        model.classifier[3] = nn.Linear(1024, num_classes)
    elif architecture == 'shufflenet_v2_x0_5':
        model = models.shufflenet_v2_x0_5(pretrained = pretrained)
        model.fc = nn.Linear(1024, num_classes)
    elif architecture == 'shufflenet_v2_x1_0':
        model = models.shufflenet_v2_x1_0(pretrained = pretrained)
        model.fc = nn.Linear(1024, num_classes)
    elif architecture == 'efficientnet_b0':
        if pretrained:
            model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)
        else:
            model = EfficientNet.from_name('efficientnet-b0', num_classes=num_classes)
    elif architecture == 'efficientnet_b1':
        if pretrained:
            model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=num_classes)
        else:
            model = EfficientNet.from_name('efficientnet-b1', num_classes=num_classes)
    elif architecture == 'efficientnet_b2':
        if pretrained:
            model = EfficientNet.from_pretrained('efficientnet-b2', num_classes=num_classes)
        else:
            model = EfficientNet.from_name('efficientnet-b2', num_classes=num_classes)
    elif architecture == 'efficientnet_b3':
        if pretrained:
            model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=num_classes)
        else:
            model = EfficientNet.from_name('efficientnet-b3', num_classes=num_classes)
    elif architecture == 'efficientnet_b4':
        if pretrained:
            model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes)
        else:
            model = EfficientNet.from_name('efficientnet-b4', num_classes=num_classes)
    return model


def initialize_finetune(model, architecture, num_ways):

    for p in model.parameters():
        p.requires_grad = False

    if architecture == 'resnet18':
        model.fc = nn.Linear(512, num_ways)
    elif architecture == 'resnet34':
        model.fc = nn.Linear(512, num_ways)
    elif architecture == 'resnet50':
        model.fc = nn.Linear(2048, num_ways)
    elif architecture == 'wide_resnet50_2':
        model.fc = nn.Linear(2048, num_ways)
    elif architecture == 'resnext50_32x4d':
        model.fc = nn.Linear(2048, num_ways)
    elif architecture == 'densenet121':
        model.classifier = nn.Linear(1024, num_ways)
    elif architecture == 'densenet161':
        model.classifier = nn.Linear(2208, num_ways)
    elif architecture == 'densenet169':
        model.classifier = nn.Linear(1664, num_ways)
    elif architecture == 'densenet201':
        model.classifier = nn.Linear(1920, num_ways)
    elif architecture == 'mnasnet':
        model.classifier[1] = nn.Linear(1280, num_ways)
    elif architecture == 'mobilenet_v3_large':
        model.classifier[3] = nn.Linear(1280, num_ways)
    elif architecture == 'mobilenet_v3_small':
        model.classifier[3] = nn.Linear(1024, num_ways)
    elif architecture == 'shufflenet_v2_x0_5':
        model.fc = nn.Linear(1024, num_ways)
    elif architecture == 'shufflenet_v2_x1_0':
        model.fc = nn.Linear(1024, num_ways)
    elif architecture == 'efficientnet_b0':
        model._fc = nn.Linear(1280, num_ways)
    elif architecture == 'efficientnet_b1':
        model._fc = nn.Linear(1280, num_ways)
    elif architecture == 'efficientnet_b2':
        model._fc = nn.Linear(1408, num_ways)
    elif architecture == 'efficientnet_b3':
        model._fc = nn.Linear(1536, num_ways)
    elif architecture == 'efficientnet_b4':
        model._fc = nn.Linear(1792, num_ways)

    return model


def get_configspace():
    cs = CS.ConfigurationSpace()

    architecture = CSH.CategoricalHyperparameter('architecture', ARCH, default_value = 'resnet18')
    lr = CSH.UniformFloatHyperparameter('lr', lower=1e-5, upper=1e-1, log=True, default_value = 1e-3)
    batch_size = CSH.UniformIntegerHyperparameter("batch_size", lower = 4, upper = 32, default_value = 16)
    optimizer = CSH.CategoricalHyperparameter('optimizer', ['SGD', 'Adam'], default_value = 'Adam')
    weight_decay = CSH.UniformFloatHyperparameter('weight_decay', lower=1e-5, upper=1e-2, log=True, default_value = 1e-3)
    momentum = CSH.UniformFloatHyperparameter('momentum', lower=0.01, upper=0.99, default_value = 0.9)
    sched_decay_interval = CSH.UniformIntegerHyperparameter("sched_decay_interval", lower = 6e1, upper = 3e2, default_value = 120)
    cs.add_hyperparameters([architecture, lr, batch_size, optimizer, weight_decay, momentum, sched_decay_interval])

    momentum_cond = CS.EqualsCondition(momentum, optimizer, 'SGD')
    cs.add_condition(momentum_cond)

    return cs


def process_images(images, size = None):
    """
    Reorder channels, resize to x224 and normalize for ImageNet pretrained networks
    """
    # HxWxC -> CxHxW
    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    # Resize
    if size:
        images = torch.nn.functional.interpolate(images, size = (size, size), mode = 'bilinear')
    # Normalize
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    images = normalize(images)

    return images

def augment(images, labels, n_aug = 5, aug_type = 'fixed_standard', aug_strength = 31):
    """
    Augment the images via TrivialAugment default.
    Max size is 30k including original images -> Larger size jobs fails on 2 CPU usually.
    """
    aug_lib.set_augmentation_space(aug_type, aug_strength)
    augmenter = aug_lib.TrivialAugment()
    
    images_PIL = [Image.fromarray((img*255).astype(np.uint8)) for img in images]
    
    augments = []
    augment_labels = []
    for i in range(n_aug):
        for img, l in zip(images_PIL, labels):
            augments.append(augmenter(img))
            augment_labels.append(l)
            if len(augments)+len(images_PIL) > int(3e4):
                break

    images_PIL = images_PIL+augments
    del augments
    images = np.stack([np.array(img, dtype = np.float32)/255 for img in images_PIL])
    del images_PIL
    labels = np.array(list(labels)+augment_labels)

    return images, labels

def do_PIL(images):
    """
    Convert images from numpy to PIL format
    """
    images_PIL = [Image.fromarray((img*255).astype(np.uint8)) for img in images]
    images = np.stack([np.array(img, dtype = np.float32)/255 for img in images_PIL])
    del images_PIL
    return images


def dump_a_custom_config(config, savepath = "experiments/custom_configs/default.json"):
    with open(savepath, 'w') as f:
        json.dump(config, f)

if __name__ == '__main__':

    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    from torchsummary import summary

    for architecture in ARCH:
        try:
            model = initialize_model(architecture, 1000).to(torch.device('cuda'))
            pytorch_total_params = sum(p.numel() for p in model.parameters())/1e6
            print(architecture, f"{round(pytorch_total_params, 3)}M")
            #summary(model, input_size=(3, 224, 224))
        except:
            print(architecture, 'Summary failed!')
    
    '''
    config = {"architecture": "resnet18", "lr": 0.001, "batch_size": 32, "optimizer": "Adam", "weight_decay": 0.001, "sched_decay_interval": 120}
    dump_a_custom_config(config, savepath)
    '''

