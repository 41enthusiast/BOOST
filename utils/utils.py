from PIL import Image
import os
import torch
import torch.nn.functional as F
import sys
from torch import nn
import matplotlib.pyplot as plt
import cv2
import itertools
from torchvision.utils import make_grid
import numpy as np
from torchvision.io import read_image
from torchvision.datasets import ImageFolder
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
import random

from sklearn.metrics import recall_score, precision_score, f1_score
from torchmetrics.functional.classification import (
    binary_auroc,
    binary_precision_recall_curve,
    binary_roc
)

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))

def check_paths(args):
    try:
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
        if args.checkpoint_model_dir is not None and not (os.path.exists(args.checkpoint_model_dir)):
            os.makedirs(args.checkpoint_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)

def load_image(filename, size=None, scale=None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img


def save_img(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)

def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1) # new_tensor for same dimension of tensor
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0) # back to tensor within 0, 1
    return (batch - mean) / std


#drops images in a batch from being seen by a subset of layers
def drop_connect(inputs, p, train_mode):  # bchw, 0-1, bool
    assert 0 <= p <= 1, 'p must be in range of [0,1]'

    #inference
    if not train_mode:
        return inputs

    bsz = inputs.shape[0]

    keep_prob = 1-p

    #binary mask for selection of weights
    rand_tensor = keep_prob
    rand_tensor += torch.rand([bsz, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    mask = torch.floor(rand_tensor)

    outputs = inputs / keep_prob*mask
    return outputs


def focal_loss(n_classes, gamma=2., alpha=4.):
    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y_pred, y):
        eps = 1e-9
        pred = torch.softmax(y_pred, dim=1) + eps
        #pred = y_pred + eps
        y_true = F.one_hot(y, n_classes)
        cross_entropy = y_true * -1*torch.log(pred)
        wt = y_true*(1-pred)**gamma
        focal_loss = alpha*wt*cross_entropy
        focal_loss = torch.max(focal_loss, dim=1)[0]
        return focal_loss.mean()
    return focal_loss_fixed

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2) # swapped ch and w*h, transpose share storage with original
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def adjust_learning_rate(optimizer, iteration_count, lr, lr_decay=5e-5):
    """Imitating the original implementation"""
    lr = lr / (1.0 + lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

mse_loss = nn.MSELoss()

def calc_content_loss( input, target):
    assert (input.size() == target.size())
    assert (target.requires_grad is False)
    return mse_loss(input, target)

def calc_style_loss(input, target):
    assert (input.size() == target.size())
    assert (target.requires_grad is False)
    input_mean, input_std = calc_mean_std(input)
    target_mean, target_std = calc_mean_std(target)
    return mse_loss(input_mean, target_mean) + \
           mse_loss(input_std, target_std)


def make_confusion_matrix(model, n_classes, loader, device):
    @torch.no_grad()
    def get_all_preds(model):
        all_preds = torch.tensor([]).to(device)
        all_tgts = torch.tensor([]).to(device)
        for batch in loader:
            images, labels = batch
            preds = model(images.to(device))[0]
            all_preds = torch.cat(
                (all_preds, preds)
                , dim=0
            )
            all_tgts = torch.cat(
                (all_tgts, labels.to(device))
                , dim=0
            )
        return all_preds, all_tgts

    # set up model predictions and targets in right format for making the confusion matrix
    preds, tgts = get_all_preds(model.to(device))
    #print(preds.argmax(dim=1).shape, tgts.shape)
    stacked = torch.stack(
        (
            tgts.squeeze()
            , preds.argmax(dim=1)
        )
        , dim=1
    )

    # make the confusion matrix
    cm = torch.zeros(n_classes, n_classes, dtype=torch.int64)
    for p in stacked:
        tl, pl = p.tolist()
        cm[int(tl), int(pl)] = cm[int(tl), int(pl)] + 1

    return cm

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        #print('Confusion matrix, without normalization')
        pass

    # set up the confusion matrix visualization
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('misc/temp_cm_logging.jpg')
    plt.close('all')
    return read_image('misc/temp_cm_logging.jpg')/255

def focal_loss_non_reduce(n_classes, gamma=2., alpha=4.):
    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y_pred, y):
        eps = 1e-9
        pred = torch.softmax(y_pred, dim=1) + eps
        #pred = y_pred + eps
        y_true = F.one_hot(y, n_classes)
        cross_entropy = y_true * -1*torch.log(pred)
        wt = y_true*(1-pred)**gamma
        focal_loss = alpha*wt*cross_entropy
        focal_loss = torch.max(focal_loss, dim=1)
        return focal_loss
    return focal_loss_fixed

def get_most_and_least_confident_predictions(model, loader, device , num_classes):
    # get model prediction for the validation dataset and all images for later use
    preds = torch.tensor([]).to(device)
    tgts = torch.tensor([]).to(device)
    all_images = torch.tensor([]).to(device)
    #loader = module.val_dataloader()
    #model = module.model.to(device)
    for batch in loader:
        images, y = batch
        pred_batch = model(images.to(device))[0]
        preds = torch.cat(
            (preds, pred_batch)
            , dim=0
        )
        tgts = torch.cat(
            (tgts, y.to(device))
            , dim=0
        )
        all_images = torch.cat(
            (all_images, images.to(device)),
            dim=0
        )
    print(preds.shape, tgts.shape)
    criterion = focal_loss_non_reduce(num_classes, 2, 2) #gamma, alpha
    confidence = criterion(preds, tgts.to(torch.int64))[0]
    #print(confidence.shape)

    # get indices with most and least confident scores
    lc_scores, least_confident = confidence.topk(4, dim=0)
    mc_scores, most_confident = confidence.topk(4, dim=0, largest=False)
    print(lc_scores, mc_scores)

    # get the images according to confidence scores, 4 each
    mc_imgs = all_images[most_confident.squeeze()]
    lc_imgs = all_images[least_confident.squeeze()]

    return (mc_scores, mc_imgs), (lc_scores, lc_imgs)

def visualize_attn(I, c, h = 256):
    # Image
    print(I.shape)
    img = I.permute((1,2,0)).cpu().numpy()
    # Heatmap
    N, C, H, W = c.size()
    a = F.softmax(c.view(N,C,-1), dim=2).view(N, C, H, W)
    up_factor = I.shape[0]/H, I.shape[1]/W
    up_factor = h/H
    #print(up_factor, I.size(), c.size())
    #if up_factor[0] > 1 or up_factor[1] > 1:
    #    a = F.interpolate(a, scale_factor=up_factor, mode='bilinear', align_corners=False)
    #if up_factor > 1:
    a = F.interpolate(a, scale_factor=up_factor, mode='bilinear', align_corners=False)
    attn = make_grid(a, nrow=4, normalize=True, scale_each=True)
    attn = attn.permute((1,2,0)).mul(255).byte().cpu().numpy()
    attn = cv2.applyColorMap(attn, cv2.COLORMAP_JET)
    attn = cv2.cvtColor(attn, cv2.COLOR_BGR2RGB)
    # Add the heatmap to the image
    vis = 0.6 * img + 0.4 * attn
    return torch.from_numpy(vis).permute(2,0,1)

def stratified_split(dataset : ImageFolder, fractions):
    indices_per_label = defaultdict(list)
    for index, label in enumerate(dataset.targets):
        indices_per_label[label].append(index)
    indices = []
    for label, idxs in indices_per_label.items():
        n_samples_for_label = round(len(idxs) * fractions[label])
        random_indices_sample = random.sample(idxs, n_samples_for_label)
        indices += random_indices_sample
    inputs = torch.utils.data.Subset(dataset, indices)
    return inputs

def get_recall_precision_f1(all_label, all_pred):
    recall = recall_score(all_label.squeeze().cpu().data.squeeze().numpy(),
                                   all_pred.cpu().data.squeeze().numpy(), average='macro')
    precision = precision_score(all_label.squeeze().cpu().data.squeeze().numpy(),
                                         all_pred.cpu().data.squeeze().numpy(), average='macro')
    f1 = f1_score(all_label.squeeze().cpu().data.squeeze().numpy(),
                           all_pred.cpu().data.squeeze().numpy(), average='macro')
                           
    return recall, precision, f1

def fpr_at_tpr(pred, target, k=0.95):
    """
    Calculate the False Positive Rate at a certain True Positive Rate

    :param pred: outlier scores
    :param target: target label
    :param k: cutoff value
    :return:
    """
    # results will be sorted in reverse order
    fpr, tpr, _ = binary_roc(pred, target)
    idx = torch.searchsorted(tpr, k)
    if idx == fpr.shape[0]:
        return fpr[idx - 1]
    return fpr[idx]