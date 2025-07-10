import wandb
from matplotlib import pyplot as plt
import matplotlib.animation as animation

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch import Tensor
from torch.utils.data import Subset

# from wikiart_ds import *
# from kaokore_ds import *
from datasets_code.pacs_ds import *
from models.stclf_model import *
from sampler.sampler_utils import *
from utils.utils import *
import importlib
import argparse




def new_ood_score_calculation(pred_scores, pred_labels, true_labels):
    """
    Calculate the OOD score based on the predicted scores, predicted labels, and true labels. 
    Uses the predicted score corresponding to the true label for the level of bias to that class.
    
    Args:
    - pred_scores (torch.Tensor): Tensor of predicted softmax scores (shape: [N, C])
    - pred_labels (torch.Tensor): Tensor of predicted labels (shape: [N])
    - true_labels (torch.Tensor): Tensor of true labels (shape: [N])
    
    Returns:
    - ood_score (torch.Tensor): OOD score
    """
    
    # Calculate the classwise OOD score. Any misclassified samples with high confidence will have a low score (biased confidentally wrong samples).
    classwise_ood_score = torch.zeros(pred_scores.shape[1])
    # print(classwise_comparison.shape, classwise_ood_score.shape)
    
    for i in range(len(classwise_ood_score)):
        class_mask = (true_labels == i)
    
        # Calculate the number of correct predictions for class 'i'
        correct_predictions = (pred_labels[class_mask] == true_labels[class_mask]).sum().float()
        
        # Calculate the total number of samples for class 'i'
        total_samples = class_mask.sum().float()
        
        classwise_ood_score[i] = torch.exp(torch.log(correct_predictions / total_samples) + torch.log(pred_scores[class_mask].max(dim = 1)[0].mean()))
        
        print(classwise_ood_score[i])
        classwise_ood_score[i] = classwise_ood_score[i]/(1+classwise_ood_score[i])
        print(classwise_ood_score[i])
        print(correct_predictions / total_samples, pred_scores[class_mask].max(dim = 1)[0].mean())
    
    print('Classwise OOD Score:', classwise_ood_score)
    
    # to make any class that gives poor scores boost the total score -> incentivize good performance across classes.
    total_ood_score = torch.exp(torch.sum(torch.log(classwise_ood_score), dim = 0))
    total_ood_score = total_ood_score/(1+total_ood_score)
    return total_ood_score, classwise_ood_score


def train_with_odin(model, dataset, totl_labels, temperature, save_path, extended_epochs, BSZ, step_sz, test_loader_out, norm_std):
    
    data_subset = dataset
    batch_sz = BSZ
    
    batch_sampler = BOOST(model, 'multiple_outputs', data_subset, totl_labels, batch_sz,
            step_sz, F.nll_loss, temperature, norm_std)
    batch_sampler.update_local(model, temperature)
    lr = 0.00008
    decay = 0.0004
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            # epochs * len(train_loader_out),
            epochs * len(train_dataset)//batch_sz,
            1,  # since lr_lambda computes multiplicative factor
            1e-6 / lr))
    
    #set for longer finetuning
    epochs = extended_epochs
    
    best_stats = ''
    # best_loss = -1000
    best_acc = 0
    loss_avg = 0.0
    temps = []
    temperature = 1
    for epoch in range(epochs):
        model.train()
        #TRAINING PHASE
        for i, batch_indices in enumerate(batch_sampler):
            batch_x, batch_y = torch.stack([data_subset[idx][0] for idx in batch_indices]), torch.tensor([dataset[idx][1] for idx in batch_indices])
            x, y = batch_x, batch_y
        # for x, y, _ in train_loader_rs:
            outputs = model(x.to(device))
            if isinstance(outputs, list):
                out = outputs[0]
            optimizer.zero_grad()
            loss = F.cross_entropy(out, y.to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()
            
        #EVALUATION PHASE
        model.eval()
        all_labels = []
        all_preds = []
        all_outputs = []
        # all_scores = []
        with torch.no_grad():
            for inputs, labels, _ in test_loader_out:
                outputs = model(inputs.cuda())
                if isinstance(outputs, list):
                    all_outputs.append(outputs[0])
                # scores = F.softmax(outputs[0])
                preds = outputs[0].argmax(dim=1)
                
                all_labels.extend(labels.numpy())
                all_preds.extend(preds.cpu().numpy())
            all_outputs = torch.cat(all_outputs, dim=0).to(device)

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        
        if accuracy > best_acc:
            best_acc = accuracy
            best_stats = f'Epoch [{epoch+1}/{epochs}], ' + f'Accuracy: {accuracy*100:.4f}, ' + f'F1 Score: {f1*100:.4f}, ' + f'Precision: {precision*100:.4f}, ' + f'Recall: {recall*100:.4f}'
        wandb.log({'accuracy': accuracy*100,
                    'f1': f1*100,
                    'recall': recall*100,
                    'precision': precision*100,
                    'loss': loss_avg,
                    'epoch': epoch})
        
        #class wise metrics
        all_preds, all_labels = np.array(all_preds), np.array(all_labels)
        kaokore_status_class_names = class_names
        cls_metrics = {'accuracy':{}, 'f1':{}, 'precision':{}, 'recall':{},
                        }
        for cls in set(all_labels):
            lbls = np.where(all_labels == cls, np.ones_like(all_labels), np.zeros_like(all_labels))
            preds = np.where(all_preds == cls, np.ones_like(all_preds), np.zeros_like(all_preds))  
            cls_metrics['accuracy'][kaokore_status_class_names[cls]] = accuracy_score(lbls, preds)
            cls_metrics['f1'][kaokore_status_class_names[cls]] = f1_score(lbls, preds)
            cls_metrics['precision'][kaokore_status_class_names[cls]] = precision_score(lbls, preds)
            cls_metrics['recall'][kaokore_status_class_names[cls]] = recall_score(lbls, preds)
            
        for metric in cls_metrics.keys():
            for cls in cls_metrics[metric].keys():
                wandb.log({f'{metric}_{cls}': cls_metrics[metric][cls]*100, 'epoch': epoch})

        if best_acc == accuracy:
            print('Saving best model and sampling weights')
            torch.save(model.state_dict(), f'{save_path}/best_model_ft_{MODEL_TYPE}_{SAMPLER_TYPE}.pth')
            
        if epoch %5 == 0:
            temperature *= 5
            batch_sampler.update_local(model, temperature)
            # batch_sampler.temperature = cosine_annealing(epoch, len(train_dataset)//batch_sz, 0, 1000)
            temps.append(temperature)
    print(best_stats)
    return model

def visualize_classwise_samples(results):
    """
    Visualize classwise correct and incorrect samples.
    
    Args:
    - results (dict): Dictionary of classwise correct/incorrect samples.
    """
    for class_label, result in results.items():
        correct_samples = result["correct"]
        incorrect_samples = result["incorrect"]
        
        # Plot correct samples
        if correct_samples:
            print(f"Class {class_label}: Correctly classified by ODIN improved Model")
            imgs = [img for img, _, _ in correct_samples]
            grid = make_grid(torch.cat(imgs, dim=0), nrow=5, padding=2)
            plt.figure(figsize=(10, 5))
            plt.imshow(grid.permute(1, 2, 0))
            plt.title(f'Correctly Classified Samples of Class {class_label} by ODIN improved Model')
            plt.axis('off')
            plt.show()
        
        # Plot incorrect samples
        if incorrect_samples:
            print(f"Class {class_label}: Incorrectly classified by ODIN improved Model")
            imgs = [img for img, _, _ in incorrect_samples]
            grid = make_grid(torch.cat(imgs, dim=0), nrow=5, padding=2)
            plt.figure(figsize=(10, 5))
            plt.imshow(grid.permute(1, 2, 0))
            plt.title(f'Incorrectly Classified Samples of Class {class_label} by ODIN improved Model')
            plt.axis('off')
            plt.show()

def ood_scores_bias_metrics(model, dataloader, train_dataset, device, temperature_scale_factor, EXP_NUM, RUN, totl_labels, BSZ, step_sz, test_dl, norm_std):
    model_path = f'saves/ood_art_tests_v2/experiment_{EXP_NUM}/model_{MODEL_TYPE}_{SAMPLER_TYPE}_{RUN}.pth'
    model.load_state_dict(torch.load(model_path))
    
    misclass_only = False
    save_model_path = f'saves/ood_art_tests_v2/experiment_{RUN}/'
    extra_epochs = 1
    
    # og_model = model
    if misclass_only:
        train_dataset = dataset
    model = train_with_odin(model, train_dataset, totl_labels,
                            temperature_scale_factor**(RUN//5), save_model_path, extra_epochs, BSZ, step_sz, test_dl, norm_std)
    
    totl_lbls = []
    totl_preds = []
    totl_pred_scores = []
    
    with torch.no_grad():
        for x, y, _ in dataloader:
            print('input shapes', x.shape, y.shape)
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, pred = torch.max(outputs[0], 1)
            pred_scores = F.softmax(outputs[0], dim = 0)
            
            if len(pred.shape) == 1:
                pred = pred.unsqueeze(0)
                y = y.unsqueeze(0)
                pred_scores = pred_scores.unsqueeze(0)
            totl_preds.extend(pred.detach().cpu())
            totl_lbls.extend(y.detach().cpu())
            totl_pred_scores.extend(pred_scores.detach().cpu())
    
    totl_lbls = torch.cat(totl_lbls, dim = 0)
    totl_pred_scores = torch.cat(totl_pred_scores, dim = 0)
    totl_preds = torch.cat(totl_preds, dim = 0)
    
    print(totl_lbls.shape, totl_preds.shape, totl_pred_scores.shape)
    
    print('OOD scores bias calculation')
    ood_score, classwise_scores = new_ood_score_calculation(totl_pred_scores, totl_preds, totl_lbls)
    wandb.log({'ood_score_new': ood_score*100, 'epoch': RUN})
    
    mab, sdb = calculate_mab_sdb(classwise_scores) 
    print('MAB:', mab, 'SDB:', sdb)
    wandb.log({'ood_mab': mab, 'ood_sdb': sdb})
    
    # return results

def calculate_mab_sdb(performance_metrics):
    """
    Calculate the Mean Absolute Bias (MAB) and Standard Deviation of Bias (SDB) based on provided performance metrics.
    
    Args:
    - performance_metrics (torch.Tensor): Tensor of performance metrics for each class (shape: [N])
    
    Returns:
    - mab (torch.Tensor): Mean Absolute Bias
    - sdb (torch.Tensor): Standard Deviation of Bias
    """
    # Ensure the performance metrics are a tensor
    performance_metrics = torch.tensor(performance_metrics, dtype=torch.float32)
    
    # Calculate the mean performance metric
    mean_performance_metric = torch.mean(performance_metrics)
    
    # Mean Absolute Bias (MAB)
    mab = torch.mean(torch.abs(performance_metrics - mean_performance_metric))
    
    # Standard Deviation of Bias (SDB)
    sdb = torch.sqrt(torch.mean((performance_metrics - mean_performance_metric) ** 2))
    
    return mab, sdb

    
def get_best_performance_metrics(model, dataloader, device, RUN, test_loader_out):
    model_path = f'saves/ood_art_tests_v2/experiment_{RUN}/best_model_{MODEL_TYPE}_{SAMPLER_TYPE}.pth'
    model.eval()
    model.load_state_dict(torch.load(model_path))
    for inputs, labels, _ in dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs[0], 1)
    all_labels = []
    all_preds = []
    all_outputs = []
    with torch.no_grad():
        for inputs, labels, _ in test_loader_out:
            outputs = model(inputs.cuda())
            if isinstance(outputs, list):
                all_outputs.append(outputs[0])
                preds = outputs[0].argmax(dim=1)
        
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
        all_outputs = torch.cat(all_outputs, dim=0).to(device)
        
    #class wise metrics
    all_preds, all_labels = np.array(all_preds), np.array(all_labels)
    cls_metrics = {'accuracy':{}, 'f1':{}, 'precision':{}, 'recall':{}}
    cls_acc = torch.zeros(NUM_CLASSES)
    cls_f1 = torch.zeros(NUM_CLASSES)
    cls_precision = torch.zeros(NUM_CLASSES)
    cls_recall = torch.zeros(NUM_CLASSES)
    print(class_names,NUM_CLASSES, np.unique(all_labels))
    for cls in set(all_labels):
        lbls = np.where(all_labels == cls, np.ones_like(all_labels), np.zeros_like(all_labels))
        preds = np.where(all_preds == cls, np.ones_like(all_preds), np.zeros_like(all_preds))  
        cls_metrics['accuracy'][class_names[cls]] = accuracy_score(lbls, preds)
        cls_metrics['f1'][class_names[cls]] = f1_score(lbls, preds)
        cls_metrics['precision'][class_names[cls]] = precision_score(lbls, preds)
        cls_metrics['recall'][class_names[cls]] = recall_score(lbls, preds)
    print(cls_metrics)
    for i in range(NUM_CLASSES):
        cls_acc[i] = cls_metrics['accuracy'][class_names[i]]*100
        cls_precision[i] = cls_metrics['precision'][class_names[i]]*100
        cls_recall[i] = cls_metrics['recall'][class_names[i]]*100
        cls_f1[i] = cls_metrics['f1'][class_names[i]]*100
    mab, sdb = calculate_mab_sdb(cls_acc)
    print('Accuracy MAB:', mab, 'SDB:', sdb)
    wandb.log({'best_accuracy_mab': mab, 'best_accuracy_sdb': sdb})
    mab, sdb = calculate_mab_sdb(cls_f1)
    print('F1 MAB:', mab, 'SDB:', sdb)
    wandb.log({'best_f1_mab': mab, 'best_f1_sdb': sdb})
    mab, sdb = calculate_mab_sdb(cls_precision)
    print('Precision MAB:', mab, 'SDB:', sdb)
    wandb.log({'best_precision_mab': mab, 'best_precision_sdb': sdb})
    mab, sdb = calculate_mab_sdb(cls_recall)
    print('Recall MAB:', mab, 'SDB:', sdb)
    wandb.log({'best_recall_mab': mab, 'best_recall_sdb': sdb})

def get_image(index, dataset):
    img, _ = dataset[index]
    return img

def parse_args():
    parser = argparse.ArgumentParser(description="BOOST Image Classification Training")

    parser.add_argument('--batch_sz', type=int, default=32,
                        help='Batch size for training (default: 32)')
    
    parser.add_argument('--dataset', type=str, default='kaokore', choices=['kaokore', 'pacs', 'custom_imgnet_style'],
                        help='Dataset name (default: kaokore)')
    
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        choices=['cuda', 'cpu'],
                        help='Computation device (default: auto-detect)')

    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')

    return parser.parse_args()

if __name__ == "__main__":
    # Usage
    wandb.init(project="BOOST_inference")
    
    args = parse_args()
    device = args.device
    
    dataset_module = importlib.import_module(f'data.{args.dataset}_ds')
    train_dataset, dataset, data_loader, totl_labels, class_names, norm_std = dataset_module.dataset_details()
  
    
    MODEL_TYPE = 'stsaclf'
    SAMPLER_TYPE = 'random'
    NUM_CLASSES = len(class_names)
    FFINETUNE = False
    DROPOUT_TYPE = 'dropout'
    DROPOUT_P = 0.23
    EXP_NUM = 1
    BEST_RUN = 8
    BSZ = args.batch_sz
    step_sz = 0.05
    
    model = AttnResNet(NUM_CLASSES,
                            ResNetN('resnet50','avgpool',
                                ['conv1', 'layer2','layer3','layer4'],
                                FFINETUNE),
                            DROPOUT_TYPE,
                            DROPOUT_P).to(device)

    get_best_performance_metrics(model, data_loader, device, BEST_RUN, data_loader, )

    ood_scores_bias_metrics(model, data_loader, train_dataset, device, 5, EXP_NUM, BEST_RUN, totl_labels, BSZ, step_sz, data_loader, norm_std)

    wandb.finish()