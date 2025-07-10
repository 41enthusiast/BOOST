import wandb

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch import Tensor

import os

from datasets import load_dataset

# reproducibility
# fix_random_seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.benchmark = False

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from data.kaokore_ds import *

from utils.utils import *
from models.stclf_model import *
from sampler.sampler_utils import *

#just to clean up, but ill advised
import warnings
import importlib
import argparse

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
  warnings.filterwarnings("ignore")

  #presets
  args = parse_args()
  device = args.device
  DS_NAME = args.dataset
  BSZ = args.batch_sz

  RUN = 1
  

  step_sz = 0.05
  criterion = torch.nn.CrossEntropyLoss()
  batch_sz = BSZ

  epochs = 50
  LR = 0.08
  momentum = 0.9
  WD = 0.0004

  temperature = 1

  dataset_module = importlib.import_module(f'data.{args.dataset}_ds')
  dataset, _, test_loader_out, totl_labels, class_names, norm_std = dataset_module.dataset_details()
  
  NUM_CLASSES = len(class_names)
  FFINETUNE = False
  DROPOUT_TYPE = 'dropout'
  DROPOUT_P = 0.23

  art_model = AttnResNet(NUM_CLASSES,
                            ResNetN('resnet50','avgpool',
                                ['conv1', 'layer2','layer3','layer4'],
                                FFINETUNE),
                            DROPOUT_TYPE,
                            DROPOUT_P).to(device)
  
  batch_sampler = BOOST(art_model, 'multiple_outputs', dataset, totl_labels, batch_sz,
            step_sz, F.nll_loss, temperature, norm_std)
  batch_sampler.update_local(art_model, temperature)

  optimizer = torch.optim.SGD(
    art_model.parameters(),
    LR, momentum=momentum,
    weight_decay=WD, nesterov=True)


  optim_name = 'SGDM'

  lr = LR
  decay = WD
  criterion = focal_loss(4, 2, 2)
  optimizer = optim.AdamW(art_model.parameters(), lr=lr, weight_decay=decay)
  optim_name = 'Adam'

  scheduler = torch.optim.lr_scheduler.LambdaLR(
      optimizer,
      lr_lambda=lambda step: cosine_annealing(
          step,
          # epochs * len(train_loader_out),
          epochs * len(dataset)//batch_sz,
          1,  # since lr_lambda computes multiplicative factor
          1e-6 / lr))

  # scheduler = torch.optim.lr_scheduler.OneCycleLR(
  #     optimizer,max_lr=0.01, 
  #                        epochs=epochs,
  #                        steps_per_epoch=len(train_dataset)//batch_sz,)

  #Visualization setup

  wandb.init(project='boost_experiments',
            name = f'experiment_{RUN}_{DS_NAME}',
            config={
              'learning_rate': lr,
              'architecture': 'stsaclf',
              'sampler': 'BOOST',
              'epochs': epochs,
              'batch_size': batch_sz,
              'temperature': temperature,
              'ood_step':step_sz,
              'dataset': 'kaokore',
              'decay': decay,
              'optimizer': optim_name,
            })

  #housekeeping code
  checkpoint_path = f'saves/ood_art_tests_v2/experiment_{RUN}'
  if not os.path.exists(checkpoint_path):
      os.makedirs(checkpoint_path, exist_ok = True)

  #TRAINING LOOP
  best_stats = ''
  best_loss = -1000
  best_acc = 0
  loss_avg = 0.0
  temps = []
  for epoch in range(epochs):
    art_model.train()
    for i, batch_indices in enumerate(batch_sampler):
      batch_x, batch_y = torch.stack([dataset[idx][0] for idx in batch_indices]), torch.tensor([dataset[idx][1] for idx in batch_indices])
      x,y = batch_x.cuda(), batch_y.cuda()
      outputs = art_model(x)
      if isinstance(outputs, list):
          out = outputs[0]
      optimizer.zero_grad()
      loss = F.cross_entropy(out, y)
      loss.backward()
      optimizer.step()
      scheduler.step()

      loss_avg = loss_avg*0.8 + float(loss)* 0.2


    # Evaluation
    art_model.eval()
    all_labels = []
    all_preds = []
    all_outputs = []
    with torch.no_grad():
        for inputs, labels, _ in test_loader_out:
            outputs = art_model(inputs.cuda())
            if isinstance(outputs, list):
              all_outputs.append(outputs[0])
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

    print(f'Epoch [{epoch+1}/{epochs}], '
          f'Accuracy: {accuracy*100:.4f}, '
          f'F1 Score: {f1*100:.4f}, '
          f'Precision: {precision*100:.4f}, '
          f'Recall: {recall*100:.4f}')
    wandb.log({'accuracy': accuracy*100,
              'f1': f1*100,
              'recall': recall*100,
              'precision': precision*100,
              'loss': loss_avg,
              'epoch': epoch})
    
    #class wise metrics
    all_preds, all_labels = np.array(all_preds), np.array(all_labels)
  #   kaokore_status_class_names = {0: 'noble', 1: 'warrior', 2: 'incarnation', 3: 'commoner'}
    print(class_names.keys())
    cls_metrics = {'accuracy':{}, 'f1':{}, 'precision':{}, 'recall':{},
                  'auroc':{}, 'fpr95': {}}
    for cls in set(all_labels):
      cls = int(cls)
      
      lbls = np.where(all_labels == cls, np.ones_like(all_labels), np.zeros_like(all_labels))
      preds = np.where(all_preds == cls, np.ones_like(all_preds), np.zeros_like(all_preds))  
      cls_metrics['accuracy'][class_names[cls]] = accuracy_score(lbls, preds)
      cls_metrics['f1'][class_names[cls]] = f1_score(lbls, preds)
      cls_metrics['precision'][class_names[cls]] = precision_score(lbls, preds)
      cls_metrics['recall'][class_names[cls]] = recall_score(lbls, preds)
      
    for metric in cls_metrics.keys():
      for cls in cls_metrics[metric].keys():
        wandb.log({f'{metric}_{cls}': cls_metrics[metric][cls]*100, 'epoch': epoch})

    print('Saving model and sampling weights')
    torch.save(art_model.state_dict(), f'{checkpoint_path}/model_{epoch}.pth')
    if best_acc == accuracy:
      torch.save(art_model.state_dict(), f'{checkpoint_path}/best_model.pth')
    
      
    if epoch %5 == 0:
      temperature *= 5
      batch_sampler.update_local(art_model, temperature)
      # batch_sampler.temperature = cosine_annealing(epoch, len(train_dataset)//batch_sz, 0, 1000)
      temps.append(temperature)

  print('train_loss', loss_avg)
