import wandb

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch import Tensor

from datasets import load_dataset
from pytorch_ood.model import WideResNet
from pytorch_ood.utils import OODMetrics, ToUnknown, fix_random_seed, is_unknown
from pytorch_ood.detector import ODIN, odin_preprocessing

# reproducibility
fix_random_seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.benchmark = False

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from torchmetrics.functional.classification import binary_auroc

# from kaokore_ds import *
from pacs_ds import *
# from wikiart_ds import *

from utils import *
from stclf_model import *
from sampler_utils import *

#just to clean up, but ill advised
import warnings
warnings.filterwarnings("ignore")

#presets
device = 'cuda'
model = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=True).to(device)

MODEL_TYPE = ['simple_odin', 'stsaclf', 'contrastive'][1]
RUN = 29
SAMPLER_TYPE = ['odin', 'stratified', 'random'][0]


if MODEL_TYPE == 'simple_odin':
  art_model = model
  art_model.fc = nn.Linear(in_features=2048, out_features=4, bias=True, device = 'cuda')
elif MODEL_TYPE == 'stsaclf':
  art_model = stclf_model.to(device)



step_sz = 0.05
criterion = torch.nn.CrossEntropyLoss()
device = 'cuda'
batch_sz = BSZ

epochs = 50
LR = 0.00008
momentum = 0.9
WD = 0.0004

temperature = 1

dataset = train_dataset

# batch_sampler = OdinSamplerRB(art_model, dataset, batch_sz,
#               step_sz, F.nll_loss, temperature, norm_std)

optimizer = torch.optim.SGD(
   art_model.parameters(),
   LR, momentum=momentum,
   weight_decay=WD, nesterov=True)


optim_name = 'SGDM'

if MODEL_TYPE == 'stsaclf':
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
        epochs * len(train_dataset)//batch_sz,
        1,  # since lr_lambda computes multiplicative factor
        1e-6 / lr))

# scheduler = torch.optim.lr_scheduler.OneCycleLR(
#     optimizer,max_lr=0.01, 
#                        epochs=epochs,
#                        steps_per_epoch=len(train_dataset)//batch_sz,)

#Visualization setup

wandb.init(project='ood_art_tests_v2',
           name = f'experiment_{RUN}_{DS_NAME}',
           config={
             'learning_rate': lr,
             'architecture': MODEL_TYPE,
             'sampler': SAMPLER_TYPE,
             'epochs': epochs,
             'batch_size': batch_sz,
             'temperature': temperature,
             'ood_step':step_sz,
             'dataset': 'kaokore',
             'decay': decay,
             'optimizer': optim_name,
           })


#TRAINING LOOP
best_stats = ''
best_loss = -1000
best_acc = 0
loss_avg = 0.0
temps = []
for epoch in range(epochs):
  art_model.train()
  # for i, batch_indices in enumerate(batch_sampler):
  #   batch_x, batch_y = torch.stack([dataset[idx][0] for idx in batch_indices]), torch.tensor([dataset[idx][1] for idx in batch_indices])
  #   x,y = batch_x.cuda(), batch_y.cuda()
  # for x, xs, y in train_loader_ws:
  #   x, xs, y = x.to(device), xs.to(device), y.to(device)
  for x, y in train_loader_out:
    x, y = x.to(device), y.to(device)
    if MODEL_TYPE == 'simple_odin':
      out = art_model(x)
    elif MODEL_TYPE == 'stsaclf':
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
      for inputs, labels in test_loader_out:
          if MODEL_TYPE == 'simple_odin':
            outputs = art_model(inputs.cuda())
            all_outputs.append(outputs[0])
            _, preds = torch.max(outputs, 1)
          elif MODEL_TYPE == 'stsaclf':
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
  kaokore_status_class_names = class_names#{0: 'noble', 1: 'warrior', 2: 'incarnation', 3: 'commoner'}
  cls_metrics = {'accuracy':{}, 'f1':{}, 'precision':{}, 'recall':{},
                 'auroc':{}, 'fpr95': {}}
  for cls in set(all_labels):
    lbls = np.where(all_labels == cls, np.ones_like(all_labels), np.zeros_like(all_labels))
    preds = np.where(all_preds == cls, np.ones_like(all_preds), np.zeros_like(all_preds))  
    cls_metrics['accuracy'][kaokore_status_class_names[cls]] = accuracy_score(lbls, preds)
    cls_metrics['f1'][kaokore_status_class_names[cls]] = f1_score(lbls, preds)
    cls_metrics['precision'][kaokore_status_class_names[cls]] = precision_score(lbls, preds)
    cls_metrics['recall'][kaokore_status_class_names[cls]] = recall_score(lbls, preds)
    
    #this forces the OvR based OOD metrics to be calculated - may not be a great idea
    cls_metrics['auroc'][kaokore_status_class_names[cls]] = binary_auroc(all_outputs[:,cls], Tensor(lbls).int().to(device))
    cls_metrics['fpr95'][kaokore_status_class_names[cls]] = fpr_at_tpr(all_outputs[:,cls], Tensor(lbls).int().to(device))
    
  for metric in cls_metrics.keys():
    for cls in cls_metrics[metric].keys():
      wandb.log({f'{metric}_{cls}': cls_metrics[metric][cls]*100, 'epoch': epoch})

  print('Saving model and sampling weights')
  torch.save(art_model.state_dict(), f'saves/ood_art_tests_v2/experiment_{RUN}/model_{MODEL_TYPE}_{SAMPLER_TYPE}_{epoch}.pth')
  if best_acc == accuracy:
    torch.save(art_model.state_dict(), f'saves/ood_art_tests_v2/experiment_{RUN}/best_model_{MODEL_TYPE}_{SAMPLER_TYPE}.pth')
   
  

  if SAMPLER_TYPE == 'odin':
    # torch.save( batch_sampler.train_sampling_probs, f'saves/ood_art_tests_v2/experiment_{RUN}/sampler_{SAMPLER_TYPE}_{AGGR_MODE}_{epoch}.pt')
    
    # visualization of results
    
    if epoch %5 == 0:
      temperature *= 8
      batch_sampler.update_local(art_model, temperature)
      # batch_sampler.temperature = cosine_annealing(epoch, len(train_dataset)//batch_sz, 0, 1000)
      temps.append(temperature)

print('train_loss', loss_avg)

print(MODEL_TYPE, 'ODIN sampling', 'macro avg')
print(temps)
