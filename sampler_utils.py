from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler, WeightedRandomSampler

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, TypeVar, Optional, Callable, List

import torch
from torch.autograd import Variable
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn

from torchvision.datasets import ImageFolder
from torchvision import transforms, utils, datasets

import pickle

# from kaokore_ds import *
# from pacs_ds import *
# from wikiart_ds import *
from wikiart_emotions_ds import *
from stclf_model import *

#with and without style transfer
# AUG_MODE = ['ST', 'VA'][1]
AGGR_MODE = ['Aggr','Indv','Both'][2]
DS_NAME = ['kaokore', 'PACS', 'WikiArt', 'WikiArt_Emotions'][3]

class OdinSamplerRB(torch.utils.data.Sampler):
    def __init__(self, model, data_source, totl_labels, batch_size, step_sz, loss, temperature, norm_std, replacement=True, device = 'cuda'):
        self.data_source = data_source
        self.data_lbls = totl_labels
        self.batch_size = batch_size
        self.replacement = replacement

        self.step_sz = step_sz
        self.loss = loss
        self.temperature = temperature
        self.norm_std = norm_std

        self.model = model

        self.sampling_probs = torch.zeros(len(data_source)).to(device)
        self.count_dict_new = []

        if len(data_source) < batch_size:
            raise ValueError("Batch size must be less than or equal to the dataset size.")

    def odin_preprocessing(
        self,
        x: Tensor,
        y: Optional[Tensor] = None,
        criterion: Optional[Callable[[Tensor], Tensor]] = None,
        eps: float = 0.05,
        temperature: float = 1000,
        norm_std: Optional[List[float]] = None,
        mode = False
        ):
        """
        Functional version of ODIN.

        :param model: module to backpropagate through
        :param x: sample to preprocess
        :param y: the label :math:`\\hat{y}` which is used to evaluate the loss. If none is given, the models
            prediction will be used
        :param criterion: loss function :math:`\\mathcal{L}` to use. If none is given, we will use negative log
                likelihood
        :param eps: step size :math:`\\epsilon` of the gradient ascend step
        :param temperature: temperature :math:`T` to use for scaling
        :param norm_std: standard deviations used during preprocessing
        """

        # we make this assignment here, because adding the default to the constructor messes with sphinx
        if criterion is None:
            criterion = F.nll_loss
        model = self.model
        with torch.inference_mode(False):
            if torch.is_inference(x):
                x = x.clone()

            with torch.enable_grad():
                x = Variable(x, requires_grad=True)
                
                #temperature based softmax scaling for confidence calibration. doesn't affect model accuracy, but influences confidence.
                if MODEL_TYPE == 'simple_odin':
                    logits = model(x) / temperature
                elif MODEL_TYPE == 'stsaclf':
                    outputs = self.model(x)
                    if isinstance(outputs, list):
                        logits = outputs[0] / temperature
                if y is None:
                    y = logits.max(dim=1).indices
                loss = criterion(logits, y)
                loss.backward()

                if mode == True:
                    #gradient noise based data preprocessing
                    gradient = torch.sign(x.grad.data)

                    if norm_std is not None:
                        for i, std in enumerate(norm_std):
                            gradient.index_copy_(
                                1,
                                torch.LongTensor([i]).to(gradient.device),
                                gradient.index_select(1, torch.LongTensor([i]).to(gradient.device)) / std,
                            )

                    x_hat = x - eps * gradient
        
        #this is in the training phase at the start of the epoch
        if mode == False:
            self.model = model
            
        if mode == True:
            return x_hat

    def predict_confidence_probs(self, x: Tensor, y: Tensor, mode, return_ind = False) -> Tensor:
        """
        Calculates softmax outlier scores on ODIN pre-processed inputs.

        :param x, x_s: input tensors
        :param y: output tensor
        :param mode: boolean to predict the confidence probabilities or not
        :return: outlier scores for each sample
        """

        x_hat = self.odin_preprocessing(
            x=x,
            y=y,
            eps=self.step_sz,
            criterion=self.loss,
            temperature=self.temperature,
            norm_std=self.norm_std,
            mode = mode
        )
        
        if mode == True:
            if MODEL_TYPE == 'simple_odin':
                results = self.model(x_hat).softmax(dim=1)
                # aug_results = self.model(x_s).softmax(dim=1)
            elif MODEL_TYPE == 'stsaclf':
                results = self.model(x_hat)[0].softmax(dim=1)
                # aug_results = self.model(x_s)[0].softmax(dim=1)
            
            #choosing to keep original or transformation - simple strat
            # if AUG_MODE == 'ST':
            #     results = torch.where(results>aug_results, results, aug_results)
            
            confidence, inds = torch.tensor(results).max(dim=1)
            if return_ind:
                return confidence, inds
            else:
                return confidence

    def update_local(self, model, temperature = 1):
        n = len(self.data_source)
        
        self.model = model
        self.temperature = temperature

        self.old_indices = torch.randperm(n).tolist()
        # print(self.old_indices)
        # print([self.data_source[i][2] for i in range(n)])
        
        for i in range(0, n, self.batch_size):
            #print('going for new batch', len(indices))
            x,y, = [], []
            seed_idxs = self.old_indices[i:i+self.batch_size]

            for idx in seed_idxs:
              x.append(self.data_source[idx][0])
              y.append(self.data_source[idx][1])

            if x == [] or y == []:
              print(seed_idxs, i)
            x = torch.stack(x).to('cuda')
            y  = torch.tensor(y).to('cuda')
            self.predict_confidence_probs(x, y, False)
            
            
            

    def __iter__(self):
        n = len(self.data_source)

        old_indices = self.old_indices
        cum_scores_class = {i: 1e-6 for i in set(self.data_lbls)}
        self.cum_sampling_probs = {i: 1e-6 for i in set(self.data_lbls)}
        all_scores = []
        all_y = []

        #get aggregate score and individual scores. precalculate for the whole dataset
        print('Calculating confidence scores for the dataset')#the time consuming part of this code - finetuning
        for i in range(0, n, self.batch_size):
            #print('going for new batch', len(indices))
            x,y = [], []
            seed_idxs = old_indices[i:i+self.batch_size]

            for idx in seed_idxs:
              x.append(self.data_source[idx][0])
              y.append(self.data_source[idx][1])
              
            x = torch.stack(x).to('cuda')
            y  = torch.tensor(y).to('cuda')
            scores = self.predict_confidence_probs(x, y, True)
            all_y.append(y)
            all_scores.append(scores)

        all_scores = torch.cat(all_scores)
        self.all_y = torch.cat(all_y)
        print('Class wise statistics for analysis')
        for i in cum_scores_class:
            temp_group = all_scores[torch.nonzero(self.all_y == i).squeeze()] #big mistake here. wasn't an actual accumulation
            cum_scores_class[i]= temp_group.mean()
        print('Cumulative scores from phase 1', cum_scores_class)
            
        print('Iterating through the dataset')
        self.count_dict_new.append({i:0 for i in set(self.data_lbls)})
        for i in range(0, n, self.batch_size):
            x,y = [], []
            seed_idxs = old_indices[i:i+self.batch_size]

            for idx in seed_idxs:
              #x.append(F.interpolate(self.data_source[idx][0], size = 32))
              x.append(self.data_source[idx][0])
              y.append(self.data_source[idx][1])
            
            x = torch.stack(x).to('cuda')
            y  = torch.tensor(y).to('cuda')
              
            probs = all_scores[seed_idxs]
            
            #rescaling
            cum_confidence = torch.zeros_like(y).float()
            for i in cum_scores_class:
                cum_confidence[torch.nonzero(y == i).squeeze()] = cum_scores_class[i]
            if AGGR_MODE == 'Both': #find a better way to accumulate the probs
                probs = probs * cum_confidence # or cum confidence as mean, the indv probs as std dev instead of multiplyin it
            elif AGGR_MODE == 'Aggr':
                probs = cum_confidence
            #need to put higher confidence on the low confidence samples - it doesn't affect it in the end
            probs = 1 - probs

            #to choose low confidence samples
            self.sampling_probs[seed_idxs] =probs
            for i in cum_scores_class:
                self.cum_sampling_probs[i] += probs[torch.nonzero(y == i).squeeze().detach().cpu()].sum()

            #weighted sampling. Changed to not require the probabilities to be normalized
            #indices for the probs tensor - need to remap to seed indices
            indices = torch.multinomial(probs, num_samples=self.batch_size, replacement=True)
            try:
                for i in y[torch.unique(indices)]:
                    self.count_dict_new[-1][i.item()]+=1
            except Exception as e:
                print('Error', e)
                print('Indices', indices)
                print('Unique', torch.unique(indices))
                

            og_indices = torch.tensor(seed_idxs, device= device)
            
            yield og_indices[indices] #major change here, mapped indices properly now
        self.train_sampling_probs = torch.zeros_like(self.sampling_probs)
        for i,j in enumerate(self.old_indices):
            self.train_sampling_probs[j] = self.sampling_probs[i].item()
        for i in self.cum_sampling_probs:
            self.cum_sampling_probs[i] /= len(torch.nonzero(self.all_y == i))
        print('Final cumulative sampling probability', self.cum_sampling_probs)


    def __len__(self):
        return len(self.data_source) // self.batch_size



def get_class_distribution(dataloader_obj, totl_labels, split):
    
    count_dict = {i: 0 for i in set(totl_labels)}
    
    if split == 'train':
        for _, lbl, _ in dataloader_obj:
            for l in lbl:
                count_dict[l.item()] += 1
    else:
        for _, lbl, _ in dataloader_obj:
            for l in lbl:
                count_dict[l.item()] += 1
            
    return count_dict


np.random.seed(0)
torch.manual_seed(0)

dataset_size = len(train_dataset)
dataset_indices = list(range(dataset_size))
#uniform random sampling with replacement
print('Making a random sampler')
dataset_indices = np.random.choice(dataset_indices, size = dataset_size)
train_sampler = SubsetRandomSampler(dataset_indices)

#weighted sampler
print('Making a weighted sampler')
if DS_NAME == 'kaokore':
    class_count = [i for i in train_dataset.count_dict.values()]
    totl_labels = train_dataset.labels
elif DS_NAME == 'WikiArt':
    class_count = [1953, 3186, 1059, 7528, 1755, 4508, 9130, 2977, 1460, 356, 894, 1574, 919, 4879, 1149, 951, 326, 1680, 3064, 4723, 821, 979, 81, 646, 232, 141, 66]
    totl_labels = ds['style']
    # with open('saves/wikiart_tot_lbls.pkl', 'rb') as f:
    #     totl_labels = pickle.load(f)    
else:
    dataloader = DataLoader(train_dataset, batch_size = BSZ,
                            shuffle = False, num_workers = 4)
    count_dict = {}
    totl_labels = []
    for _, labels, __ in tqdm(dataloader):
        for label in labels:
            if label.item() not in count_dict:
                count_dict[label.item()] = 1
            count_dict[label.item()] += 1
            totl_labels.append(label.item())
    print('Total labels:', count_dict, totl_labels)
    class_count = [i for i in count_dict.values()]
    with open(f'saves/{DS_NAME}_tot_lbls.pkl', 'wb') as f:
        pickle.dump(totl_labels, f)
class_weights = 1./torch.tensor(class_count, dtype=torch.float) 
print(class_weights.shape)# simple form of aggregate
print('Class Count:', class_count, 'Total length', len(class_count))

weighted_train_sampler = WeightedRandomSampler(weights = class_weights[totl_labels],
                                            num_samples = len(train_dataset),
                                            replacement = True)

#random shuffling

train_loader_rs = DataLoader(dataset = train_dataset, shuffle = False, batch_size = BSZ,
                            sampler = train_sampler)
train_loader_ws = DataLoader(dataset = train_dataset, shuffle = False, batch_size = BSZ,
                            sampler = weighted_train_sampler)
# print(get_class_distribution(train_loader_rs, totl_labels, 'train'))
# print(get_class_distribution(train_loader_ws, totl_labels, 'train'))


print('Making the ODIN sampler')
temperature = 1
dataset = train_dataset
batch_sz = BSZ
device = 'cuda'
step_sz = 0.05
art_model = stclf_model.to(device)
MODEL_TYPE = ['stsaclf', 'simple_odin'][0]
print('Total labels', totl_labels, set(totl_labels))
batch_sampler = OdinSamplerRB(art_model, dataset, totl_labels, batch_sz,
            step_sz, F.nll_loss, temperature, norm_std)
batch_sampler.update_local(art_model, temperature)


# for i, batch_indices in enumerate(batch_sampler):
#     batch_x, batch_y = torch.stack([dataset[idx][0] for idx in batch_indices]), torch.tensor([dataset[idx][1] for idx in batch_indices])
#     x,y = batch_x.cuda(), batch_y.cuda()
    # print(batch_sampler.sampling_probs.mean(), batch_sampler.sampling_probs.std())
    # print(batch_sampler.count_dict_new)

