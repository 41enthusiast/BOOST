import torch
from typing import Optional, Callable, List
from torch import Tensor
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, SubsetRandomSampler
import pickle
import numpy as np


class BOOST(torch.utils.data.Sampler):
    def __init__(self, model, model_type, data_source, totl_labels, batch_size, step_sz, loss, temperature, norm_std, replacement=True, device = 'cuda', aggr_mode = 'BOTH'):
        self.data_source = data_source
        self.data_lbls = totl_labels
        self.batch_size = batch_size
        self.replacement = replacement
        self.device = device

        self.step_sz = step_sz
        self.loss = loss
        self.temperature = temperature
        self.norm_std = norm_std

        self.model = model
        self.model_type = model_type

        self.sampling_probs = torch.zeros(len(data_source)).to(device)
        self.aggr_mode = aggr_mode
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
                if self.model_type == 'single_output':
                    logits = model(x) / temperature
                elif self.model_type == 'multiple_outputs':
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
            if self.model_type == 'single_output':
                results = self.model(x_hat).softmax(dim=1)
            elif self.model_type == 'multiple_outputs':
                results = self.model(x_hat)[0].softmax(dim=1)

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

        for i in range(0, n, self.batch_size):
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
            temp_group = all_scores[torch.nonzero(self.all_y == i).squeeze()]
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
            if self.aggr_mode == 'Both': #find a better way to accumulate the probs
                probs = probs * cum_confidence # or cum confidence as mean, the indv probs as std dev instead of multiplyin it
            elif self.aggr_mode == 'Aggr':
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


            og_indices = torch.tensor(seed_idxs, device= self.device)

            yield og_indices[indices] #major change here, mapped indices properly now
        self.train_sampling_probs = torch.zeros_like(self.sampling_probs)
        for i,j in enumerate(self.old_indices):
            self.train_sampling_probs[j] = self.sampling_probs[i].item()
        for i in self.cum_sampling_probs:
            self.cum_sampling_probs[i] /= len(torch.nonzero(self.all_y == i))
        print('Final cumulative sampling probability', self.cum_sampling_probs)


    def __len__(self):
        return len(self.data_source) // self.batch_size

    def ordered_sampler(self, model, class_names, display_dataset, num_rows = 16, sort_order = 'descending', save_path = None, temperature = 1):
        temp_dataset = self.data_source
        if save_path:
            self.train_sampling_probs = torch.load(f'saves/{save_path[0]}/sampler_odin_Both_{save_path[1]}.pt') # experiment name and epoch
        else:
            self.update_local(model, temperature)
            for _ in self:
                pass
        self.data_source = display_dataset
        for cls in list(class_names.keys()):
            cls_indices = (self.all_y == cls).nonzero()
            og_indices = torch.tensor(self.old_indices).to(self.device)
            sampling_probs_cls = self.sampling_probs[cls_indices]

            sampling_scores, indices=torch.sort(sampling_probs_cls, dim = 0)
            rev_sampling_scores, rev_indices=torch.sort(sampling_probs_cls, dim = 0, descending = True)
            sampler_indices, rev_sampler_indices = cls_indices[indices], cls_indices[rev_indices]
            og_indices, rev_og_indices = og_indices[sampler_indices], og_indices[rev_sampler_indices]
            indices, rev_indices = og_indices.squeeze().tolist(), rev_og_indices.squeeze().tolist()
            sampling_scores, rev_sampling_scores = sampling_scores.squeeze(), rev_sampling_scores.squeeze()

            #careful about the mapping, both the sampler and the train dataset needed to be mapped
            print(cls, indices, self.all_y[sampler_indices].squeeze().tolist(),
                [self.data_source[i][1] for i in indices], sampling_scores)

            if sort_order == 'descending':
                for i in range(0, len(sampling_scores), num_rows):
                    high_sampl_scores = [[i, None] for i in sampling_scores[i:i+num_rows]]
                    score_labels = [self.data_source[ind][1] for ind in indices[i:i+num_rows]]
                    score_indices = [self.data_source[ind][2] for ind in indices[i:i+num_rows]]
                    for j,ind in enumerate(indices[i:i+num_rows]):
                        high_sampl_scores[j][1] = self.data_source[ind][0]
                    yield high_sampl_scores, score_labels, score_indices
            else:
                for i in range(len(sampling_scores), 0, -num_rows):
                    rev_indices = indices[max(0, i-num_rows):i]
                    low_sampl_scores = [[i, None] for i in rev_sampling_scores[:num_rows]]
                    score_labels = [self.data_source[ind][1] for ind in rev_indices[i:i+num_rows]]
                    score_indices = [self.data_source[ind][2] for ind in rev_indices[i:i+num_rows]]
                    for j,ind in enumerate(rev_indices):
                        low_sampl_scores[j][1] = self.data_source[ind][0]
                    yield low_sampl_scores, score_labels, score_indices
            self.data_source = temp_dataset

def get_total_labels(dataset, dataset_name, batch_sz, save = False):
    if dataset_name == 'kaokore':
        class_count = [i for i in dataset.count_dict.values()]
        totl_labels = dataset.labels
    elif dataset_name == 'WikiArt':
        class_count = [1953, 3186, 1059, 7528, 1755, 4508, 9130, 2977, 1460, 356, 894, 1574, 919, 4879, 1149, 951, 326, 1680, 3064, 4723, 821, 979, 81, 646, 232, 141, 66]
        totl_labels = dataset['style']
    else:
        dataloader = DataLoader(dataset, batch_size = batch_sz,
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

    class_weights = 1./torch.tensor(class_count, dtype=torch.float)
    print('Class Count:', class_count, 'Total length', len(class_count))

    if save:
        with open(f'saves/{dataset_name}_tot_lbls.pkl', 'wb') as f:
            pickle.dump(totl_labels, f)

    return totl_labels, class_weights

def make_BOOST_sampler(art_model, dataset, dataset_name, norm_std, batch_sz = 16, step_sz = 0.05, temperature = 1, loss_fn = F.nll_loss, device = 'cuda'):

    totl_labels, _ = get_total_labels(dataset, dataset_name, batch_sz)
    art_model.to(device)

    batch_sampler = BOOST(art_model, 'multiple_outputs', dataset, totl_labels, batch_sz,
            step_sz, loss_fn, temperature, norm_std)
    batch_sampler.update_local(art_model, temperature)
    return batch_sampler

def make_random_sampler(dataset, batch_size = 16):
    dataset_size = len(dataset)
    dataset_indices = list(range(dataset_size))
    #uniform random sampling with replacement
    print('Making a random sampler')
    dataset_indices = np.random.choice(dataset_indices, size = dataset_size)
    random_sampler = SubsetRandomSampler(dataset_indices)

    data_loader_rs = DataLoader(dataset = dataset, shuffle = False, batch_size = batch_size,
                            sampler = random_sampler)
    return data_loader_rs, random_sampler