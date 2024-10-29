import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import wandb
import numpy as np

from kaokore_ds import *
from sampler_utils import *

def plot_softmax_distribution(softmax, class_names):
    # Create a bar plot for the softmax probabilities
    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = np.arange(len(class_names))
    ax.barh(y_pos, softmax.detach().cpu().numpy(), align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_names)
    ax.invert_yaxis()  # Invert to have the highest score on top
    ax.set_xlabel('Softmax Probability')
    ax.set_title('Softmax Distribution')
    return fig

def log_samples_to_wandb(samples, category):
    for class_name, sample_list in samples.items():
        for img_tensor, softmax in sample_list:
            # Convert image tensor back to PIL image for WandB logging
            img = transforms.ToPILImage()(img_tensor.squeeze())

            # Plot the softmax distribution
            fig = plot_softmax_distribution(softmax, class_names)

            # Log the image and the softmax distribution to WandB
            wandb.log({
                f"{category}_image_{class_name}": wandb.Image(img),
                f"{category}_softmax_distribution_{class_name}": wandb.Image(fig)
            })

            # Close the plot to free memory
            plt.close(fig)
            
DS_NAME = ['kaokore', 'wikiart', 'pacs'][0]

#sampler visualizations
wandb.init(project='ood_art_sampler_visualizations')

if DS_NAME == 'kaokore':
    class_names = ['noble', 'warrior', 'incarnation', 'commoner']#{0: 'noble', 1: 'warrior', 2: 'incarnation', 3: 'commoner'}
columns = ['Epoch', 'Index', 'Class', 'Confidence score', 'Image']
RUN = 17
MODEL_TYPE = 'stsaclf'
SAMPLER_TYPE = 'odin'
epochs = [33,]#[0,33,49]
device = 'cuda'

print('Getting label distributions with ODIN sampling')
root_path = f'saves/ood_art_tests_v2/experiment_{RUN}'
model = stclf_model.to(device)
dataset = test_dataset
test_loader = DataLoader(test_dataset, batch_size = 1)

#tracking metrics
correct_samples = {class_name: [] for class_name in class_names}
incorrect_samples = {class_name: [] for class_name in class_names}

for epoch in epochs:
    
    # Dictionary to store the batch indices for each class
    class_indices = defaultdict(list)
    
    model.load_state_dict(torch.load(f"{root_path}/model_{MODEL_TYPE}_{SAMPLER_TYPE}_{epoch}.pth"))
    if SAMPLER_TYPE == 'odin':
        temperature = 125
    else:
        temperature = 1.0
    batch_sampler.update_local(model, temperature)
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            true_class = target.item()
            
            output = model(data)
            if SAMPLER_TYPE == 'odin':
                softmax = F.softmax(output[0]/temperature, dim=1)
                criterion = F.nll_loss
                eps = 0.005
                with torch.inference_mode(False):
                    x = data.clone()

                    with torch.enable_grad():
                        x = Variable(x, requires_grad=True)
                        
                        #temperature based softmax scaling for confidence calibration. doesn't affect model accuracy, but influences confidence.
                        
                        outputs = model(x)
                        if isinstance(outputs, list):
                            logits = outputs[0] / temperature
                        if target is None:
                            target = logits.max(dim=1).indices
                        # print(logits.shape, target.shape, x.shape, len(outputs))
                        loss = criterion(logits, target)
                        loss.backward()
                        
                        #gradient noise based data preprocessing
                        gradient = torch.sign(x.grad.data)

                        if norm_std is not None:
                            for i, std in enumerate(norm_std):
                                gradient.index_copy_(
                                    1,
                                    torch.LongTensor([i]).to(gradient.device),
                                    gradient.index_select(1, torch.LongTensor([i]).to(gradient.device)) / std,
                                )

                        x_hat = eps * gradient
                        data = torch.cat((x, x_hat), dim =2)
                        
            else:
                softmax = F.softmax(output[0], dim=1)
            _, pred = torch.max(output[0], 1)
            
            
            # Separate correct and incorrect classifications
            if pred == true_class:
                correct_samples[class_names[true_class]].append((data.detach().cpu(), softmax.squeeze()))
            else:
                incorrect_samples[class_names[true_class]].append((data.detach().cpu(), softmax.squeeze()))

    #get repeated entries for each class
    # for i, batch_indices in enumerate(batch_sampler):
    #     batch_x, batch_y = torch.stack([dataset[idx][0] for idx in batch_indices]), torch.tensor([dataset[idx][1] for idx in batch_indices])
    



    # Log correct and incorrect samples
log_samples_to_wandb(correct_samples, "correct")
log_samples_to_wandb(incorrect_samples, "incorrect")

# def class_distro_chart(classes_dict, class_names, file_name):
#     plt.figure(figsize=(15,8))
#     class_distro = {class_names[i]: classes_dict[i] for i in classes_dict}
#     chart = sns.barplot(data = pd.DataFrame.from_dict([class_distro]).melt(), x = "variable", y="value", hue="variable").set_title('Natural Images Class Distribution')
#     fig = chart.get_figure()
#     fig.savefig(f'figures/{file_name}.png')
    
# if __name__ == '__main__':
#     kaokore_status_class_names = {0: 'noble', 1: 'warrior', 2: 'incarnation', 3: 'commoner'}
#     # class_distro_chart(train_dataset.count_dict,
#     #                    kaokore_status_class_names,
#     #                    'original_train_status_classdistro')
#     # class_distro_chart(get_class_distribution(train_loader_ws, train_dataset, 'train'),
#     #                    kaokore_status_class_names,
#     #                    'classweighted_sampling_status_classdistro')
    
#     # class_distro_chart(batch_sampler.count_dict_new,
#     #                    kaokore_status_class_names,
#     #                    'odin_sampling_status_classdistro')
    
#     wrun = wandb.init(project='ood_art_visualization_tests')
    
#     columns = ['Epoch', 'Index', 'Class', 'Confidence score', 'Image']
#     RUN = 16
#     EPOCH = 29
#     least_conf = wandb.Table(columns = columns)
#     most_conf = wandb.Table(columns = columns)
#     num_rows = 64
#     root_path = 'ood_art_tests_v2/experiment_{RUN}/'
    
#     batch_sampler.train_sampling_probs = torch.load(f'saves/{root_path}/sampler_odin_Both_{EPOCH}.pt')
    
    
    
    
#     for cls in list(kaokore_status_class_names.keys()):
#         cls_indices = (batch_sampler.all_y == cls).nonzero()
#         og_indices = torch.tensor(batch_sampler.old_indices).to(device)
#         sampling_probs_cls = batch_sampler.sampling_probs[cls_indices]
        
#         sampling_scores, indices=torch.sort(sampling_probs_cls, dim = 0)
#         rev_sampling_scores, rev_indices=torch.sort(sampling_probs_cls, dim = 0, descending = True)
#         sampler_indices, rev_sampler_indices = cls_indices[indices], cls_indices[rev_indices]
#         og_indices, rev_og_indices = og_indices[sampler_indices], og_indices[rev_sampler_indices]
#         indices, rev_indices = og_indices.squeeze().tolist(), rev_og_indices.squeeze().tolist()
#         sampling_scores, rev_sampling_scores = sampling_scores.squeeze(), rev_sampling_scores.squeeze()
        
#         #careful about the mapping, both the sampler and the train dataset needed to be mapped
#         print(cls, indices, batch_sampler.all_y[sampler_indices].squeeze().tolist(),
#               [train_dataset[i][1] for i in indices], sampling_scores)
    
#         most_confident_vals = [[i, None] for i in sampling_scores[:num_rows]]
#         for i,ind in enumerate(indices[:num_rows]):
#             most_confident_vals[i][1] = train_dataset[ind][0]
            
#         rev_indices = indices[:len(indices)-num_rows-1:-1] #reverse the last nrows indices to not sort a second time
#         # print(len(rev_indices))
#         # print(sampling_scores[:num_rows])
#         least_confident_vals = [[i, None] for i in rev_sampling_scores[:num_rows]]
#         for i,ind in enumerate(rev_indices):
#             least_confident_vals[i][1] = train_dataset[ind][0]
        
#         #sanity check
#         for i in range(len(most_confident_vals)):
#             if most_confident_vals[i][1] is not None and least_confident_vals[i][1] is not None:
#                 print(most_confident_vals[i][0], least_confident_vals[i][0])
#             else:
#                 assert False, "Sampling score images not indexed"
                
#         for i in range(num_rows):
#             l_img = wandb.Image(least_confident_vals[i][1])
#             m_img = wandb.Image(most_confident_vals[i][1])
#             row_tab1 = [EPOCH, 
#                         indices[i],
#                    kaokore_status_class_names[train_dataset[indices[i]][1]], 
#                    least_confident_vals[i][0].item(),
#                    l_img]
#             row_tab2 = [EPOCH,
#                         rev_indices[i],
#                    kaokore_status_class_names[train_dataset[rev_indices[i]][1]], 
#                    most_confident_vals[i][0].item(),
#                    m_img]
#             least_conf.add_data(*row_tab1)
#             most_conf.add_data(*row_tab2)
        
#     # Create a wandb Artifact
#     artifact = wandb.Artifact(
#         name="sampling_score_images",
#         type="kaokore_dataset",
#     )
    
#     artifact.add(least_conf, "least_scores")
#     artifact.add(most_conf, "most_scores")
#     wrun.log_artifact(artifact)
    
#     wrun.finish()
     
           