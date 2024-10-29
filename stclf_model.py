import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import ProjectorBlock, SpatialAttn
from torchvision import models
from collections import OrderedDict


from utils import drop_connect
from pretrained_models import *
# from pacs_ds import class_names
# from kaokore_ds import class_names
# from wikiart_ds import class_names
from wikiart_emotions_ds import class_names

#previous best
class AttnResNet(nn.Module): #the vgg n densnet versions
    def __init__(self, num_classes, backbone:ResNetN, dropout_mode, p, attention=True, normalize_attn=True):
        super(AttnResNet, self).__init__()
        # conv blocks
        self.pretrained = backbone

        self.fhooks = []
        self.selected_out = OrderedDict()

        # attention blocks
        self.attention = attention
        if self.attention:

            self.project_ins = backbone.project_ins
            for i,p_name in enumerate(['projector0', 'projector1', 'projector2', 'projector3']):
                if backbone.project_ins[i] != backbone.project_ins[-1]:
                    setattr(self, p_name, ProjectorBlock(backbone.project_ins[i], backbone.project_ins[-1]))

            self.attn0 = SpatialAttn(in_features=backbone.project_ins[-1], normalize_attn=normalize_attn)# (batch_size,1,H,W), (batch_size,C)
            self.attn1 = SpatialAttn(in_features=backbone.project_ins[-1], normalize_attn=normalize_attn)
            self.attn2 = SpatialAttn(in_features=backbone.project_ins[-1], normalize_attn=normalize_attn)
            self.attn3 = SpatialAttn(in_features=backbone.project_ins[-1], normalize_attn=normalize_attn)

        # dropout selection for type of regularization
        self.dropout_mode, self.p = dropout_mode, p

        if self.dropout_mode == 'dropout':
            self.dropout = nn.Dropout(self.p)
        elif self.dropout_mode == 'dropconnect':
            self.dropout = drop_connect

        # final classification layer
        if self.attention:
            self.fc1 = nn.Linear(in_features=backbone.project_ins[-1] * 4, out_features=backbone.project_ins[-1], bias=True)
            self.classify = nn.Linear(in_features=backbone.project_ins[-1], out_features=num_classes, bias=True)
        else:
            self.fc1 = nn.Linear(in_features=backbone.project_ins[-1], out_features=backbone.project_ins[-1]//2, bias=True)
            self.classify = nn.Linear(in_features=backbone.project_ins[-1]//2, out_features=num_classes, bias=True)


    def forward_hook(self, layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = output
        return hook

    def forward(self, x):

        l0, l1, l2, l3, g = self.pretrained(x)

        # attention
        if self.attention:
            for i in range(4):
                if hasattr(self,f'projector{i}'):
                    locals()[f'c{i}'], locals()[f'g{i}'] = getattr(self,f'attn{i}')(getattr(self,f'projector{i}')(locals()[f'l{i}']), g)
                else:
                    locals()[f'c{i}'], locals()[f'g{i}'] = getattr(self,f'attn{i}')(locals()[f'l{i}'], g)
            
            all_locals = locals()
            global_feats = [all_locals[f'g{i}'] for i in range(4)]
            attn_maps = [all_locals[f'c{i}'] for i in range(4)]
            g = torch.cat(global_feats, dim=1) # batch_sizex3C

            # fc layer
            out = torch.relu(self.fc1(g)) # batch_sizexnum_classes

            if self.dropout_mode == 'dropout':
                out = self.dropout(out)
            elif self.dropout_mode == 'dropconnect':
                out = self.dropout(out, self.p, self.training)

        else:
            attn_maps = [None, None, None, None]
            out = self.fc1(torch.squeeze(g))

            if self.dropout_mode == 'dropout':
                out = self.dropout(out)
            elif self.dropout_mode == 'dropconnect':
                out = self.dropout(out, self.p, self.training)

        out = self.classify(out)
        return [out,]+ attn_maps + [g,]
    

# old best model
FFINETUNE = False
DS_NAME = ['Kaokore', 'PACS', 'WikiArt'][1]

NUM_CLASSES = len(class_names)
print('Num classes',NUM_CLASSES, class_names)

DROPOUT_P = 0.23
DROPOUT_TYPE = 'dropout'
DROPOUT_P = 0.23

stclf_model = AttnResNet(NUM_CLASSES,
                            ResNetN('resnet50','avgpool',
                                ['conv1', 'layer2','layer3','layer4'],
                                FFINETUNE),
                            DROPOUT_TYPE,
                            DROPOUT_P)