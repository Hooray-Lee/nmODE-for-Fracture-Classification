import torchvision.models as models
from torch.nn import Parameter
from util import *
import torch
import torch.nn as nn
from transformer import AttentionBlock
from acsconv.converters import ACSConverter

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCNResnet(nn.Module):
    def __init__(self, model, num_classes, in_channel=300, t=0, adj_file=None):
        super(GCNResnet, self).__init__()
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        self.num_classes = num_classes
        self.pooling = nn.MaxPool2d(14, 14)

        self.gc1 = GraphConvolution(in_channel, 1024)
        self.gc2 = GraphConvolution(1024, 2048)
        self.relu = nn.LeakyReLU(0.2)

        _adj = gen_A(num_classes, t, adj_file)
        self.A = Parameter(torch.from_numpy(_adj).float())
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, feature, inp):
        feature = self.features(feature)
        feature = self.pooling(feature)
        feature = feature.view(feature.size(0), -1)


        inp = inp[0]
        adj = gen_adj(self.A).detach()
        x = self.gc1(inp, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)

        x = x.transpose(0, 1)
        x = torch.matmul(feature, x)
        return x

    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.gc1.parameters(), 'lr': lr},
                {'params': self.gc2.parameters(), 'lr': lr},
                ]



def gcn_resnet101(num_classes, t, pretrained=True, adj_file=None, in_channel=300):
    model = models.resnet101(pretrained=pretrained)
    return GCNResnet(model, num_classes, t=t, adj_file=adj_file, in_channel=in_channel)


class GCN_DETR_Resnet(nn.Module):
    def __init__(self, num_classes, t, adj_file, d_model=256, num_heads=4, pretrained=True):
        super().__init__()
        self.num_classes = num_classes

        # Backbone
        backbone = models.resnet101(pretrained=pretrained)
        self.features = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu,
            backbone.maxpool, backbone.layer1, backbone.layer2,
            backbone.layer3, backbone.layer4
        )
        self.input_proj = nn.Conv2d(2048, d_model, kernel_size=1)
        self.pos_encoder = PositionalEncoding2D(d_model, 14, 14)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward=1024)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        decoder_layer = nn.TransformerDecoderLayer(d_model, num_heads, dim_feedforward=1024)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)

        # Learnable Query
        self.query_embed = nn.Parameter(torch.randn(num_classes, d_model))

        # GCN
        self.gc1 = GraphConvolution(d_model, 1024)
        self.gc2 = GraphConvolution(1024, d_model)
        self.relu = nn.LeakyReLU(0.2)

        _adj = gen_A(num_classes, t, adj_file)
        self.A = Parameter(torch.from_numpy(_adj).float(), requires_grad=False)
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, x, _=None):
        B = x.size(0)
        x = self.features(x)                     
        x = self.input_proj(x)          
        x = self.pos_encoder(x)
        x = x.flatten(2).permute(2, 0, 1)       

        memory = self.encoder(x)               

        tgt = self.query_embed.unsqueeze(1).repeat(1, B, 1) 
        hs = self.decoder(tgt, memory)            
        hs = hs.transpose(0, 1)                  

        adj = gen_adj(self.A).detach()
        gcn_feat = self.gc1(self.query_embed, adj)  
        gcn_feat = self.relu(gcn_feat)
        gcn_feat = self.gc2(gcn_feat, adj)       
        gcn_feat = gcn_feat.unsqueeze(0).repeat(B, 1, 1) 

        out = (hs * gcn_feat).sum(-1)      
        return out, hs, self.query_embed
    
    def get_config_optim(self, lr, lrp):
        return [
            {'params': self.features.parameters(), 'lr': lr * lrp},
            {'params': self.input_proj.parameters(), 'lr': lr},
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.decoder.parameters(), 'lr': lr},
            {'params': self.gc1.parameters(), 'lr': lr},
            {'params': self.gc2.parameters(), 'lr': lr},
            {'params': [self.query_embed], 'lr': lr}, 
        ]

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, height, width):
        super().__init__()
        self.height = height
        self.width = width
        self.d_model = d_model
        self.register_buffer('pe', self._build_pe())

    def _build_pe(self):
        pe = torch.zeros(self.d_model, self.height, self.width)
        if self.d_model % 4 != 0:
            raise ValueError("d_model must be divisible by 4 for 2D encoding.")
        d_model = int(self.d_model / 2)

        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., self.width).unsqueeze(1)
        pos_h = torch.arange(0., self.height).unsqueeze(1)

        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, self.height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, self.height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, self.width)
        pe[d_model+1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, self.width)

        return pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :, :x.size(2), :x.size(3)]

class GCN_DETR_GATN_Resnet(nn.Module):
    def __init__(self, num_classes, t, adj_file, d_model=256, num_heads=4, pretrained=True):
        super().__init__()
        self.num_classes = num_classes

        # Backbone
        backbone = models.resnet101(pretrained=pretrained)
        self.features = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu,
            backbone.maxpool, backbone.layer1, backbone.layer2,
            backbone.layer3, backbone.layer4
        )
        self.input_proj = nn.Conv2d(2048, d_model, kernel_size=1)
        self.pos_encoder = PositionalEncoding2D(d_model, 14, 14)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward=1024)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        decoder_layer = nn.TransformerDecoderLayer(d_model, num_heads, dim_feedforward=1024)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)

        # Learnable Query
        self.query_embed = nn.Parameter(torch.randn(num_classes, d_model))

        # GCN
        self.gc1 = GraphConvolution(d_model, 1024)
        self.gc2 = GraphConvolution(1024, d_model)
        self.relu = nn.LeakyReLU(0.2)

        _adj = gen_A(num_classes, t, adj_file)
        self.A = Parameter(torch.from_numpy(_adj).float(), requires_grad=False)

        # Graph Transformer Layer
        self.graph_transformer = AttentionBlock(num_classes)

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, x, _=None):
        B = x.size(0)
        x = self.features(x)                      
        x = self.input_proj(x)                
        x = self.pos_encoder(x)
        x = x.flatten(2).permute(2, 0, 1)      

        memory = self.encoder(x)     

        tgt = self.query_embed.unsqueeze(1).repeat(1, B, 1)
        hs = self.decoder(tgt, memory)            
        hs = hs.transpose(0, 1)               

        A_1 = self.A.unsqueeze(0).cuda() 
        A_2 = self.A.unsqueeze(0).cuda()
        adj_trans, _ = self.graph_transformer(A_1, A_2)
        adj_trans = torch.squeeze(adj_trans, 0) + torch.eye(self.num_classes).type_as(adj_trans)
        adj_trans = gen_adj(adj_trans)

        gcn_feat = self.gc1(self.query_embed, adj_trans)
        gcn_feat = self.relu(gcn_feat)
        gcn_feat = self.gc2(gcn_feat, adj_trans)
        gcn_feat = gcn_feat.unsqueeze(0).repeat(B, 1, 1) 

        out = (hs * gcn_feat).sum(-1)      
        return out, hs, self.query_embed

    def get_config_optim(self, lr, lrp):
        return [
            {'params': self.features.parameters(), 'lr': lr * lrp},
            {'params': self.input_proj.parameters(), 'lr': lr},
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.decoder.parameters(), 'lr': lr},
            {'params': self.gc1.parameters(), 'lr': lr},
            {'params': self.gc2.parameters(), 'lr': lr},
            {'params': [self.query_embed], 'lr': lr},
            {'params': self.graph_transformer.parameters(), 'lr': lr},
        ]

class SCUCT_DETR_Resnet(nn.Module):
    def __init__(self, num_classes, d_model=256, num_heads=4, pretrained=True):
        super().__init__()

        backbone2d = models.resnet18(pretrained=pretrained)
        self.backbone = ACSConverter(backbone2d)
        self.input_proj = nn.Conv3d(512, d_model, kernel_size=1)  
        self.pos_encoder = nn.Identity() 

        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward=1024)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        decoder_layer = nn.TransformerDecoderLayer(d_model, num_heads, dim_feedforward=1024)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)

        self.query_embed = nn.Parameter(torch.randn(num_classes, d_model))
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        feat = self.backbone(x) 
        feat = self.input_proj(feat) 
        B, C, D, H, W = feat.shape
        feat_flat = feat.view(B, C, -1).permute(2, 0, 1)  
        memory = self.encoder(feat_flat) 
        query = self.query_embed.unsqueeze(1).repeat(1, B, 1)  
        tgt = torch.zeros_like(query)
        hs = self.decoder(tgt, memory)  
        hs = hs.permute(1, 0, 2) 
        out = self.classifier(hs) 
        out = out.mean(dim=-1) 
        return out

    def get_config_optim(self, lr, lrp):
        return [
            {'params': self.backbone.parameters(), 'lr': lrp},
            {'params': self.input_proj.parameters()},
            {'params': self.encoder.parameters()},
            {'params': self.decoder.parameters()},
            {'params': self.classifier.parameters()},
        ]
        
class ACSBackbone(nn.Module):
    def __init__(self, pretrained=True, use_r3d_fallback=False):
        super().__init__()
        self.use_r3d = False
        
        if not use_r3d_fallback:
            
            backbone2d_full = models.resnet18(pretrained=pretrained)
            modules = list(backbone2d_full.children())[:-2]
            backbone2d = nn.Sequential(*modules)
            self.backbone = ACSConverter(backbone2d)
            self.out_ch = 512
        else:
            try:
                r3d = models.video.r3d_18(pretrained=pretrained)
                modules = list(r3d.children())[:-2]
                self.backbone = nn.Sequential(*modules)
                self.out_ch = 512
                self.use_r3d = True
            except Exception:
                backbone2d = models.resnet18(pretrained=pretrained)
                self.backbone = ACSConverter(backbone2d)
                self.out_ch = 512

    def forward(self, x):
        self.backbone.to(x.device)
        if x.dim() == 5 and x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1, 1)
        return self.backbone(x)


class SequenceModule(nn.Module):
    def __init__(self, in_dim, d_model=256, lstm_hidden=256):
        super().__init__()
        self.pool_to_d = nn.Linear(in_dim, d_model)
        self.lstm = nn.LSTM(input_size=d_model,
                            hidden_size=lstm_hidden,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.out_proj = nn.Linear(2 * lstm_hidden, d_model)

    def forward(self, vec_seq):
        x = self.pool_to_d(vec_seq)   
        lstm_out, _ = self.lstm(x)   
        out = self.out_proj(lstm_out)  
        return out 


class DETRModule(nn.Module):
    def __init__(self, num_classes, d_model=256, num_heads=4, num_encoder_layers=1, num_decoder_layers=3):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward=1024)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model, num_heads, dim_feedforward=1024)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.query_embed = nn.Parameter(torch.randn(num_classes, d_model))

    def forward(self, memory):
        
        mem = self.encoder(memory) 
        B = mem.size(1)
        tgt = self.query_embed.unsqueeze(1).repeat(1, B, 1) 
        hs = self.decoder(tgt, mem) 
        hs = hs.transpose(0, 1) 
        return hs, self.query_embed


class GCNModule(nn.Module):
    def __init__(self, num_classes, d_model, t=0, adj_file=None):
        super().__init__()
        self.gc1 = GraphConvolution(d_model, 1024)
        self.gc2 = GraphConvolution(1024, d_model)
        self.relu = nn.LeakyReLU(0.2)
        _adj = gen_A(num_classes, t, adj_file)
        
        self.A = Parameter(torch.from_numpy(_adj).float(), requires_grad=False)

    def forward(self, query_embed):
        
        adj = gen_adj(self.A).detach()
        x = self.gc1(query_embed, adj)
        x = self.relu(x)
        x = self.gc2(x, adj) 
        return x

class Integrated3DModel(nn.Module):
    def __init__(self,
                 num_classes,
                 use_acs=True,
                 use_sequence=False,
                 use_detr=False,
                 use_gcn=False,
                 d_model=256,
                 lstm_hidden=256,
                 num_heads=4,
                 pretrained=True,
                 t=0,
                 adj_file=None):
        super().__init__()
        self.num_classes = num_classes
        self.use_sequence = use_sequence
        self.use_detr = use_detr
        self.use_gcn = use_gcn
        self.d_model = d_model

        if use_acs:
            self.backbone = ACSBackbone(pretrained=pretrained)
        else:
            self.backbone = ACSBackbone(pretrained=pretrained)
        backbone_ch = self.backbone.out_ch

        self.input_proj3d = nn.Conv3d(backbone_ch, d_model, kernel_size=1)

        self.pool_proj = nn.Linear(backbone_ch, d_model)

        if use_sequence:
            self.sequence = SequenceModule(in_dim=backbone_ch, d_model=d_model, lstm_hidden=lstm_hidden)

        if use_detr:
            self.detr = DETRModule(num_classes, d_model=d_model, num_heads=num_heads)

        if use_gcn:
            self.gcn = GCNModule(num_classes, d_model, t=t, adj_file=adj_file)

        if not use_gcn:
            self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        B, T = x.size(0), x.size(1)
        H, W, D = x.size(2), x.size(3), x.size(4)

        x_flat = x.view(B * T, 1, H, W, D)            
        feat3d = self.backbone(x_flat)                 

        if self.use_sequence:
            pooled = feat3d.mean(dim=[2, 3, 4])             
            vec = pooled.view(B, T, -1)                     
            seq_feat = self.sequence(vec)                    
            seq_flat = seq_feat.contiguous().view(B * T, -1) 

            if self.use_detr:
                memory = seq_flat.unsqueeze(0)               
                hs_flat, query = self.detr(memory)            
                hs = hs_flat.view(B, T, self.num_classes, -1) 

                if self.use_gcn:
                    gcn_feat = self.gcn(query)               
                    gcn_feat_bt = gcn_feat.unsqueeze(0).unsqueeze(0).repeat(B, T, 1, 1)
                    out = (hs * gcn_feat_bt).sum(-1)
                    return out, hs, query
                else:
                    cls = self.classifier(hs)                
                    out = cls.mean(dim=-1)                  
                    return out, hs, query

            else:
                if self.use_gcn:
                    query_embed = self.detr.query_embed if hasattr(self, 'detr') else torch.randn(self.num_classes, self.d_model, device=seq_flat.device)
                    gcn_feat = self.gcn(query_embed)    
                    out_flat = torch.matmul(seq_flat, gcn_feat.t()) 
                else:
                    out_flat = self.classifier(seq_flat)   
                out = out_flat.view(B, T, -1)
                return out, None, None

        if self.use_detr:
            feat_proj = self.input_proj3d(feat3d)  
            B_T = B * T
            _, Cd, Dp, Hp, Wp = feat_proj.shape
            memory = feat_proj.contiguous().view(B_T, Cd, -1).permute(2, 0, 1) 
            hs = self.detr(memory)                       
            if isinstance(hs, tuple):
                hs_flat, query = hs
            else:
                hs_flat, query = hs, self.detr.query_embed
            hs_flat = hs_flat.view(B, T, self.num_classes, -1)  

            if self.use_gcn:
                gcn_feat = self.gcn(query)                    
                gcn_feat_bt = gcn_feat.unsqueeze(0).unsqueeze(0).repeat(B, T, 1, 1)
                out = (hs_flat * gcn_feat_bt).sum(-1)
                return out, hs_flat, query
            else:
                cls = self.classifier(hs_flat)                 
                out = cls.mean(dim=-1)                       
                return out, hs_flat, query
        pooled = feat3d.mean(dim=[2, 3, 4])                  
        proj = self.pool_proj(pooled)           

        if self.use_gcn:
            query_embed = self.detr.query_embed if hasattr(self, 'detr') else torch.randn(self.num_classes, self.d_model, device=proj.device)
            gcn_feat = self.gcn(query_embed)          
            out_flat = torch.matmul(proj, gcn_feat.t())      
        else:
            out_flat = self.classifier(proj)           

        out = out_flat.view(B, T, -1)                   
        return out, None, None

    def _make_dummy_query(self):
        return nn.Parameter(torch.randn(self.num_classes, self.d_model))

    def get_config_optim(self, lr, lrp):
        params = []
        if hasattr(self, 'backbone'):
            params.append({'params': self.backbone.parameters(), 'lr': lr * lrp})
        if hasattr(self, 'input_proj3d'):
            params.append({'params': self.input_proj3d.parameters(), 'lr': lr})
        if hasattr(self, 'pool_proj'):
            params.append({'params': self.pool_proj.parameters(), 'lr': lr})
        if hasattr(self, 'sequence'):
            params.append({'params': self.sequence.parameters(), 'lr': lr})
        if hasattr(self, 'detr'):
            params.append({'params': self.detr.parameters(), 'lr': lr})
        if hasattr(self, 'gcn'):
            params.append({'params': self.gcn.parameters(), 'lr': lr})
        if hasattr(self, 'classifier'):
            params.append({'params': self.classifier.parameters(), 'lr': lr})
        return params