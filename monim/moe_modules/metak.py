import torch
import numpy as np
from torch import nn
import math
import os

# 定义
def hook(module,input,output):
    print('----')
    print(output.shape)
    print(output)
# 注册


class AdaptiveCombiner(nn.Module):
    r""" Adaptive knn-mt Combiner """
    def __init__(self, 
                max_k,
                use_k=32,
                k_trainable = True,
                lambda_trainable = True,
                temperature_trainable = True, 
                **kwargs
                ):
        super().__init__()
        self.model = MetaKNetwork(max_k, use_k=use_k, 
        k_trainable=k_trainable, lambda_trainable=lambda_trainable, temperature_trainable=temperature_trainable,**kwargs)
        
        self.max_k = max_k
        self.use_k = use_k
        self.k_trainable = k_trainable
        self.lambda_trainable = lambda_trainable
        self.temperature_trainable = temperature_trainable
        self.kwargs = kwargs 
        self.mask_for_distance = None

        # check 
        assert self.lambda_trainable or "log_lambda" in kwargs, \
            "if lambda is not trainable, you should provide a fixed Tensor([1, 2]) log_lambda value"
        assert self.temperature_trainable or "temperature" in kwargs, \
            "if temperature is not trainable, you should provide a fixed temperature"
        
        # self.k = None if self.k_trainable else kwargs["k"]
        self.log_lambda = None if self.lambda_trainable else kwargs["log_lambda"]
        self.temperature = None if self.temperature_trainable else kwargs["temperature"]


    def get_knn_prob(self, inputs, device="cuda:0"):
        vals, distances, tgt = inputs['val_id'], inputs['distance'], inputs['tgt']
        metak_outputs = self.model(inputs)
        # print(metak_outputs["lambda_net_output"])
        # print(metak_outputs["k_net_output"][0])
        # print(metak_outputs["temperature_net_output"])
        # import pdb; pdb.set_trace()
        # if self.lambda_trainable:
        #     self.log_lambda = metak_outputs["lambda_net_output"]
        
        if self.temperature_trainable:
            self.temperature = metak_outputs["temperature_net_output"]
        
        if self.lambda_trainable:
            # generate mask_for_distance just for once
            if not hasattr(self, "mask_for_distance") or self.mask_for_distance is None:
                self.mask_for_distance = self._generate_mask_for_distance(self.max_k, device) # [R_K, K]
            
            probs = metak_outputs["lambda_k_net_output"] # [B, R_K]
            self.log_lambda = probs[:, 0]
            self.k_probs = probs[:, 1:]
            B, K = vals.size()
            R_K = self.k_probs.size(-1)

            distances = distances.unsqueeze(-2).expand(B, R_K, K)
            distances = distances * self.mask_for_distance  # [B, R_K, K]
            if self.temperature_trainable:
                temperature = self.temperature.unsqueeze(-1).expand(B, R_K, K)
            else:
                temperature = self.temperature
            # print(distances)
            # print(self.temperature.squeeze(-1))
            distances = - distances / temperature
            distances = torch.clip(distances, min=-1e5)
            knn_weight = torch.log_softmax(distances, dim=-1)  # [B, R_K, K]
            # [B, K]
            weight_sum_knn_weight = self.k_probs.unsqueeze(-1) + knn_weight
            weight_sum_knn_weight = torch.logsumexp(weight_sum_knn_weight, dim=-2)
            # [B, K]
            index_mask = torch.eq(vals, tgt.unsqueeze(-1)).int() # 没有这个int，是不能判断==0的...
            index_mask[index_mask == 0] = -10000 # for stability
            index_mask[index_mask == 1] = 0
            # [B]
            knn_scores = torch.logsumexp(index_mask + weight_sum_knn_weight, dim=-1)
            

        # else:
        #     # if k is not trainable, the process of calculate knn probs is same as vanilla knn-mt
        #     knn_prob = calculate_knn_prob(vals, distances, self.probability_dim,
        #                 self.temperature, device=device)

        return knn_scores
    
    def get_prob(self, inputs, lm_scores=None, knn_scores=None, device="cuda:0", **kwargs):
        extra = {}
        inputs['distance'] = inputs['old_distance']
        knn_scores = self.get_knn_prob(inputs, device=device)
        extra['log_lambda'] = self.log_lambda
        extra['knn_scores'] = knn_scores
        extra['temp'] = self.temperature
        if lm_scores is None:
            return None, extra
        
        prob = torch.stack((knn_scores, self.log_lambda+lm_scores), dim=-1)
        prob = torch.logsumexp(prob, dim=-1)
        return prob, extra

    def get_loss(self, inputs, lm_scores=None, knn_scores=None, device="cuda:0", **kwargs):
        prob, extra = self.get_prob(inputs, lm_scores, knn_scores, device, **kwargs)
        cross_entropy = -prob
        loss = cross_entropy.mean() / np.log(2)
        if kwargs.get('l1', 0):
            loss = loss + kwargs['l1'] * torch.abs(self.k_probs.exp()).sum() / self.log_lambda.shape[0]
        else:
            print('shit!')
            1/0
        return loss, extra


    @staticmethod
    def _generate_mask_for_distance(max_k, device):
        k_mask = torch.empty((max_k, max_k)).fill_(999.)
        k_mask = torch.triu(k_mask, diagonal=1) + 1
        # power_index = torch.tensor([pow(2, i) - 1 for i in range(0, int(math.log(max_k, 2)) + 1)])
        power_index = [1023]
        k_mask = k_mask[power_index]
        k_mask.requires_grad = False
        k_mask = k_mask.to(device)
        return k_mask


class LeakyReLUNet(nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_feat, out_feat),
            nn.LeakyReLU(),
            nn.Linear(out_feat, out_feat),
        )

    def forward(self, features):
        return self.model(features)


class MLPMOE(nn.Module):
    def __init__(self,
                 feature_size=None,
                 hidden_units=128,
                 dis_hidden_units=256,
                 nlayers=3,
                 dropout=0,
                 non_ctxt_dim=512,
                 ):
        super().__init__()

        if 'ctxt' in feature_size:
            non_ctxt_dim = feature_size['ctxt']

        non_ctxt_size = len([x for x in feature_size if x != 'ctxt'])

        if non_ctxt_size != 0:
            non_ctxt_dim = (non_ctxt_dim // non_ctxt_size) * non_ctxt_size
        else:
            non_ctxt_dim = 0

        ctxt_dim = feature_size.get('ctxt', 0)
        input_units = ctxt_dim + non_ctxt_dim
        
        input_units += dis_hidden_units
        models = [nn.Linear(input_units, hidden_units), nn.Dropout(p=dropout), nn.ReLU()]

        for _ in range(nlayers-1):
            models.extend([nn.Linear(hidden_units, hidden_units), nn.Dropout(p=dropout), nn.ReLU()])

        # models.append(nn.Linear(hidden_units, int(math.log(1024, 2))+2))
        models.append(nn.Linear(hidden_units, 2))
        models.append(nn.LogSoftmax(dim=-1))

        self.model = nn.Sequential(*models)

        input_layer = {}
        if non_ctxt_size != 0:
            ndim = non_ctxt_dim // non_ctxt_size
            for k in feature_size:
                if k != 'ctxt':
                    input_layer[k] = LeakyReLUNet(feature_size[k], ndim)

        self.input_layer = nn.ModuleDict(input_layer)

        self.feature_size = feature_size

    def forward(self, features, dis_hidden):
        features_cat = [features['ctxt']] if 'ctxt' in self.feature_size else []

        for k in self.feature_size:
            if k != 'ctxt':
                features_cat.append(self.input_layer[k](features[k]))
        features_cat.append(dis_hidden)
        return self.model(torch.cat(features_cat, -1))

class MetaKNetwork(nn.Module):
    r""" meta k network of adaptive knn-mt """
    def __init__(
        self,
        max_k = 1024,
        use_k = 32,
        k_trainable = True,
        lambda_trainable = True,
        temperature_trainable = True,
        k_net_hid_size = 256,
        lambda_net_hid_size = 128,
        temperature_net_hid_size = 128,
        k_net_dropout_rate = 0.2,
        lambda_net_dropout_rate = 0.2,
        temperature_net_dropout_rate = 0.2,
        lambda_net_feature_size=None, 
        lambda_net_nlayers=3,
        label_count_as_feature = True,
        relative_label_count = False,
        device = "cuda:0",
        **kwargs,
    ):
        super().__init__()
        self.max_k = max_k    
        self.use_k = use_k
        self.k_trainable = k_trainable
        self.lambda_trainable = lambda_trainable
        self.temperature_trainable = temperature_trainable
        self.label_count_as_feature = label_count_as_feature
        self.relative_label_count = relative_label_count
        self.device = device
        self.mask_for_label_count = None

        if k_trainable:
            self.distance_to_k_hidden = LeakyReLUNet(self.use_k*2 if self.label_count_as_feature else self.use_k, k_net_hid_size)
            # self.distance_to_k_hidden = nn.Sequential(
            #         nn.Linear(self.use_k*2 if self.label_count_as_feature else self.use_k, k_net_hid_size),
            #         nn.ReLU(),
            #         nn.Dropout(p=k_net_dropout_rate),
            #         # nn.Linear(k_net_hid_size, int(math.log(self.max_k, 2))+1),
            #         nn.Linear(k_net_hid_size, k_net_hid_size),
            #         # nn.Softmax(dim=-1)
            #     ) # [1 neighbor, 2 neighbor, 4 neighbor, 8 neighbor, ..]
            # if self.label_count_as_feature:
            #     nn.init.normal_(self.distance_to_k_hidden[0].weight[:, :self.use_k], mean=0, std=0.01)
            #     nn.init.normal_(self.distance_to_k_hidden[0].weight[:, self.use_k:], mean=0, std=0.1)
            # else:
            #     nn.init.normal_(self.distance_to_k_hidden[0].weight, mean=0, std=0.01)

        if lambda_trainable:
            # self.distance_to_lambda = nn.Sequential(
            #         nn.Linear(self.max_k*2 if self.label_count_as_feature else self.max_k, lambda_net_hid_size),
            #         nn.ReLU(),
            #         nn.Dropout(p=lambda_net_dropout_rate),
            #         nn.Linear(lambda_net_hid_size, 1),
            #         nn.Sigmoid()
            #     )
            self.distance_to_lambda = MLPMOE(
                feature_size=lambda_net_feature_size,
                hidden_units=lambda_net_hid_size,
                nlayers=lambda_net_nlayers,
                dropout=lambda_net_dropout_rate,
                dis_hidden_units=k_net_hid_size,
                )

            # if self.label_count_as_feature:
            #     nn.init.xavier_normal_(self.distance_to_lambda[0].weight[:, :self.max_k], gain=0.01)
            #     nn.init.xavier_normal_(self.distance_to_lambda[0].weight[:, self.max_k:], gain=0.1)
            #     nn.init.xavier_normal_(self.distance_to_lambda[-2].weight)
            # else:
            #     nn.init.normal_(self.distance_to_lambda[0].weight, mean=0, std=0.01)
        
        if temperature_trainable:
            self.distance_to_temperature = nn.Sequential(
                    nn.Linear(self.use_k*2 if self.label_count_as_feature else self.use_k,
                            temperature_net_hid_size),
                    nn.ReLU(),
                    nn.Dropout(p=temperature_net_dropout_rate),
                    nn.Linear(temperature_net_hid_size, 1)
                )
            if self.label_count_as_feature:
                nn.init.xavier_normal_(self.distance_to_temperature[0].weight[:, :self.use_k], gain=0.001)
                nn.init.xavier_normal_(self.distance_to_temperature[0].weight[:, self.use_k:], gain=0.001)
                nn.init.xavier_normal_(self.distance_to_temperature[-1].weight)
            else:
                nn.init.normal_(self.distance_to_temperature[0].weight, mean=0, std=0.001)


    def forward(self, inputs):
        val_counts, distances = inputs['val_count'][:, :self.use_k], inputs['distance'][:, :self.use_k]
        network_inputs = torch.cat((distances.detach(), val_counts.detach().float()), dim=-1)
        dis_network_inputs = network_inputs if self.label_count_as_feature else network_inputs[:, :self.use_k]
        temp_network_inputs = network_inputs if self.label_count_as_feature else network_inputs[:, :self.use_k]
        results = {}
        
        k_hidden = self.distance_to_k_hidden(dis_network_inputs) if self.k_trainable else None
        results["lambda_k_net_output"] = self.distance_to_lambda(inputs, k_hidden) if self.lambda_trainable else None
        # handle0 = self.distance_to_temperature[0].register_forward_hook(hook)
        # handle1 = self.distance_to_temperature[1].register_forward_hook(hook)
        # handle2 = self.distance_to_temperature[2].register_forward_hook(hook)
        # print(network_inputs)
        results["temperature_net_output"] = self.distance_to_temperature(temp_network_inputs) \
                    if self.temperature_trainable else None # (0, +∞)
        
        # print(results["temperature_net_output"].squeeze(-1))
        results["temperature_net_output"] = torch.exp(results["temperature_net_output"])
        # print(results["temperature_net_output"].squeeze(-1))
        if (results["temperature_net_output"] == 0).any():
            min_temp = torch.max(distances) / 100
            print(f'sum: {(results["temperature_net_output"] == 0).sum()}, clip temperature to {min_temp}')
            results["temperature_net_output"] = torch.clip(results["temperature_net_output"], min=min_temp)
            if (results["temperature_net_output"] == 0).all():
                print('meijiule')
                1/0
        return results
    
    def epoch_update(self):
        pass