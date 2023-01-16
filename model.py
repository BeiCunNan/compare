import math

import torch
from torch import nn



class Transformer_CLS(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.block = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(base_model.config.hidden_size, 192),
            nn.Linear(192, 24),
            nn.Linear(24, num_classes),
            nn.Softmax(dim=1)
        )
        for param in base_model.parameters():
            param.requires_grad = (True)

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        cls_feats = raw_outputs.last_hidden_state[:, 0, :]
        predicts = self.block(cls_feats)
        return predicts



class Self_Attention(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes

        for param in base_model.parameters():
            param.requires_grad = (True)

        self.key_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self.query_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self.value_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self._norm_fact = 1 / math.sqrt(self.base_model.config.hidden_size)
        self.fnn = nn.Linear(self.base_model.config.hidden_size, num_classes)

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        tokens = raw_outputs.last_hidden_state
        K = self.key_layer(tokens)
        Q = self.query_layer(tokens)
        V = self.value_layer(tokens)
        attention = nn.Softmax(dim=-1)((torch.bmm(Q, K.permute(0, 2, 1))) * self._norm_fact)
        output = torch.bmm(attention, V)
        output = torch.sum(output, dim=1)
        predicts = self.fnn(output)
        return predicts

class Self_Attention_New(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes

        for param in base_model.parameters():
            param.requires_grad = (True)

        self.nsakey_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self.nsaquery_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self.nsavalue_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self.nsa_norm_fact = 1 / math.sqrt(self.base_model.config.hidden_size)

        self.key_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self.query_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self.value_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self._norm_fact = 1 / math.sqrt(self.base_model.config.hidden_size)

        self.fnn = nn.Linear(self.base_model.config.hidden_size * 4, num_classes)
        self.sgsa = nn.Linear(self.base_model.config.hidden_size , 1)

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        tokens = raw_outputs.last_hidden_state

        # SA
        K = self.key_layer(tokens)
        Q = self.query_layer(tokens)
        V = self.value_layer(tokens)
        attention = nn.Softmax(dim=-1)((torch.bmm(Q, K.permute(0, 2, 1))) * self.nsa_norm_fact)
        output = torch.bmm(attention, V)

        # Layer_Normalizaton
        norm = nn.LayerNorm([output.shape[1], output.shape[2]], eps=1e-8).cuda()
        output_LN = norm(output)

        # NSA
        K_N = self.key_layer(output_LN)
        Q_N = self.query_layer(output_LN)
        V_N = self.value_layer(output_LN)
        attention_N = nn.Softmax(dim=-1)((torch.bmm(Q_N.permute(0, 2, 1), K_N) * self._norm_fact))
        output_N = torch.bmm(V_N, attention_N)

        # SGSA
        output_SGSA = self.sgsa(output_N) * output_N

        # Layer_Normalization
        # norm = nn.LayerNorm([output_SGSA.shape[1], output_SGSA.shape[2]], eps=1e-8).cuda()
        # output_LN = norm(output_SGSA)

        # Add
        output_N = torch.cat((tokens, output_SGSA), 2)
        # output_N = torch.add(tokens,output_N)

        # Pooling
        output_A = torch.mean(output_N, dim=1)
        output_B, _ = torch.max(output_N, dim=1)

        predicts = self.fnn(torch.cat((output_A, output_B), 1))
        return predicts
