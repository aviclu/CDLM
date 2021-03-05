import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import *


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.uniform_(m.bias)


class SpanEmbedder(nn.Module):
    def __init__(self, config, device):
        super(SpanEmbedder, self).__init__()
        self.bert_hidden_size = config.bert_hidden_size
        self.with_width_embedding = config.with_mention_width
        self.use_head_attention = config.with_head_attention
        self.device = device
        self.dropout = config.dropout
        self.padded_vector = torch.zeros(self.bert_hidden_size, device=device)
        self.self_attention_layer = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.bert_hidden_size, config.hidden_layer),
            # nn.Dropout(config['dropout']),
            nn.ReLU(),
            nn.Linear(config.hidden_layer, 1)
        )
        self.self_attention_layer.apply(init_weights)
        self.width_feature = nn.Embedding(5, config.embedding_dimension)


    def pad_continous_embeddings(self, continuous_embeddings):
        max_length = max(len(v) for v in continuous_embeddings)
        padded_tokens_embeddings = torch.stack(
            [torch.cat((emb, self.padded_vector.repeat(max_length - len(emb), 1)))
             for emb in continuous_embeddings]
        )
        masks = torch.stack(
            [torch.cat(
                (torch.ones(len(emb), device=self.device),
                 torch.zeros(max_length - len(emb), device=self.device)))
             for emb in continuous_embeddings]
        )
        return padded_tokens_embeddings, masks


    def forward(self, start_end, continuous_embeddings, width):
        vector = start_end
        if self.use_head_attention:
            padded_tokens_embeddings, masks = self.pad_continous_embeddings(continuous_embeddings)
            attention_scores = self.self_attention_layer(padded_tokens_embeddings).squeeze(-1)
            attention_scores *= masks
            attention_scores = torch.where(attention_scores != 0, attention_scores,
                                           torch.tensor(-9e9, device=self.device))
            attention_scores = F.softmax(attention_scores, dim=1)
            weighted_sum = (attention_scores.unsqueeze(-1) * padded_tokens_embeddings).sum(dim=1)
            vector = torch.cat((vector, weighted_sum), dim=1)

        if self.with_width_embedding:
            width = torch.clamp(width, max=4)
            width_embedding = self.width_feature(width)
            vector = torch.cat((vector, width_embedding), dim=1)

        return vector



class SpanScorer(nn.Module):
    def __init__(self, config):
        super(SpanScorer, self).__init__()
        self.input_layer = config.bert_hidden_size * 3
        if config.with_mention_width:
            self.input_layer += config.embedding_dimension
        self.mlp = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.input_layer, config.hidden_layer),
            # nn.Dropout(config['dropout']),
            nn.ReLU(),
            nn.Linear(config.hidden_layer, 1)
        )
        self.mlp.apply(init_weights)


    def forward(self, span_embedding):
        return self.mlp(span_embedding)




class SimplePairWiseClassifier(nn.Module):
    def __init__(self, config):
        super(SimplePairWiseClassifier, self).__init__()
        self.input_layer = config.bert_hidden_size * 3 if config.with_head_attention else config.bert_hidden_size * 2
        if config.with_mention_width:
            self.input_layer += config.embedding_dimension
        self.input_layer *= 3
        self.hidden_layer = config.hidden_layer
        self.pairwise_mlp = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.input_layer, self.hidden_layer),
            nn.ReLU(),
            nn.Linear(self.hidden_layer, self.hidden_layer),
            nn.Dropout(config.dropout),
            nn.ReLU(),
            nn.Linear(self.hidden_layer, 1),
        )
        self.pairwise_mlp.apply(init_weights)

    def forward(self, first, second):
        return self.pairwise_mlp(torch.cat((first, second, first * second), dim=1))





class FullCrossEncoder(nn.Module):
    def __init__(self, config, is_training=True, long=False):
        super(FullCrossEncoder, self).__init__()
        self.segment_size = config.segment_window * 2
        self.tokenizer = LongformerTokenizer.from_pretrained(config.cdlm_path)


        self.tokenizer.add_tokens(['<m>', '</m>'], special_tokens = True)
        self.tokenizer.add_tokens(['<g>'], special_tokens = True)
        self.start_id = self.tokenizer.encode('<m>', add_special_tokens=False)[0]
        self.end_id = self.tokenizer.encode('</m>', add_special_tokens=False)[0]

        self.vals = [self.start_id, self.end_id]
        self.long = long
        if not is_training and config.pretrained_model:
            self.model = AutoModel.from_pretrained(config.cdlm_path)
        else:
            self.model = AutoModel.from_pretrained(config.cdlm_path)
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.hidden_size = self.model.config.hidden_size
        if not self.long:
            self.linear = nn.Sequential(
                nn.Linear(self.hidden_size, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
        else:
            self.linear = nn.Sequential(
                nn.Linear(self.hidden_size*4, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
        self.linear.apply(init_weights)



    def forward(self, input_ids, attention_mask=None, arg1=None, arg2=None):
        output, _ = self.model(input_ids, attention_mask=attention_mask)
        if self.long:
            arg1_vec = (output*arg1.unsqueeze(-1)).sum(1)
            arg2_vec = (output*arg2.unsqueeze(-1)).sum(1)
        cls_vector = output[:, 0, :]
        if not self.long:
            scores = self.linear(cls_vector)
        else:
            scores = self.linear(torch.cat([cls_vector, arg1_vec, arg2_vec, arg1_vec*arg2_vec],dim=1))
        return scores