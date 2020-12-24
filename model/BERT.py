# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 20:44:19 2020

@author: yuanr
"""

import math

from copy import deepcopy as cp



import torch

import torch.nn as nn

import torch.nn.functional as F

from pytorch_pretrained_bert.modeling import BertLayerNorm, BertSelfOutput, BertOutput, BertIntermediate, BertPooler

from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertLMPredictionHead

from scipy import signal

from torch.autograd import Variable





def clones(module, N):

    return nn.ModuleList([cp(module) for _ in range(N)])





class BertEmbeddings(nn.Module):

    def __init__(self, config):

        super(BertEmbeddings, self).__init__()

        self.word_embeddings = nn.Embedding(

            config.vocab_size, config.hidden_size, padding_idx=0)

        self.position_embeddings = nn.Embedding(

            config.max_position_embeddings, config.hidden_size)

        self.token_type_embeddings = nn.Embedding(

            config.type_vocab_size, config.hidden_size)



        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load

        # any TensorFlow checkpoint file

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)



    def forward(self, input_ids, position_ids=None, token_type_ids=None):

        seq_length = input_ids.size(1)



        if position_ids is None:

            position_ids = torch.arange(

                seq_length, dtype=torch.long, device=input_ids.device)

            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)



        if token_type_ids is None:

            token_type_ids = torch.zeros_like(input_ids)



        words_embeddings = self.word_embeddings(input_ids)

        position_embeddings = self.position_embeddings(position_ids)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)



        embeddings = words_embeddings + position_embeddings + token_type_embeddings
#        embeddings = words_embeddings + token_type_embeddings        

        embeddings = self.LayerNorm(embeddings)

        embeddings = self.dropout(embeddings)

        return embeddings





class BertSelfAttention(nn.Module):

    def __init__(self, config):

        super(BertSelfAttention, self).__init__()

        if config.hidden_size % config.num_attention_heads != 0:

            raise ValueError(

                "The hidden size (%d) is not a multiple of the number of attention "

                "heads (%d)" % (config.hidden_size, config.num_attention_heads))

        self.num_attention_heads = config.num_attention_heads

        self.attention_head_size = int(

            config.hidden_size / config.num_attention_heads)

        self.all_head_size = self.num_attention_heads * self.attention_head_size



        self.query = nn.Linear(config.hidden_size, self.all_head_size)

        self.key = nn.Linear(config.hidden_size, self.all_head_size)

        self.value = nn.Linear(config.hidden_size, self.all_head_size)



        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)



    def transpose_for_scores(self, x):

        new_x_shape = x.size()[

                      :-1] + (self.num_attention_heads, self.attention_head_size)

        x = x.view(*new_x_shape)

        return x.permute(0, 2, 1, 3)



    def forward(self, hidden_states, attention_mask):

        mixed_query_layer = self.query(hidden_states)

        mixed_key_layer = self.key(hidden_states)

        mixed_value_layer = self.value(hidden_states)



        query_layer = self.transpose_for_scores(mixed_query_layer)

        key_layer = self.transpose_for_scores(mixed_key_layer)

        value_layer = self.transpose_for_scores(mixed_value_layer)



        # Take the dot product between "query" and "key" to get the raw attention scores.

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores/math.sqrt(self.attention_head_size)

        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)

        attention_scores = attention_scores + attention_mask



        # Normalize the attention scores to probabilities.

        attention_probs_ = nn.Softmax(dim=-1)(attention_scores)



        # This is actually dropping out entire tokens to attend to, which might

        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.dropout(attention_probs_)



        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[

                                  :-2] + (self.all_head_size,)

        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer, attention_probs





class BertAttention(nn.Module):

    def __init__(self, config):

        super(BertAttention, self).__init__()

        self.self = BertSelfAttention(config)

        self.output = BertSelfOutput(config)



    def forward(self, input_tensor, attention_mask):

        self_output, attns = self.self(input_tensor, attention_mask)

        attention_output = self.output(self_output, input_tensor)

        return attention_output, attns





class BertLayer(nn.Module):

    def __init__(self, config):

        super(BertLayer, self).__init__()

        self.attention = BertAttention(config)

        self.intermediate = BertIntermediate(config)

        self.output = BertOutput(config)



    def forward(self, hidden_states, attention_mask):

        attention_output, attns = self.attention(hidden_states, attention_mask)

        intermediate_output = self.intermediate(attention_output)

        layer_output = self.output(intermediate_output, attention_output)

        return layer_output, attns





class BertEncoder(nn.Module):

    def __init__(self, config):

        super(BertEncoder, self).__init__()

        layer = BertLayer(config)

        self.layer = clones(layer, config.num_hidden_layers)



    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=False):

        outputs = []

        attentions = []

        for layer_module in self.layer:

            hidden_states, attention = layer_module(hidden_states, attention_mask)

            if output_all_encoded_layers:

                outputs.append(hidden_states)

                attentions.append(attention)

        if not output_all_encoded_layers:

            outputs.append(hidden_states)

            attentions.append(attention)

        return outputs, attentions





class BertModel(BertPreTrainedModel):

    def __init__(self, config):

        super(BertModel, self).__init__(config)

        self.embeddings = BertEmbeddings(config)

        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config)

        self.apply(self.init_bert_weights)



    def forward(self, input_ids, position_ids, token_type_ids, attention_mask, output_all_encoded_layers=True):

        extended_attention_mask = attention_mask.unsqueeze(1)

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility

        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0



        embedding_output = self.embeddings(

            input_ids, position_ids, token_type_ids)

        outputs, attentions = self.encoder(embedding_output,

                                           extended_attention_mask,

                                           output_all_encoded_layers=output_all_encoded_layers)

        sequence_output = outputs[-1]

        pooled_output = self.pooler(sequence_output)

        return sequence_output, pooled_output, attentions





class Classifier(nn.Module):

    def __init__(self, hidden_size):

        super(Classifier, self).__init__()

        self.linear1 = nn.Linear(hidden_size, 1)

        self.sigmoid = nn.Sigmoid()



    def forward(self, x, mask_cls):

        h = self.linear1(x).squeeze(-1)

#        sent_scores = self.sigmoid(h) * mask_cls.float()
        sent_scores = h * mask_cls.float()
        return sent_scores
    
    
class ExtSummarizer(nn.Module):

    def __init__(self):

        super(ExtSummarizer, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.ext_layer = Classifier(768*2)



    def forward(self,  input_ids, position_ids, token_type_ids, attention_mask, clss, clsf, mask_clsf):

        top_vec, _, attention = self.bert(input_ids, position_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # fact level embedding
        facts_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clsf]
        # sentence level embedding
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        # document level embedding
#        docus_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), 0]
        
#        docus_vec = docus_vec.expand(facts_vec.size()[0],facts_vec.size()[1],facts_vec.size()[2])
        
        facts_vec= torch.cat((facts_vec,sents_vec),2)
        
#        facts_vec= torch.cat((facts_vec,docus_vec),2)
        
#        facts_vec= torch.cat((facts_vec,sents_vec,docus_vec),2)

        facts_vec = facts_vec * mask_clsf[:, :, None].float()

        sent_scores = self.ext_layer(facts_vec, mask_clsf).squeeze(-1)

        return sent_scores, mask_clsf 
