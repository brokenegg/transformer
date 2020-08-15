# Copyright Katsuya Iida.

import torch
import math
from torch import nn
from torch.nn import functional as F
from transformers.modeling_bert import BertForPretrainingOutput

def linear(input, weight, bias):
    assert weight.dim() == 2
    assert bias.dim() == 1
    return input @ weight.transpose(0, 1) + bias

def layer_norm(weights, bias, x, epsilon=1e-5):
    mean = torch.mean(x, dim=2)
    var = torch.var(x, unbiased=False, dim=2)
    mean = mean[:, :, None]
    var = var[:, :, None]
    normalized_input = (x - mean) / torch.sqrt(var + epsilon)
    return weights * normalized_input + bias

class MyBert(nn.Module):
    # bert.embeddings.position_ids [1, 512]
    # bert.embeddings.word_embeddings.weight [30522, 768]
    # bert.embeddings.position_embeddings.weight [512, 768]
    # bert.embeddings.token_type_embeddings.weight [2, 768]
    # bert.embeddings.LayerNorm.weight [768]
    # bert.embeddings.LayerNorm.bias [768]
    # bert.encoder.layer.0.attention.self.query.weight [768, 768]
    # bert.encoder.layer.0.attention.self.query.bias [768]
    # bert.encoder.layer.0.attention.self.key.weight [768, 768]
    # bert.encoder.layer.0.attention.self.key.bias [768]
    # bert.encoder.layer.0.attention.self.value.weight [768, 768]
    # bert.encoder.layer.0.attention.self.value.bias [768]
    # bert.encoder.layer.0.attention.output.dense.weight [768, 768]
    # bert.encoder.layer.0.attention.output.dense.bias [768]
    # bert.encoder.layer.0.attention.output.LayerNorm.weight [768]
    # bert.encoder.layer.0.attention.output.LayerNorm.bias [768]
    # bert.encoder.layer.0.intermediate.dense.weight [3072, 768]
    # bert.encoder.layer.0.intermediate.dense.bias [3072]
    # bert.encoder.layer.0.output.dense.weight [768, 3072]
    # bert.encoder.layer.0.output.dense.bias [768]
    # bert.encoder.layer.0.output.LayerNorm.weight [768]
    # bert.encoder.layer.0.output.LayerNorm.bias [768]
    # bert.pooler.dense.weight [768, 768]
    # bert.pooler.dense.bias [768]
    # cls.predictions.bias [30522]
    # cls.predictions.transform.dense.weight [768, 768]
    # cls.predictions.transform.dense.bias [768]
    # cls.predictions.transform.LayerNorm.weight [768]
    # cls.predictions.transform.LayerNorm.bias [768]
    # cls.predictions.decoder.weight [30522, 768]
    # cls.predictions.decoder.bias [30522]
    # cls.seq_relationship.weight [2, 768]
    # cls.seq_relationship.bias [2]

    def __init__(self, state_dict):
        super(MyBert, self).__init__()
        for k, v in state_dict.items():
            k = k.replace('.', '__')
            #print(k, v.dtype)
            requires_grad = v.dtype == torch.float32
            v = nn.Parameter(v, requires_grad=requires_grad)
            setattr(self, k, v)
        self.num_attention_heads = 12
        self.num_hidden_layers = 12
        self.hidden_size = 768

        self.attention_head_size = self.hidden_size // self.num_attention_heads

    def forward(self, input_ids, token_type_ids, attention_mask):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        return BertForPretrainingOutput(
            loss=None,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=sequence_output,
            attentions=None
        )

    def bert(self, input_ids, token_type_ids, attention_mask):
        embedding_output = self.embeddings(input_ids, token_type_ids)
        extended_attention_mask = attention_mask[:, None, None, :]
        hidden_states = self.encoder(embedding_output, extended_attention_mask)
        pooled_output = self.pooler(hidden_states)
        return hidden_states, pooled_output

    def embeddings(self, input_ids, token_type_ids):

        inputs_embeds = self.bert__embeddings__word_embeddings__weight[input_ids]

        seq_length = input_ids.size()[1]
        position_ids = self.bert__embeddings__position_ids[:, 0:seq_length]
        position_embeddings = self.bert__embeddings__position_embeddings__weight[position_ids]

        token_type_embeddings = self.bert__embeddings__token_type_embeddings__weight[token_type_ids]

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings

        embeddings = layer_norm(
            self.bert__embeddings__LayerNorm__weight,
            self.bert__embeddings__LayerNorm__bias,
            embeddings, epsilon=1e-12)

        return embeddings

    def encoder(self, hidden_states, attention_mask):
        for i in range(self.num_hidden_layers):
            hidden_states = self.bert_layer(i, hidden_states, attention_mask)
        return hidden_states

    def bert_layer(self, i, hidden_states, attention_mask):
        attention_output = self.attention(i, hidden_states, attention_mask)
        intermediate_output = self.intermediate(i, attention_output)
        layer_output = self.output(i, intermediate_output, attention_output)
        return layer_output

    def attention(self, i, hidden_states, attention_mask):
        self_outputs = self.attention_self(i, hidden_states, attention_mask)
        attention_output = self.attention_output(i, self_outputs, hidden_states)
        return attention_output

    def attention_self(self, i, hidden_states, attention_mask):
        def dense(t, x):
            weight = getattr(self, f'bert.encoder.layer.{i}.attention.self.{t}.weight'.replace('.', '__'))
            bias = getattr(self, f'bert.encoder.layer.{i}.attention.self.{t}.bias'.replace('.', '__'))
            x = linear(x, weight, bias)
            x_shape = x.size()
            x = x.view(x_shape[0], x_shape[1], self.num_attention_heads, -1)
            x = x.permute(0, 2, 1, 3) # [batch, head, len, val]
            return x
        query = dense('query', hidden_states)
        key = dense('key', hidden_states)
        value = dense('value', hidden_states)

        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores += attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores) # [batch, head, len, len]

        context_layer = torch.matmul(attention_probs, value) # [batch, head, len, val]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() # [batch, len, head, val]
        context_layer_shape = context_layer.size()
        context_layer = context_layer.view(context_layer_shape[0], context_layer_shape[1], -1)

        return context_layer

    def attention_output(self, i, hidden_states, input_tensor):
        weight = getattr(self, f'bert.encoder.layer.{i}.attention.output.dense.weight'.replace('.', '__'))
        bias = getattr(self, f'bert.encoder.layer.{i}.attention.output.dense.bias'.replace('.', '__'))
        hidden_states = linear(hidden_states, weight, bias)

        weight = getattr(self, f'bert.encoder.layer.{i}.attention.output.LayerNorm.weight'.replace('.', '__'))
        bias = getattr(self, f'bert.encoder.layer.{i}.attention.output.LayerNorm.bias'.replace('.', '__'))
        hidden_states = layer_norm(weight, bias, hidden_states + input_tensor, 1e-12)

        return hidden_states

    def intermediate(self, i, hidden_states):
        weight = getattr(self, f'bert.encoder.layer.{i}.intermediate.dense.weight'.replace('.', '__'))
        bias = getattr(self, f'bert.encoder.layer.{i}.intermediate.dense.bias'.replace('.', '__'))
        hidden_states = F.gelu(linear(hidden_states, weight, bias))
        return hidden_states

    def output(self, i, hidden_states, input_tensor):
        weight = getattr(self, f'bert.encoder.layer.{i}.output.dense.weight'.replace('.', '__'))
        bias = getattr(self, f'bert.encoder.layer.{i}.output.dense.bias'.replace('.', '__'))
        hidden_states = linear(hidden_states, weight, bias)

        weight = getattr(self, f'bert.encoder.layer.{i}.output.LayerNorm.weight'.replace('.', '__'))
        bias = getattr(self, f'bert.encoder.layer.{i}.output.LayerNorm.bias'.replace('.', '__'))
        hidden_states = layer_norm(weight, bias, hidden_states + input_tensor, 1e-12)

        return hidden_states

    def pooler(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        weight = getattr(self, 'bert.pooler.dense.weight'.replace('.', '__'))
        bias = getattr(self, 'bert.pooler.dense.bias'.replace('.', '__'))
        pooled_output = linear(first_token_tensor, weight, bias)
        pooled_output = nn.Tanh()(pooled_output)
        return pooled_output

    def cls(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score

    def predictions(self, hidden_states):
        hidden_states = self.transform(hidden_states)

        weight = getattr(self, 'cls.predictions.decoder.weight'.replace('.', '__'))
        bias = getattr(self, 'cls.predictions.decoder.bias'.replace('.', '__'))
        hidden_states = linear(hidden_states, weight, bias)

        return hidden_states

    def transform(self, hidden_states):
        weight = getattr(self, 'cls.predictions.transform.dense.weight'.replace('.', '__'))
        bias = getattr(self, 'cls.predictions.transform.dense.bias'.replace('.', '__'))
        hidden_states = linear(hidden_states, weight, bias)

        hidden_states = F.gelu(hidden_states)

        weight = getattr(self, 'cls.predictions.transform.LayerNorm.weight'.replace('.', '__'))
        bias = getattr(self, 'cls.predictions.transform.LayerNorm.bias'.replace('.', '__'))
        hidden_states = layer_norm(weight, bias, hidden_states, 1e-12)

        return hidden_states

    def seq_relationship(self, hidden_states):
        weight = getattr(self, 'cls.seq_relationship.weight'.replace('.', '__'))
        bias = getattr(self, 'cls.seq_relationship.bias'.replace('.', '__'))
        hidden_states = linear(hidden_states, weight, bias)
        return hidden_states