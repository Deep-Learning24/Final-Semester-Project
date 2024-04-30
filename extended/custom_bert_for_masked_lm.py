import math
from transformers import AutoModelForMaskedLM, BertConfig
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.models.bert.modeling_bert import BertSelfAttention, BertPredictionHeadTransform, BertLMPredictionHead


class CustomBertForMaskedLM(AutoModelForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.bert.encoder.layer = nn.ModuleList([CustomBertLayer(config) for _ in range(config.num_hidden_layers)])


class CustomBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = CustomBertAttention(config)
        self.intermediate = config.intermediate_class(config)
        self.output = config.output_class(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, output_attentions=False):
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask, output_attentions)
        attention_output = self_attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + self_attention_outputs[1:]  # add attentions if we output them
        return outputs


class CustomBertAttention(BertSelfAttention):
    def __init__(self, config):
        super().__init__(config)
        self.causal_mask = self._get_causal_mask(0, config.max_position_embeddings)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        self._set_attention_mask(attention_mask)

        mixed_query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)

        attention_scores = torch.matmul(mixed_query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + self.causal_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

    def _get_causal_mask(self, begin_idx, end_idx):
        mask = torch.triu(torch.ones(end_idx, end_idx), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask



config = BertConfig.from_pretrained("jean-paul/KinyaBERT-large")
model = CustomBertForMaskedLM.from_pretrained("jean-paul/KinyaBERT-large", config=config)

print(model)

