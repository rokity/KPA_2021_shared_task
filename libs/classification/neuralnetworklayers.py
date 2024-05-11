from transformers import AutoModel
import torch.nn as nn
from torch import reshape, cat


class NeuralNetworkLayers(nn.Module):
    # define layer Neural Network

    def __init__(self, model_name, drop_out, out_unit_1):
        super(NeuralNetworkLayers, self).__init__()

        self.model_name = model_name  # bert, roberta, albert or deberta in base version
        # transformer model
        self.transformer_layer = AutoModel.from_pretrained(self.model_name)

        # add dense layers with dropuot
        self.dense_layer_1 = nn.Linear(769, out_unit_1)
        self.drop_out = nn.Dropout(drop_out)
        self.dense_layer_2 = nn.Linear(out_unit_1, 1)

        # apply sigmoid act. function in output (last dense layer)
        self.act_function = nn.Sigmoid()

    def forward(self, ids, mask, stance, tti=None):
        # execute feed-forward

        # manage transformer ouput based on the given transformer
        if self.model_name.startswith("bert-") == True:
            x = self.transformer_layer(
                input_ids=ids, token_type_ids=tti, attention_mask=mask
            ).pooler_output
        else:
            hidden_state = self.transformer_layer(input_ids=ids, attention_mask=mask)[0]
            x = hidden_state[:, 0]

        # concatenate transformer output with stance
        stance = reshape(stance, (len(stance), 1))
        concat = cat((x, stance), dim=1)

        x1 = self.dense_layer_1(concat)
        x1 = self.drop_out(x1)
        x2 = self.dense_layer_2(x1)
        x2 = self.drop_out(x2)

        out = self.act_function(x2)

        return out
