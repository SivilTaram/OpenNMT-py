"""
Implementation from: https://raw.githubusercontent.com/Zenglinxiao/OpenNMT-py/bert/onmt/encoders/bert.py
@Author: Zenglinxiao
"""

import torch.nn as nn
from onmt.encoders.transformer import TransformerEncoderLayer
from onmt.utils.misc import sequence_mask
import torch


class BertEncoder(nn.Module):
    """BERT Encoder: A Transformer Encoder with LayerNorm and BertPooler.
    :cite:`DBLP:journals/corr/abs-1810-04805`

    Args:
       embeddings (onmt.modules.BertEmbeddings): embeddings to use
       num_layers (int): number of encoder layers.
       d_model (int): size of the model
       heads (int): number of heads
       d_ff (int): size of the inner FF layer
       dropout (float): dropout parameters
    """

    def __init__(self, embeddings, num_layers=12, d_model=768, heads=12,
                 d_ff=3072, dropout=0.1, attention_dropout=0.1,
                 max_relative_positions=0, pretrained_file=None):
        super(BertEncoder, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.heads = heads
        self.dropout = dropout
        # Feed-Forward size should be 4*d_model as in paper
        self.d_ff = d_ff

        self.embeddings = embeddings
        # Transformer Encoder Block
        self.encoder = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff,
             dropout, attention_dropout,
             max_relative_positions=max_relative_positions,
             activation='gelu', is_bert=True) for _ in range(num_layers)])

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-12)
        self.pooler = BertPooler(d_model)

        if pretrained_file:
            state_dict = torch.load(pretrained_file)
            self.load_state_dict(state_dict['model'])

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            embeddings,
            opt.layers,
            opt.word_vec_size,
            opt.heads,
            opt.transformer_ff,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            opt.attention_dropout[0] if type(opt.attention_dropout)
            is list else opt.attention_dropout,
            opt.max_relative_positions,
            opt.pretrained_file
        )

    def forward(self, input_ids, lengths, token_type_ids=None):
        """
        Args:
            input_ids (Tensor): ``(seq_len, batch_size, feature_dim)``, padding ids=0
            lengths (Tensor): ``(batch_size)``, record length of sequence
            token_type_ids (seq_len, batch_size): ``(B, S)``, A(0), B(1), pad(0)
        Returns:
            all_encoder_layers (list of Tensor): ``(B, S, H)``, token level
            pooled_output (Tensor): ``(B, H)``, sequence level
        """
        # remove the feature dimension
        # seq_len x batch_size

        emb = self.embeddings(input_ids, token_type_ids)

        out = emb.transpose(0, 1).contiguous()
        # [batch, seq] -> [batch, 1, seq]
        mask = ~sequence_mask(lengths).unsqueeze(1)

        for layer in self.encoder:
            out = layer(out, mask)
        out = self.layer_norm(out)

        return emb, out.transpose(0, 1).contiguous(), lengths

    def update_dropout(self, dropout):
        self.dropout = dropout
        self.embeddings.update_dropout(dropout)
        for layer in self.encoder:
            layer.update_dropout(dropout)


class BertPooler(nn.Module):
    def __init__(self, hidden_size):
        """A pooling block (Linear layer followed by Tanh activation).

        Args:
            hidden_size (int): size of hidden layer.
        """

        super(BertPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation_fn = nn.Tanh()

    def forward(self, hidden_states):
        """hidden_states[:, 0, :] --> {Linear, Tanh} --> Returns.

        Args:
            hidden_states (Tensor): last layer's hidden_states, ``(B, S, H)``
        Returns:
            pooled_output (Tensor): transformed output of last layer's hidden
        """

        first_token_tensor = hidden_states[:, 0, :]  # [batch, d_model]
        pooled_output = self.activation_fn(self.dense(first_token_tensor))
        return pooled_output