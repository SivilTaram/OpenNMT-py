"""
Implementation of "Attention is All You Need"
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from onmt.decoders.decoder import DecoderBase
from onmt.modules import MultiHeadedAttention, AverageAttention
from onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.utils.misc import sequence_mask


class TransformerDecoderLayer(nn.Module):
    """Transformer Decoder layer block in Pre-Norm style.
    Pre-Norm style is an improvement w.r.t. Original paper's Post-Norm style,
    providing better converge speed and performance. This is also the actual
    implementation in tensor2tensor and also avalable in fairseq.
    See https://tunz.kr/post/4 and :cite:`DeeperTransformer`.

    .. mermaid::

        graph LR
        %% "*SubLayer" can be self-attn, src-attn or feed forward block
            A(input) --> B[Norm]
            B --> C["*SubLayer"]
            C --> D[Drop]
            D --> E((+))
            A --> E
            E --> F(out)


    Args:
        d_model (int): the dimension of keys/values/queries in
            :class:`MultiHeadedAttention`, also the input size of
            the first-layer of the :class:`PositionwiseFeedForward`.
        heads (int): the number of heads for MultiHeadedAttention.
        d_ff (int): the second-layer of the :class:`PositionwiseFeedForward`.
        dropout (float): dropout in residual, self-attn(dot) and feed-forward
        attention_dropout (float): dropout in context_attn (and self-attn(avg))
        self_attn_type (string): type of self-attention scaled-dot, average
        max_relative_positions (int):
            Max distance between inputs in relative positions representations
        aan_useffn (bool): Turn on the FFN layer in the AAN decoder
        full_context_alignment (bool):
            whether enable an extra full context decoder forward for alignment
        alignment_heads (int):
            N. of cross attention heads to use for alignment guiding
    """

    def __init__(self, d_model, heads, d_ff, dropout, attention_dropout,
                 self_attn_type="scaled-dot", max_relative_positions=0,
                 aan_useffn=False, full_context_alignment=False,
                 alignment_heads=0, activation='relu'):
        super(TransformerDecoderLayer, self).__init__()

        if self_attn_type == "scaled-dot":
            self.self_attn = MultiHeadedAttention(
                heads, d_model, dropout=attention_dropout,
                max_relative_positions=max_relative_positions)
        elif self_attn_type == "average":
            self.self_attn = AverageAttention(d_model,
                                              dropout=attention_dropout,
                                              aan_useffn=aan_useffn)

        self.his_context_attn = MultiHeadedAttention(
            heads, d_model, dropout=attention_dropout)
        self.cur_context_attn = MultiHeadedAttention(
            heads, d_model, dropout=attention_dropout)
        # [his, cur]
        self.his_cur_feed_forward = nn.Linear(d_model * 2, d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout, activation)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_3 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)
        self.full_context_alignment = full_context_alignment
        self.alignment_heads = alignment_heads

    def forward(self, *args, **kwargs):
        """ Extend `_forward` for (possibly) multiple decoder pass:
        Always a default (future masked) decoder forward pass,
        Possibly a second future aware decoder pass for joint learn
        full context alignement, :cite:`garg2019jointly`.

        Args:
            * All arguments of _forward.
            with_align (bool): whether return alignment attention.

        Returns:
            (FloatTensor, FloatTensor, FloatTensor or None):

            * output ``(batch_size, T, model_dim)``
            * top_his_attn ``(batch_size, T, src_len)``
            * attn_align ``(batch_size, T, src_len)`` or None
        """
        with_align = kwargs.pop('with_align', False)
        output, his_attns, cur_attns, his_mid, cur_mid = self._forward(*args, **kwargs)

        # batch_size x T x src_len
        top_his_attn = his_attns[:, 0, :, :].contiguous()
        top_cur_attn = cur_attns[:, 0, :, :].contiguous()

        # batch_size x T x hidden_size
        top_his_mid = his_mid
        top_cur_mid = cur_mid

        attn_align = None
        if with_align:
            if self.full_context_alignment:
                # return _, (B, Q_len, K_len)
                _, his_attns = self._forward(*args, **kwargs, future=True)

            if self.alignment_heads > 0:
                his_attns = his_attns[:, :self.alignment_heads, :, :].contiguous()
            # layer average attention across heads, get ``(B, Q, K)``
            # Case 1: no full_context, no align heads -> layer avg baseline
            # Case 2: no full_context, 1 align heads -> guided align
            # Case 3: full_context, 1 align heads -> full cte guided align
            attn_align = his_attns.mean(dim=1)
        return output, top_his_attn, top_cur_attn, top_his_mid, top_cur_mid, attn_align

    def _forward(self, inputs, history_memory_bank, current_memory_bank,
                 history_pad_mask, current_pad_mask, tgt_pad_mask,
                 layer_cache=None, step=None, future=False):
        """ A naive forward pass for transformer decoder.

        # T: could be 1 in the case of stepwise decoding or tgt_len

        Args:
            inputs (FloatTensor): ``(batch_size, T, model_dim)``
            history_memory_bank (FloatTensor): ``(batch_size, src_len, model_dim)``
            history_pad_mask (LongTensor): ``(batch_size, 1, src_len)``
            tgt_pad_mask (LongTensor): ``(batch_size, 1, T)``
            layer_cache (dict or None): cached layer info when stepwise decode
            step (int or None): stepwise decoding counter
            future (bool): If set True, do not apply future_mask.

        Returns:
            (FloatTensor, FloatTensor):

            * output ``(batch_size, T, model_dim)``
            * his_attns ``(batch_size, head, T, src_len)``

        """
        dec_mask = None

        if step is None:
            tgt_len = tgt_pad_mask.size(-1)
            if not future:  # apply future_mask, result mask in (B, T, T)
                future_mask = torch.ones(
                    [tgt_len, tgt_len],
                    device=tgt_pad_mask.device,
                    dtype=torch.uint8)
                future_mask = future_mask.triu_(1).view(1, tgt_len, tgt_len)
                # BoolTensor was introduced in pytorch 1.2
                try:
                    future_mask = future_mask.bool()
                except AttributeError:
                    pass
                dec_mask = torch.gt(tgt_pad_mask + future_mask, 0)
            else:  # only mask padding, result mask in (B, 1, T)
                dec_mask = tgt_pad_mask

        input_norm = self.layer_norm_1(inputs)

        if isinstance(self.self_attn, MultiHeadedAttention):
            query, _ = self.self_attn(input_norm, input_norm, input_norm,
                                      mask=dec_mask,
                                      layer_cache=layer_cache,
                                      attn_type="self")
        elif isinstance(self.self_attn, AverageAttention):
            query, _ = self.self_attn(input_norm, mask=dec_mask,
                                      layer_cache=layer_cache, step=step)

        query = self.drop(query) + inputs
        query_norm = self.layer_norm_2(query)

        his_mid, his_attns = self.his_context_attn(history_memory_bank, history_memory_bank, query_norm,
                                                   mask=history_pad_mask,
                                                   layer_cache=layer_cache,
                                                   attn_type="context")
        cur_mid, cur_attns = self.cur_context_attn(current_memory_bank, current_memory_bank, query_norm,
                                                   mask=current_pad_mask,
                                                   layer_cache=layer_cache,
                                                   attn_type="context")

        mid = self.his_cur_feed_forward(torch.cat([his_mid, cur_mid], dim=2))

        # the 3-rd layer norm is performed inside self.feed_forward
        # here we use another feed forward to retain the residual connection
        output = self.feed_forward(self.drop(mid) + query)

        return output, his_attns, cur_attns, his_mid, cur_mid

    def update_dropout(self, dropout, attention_dropout):
        self.self_attn.update_dropout(attention_dropout)
        self.his_context_attn.update_dropout(attention_dropout)
        self.feed_forward.update_dropout(dropout)
        self.drop.p = dropout


class TransformerDecoder(DecoderBase):
    """The Transformer decoder from "Attention is All You Need".
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          BB[multi-head src-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> BB
          BB --> C
          C --> O


    Args:
        num_layers (int): number of encoder layers.
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        copy_attn (bool): if using a separate copy attention
        self_attn_type (str): type of self-attention scaled-dot, average
        dropout (float): dropout in residual, self-attn(dot) and feed-forward
        attention_dropout (float): dropout in context_attn (and self-attn(avg))
        embeddings (onmt.modules.Embeddings):
            embeddings to use, should have positional encodings
        max_relative_positions (int):
            Max distance between inputs in relative positions representations
        aan_useffn (bool): Turn on the FFN layer in the AAN decoder
        full_context_alignment (bool):
            whether enable an extra full context decoder forward for alignment
        alignment_layer (int): N° Layer to supervise with for alignment guiding
        alignment_heads (int):
            N. of cross attention heads to use for alignment guiding
    """

    def __init__(self, num_layers, d_model, heads, d_ff,
                 copy_attn, self_attn_type, dropout, attention_dropout,
                 embeddings, max_relative_positions, aan_useffn,
                 full_context_alignment, alignment_layer,
                 alignment_heads):
        super(TransformerDecoder, self).__init__()

        self.embeddings = embeddings

        # Decoder State
        self.state = {}

        self.transformer_layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, heads, d_ff, dropout,
                                     attention_dropout, self_attn_type=self_attn_type,
                                     max_relative_positions=max_relative_positions,
                                     aan_useffn=aan_useffn,
                                     full_context_alignment=full_context_alignment,
                                     alignment_heads=alignment_heads,
                                     activation='gelu')
             for i in range(num_layers)])

        # previously, there was a GlobalAttention module here for copy
        # attention. But it was never actually used -- the "copy" attention
        # just reuses the context attention.
        self._copy = copy_attn
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.alignment_layer = alignment_layer

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.dec_layers,
            opt.dec_rnn_size,
            opt.heads,
            opt.transformer_ff,
            opt.copy_attn,
            opt.self_attn_type,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            opt.attention_dropout[0] if type(opt.attention_dropout)
                                        is list else opt.dropout,
            embeddings,
            opt.max_relative_positions,
            opt.aan_useffn,
            opt.full_context_alignment,
            opt.alignment_layer,
            alignment_heads=opt.alignment_heads)

    def init_state(self, src, memory_bank, enc_hidden):
        """Initialize decoder state."""
        self.state["src"] = src
        self.state["cache"] = None

    def map_state(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)

        self.state["src"] = fn(self.state["src"], 1)
        if self.state["cache"] is not None:
            _recursive_map(self.state["cache"])

    def detach_state(self):
        self.state["src"] = self.state["src"].detach()

    def forward(self, tgt, memory_bank, step=None, **kwargs):
        """Decode, possibly stepwise."""
        if step == 0:
            self._init_cache(memory_bank)

        sep_id = kwargs['sep_id']

        src_origin_ids = kwargs['src_origin']
        src_origin_ids = src_origin_ids.transpose(0, 1).squeeze(dim=-1)

        emb = self.embeddings(tgt, step=step)
        assert emb.dim() == 3  # len x batch x embedding_dim

        output = emb.transpose(0, 1).contiguous()

        # batch (maybe x beam size) x len x hidden_size
        src_memory_bank = memory_bank.transpose(0, 1).contiguous()

        input_batch_size = src_origin_ids.shape[0]
        mem_batch_size = src_memory_bank.shape[0]

        if input_batch_size != mem_batch_size:
            batch_offset = kwargs['batch_offset']
            assert batch_offset is not None
            # there are beams in translation
            input_batch_size = batch_offset.shape[0]
            assert mem_batch_size % input_batch_size == 0
            # re-shape the src_origin_ids
            beam_size = mem_batch_size // input_batch_size
            src_origin_ids = src_origin_ids[batch_offset, :].unsqueeze(dim=1).repeat(1, beam_size, 1)
            src_origin_ids = src_origin_ids.view(mem_batch_size, -1).contiguous()

        borders = []
        for i in range(mem_batch_size):
            borders.append((src_origin_ids[i] == sep_id).nonzero()[-1])

        tgt_words = tgt[:, :, 0].transpose(0, 1)

        pad_idx = self.embeddings.word_padding_idx
        src_lens = kwargs["memory_lengths"]
        src_max_len = self.state["src"].shape[0]

        # batch_size x 1 x seq_len
        src_pad_mask = ~sequence_mask(src_lens, src_max_len).unsqueeze(1)

        history_memory_bank = []
        current_memory_bank = []
        history_pad_mask = []
        current_pad_mask = []
        # use borders to split src_memory_bank into history / current ones
        for i in range(mem_batch_size):
            # we should keep the gradients as well as the separate values for self-attention
            blank_memory = src_memory_bank.data.new(src_memory_bank[i].shape).fill_(pad_idx)
            blank_memory[:borders[i]] = src_memory_bank[i, :borders[i]]
            history_memory_bank.append(blank_memory)
            # 0 means valid, 1 means need to be masked
            pad_mask = src_pad_mask[i].data.new(src_pad_mask[i].shape).fill_(1)
            pad_mask[0, :borders[i]].fill_(0)
            history_pad_mask.append(src_pad_mask[i] | pad_mask)
            # copy the remaining memory
            blank_memory = src_memory_bank.data.new(src_memory_bank[i].shape).fill_(pad_idx)
            blank_memory[borders[i]:] = src_memory_bank[i, borders[i]:]
            current_memory_bank.append(blank_memory)
            # mask
            pad_mask = src_pad_mask[i].data.new(src_pad_mask[i].shape).fill_(1)
            pad_mask[0, borders[i]:].fill_(0)
            current_pad_mask.append(src_pad_mask[i] | pad_mask)

        # TODO: the speed may be a bottleneck for training
        history_memory_bank = pad_sequence(history_memory_bank, batch_first=True)
        current_memory_bank = pad_sequence(current_memory_bank, batch_first=True)
        history_pad_mask = pad_sequence(history_pad_mask, batch_first=True)
        current_pad_mask = pad_sequence(current_pad_mask, batch_first=True)

        tgt_pad_mask = tgt_words.data.eq(pad_idx).unsqueeze(1)  # [B, 1, T_tgt]

        with_align = kwargs.pop('with_align', False)
        attn_aligns = []

        for i, layer in enumerate(self.transformer_layers):
            layer_cache = self.state["cache"]["layer_{}".format(i)] \
                if step is not None else None
            output, his_attn, cur_attn, his_mid, cur_mid, attn_align = layer(
                output,
                history_memory_bank,
                current_memory_bank,
                history_pad_mask,
                current_pad_mask,
                tgt_pad_mask,
                step=step,
                with_align=with_align)
            if attn_align is not None:
                attn_aligns.append(attn_align)

        output = self.layer_norm(output)
        dec_outs = output.transpose(0, 1).contiguous()
        his_attn = his_attn.transpose(0, 1).contiguous()
        cur_attn = cur_attn.transpose(0, 1).contiguous()
        his_mid = his_mid.transpose(0, 1).contiguous()
        cur_mid = cur_mid.transpose(0, 1).contiguous()

        # TODO: coverage may not work in our scenario (split attention)
        attns = {"std": his_attn}
        if self._copy:
            attns["his_copy"] = his_attn
            attns["cur_copy"] = cur_attn
            attns["his_mid"] = his_mid
            attns["cur_mid"] = cur_mid

        if with_align:
            attns["align"] = attn_aligns[self.alignment_layer]  # `(B, Q, K)`
            # attns["align"] = torch.stack(attn_aligns, 0).mean(0)  # All avg

        # TODO change the way attns is returned dict => list or tuple (onnx)
        return dec_outs, attns

    def _init_cache(self, memory_bank):
        self.state["cache"] = {}
        batch_size = memory_bank.size(1)
        depth = memory_bank.size(-1)

        for i, layer in enumerate(self.transformer_layers):
            layer_cache = {"memory_keys": None, "memory_values": None}
            if isinstance(layer.self_attn, AverageAttention):
                layer_cache["prev_g"] = torch.zeros((batch_size, 1, depth),
                                                    device=memory_bank.device)
            else:
                layer_cache["self_keys"] = None
                layer_cache["self_values"] = None
            self.state["cache"]["layer_{}".format(i)] = layer_cache

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.transformer_layers:
            layer.update_dropout(dropout, attention_dropout)
