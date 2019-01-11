import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .bilstm import BiLSTM
from .crf import CRF

import logging

logger = logging.getLogger('ourlogger')


class BiLSTM_CRF(nn.Module):
    def __init__(self, data, use_w2c=False, use_attn=False):
        super(BiLSTM_CRF, self).__init__()
        logger.info(("build batched lstmcrf..."))
        self.gpu = data.HP_gpu
        self.use_w2c = use_w2c
        self.use_attn = use_attn

        ## add two more label for downlayer lstm, use original label size for CRF
        label_size = data.label_alphabet_size

        self.lstm = BiLSTM(data, use_w2c=self.use_w2c, use_attn=self.use_attn)
        # buyao -2!
        self.crf = CRF(label_size, self.gpu)

    def neg_log_likelihood_loss(self, gaz_list, word_inputs, biword_inputs, word_seq_lengths, char_inputs,
                                char_seq_lengths, char_seq_recover, batch_label, mask):
        outs = self.lstm.get_output_score(gaz_list, word_inputs, biword_inputs, word_seq_lengths, char_inputs,
                                          char_seq_lengths, char_seq_recover)
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        total_loss = self.crf.neg_log_likelihood_loss(outs, mask, batch_label)
        scores, tag_seq = self.crf._viterbi_decode(outs, mask)
        return total_loss, tag_seq

    def forward(self, gaz_list, word_inputs, biword_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                char_seq_recover, mask):
        outs = self.lstm.get_output_score(gaz_list, word_inputs, biword_inputs, word_seq_lengths, char_inputs,
                                          char_seq_lengths, char_seq_recover)
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        scores, tag_seq = self.crf._viterbi_decode(outs, mask)
        return tag_seq

    def get_lstm_features(self, gaz_list, word_inputs, biword_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                          char_seq_recover):
        return self.lstm.get_lstm_features(gaz_list, word_inputs, biword_inputs, word_seq_lengths, char_inputs,
                                           char_seq_lengths, char_seq_recover)


