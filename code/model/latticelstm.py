
"""Implementation of batch-normalized LSTM."""
import torch
from torch import nn
import torch.autograd as autograd
from torch.autograd import Variable
from torch.nn import functional, init
import numpy as np

import logging

logger = logging.getLogger('ourlogger')
from model.attention import MultiHeadAttention

import json
with open("config.json", 'r') as load_f:
    config = json.load(load_f)



class WordLSTMCell(nn.Module):
    """A basic LSTM cell."""

    def __init__(self, input_size, hidden_size, use_bias=True):
        """
        Most parts are copied from torch.nn.LSTMCell.
        """

        super(WordLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.weight_ih = nn.Parameter(
            torch.FloatTensor(input_size, 3 * hidden_size))
        self.weight_hh = nn.Parameter(
            torch.FloatTensor(hidden_size, 3 * hidden_size))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(3 * hidden_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters following the way proposed in the paper.
        """
        init.orthogonal(self.weight_ih.data)
        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 3)
        self.weight_hh.data.set_(weight_hh_data)
        # The bias is just set to zero vectors.
        if self.use_bias:
            init.constant(self.bias.data, val=0)

    def forward(self, input_, hx):
        """
        Args:
            input_: A (batch, input_size) tensor containing input
                features.
            hx: A tuple (h_0, c_0), which contains the initial hidden
                and cell state, where the size of both states is
                (batch, hidden_size).
        Returns:
            h_1, c_1: Tensors containing the next hidden and cell state.
        """

        h_0, c_0 = hx
        batch_size = h_0.size(0)
        bias_batch = (self.bias.unsqueeze(0).expand(batch_size, *self.bias.size()))
        '''bias_batch + h_0*self.weight_hh'''
        wh_b = torch.addmm(bias_batch, h_0, self.weight_hh)
        '''input_*self.weight_ih'''
        wi = torch.mm(input_, self.weight_ih)
        f, i, g = torch.split(wh_b + wi, split_size=self.hidden_size, dim=1)
        c_1 = torch.sigmoid(f) * c_0 + torch.sigmoid(i) * torch.tanh(g)
        return c_1

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class MultiInputLSTMCell(nn.Module):
    """A basic LSTM cell."""

    # (hx, cx) = self.rnn(input[t], input_c_list[t], (hx, cx))

    def __init__(self, input_size, hidden_size, use_bias=True, use_w2c=False, use_attn=False):
        """
        Most parts are copied from torch.nn.LSTMCell.
        """

        super(MultiInputLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.use_w2c = use_w2c
        self.use_attn = use_attn

        self.weight_ih = nn.Parameter(
            torch.FloatTensor(input_size, 3 * hidden_size))
        self.weight_hh = nn.Parameter(
            torch.FloatTensor(hidden_size, 3 * hidden_size))
        self.alpha_weight_ih = nn.Parameter(
            torch.FloatTensor(input_size, hidden_size))
        self.alpha_weight_hh = nn.Parameter(
            torch.FloatTensor(hidden_size, hidden_size))

        '''concat'''
        if self.use_attn and config['combination'] == 0:
            self.beta_weight = nn.Parameter(
                torch.FloatTensor(hidden_size + input_size, input_size))
        else:
            self.beta_weight = nn.Parameter(
                torch.FloatTensor(hidden_size, input_size))

        self.theta_weight_ih = nn.Parameter(
            torch.FloatTensor(input_size, 3 * hidden_size)
        )
        self.theta_weight_hh = nn.Parameter(
            torch.FloatTensor(hidden_size, 3 * hidden_size)
        )
        if self.use_w2c:
            self.w2c_weight = nn.Parameter(
                torch.FloatTensor(input_size, input_size)
            )

        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(3 * hidden_size))
            self.alpha_bias = nn.Parameter(torch.FloatTensor(hidden_size))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('alpha_bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters following the way proposed in the paper.
        """
        init.orthogonal(self.weight_ih.data)
        init.orthogonal(self.alpha_weight_ih.data)
        init.orthogonal(self.theta_weight_ih.data)

        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 3)
        # w2c_weight_data = torch.eye(self.input_size)

        theta_weight_hh_data = torch.eye(self.hidden_size)
        theta_weight_hh_data = theta_weight_hh_data.repeat(1, 3)
        self.weight_hh.data.set_(weight_hh_data)
        self.theta_weight_hh.data.set_(theta_weight_hh_data)

        alpha_weight_hh_data = torch.eye(self.hidden_size)
        alpha_weight_hh_data = alpha_weight_hh_data.repeat(1, 1)
        self.alpha_weight_hh.data.set_(alpha_weight_hh_data)

        init.orthogonal(self.beta_weight.data)

        if self.use_w2c:
            init.orthogonal(self.w2c_weight.data)

        # The bias is just set to zero vectors.
        if self.use_bias:
            init.constant(self.bias.data, val=0)
            init.constant(self.alpha_bias.data, val=0)

    def forward(self, input_, word_emb, hx, sent_info=None):
        h_0, c_0 = hx
        batch_size = h_0.size(0)
        assert (batch_size == 1)
        bias_batch = (self.bias.unsqueeze(0).expand(batch_size, *self.bias.size()))

        if type(word_emb) == type([]):
            X = input_

        else:
            if self.use_w2c:
                word_emb_ = torch.mm(self.w2c_weight, word_emb.t())
                X = torch.cat((word_emb_, input_.t()), 1)  # ix(n+1)
            else:
                X = torch.cat((word_emb.t(), input_.t()), 1)  # ix(n+1)

            if self.use_attn:
                if config['combination'] == 0:
                    s = torch.mm(torch.mm(torch.cat((h_0, sent_info), dim=1), self.beta_weight), X)  # 1x(n+1)
                elif config['combination'] == 1:
                    s = torch.mm(torch.mm(h_0 * sent_info, self.beta_weight), X)
                else:
                    print('error @ multicell forward.')
                    exit(0)
            else:
                s = torch.mm(torch.mm(h_0, self.beta_weight), X)  # 1x(n+1)
            s = functional.softmax(s, dim=0)
            X = torch.mm(s, X.t())  # 1xi
        wh_b = torch.addmm(bias_batch, h_0, self.theta_weight_hh)
        wi = torch.mm(X, self.theta_weight_ih)
        i, o, g = torch.split(wh_b + wi, split_size=self.hidden_size, dim=1)
        i = torch.sigmoid(i)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        f = 1 - i
        c_1 = f * c_0 + i * g
        h_1 = o * torch.tanh(c_1)
        return h_1, c_1

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class LatticeLSTM(nn.Module):
    """A module that runs multiple steps of LSTM."""

    # input_dim=50
    # hidden_dim=100

    def __init__(self, input_dim, hidden_dim, word_drop, word_alphabet_size, word_emb_dim, pretrain_word_emb=None,
                 left2right=True, fix_word_emb=True, gpu=True, use_bias=True, n_head=5, use_attn=False, use_w2c=False):
        super(LatticeLSTM, self).__init__()
        skip_direction = "forward" if left2right else "backward"
        print("build LatticeLSTM... ", skip_direction, ", Fix emb:", fix_word_emb, " gaz drop:", word_drop)
        self.gpu = gpu
        self.hidden_dim = hidden_dim
        self.word_emb = nn.Embedding(word_alphabet_size, word_emb_dim)
        self.n_head = config['n_head']
        self.use_attn = use_attn
        self.use_w2c = use_w2c
        logger.info(("self.use_attn: ", self.use_attn))
        logger.info(("self.use_w2c: ", self.use_w2c))

        if pretrain_word_emb is not None:
            print("load pretrain word emb...", pretrain_word_emb.shape)
            self.word_emb.weight.data.copy_(torch.from_numpy(pretrain_word_emb))

        else:
            self.word_emb.weight.data.copy_(torch.from_numpy(self.random_embedding(word_alphabet_size, word_emb_dim)))
        if fix_word_emb:
            self.word_emb.weight.requires_grad = False

        self.word_dropout = nn.Dropout(word_drop)

        self.rnn = MultiInputLSTMCell(input_dim, hidden_dim, use_attn=self.use_attn, use_w2c=self.use_w2c)

        # self.word_rnn = WordLSTMCell(word_emb_dim, hidden_dim)
        self.left2right = left2right
        if self.gpu:
            self.rnn = self.rnn.cuda()
            self.word_emb = self.word_emb.cuda()
            self.word_dropout = self.word_dropout.cuda()
            # self.word_rnn = self.word_rnn.cuda()

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def forward(self, input, skip_input_list, hidden=None, attn=None):
        """
            input: variable (batch, seq_len), batch = 1
            skip_input_list: [skip_input, volatile_flag]
            skip_input: three dimension list, with length is seq_len. Each element is a list of matched word id and its length.
                        example: [[], [[25,13],[2,3]]] 25/13 is word id, 2,3 is word length .
        """

        volatile_flag = skip_input_list[1]
        skip_input = skip_input_list[0]
        if not self.left2right:
            skip_input = convert_forward_gaz_to_backward(skip_input)
        input = input.transpose(1, 0)
        seq_len = input.size(0)

        if self.use_attn and not config['shared_attn']:
            Attn = MultiHeadAttention(self.n_head, input.size(2), int(input.size(2)/self.n_head), int(input.size(2)/self.n_head))
            if self.gpu:
                Attn = Attn.cuda()


            q = input

            if self.gpu:
                q = q.cuda()
            q = q.permute(1, 0, 2)

            output, _ = Attn.forward(q, q, q)


        batch_size = input.size(1)
        assert (batch_size == 1)
        hidden_out = []
        memory_out = []
        if hidden:
            (hx, cx) = hidden
        else:
            hx = autograd.Variable(torch.zeros(batch_size, self.hidden_dim))
            cx = autograd.Variable(torch.zeros(batch_size, self.hidden_dim))
            if self.gpu:
                hx = hx.cuda()
                cx = cx.cuda()

        id_list = list(range(seq_len))
        if not self.left2right:
            id_list = list(reversed(id_list))

        '''i deleted'''
        # input_c_list = init_list_of_objects(seq_len)

        for t in id_list:
            if skip_input[t]:
                matched_num = len(skip_input[t][0])
                word_var = autograd.Variable(torch.LongTensor(skip_input[t][0]), volatile=volatile_flag)
                if self.gpu:
                    word_var = word_var.cuda()
                word_emb = self.word_emb(word_var)
                word_emb = self.word_dropout(word_emb)
                # ct=self.word_rnn(word_emb,(hx,cx))
                # assert(ct.size(0)==len(skip_input[t][1]))
                # for idx in range(matched_num):
                #     length=skip_input[t][1][idx]
                #     if self.left2right:
                #         input_c_list[t].append(ct[idx,:].unsqueeze(0))
                if self.use_attn:
                    if config['shared_attn']:
                        hx, cx = self.rnn(input[t], word_emb, (hx, cx), attn)
                    if not config['shared_attn']:
                        hx, cx = self.rnn(input[t], word_emb, (hx, cx), output)
                else:
                    hx, cx = self.rnn(input[t], word_emb, (hx, cx))
            else:
                if self.use_attn:
                    if config['shared_attn']:
                        hx, cx = self.rnn(input[t], [], (hx, cx), attn)
                    if not config['shared_attn']:
                        hx, cx = self.rnn(input[t], [], (hx, cx), output)
                else:
                    hx, cx = self.rnn(input[t], [], (hx, cx))

            hidden_out.append(hx)
            memory_out.append(cx)

        if not self.left2right:
            hidden_out = list(reversed(hidden_out))
            memory_out = list(reversed(memory_out))
        output_hidden, output_memory = torch.cat(hidden_out, 0), torch.cat(memory_out, 0)
        # (batch, seq_len, hidden_dim)
        # print output_hidden.size()
        return output_hidden.unsqueeze(0), output_memory.unsqueeze(0)


def init_list_of_objects(size):
    list_of_objects = list()
    for i in range(0, size):
        list_of_objects.append(list())
    return list_of_objects


def convert_forward_gaz_to_backward(forward_gaz):
    # print forward_gaz
    length = len(forward_gaz)
    backward_gaz = init_list_of_objects(length)
    for idx in range(length):
        if forward_gaz[idx]:
            assert (len(forward_gaz[idx]) == 2)
            num = len(forward_gaz[idx][0])
            for idy in range(num):
                the_id = forward_gaz[idx][0][idy]
                the_length = forward_gaz[idx][1][idy]
                new_pos = idx + the_length - 1
                if backward_gaz[new_pos]:
                    backward_gaz[new_pos][0].append(the_id)
                    backward_gaz[new_pos][1].append(the_length)
                else:
                    backward_gaz[new_pos] = [[the_id], [the_length]]
    return backward_gaz



