import numpy as np
import os
import theano
import theano.tensor as T
from deep_disfluency.rnn.lstm import LSTM
from theano import shared
from collections import OrderedDict
class BiLstm(object):
    def __init__(self, input, input_dim, hidden_dim, output_dim,
                 mini_batch=False, params=None):
        self.mini_batch = mini_batch
        input_f = input
        if mini_batch:
            input_b = input[::, ::-1]
        else:
            input_b = input[::-1]
        if params is None:
            # self.fwd_lstm = LSTM(input=input_f, input_dim=input_dim, hidden_dim=hidden_dim,
            #                      output_dim=output_dim, mini_batch=mini_batch)
            # self.bwd_lstm = LSTM(input=input_b, input_dim=input_dim, hidden_dim=hidden_dim,
            #                      output_dim=output_dim, mini_batch=mini_batch)
            self.fwd_lstm = LSTM(ne=vocab_size,
                         de=emb_dimension,
                         n_lstm=n_hidden,
                         na=n_extra,
                         n_out=n_classes,
                         cs=self.window_size,
                         npos=n_pos,
                         lr=lr,
                         single_output=True,
                         cost_function='nll',
                         bcw=False)
            self.bwd_lstm = LSTM(ne=vocab_size,
                         de=emb_dimension,
                         n_lstm=n_hidden,
                         na=n_extra,
                         n_out=n_classes,
                         cs=self.window_size,
                         npos=n_pos,
                         lr=lr,
                         single_output=True,
                         cost_function='nll',
                         bcw=True)
            self.V_f = theano.shared(
                value=get(identifier='uniform', shape=(hidden_dim, output_dim)),
                name='V_f',
                borrow=True
            )
            self.V_b = theano.shared(
                value=get(identifier='uniform', shape=(hidden_dim, output_dim)),
                name='V_b',
                borrow=True
            )
            self.by = theano.shared(
                value=get('zero', shape=(output_dim,)),
                name='by',
                borrow=True)

        else:
            # To support loading from persistent storage, the current implementation of Lstm() will require a
            # change and is therefore not supported.
            # An elegant way would be to implement BiLstm() without using Lstm() [is a trivial thing to do].
            raise NotImplementedError

        # since now bilstm is doing the actual classification ; we don't need 'Lstm().V & Lstm().by' as they
        # are not part of computational graph (separate logistic-regression unit/layer is probably the best way to
        # handle this). Here's the ugly workaround -_-
        self.params = [self.fwd_lstm.W_i, self.fwd_lstm.U_i, self.fwd_lstm.b_i,
                       self.fwd_lstm.W_f, self.fwd_lstm.U_f, self.fwd_lstm.b_f,
                       self.fwd_lstm.W_c, self.fwd_lstm.U_c, self.fwd_lstm.b_c,
                       self.fwd_lstm.W_o, self.fwd_lstm.U_o, self.fwd_lstm.b_o,

                       self.bwd_lstm.W_i, self.bwd_lstm.U_i, self.bwd_lstm.b_i,
                       self.bwd_lstm.W_f, self.bwd_lstm.U_f, self.bwd_lstm.b_f,
                       self.bwd_lstm.W_c, self.bwd_lstm.U_c, self.bwd_lstm.b_c,
                       self.bwd_lstm.W_o, self.bwd_lstm.U_o, self.bwd_lstm.b_o,

                       self.V_f, self.V_b, self.by]

        self.bwd_lstm.h_t = self.bwd_lstm.h_t[::-1]
        # Take the weighted sum of forward & backward lstm's hidden representation
        self.h_t = T.dot(self.fwd_lstm.h_t, self.V_f) + T.dot(self.bwd_lstm.h_t, self.V_b)

        if mini_batch:
            # T.nnet.softmax cannot operate on tensor3, here's a simple reshape trick to make it work.
            h_t = self.h_t + self.by
            h_t_t = T.reshape(h_t, (h_t.shape[0] * h_t.shape[1], -1))
            y_t = T.nnet.softmax(h_t_t)
            self.y_t = T.reshape(y_t, h_t.shape)
            self.y = T.argmax(self.y_t, axis=2)
        else:
            self.y_t = T.nnet.softmax(self.h_t + self.by)
            self.y = T.argmax(self.y_t, axis=1)

    def cross_entropy(self, y):
        if self.mini_batch:
            return T.mean(T.sum(T.nnet.categorical_crossentropy(self.y_t, y), axis=1))  # naive batch-normalization
        else:
            return T.sum(T.nnet.categorical_crossentropy(self.y_t, y))

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.y_t)[:, y])

    def errors(self, y):
        return T.mean(T.neq(self.y, y))

    # TODO: Find a way of sampling (running forward + backward lstm manually is really ugly and therefore, avoided).
    def generative_sampling(self, seed, emb_data, sample_length):
        return NotImplementedError