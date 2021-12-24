import numpy as np
import os
import sys
import theano
import theano.tensor as T
from theano import shared
from collections import OrderedDict
from keras import Input, Model, initializers
from keras.models import Sequential
from keras import layers
import tensorflow as tf
# import tensorflow_addons as tfa
THIS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(THIS_DIR + "/../../")
# from keras.layers import Dense, LSTM,Flatten, Dropout, Conv1D, MaxPooling1D, \
#      GlobalMaxPooling1D, concatenate,Bidirectional, Layer
# from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping,TensorBoard
from keras.optimizers import Adagrad, SGD
from keras import backend as K

def init_weight(shape, name, sample='uni', seed=None):
    rng = np.random.RandomState(seed)
    if sample == 'unishape':
        values = rng.uniform(
            low=-np.sqrt(6. / (shape[0] + shape[1])),
            high=np.sqrt(6. / (shape[0] + shape[1])),
            size=shape).astype(dtype)
    elif sample == 'svd':
        values = rng.uniform(low=-1., high=1., size=shape).astype(dtype)
        _, svs, _ = np.linalg.svd(values)
        # svs[0] is the largest singular value
        values = values / svs[0]
    elif sample == 'uni':
        values = rng.uniform(low=-0.1, high=0.1, size=shape).astype(dtype)
    elif sample == 'zero':
        values = np.zeros(shape=shape, dtype=dtype)
    else:
        raise ValueError("Unsupported initialization scheme: %s"
                         % sample)
    #shared variable for fixed set of values
    return shared(values, name=name, borrow=True)

dtype = T.config.floatX  # @UndefinedVariable
#FOR retraing using theano LSTM weights
# class keras_LSTM(object):
#     def __init__(self, sequence_len,emb,folder,ne, de, na, n_lstm, n_out, cs, npos, lr=0.05,
#                  single_output=True, output_activation=T.nnet.softmax,
#                  cost_function='nll',bcw=False):
#         inputs = np.random.random([30, 23, 9]).astype(np.float32)
#         char_lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(50,
#                                                     use_peepholes=True,
#                                                     state_is_tuple=True,
#
#                                                     reuse=False)
#
#     # def __init__(self, sequence_len,emb,folder,ne, de, na, n_lstm, n_out, cs, npos, lr=0.05,
#     #              single_output=True, output_activation=T.nnet.softmax,
#     #              cost_function='nll',bcw=False):
#     #     '''
#     #     ne :: number of word embeddings in the vocabulary
#     #     de :: dimension of the word embeddings
#     #     na :: number of acoustic or language model features at each word step
#     #             (acoustic context size in frames * number of features)
#     #     n_lstm :: dimension of the lstm layer
#     #     n_out :: number of classes
#     #     cs :: word window context size
#     #     npos :: number of pos tags
#     #     '''
#     #     #dimension initialization
#     #     self.sequence_len=sequence_len
#     #     self.emb = emb
#     #     self.vocab=ne+1
#     #     self.word_in = (de * cs)
#     #     self.window=cs
#     #     self.emb_dim=de
#     #     self.pos_in=(125 * cs)
#     #     self.n_lstm = n_lstm
#     #     self.n_out = n_out
#     #     self.lr=lr
#     #     self.names = ["W_xi", "W_hi", "W_ci", "b_i",
#     #                   "W_xf", "W_hf", "W_cf", "b_f",
#     #                   "W_xc", "W_hc", "b_c",
#     #                   "W_xo", "W_ho", "W_co", "b_o",
#     #                   "W_hy", "b_y", "embeddings"]
#     #     #input/output shape
#     #
#     #     word_inputs = Input(shape=(self.sequence_len,self.window))
#     #     print('word')
#     #     print(word_inputs.shape)
#     #     pos_inputs=Input(shape=(self.sequence_len,self.pos_in))
#     #     print('pos')
#     #     print(pos_inputs.shape)
#     #     embedding = layers.Embedding(self.vocab, self.emb_dim, input_length=(self.sequence_len,self.window), weights=[self.emb],
#     #                           trainable=True)(word_inputs) # line A
#     #     print('vocab')
#     #     print(self.vocab)
#     #     print('emb')
#     #
#     #     print(embedding.shape)
#     #     embedding=layers.Reshape((self.sequence_len,self.word_in))(embedding)
#     #     print(embedding.shape)
#     #     word_pos_inputs= layers.concatenate([embedding,pos_inputs],axis=2)
#     #     print('concat')
#     #     print(word_pos_inputs.shape)
#     #     #model architectute
#     #
#     #     lstm=layers.LSTM(units=self.n_lstm,return_sequences=True,trainable=False)(word_pos_inputs)
#     #
#     #     output=layers.Dense(output_dim=self.n_out, activation='softmax')(lstm)
#     #     model = Model(inputs=[word_inputs, pos_inputs], outputs=output)
#     #
#     #
#     #
#     #
#     #
#     #
#     #     # #loading pretrained weight
#     lstm_input,lstm_hidden,lstm_bias, output_weight = self.load_retrain_weights(folder)
#     #     print('lstm_input')
#     #     print(lstm_input.shape)
#     #     print('lstm_hidden')
#     #     print(lstm_hidden.shape)
#     #     print('lstm_bias')
#     #     print(lstm_bias.shape)
#     #
#     #     # print(model.get_layer('lstm_1').get_weights())
#     #     for elem in model.get_layer('lstm_1').get_weights():
#     #         print('elem')
#     #         print(elem.shape)
#     #
#     model.get_layer('lstm_1').set_weights((lstm_input,lstm_hidden,lstm_bias))
#     model.get_layer('dense_1').set_weights(output_weight)
#     #     # #model compile
#     #     model.compile(loss='categorical_crossentropy', optimizer=SGD(self.lr), metrics=['accuracy'])
#     #     model.summary()
#     def load_retrain_weights(self, folder):
#         print('shape of weight')
#         lstm_input=[]
#         lstm_hidden = []
#         lstm_bias = []
#         output_weight=[]
#         for name in self.names:
#             if name in ["W_xi",
#                       "W_xf",
#                       "W_xc"]:
#
#                 lstm_input.append(np.load(os.path.join(folder, name + ".npy")))
#             elif name in [ "W_hi", "W_ci",
#                       "W_hf", "W_cf",
#                        "W_hc",
#                       "W_ho", "W_co" ]:
#
#                 lstm_hidden.append(np.load(os.path.join(folder, name + ".npy")))
#             elif name in ["b_i",
#                        "b_f",
#                       "b_c",
#                        "b_o"
#                        ]:
#
#                 lstm_bias.append(np.load(os.path.join(folder, name + ".npy")))
#             elif name in ["W_hy", "b_y"]:
#                 output_weight.append(np.load(os.path.join(folder, name + ".npy")))
#             elif name in ["embeddings"]:
#                 print("embeddings")
#                 np.load(os.path.join(folder, name + ".npy")).shape
#             elif name in ["W_xo"]:
#                 lstm_input.append(init_weight((self.word_in+self.pos_in, self.n_lstm), 'W_xo'))
#
#
#
#         return  np.array(lstm_input),np.array(lstm_hidden),np.array(lstm_bias),np.array(output_weight)
#
#
#




class my_LSTM(object):

    def __init__(self, ne, de, na, n_lstm, n_out, cs, npos, lr=0.05,
                 single_output=True, output_activation=T.nnet.softmax,
                 cost_function='nll',bcw=False):
        '''
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        na :: number of acoustic or language model features at each word step
                (acoustic context size in frames * number of features)
        n_lstm :: dimension of the lstm layer
        n_out :: number of classes
        cs :: word window context size
        npos :: number of pos tags
        '''

        self.emb = init_weight((ne+1, de), 'emb')  # add one to ne for PADDING
        self.n_in = (de * cs)+(npos * cs)
        self.n_lstm = n_lstm
        self.n_out = n_out
        self.W_xi = init_weight((self.n_in, self.n_lstm), 'W_xi')
        self.W_hi = init_weight((self.n_lstm, self.n_lstm), 'W_hi', 'svd')
        self.W_ci = init_weight((self.n_lstm, self.n_lstm), 'W_ci', 'svd')
        # bias to the input:
        self.b_i = shared(np.cast[dtype](np.random.uniform(-0.5, .5,
                                                           size=n_lstm)))
        # forget gate weights:
        self.W_xf = init_weight((self.n_in, self.n_lstm), 'W_xf')
        self.W_hf = init_weight((self.n_lstm, self.n_lstm), 'W_hf', 'svd')
        self.W_cf = init_weight((self.n_lstm, self.n_lstm), 'W_cf', 'svd')
        # bias
        self.b_f = shared(np.cast[dtype](np.random.uniform(0, 1.,
                                                           size=n_lstm)))
        # memory cell gate weights:
        self.W_xc = init_weight((self.n_in, self.n_lstm), 'W_xc')
        self.W_hc = init_weight((self.n_lstm, self.n_lstm), 'W_hc', 'svd')
        # bias to the memory cell:
        self.b_c = shared(np.zeros(n_lstm, dtype=dtype))
        # output gate weights:
        self.W_xo = init_weight((self.n_in, self.n_lstm), 'W_xo')
        self.W_ho = init_weight((self.n_lstm, self.n_lstm), 'W_ho', 'svd')
        self.W_co = init_weight((self.n_lstm, self.n_lstm), 'W_co', 'svd')
        # bias on output gate:
        self.b_o = shared(np.cast[dtype](np.random.uniform(-0.5, .5,
                                                           size=n_lstm)))
        #adding hidden layer
        self.W_feed = init_weight((self.n_lstm, self.n_lstm), 'W_feed')
        self.b_feed = shared(np.zeros(self.n_lstm, dtype=dtype))  # output bias
        # hidden to y matrix weights:
        self.W_hy = init_weight((self.n_lstm, self.n_out), 'W_hy')
        self.b_y = shared(np.zeros(n_out, dtype=dtype))  # output bias

        # Weights for L1 and L2
        self.L1_reg = 0.0
        self.L2_reg = 0.00001

        self.params = [self.W_xi, self.W_hi, self.W_ci, self.b_i,
                       self.W_xf, self.W_hf, self.W_cf, self.b_f,
                       self.W_xc, self.W_hc, self.b_c,
                       self.W_ho, self.W_co, self.W_co, self.b_o,
                       self.W_hy, self.b_y, self.emb]
        self.names = ["W_xi", "W_hi", "W_ci", "b_i",
                      "W_xf", "W_hf", "W_cf", "b_f",
                      "W_xc", "W_hc", "b_c",
                      "W_ho", "W_co", "W_co", "b_o",
                      "W_hy", "b_y", "embeddings"]

        self.save_params = [self.W_xi, self.W_hi, self.W_ci, self.b_i,
                       self.W_xf, self.W_hf, self.W_cf, self.b_f,
                       self.W_xc, self.W_hc, self.b_c,
                       self.W_xo, self.W_ho, self.W_co, self.b_o,
                       self.W_hy, self.b_y, self.emb]
        self.save_names = ["W_xi", "W_hi", "W_ci", "b_i",
                          "W_xf", "W_hf", "W_cf", "b_f",
                          "W_xc", "W_hc", "b_c",
                          "W_xo", "W_ho", "W_co", "b_o",
                          "W_hy", "b_y", "embeddings"]

        def step_lstm(x_t, h_tm1, c_tm1):
            i_t = T.nnet.sigmoid(T.dot(x_t, self.W_xi) +
                                 T.dot(h_tm1, self.W_hi) +
                                 T.dot(c_tm1, self.W_ci) + self.b_i)
            f_t = T.nnet.sigmoid(T.dot(x_t, self.W_xf) +
                                 T.dot(h_tm1, self.W_hf) +
                                 T.dot(c_tm1, self.W_cf) + self.b_f)
            c_t = f_t * c_tm1 + i_t * T.tanh(T.dot(x_t, self.W_xc) +
                                             T.dot(h_tm1, self.W_hc) +
                                             self.b_c)
            o_t = T.nnet.sigmoid(T.dot(x_t, self.W_xo) +
                                 T.dot(h_tm1, self.W_ho) +
                                 T.dot(c_t, self.W_co) + self.b_o)
            h_t = o_t * T.tanh(c_t)
            y_feed=T.nnet.relu(T.dot(h_t, self.W_feed) + self.b_feed)
            # y_t = T.nnet.softmax(T.dot(h_t, self.W_hy) + self.b_y)
            y_t = T.nnet.softmax(T.dot(y_feed, self.W_hy) + self.b_y)
            return [h_t, c_t, y_t]

        # batch of sequence of vectors
        self.idxs = T.imatrix()
        self.pos_idxs = T.imatrix()

        # The eye function (diagonal 1s) for the POS, small in memory
        self.pos = T.eye(npos, npos, 0)
        # TODO No pos
        # x = self.emb[self.idxs].reshape((self.idxs.shape[0], de*cs))
        # POS version
        # print('inside lstm')
        # print(self.idxs.shape)
        # print(self.pos_idxs.shape)
        x = T.concatenate((self.emb[self.idxs].reshape((self.idxs.shape[0],
                                                        de*cs)),
                           self.pos[self.pos_idxs].reshape(
                                            (self.pos_idxs.shape[0],
                                             npos*cs))), 1)
        # print(x.shape)
        self.y = T.iscalar('y')
        # initial hidden state
        self.h0 = shared(np.zeros(shape=self.n_lstm, dtype=dtype))
        self.c0 = shared(np.zeros(shape=self.n_lstm, dtype=dtype))
        self.lr = T.scalar('lr')
        [h_vals, c_vals, y_vals], _ = theano.scan(
                                        fn=step_lstm,
                                        sequences=x,
                                        outputs_info=[self.h0, self.c0, None],
                                        n_steps=x.shape[0])
        self.output = y_vals
        p_y_given_x_lastword = self.output[-1, 0, :]
        p_y_given_x_sentence = self.output[:, 0, :]
        p_y_given_x_sentence_hidden = (h_vals, c_vals, self.output[:, 0, :])
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)
        # y_pred_word = T.argmax(p_y_given_x_lastword)

        self.cxe = T.mean(T.nnet.binary_crossentropy(self.output, self.y))
        self.nll = -T.mean(T.log(p_y_given_x_lastword)[self.y])
        self.mse = T.mean((self.output - self.y) ** 2)

        self.sentence_nll = -T.mean(T.log(p_y_given_x_sentence)
                                    [T.arange(x.shape[0]), self.y])

        self.L2_sqr = sum([(p ** 2).sum() for p in self.params])
        self.cost = self.nll + self.L2_reg * self.L2_sqr
        if cost_function == 'mse':
            self.cost = self.mse + self.L2_reg * self.L2_sqr
        elif cost_function == 'cxe':
            self.cost = self.cxe + self.L2_reg * self.L2_sqr
        self.debug = theano.function(inputs=[x, self.y],
                                     outputs=[x.shape, self.y.shape,
                                              y_vals.shape, self.cost.shape])
        gradients = T.grad(self.cost, self.params)
        self.updates = OrderedDict((p, p-self.lr*g)
                                   for p, g in zip(self.params, gradients))
        self.loss = theano.function(inputs=[x, self.y], outputs=self.cost)
        # if na == 0: #assume no acoustic features for now
        # simply outputs the soft_max distribution for each word in utterance
        self.soft_max = theano.function(inputs=[self.idxs, self.pos_idxs],
                                        outputs=p_y_given_x_sentence)
        self.soft_max_return_hidden_layer = theano.function(
                                        inputs=[self.idxs, self.pos_idxs],
                                        outputs=p_y_given_x_sentence_hidden)
        if na == 0:
            self.train = theano.function(inputs=[self.idxs, self.pos_idxs,
                                                 self.y, self.lr],
                                         outputs=self.cost,
                                         updates=self.updates)
            self.classify = theano.function(inputs=[self.idxs, self.pos_idxs],
                                        outputs=y_pred)
        else:
            self.train = theano.function(inputs=[self.idxs, self.pos_idxs,
                                                 self.acoustic,
                                                 self.y, self.lr],
                                         outputs=self.cost,
                                         updates=self.updates)
            self.classify = theano.function(inputs=[self.idxs, self.pos_idxs,
                                                    self.acoustic],
                                        outputs=y_pred)
        self.normalize = theano.function(
                                         inputs=[],
                                         updates={self.emb:
                                                  self.emb/T.sqrt(
                                                                (self.emb**2).
                                                                sum(axis=1))
                                                  .dimshuffle(0, 'x')})

    def classify_by_index(self, word_idx, indices, pos_idx=None,
                          extra_features=None):
        """Classification method which assumes the dialogue matrix is
        in the right format.

        :param word_idx: window size * dialogue length matrix
        :param labels: vector dialogue length long
        :param indices: 2 * dialogue length matrix for start, stop indices
        :param pos_idx: pos window size * dialogue length matrix
        :param extra_features: number of features * dialogue length matrix
        """
        output = []
        for start, stop in indices:

            if extra_features:

                output.extend(self.classify(word_idx[start:stop+1, :],
                                            pos_idx[start:stop+1, :],
                                            np.asarray(
                                            extra_features[start:stop+1, :],
                                            dtype='float32')
                                            )
                              )
            else:
                output.extend(self.classify(word_idx[start:stop+1, :],
                                            pos_idx[start:stop+1, :]
                                            )
                              )
        return output

    def fit(self, word_idx, labels, lr, indices, pos_idx=None,
            extra_features=None):
        """Fit method which assumes the dialogue matrix is in the right
        format.

        :param word_idx: window size * dialogue length matrix
        :param labels: vector dialogue length long
        :param indices: 2 * dialogue length matrix for start, stop indices
        :param pos_idx: pos window size * dialogue length matrix
        :param extra_features: number of features * dialogue length matrix
        """
        loss = 0
        test = 0
        testing = False
        for start, stop in indices:
            if testing:
                test += 1
                if test > 50:
                    break

            if extra_features:

                x = self.train(word_idx[start:stop+1, :],
                               pos_idx[start:stop+1, :],
                               np.asarray(extra_features[start:stop+1, :],
                                          dtype='float32'),
                               labels[stop],
                               lr)
            else:
                # print(word_idx.shape)
                # print(pos_idx.shape)
                # print(labels.shape)

                x = self.train(word_idx[start:stop+1, :],
                               pos_idx[start:stop+1, :],
                               labels[stop],
                               lr)
            loss += x
            self.normalize()
        return loss

    def shared_dataset(self, mycorpus, borrow=True):
        """ Load the dataset into shared variables """
        return theano.shared(np.asarray(mycorpus, dtype='int32'), borrow=True)

    def load_weights_from_folder(self, folder):
        # print('shape of weight')
        for name, param in zip(self.names, self.params):
            # print(name)
            # print(np.load(os.path.join(folder, name + ".npy")).shape)
            param.set_value(np.load(os.path.join(folder, name + ".npy")))

    def load_lstm_weights_from_folder(self, folder):
        # print('shape of weight')
        for name, param in zip(self.names, self.params):
            if name not in [ "W_hy", "b_y", "embeddings"]:
                # print(name)
                # print(np.load(os.path.join(folder, name + ".npy")).shape)
                param.set_value(np.load(os.path.join(folder, name + ".npy")))
    def load_output_weight(self,weight,name,seed=None):
        rng = np.random.RandomState(seed)

        if name=='W_hy':
            values = rng.uniform(low=-0.1, high=0.1, size=(self.n_lstm,self.n_out)).astype(dtype)
            for i in range(0,len(weight)):
                for j in range(0,len(weight[i])):
                    values[i][j]=weight[i][j]
            return values
        else:
            values=np.zeros(shape=self.n_out, dtype=dtype)
            for i in range(0,len(weight)):
                values[i]=weight[i]
            return values


    def load_output_weights_from_folder(self, folder):
        # print('shape of weight')
        for name, param in zip(self.names, self.params):
            if name  in [ "W_hy", "b_y"]:
                # print(name)
                # print(np.load(os.path.join(folder, name + ".npy")).shape)
                output=self.load_output_weight(np.load(os.path.join(folder, name + ".npy")),name)
                param.set_value(output)


    def load_weights(self, emb=None, c0=None, h0=None):
        if emb is not None:
            self.emb.set_value(emb)
        if c0 is not None:
            self.c0.set_value(c0)
        if h0 is not None:
            self.h0.set_value(h0)

    def save(self, folder):
        for save_param, save_name in zip(self.save_params, self.save_names):
            np.save(os.path.join(folder, save_name + '.npy'), save_param.get_value())


class LstmMiniBatch:
    def __init__(self, n_in, n_lstm, n_out, lr=0.05, batch_size=64,
                 single_output=True,
                 output_activation=T.nnet.softmax, cost_function='nll'):
        self.n_in = n_in
        self.n_lstm = n_lstm
        self.n_out = n_out
        self.W_xi = init_weight((self.n_in, self.n_lstm), 'W_xi')
        self.W_hi = init_weight((self.n_lstm, self.n_lstm), 'W_hi', 'svd')
        self.W_ci = init_weight((self.n_lstm, self.n_lstm), 'W_ci', 'svd')
        self.b_i = shared(np.cast[dtype](np.random.uniform(-0.5, .5,
                                                           size=n_lstm)))
        self.W_xf = init_weight((self.n_in, self.n_lstm), 'W_xf')
        self.W_hf = init_weight((self.n_lstm, self.n_lstm), 'W_hf', 'svd')
        self.W_cf = init_weight((self.n_lstm, self.n_lstm), 'W_cf', 'svd')
        self.b_f = shared(np.cast[dtype](np.random.uniform(0, 1.,
                                                           size=n_lstm)))
        self.W_xc = init_weight((self.n_in, self.n_lstm), 'W_xc')
        self.W_hc = init_weight((self.n_lstm, self.n_lstm), 'W_hc',
                                'svd')
        self.b_c = shared(np.zeros(n_lstm, dtype=dtype))
        self.W_xo = init_weight((self.n_in, self.n_lstm), 'W_xo')
        self.W_ho = init_weight((self.n_lstm, self.n_lstm), 'W_ho', 'svd')
        self.W_co = init_weight((self.n_lstm, self.n_lstm), 'W_co', 'svd')
        self.b_o = shared(np.cast[dtype](np.random.uniform(-0.5,
                                                           .5,
                                                           size=n_lstm)))
        self.W_hy = init_weight((self.n_lstm, self.n_out),
                                'W_hy')
        self.b_y = shared(np.zeros(n_out, dtype=dtype))
        self.params = [self.W_xi, self.W_hi, self.W_ci, self.b_i,
                       self.W_xf, self.W_hf, self.W_cf, self.b_f,
                       self.W_xc, self.W_hc, self.b_c,
                       self.W_ho, self.W_co, self.W_co, self.b_o,
                       self.W_hy, self.b_y]

        def step_lstm(x_t, h_tm1, c_tm1):
            i_t = T.nnet.sigmoid(T.dot(x_t, self.W_xi) +
                                 T.dot(h_tm1, self.W_hi) +
                                 T.dot(c_tm1, self.W_ci) + self.b_i
                                 )
            f_t = T.nnet.sigmoid(T.dot(x_t, self.W_xf) +
                                 T.dot(h_tm1, self.W_hf) +
                                 T.dot(c_tm1, self.W_cf) + self.b_f
                                 )
            c_t = f_t * c_tm1 + i_t * T.tanh(T.dot(x_t, self.W_xc) +
                                             T.dot(h_tm1, self.W_hc) +
                                             self.b_c
                                             )
            o_t = T.nnet.sigmoid(T.dot(x_t, self.W_xo) +
                                 T.dot(h_tm1, self.W_ho) +
                                 T.dot(c_t, self.W_co) + self.b_o
                                 )
            h_t = o_t * T.tanh(c_t)
            y_t = T.nnet.softmax(T.dot(h_t, self.W_hy) + self.b_y)
            return [h_t, c_t, y_t]

        X = T.tensor3()  # batch of sequence of vector
        Y = T.tensor3()  # batch of sequence of vector
        # initial hidden states:
        h0 = shared(np.zeros(shape=(batch_size, self.n_lstm), dtype=dtype))
        c0 = shared(np.zeros(shape=(batch_size, self.n_lstm), dtype=dtype))
        self.lr = shared(np.cast[dtype](lr))

        [_, _, y_vals], _ = theano.scan(
                                            fn=step_lstm,
                                            sequences=X.dimshuffle(1, 0, 2),
                                            outputs_info=[h0, c0, None]
                                                  )

        if single_output:
            self.output = y_vals[-1]
        else:
            self.output = y_vals.dimshuffle(1, 0, 2)

        cxe = T.mean(T.nnet.binary_crossentropy(self.output, Y))
        nll = -T.mean(Y * T.log(self.output) + (1. - Y) *
                      T.log(1. - self.output))
        mse = T.mean((self.output - Y) ** 2)

        cost = 0
        if cost_function == 'mse':
            cost = mse
        elif cost_function == 'cxe':
            cost = cxe
        else:
            cost = nll

        gparams = T.grad(cost, self.params)
        updates = OrderedDict()
        for param, gparam in zip(self.params, gparams):
            updates[param] = param - gparam * self.lr

        self.loss = theano.function(inputs=[X, Y], outputs=[cxe, mse, cost])
        self.train = theano.function(inputs=[X, Y], outputs=cost,
                                     updates=updates)
        self.predictions = theano.function(inputs=[X],
                                           outputs=y_vals.dimshuffle(1, 0, 2))
        self.debug = theano.function(inputs=[X, Y],
                                     outputs=[X.shape, Y.shape,
                                              y_vals.shape, cxe.shape])
