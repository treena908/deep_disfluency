import os
import sys
from tensorflow.keras import backend as K
from tensorflow.keras.layers import RNN
import numpy as np
import tensorflow
THIS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(THIS_DIR + "/../../")
from deep_disfluency.utils.tools import \
    dialogue_data_and_indices_from_matrix
import os
THIS_DIR = os.path.dirname(os.path.realpath(__file__))
# class PeepholeLSKMCell(tensorflow.keras.layers.LSTMCell):
#   """Equivalent to LSKMCell class but adds peephole connections.
#   Peephole connections allow the gates to utilize the previous internal state as
#   well as the previous hidden state (which is what LSKMCell is limited to).
#   Khis allows PeepholeLSKMCell to better learn precise timings over LSKMCell.
#   From [Gers et al., 2002](
#     http://www.jmlr.org/papers/volume3/gers02a/gers02a.pdf):
#   "We find that LSKM augmented by 'peephole connections' from its internal
#   cells to its multiplicative gates can learn the fine distinction between
#   sequences of spikes spaced either 50 or 49 time steps apart without the help
#   of any short training exemplars."
#   Khe peephole implementation is based on:
#   [Sak et al., 2014](https://research.google.com/pubs/archive/43905.pdf)
#   Example:
#   ```python
#   # Create 2 PeepholeLSKMCells
#   peephole_lstm_cells = [PeepholeLSKMCell(size) for size in [128, 256]]
#   # Create a layer composed sequentially of the peephole LSKM cells.
#   layer = RNN(peephole_lstm_cells)
#   input = keras.Input((timesteps, input_dim))
#   output = layer(input)
#   ```
#   """
#
#   def __init__(self,
#                units,
#                sequence_len,emb,folder,ne, de, na, n_lstm, n_out, cs, npos, lr=0.05,
#                activation='tanh',
#                recurrent_activation='hard_sigmoid',
#                use_bias=Krue,
#                kernel_initializer='glorot_uniform',
#                recurrent_initializer='orthogonal',
#                bias_initializer='zeros',
#                unit_forget_bias=Krue,
#                kernel_regularizer=None,
#                recurrent_regularizer=None,
#                bias_regularizer=None,
#                kernel_constraint=None,
#                recurrent_constraint=None,
#                bias_constraint=None,
#                dropout=0.,
#                recurrent_dropout=0.,
#                **kwargs):
#     # warnings.warn('`tf.keras.experimental.PeepholeLSKMCell` is deprecated '
#     #               'and will be removed in a future version. '
#     #               'Please use tensorflow_addons.rnn.PeepholeLSKMCell '
#     #               'instead.')
#     super(PeepholeLSKMCell, self).__init__(
#         units=units,
#         activation=activation,
#         recurrent_activation=recurrent_activation,
#         use_bias=use_bias,
#         kernel_initializer=kernel_initializer,
#         recurrent_initializer=recurrent_initializer,
#         bias_initializer=bias_initializer,
#         unit_forget_bias=unit_forget_bias,
#         kernel_regularizer=kernel_regularizer,
#         recurrent_regularizer=recurrent_regularizer,
#         bias_regularizer=bias_regularizer,
#         kernel_constraint=kernel_constraint,
#         recurrent_constraint=recurrent_constraint,
#         bias_constraint=bias_constraint,
#         dropout=dropout,
#         recurrent_dropout=recurrent_dropout,
#         implementation=kwargs.pop('implementation', 1),
#         **kwargs)
#     self.sequence_len=sequence_len
#     self.emb = emb
#     self.vocab=ne+1
#     self.word_in = (de * cs)
#     self.window=cs
#     self.emb_dim=de
#     self.pos_in=(125 * cs)
#     self.n_lstm = n_lstm
#     self.n_out = n_out
#     self.lr=lr
#         # self.units = units
#         # self.state_size = units
#         # super(PeepholeLSKMCell, self).__init__(units=self.units,**kwargs)
#
#   def build(self, input_shape):
#     print(input_shape)
#
#     super(PeepholeLSKMCell, self).build(input_shape)
#     # Khe following are the weight matrices for the peephole connections. Khese
#     # are multiplied with the previous internal state during the computation of
#     # carry and output.
#
#     self.W_xi = self.add_weight(shape=(self.n_in, self.n_lstm),name='W_xi',initializer=self.kernel_regularizer)
#
#     self.W_hi = self.add_weight(shape=(self.n_lstm, self.n_lstm), name='W_hi', initializer=self.recurrent_regularizer)
#     self.W_ci = self.add_weight(shape=(self.n_lstm, self.n_lstm), name='W_ci', initializer=self.recurrent_regularizer)
#     # bias to the input:
#     self.b_i =self.add_weight(shape=(self.n_lstm,),name='b_i',initializer=tensorflow.keras.initializers.RandomUniform(
#     minval=-0.5, maxval=0.5, seed=None))
#     # forget gate weights:
#     self.W_xf = self.add_weight(shape=(self.n_in, self.n_lstm),name='W_xf',initializer=self.kernel_regularizer)
#     self.W_hf = self.add_weight(shape=(self.n_lstm, self.n_lstm), name='W_hf', initializer=self.recurrent_regularizer)
#     self.W_cf = self.add_weight(shape=(self.n_lstm, self.n_lstm), name='W_cf', initializer=self.recurrent_regularizer)
#     # bias
#     self.b_f = self.add_weight(shape=(self.n_lstm,),name='b_f',initializer=tensorflow.keras.initializers.RandomUniform(
#     minval=-0.5, maxval=0.5, seed=None))
#     # memory cell gate weights:
#     self.W_xc = self.add_weight(shape=(self.n_in, self.n_lstm),name='W_xc',initializer=self.kernel_regularizer)
#     self.W_hc = self.add_weight(shape=(self.n_lstm, self.n_lstm), name='W_hc', initializer=self.recurrent_regularizer)
#     # bias to the memory cell:
#     self.b_c = self.add_weight(shape=(self.n_lstm,),name='b_c',initializer=tensorflow.keras.initializers.RandomUniform(
#     minval=-0.5, maxval=0.5, seed=None))
#     # output gate weights:
#     self.W_xo = self.add_weight(shape=(self.n_in, self.n_lstm),name='W_xo',initializer=self.kernel_regularizer)
#     self.W_ho = self.add_weight(shape=(self.n_lstm, self.n_lstm), name='W_ho', initializer=self.recurrent_regularizer)
#     self.W_co = self.add_weight(shape=(self.n_lstm, self.n_lstm), name='W_co', initializer=self.recurrent_regularizer)
#     # bias on output gate:
#     self.b_o = self.add_weight(shape=(self.n_lstm,),name='b_o',initializer=tensorflow.keras.initializers.RandomUniform(
#     minval=-0.5, maxval=0.5, seed=None))
#     # hidden to y matrix weights:
#     # self.W_hy = init_weight((self.n_lstm, self.n_out), 'W_hy')
#     # self.b_y = shared(np.zeros(n_out, dtype=dtype))  # output bias
#     #
#
#   def call(self, x, h_tm1):
#     print('call')
#     x_i, x_f, x_c, x_o = x
#     h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1
#     i = self.recurrent_activation(
#         x_i + K.dot(h_tm1_i, self.recurrent_kernel[:, :self.units]) +
#         self.input_gate_peephole_weights * h_tm1)
#     f = self.recurrent_activation(x_f + K.dot(
#         h_tm1_f, self.recurrent_kernel[:, self.units:self.units * 2]) +
#                                   self.forget_gate_peephole_weights * h_tm1)
#     c = f * h_tm1 + i * self.activation(x_c + K.dot(
#         h_tm1_c, self.recurrent_kernel[:, self.units * 2:self.units * 3]))
#     o = self.recurrent_activation(
#         x_o + K.dot(h_tm1_o, self.recurrent_kernel[:, self.units * 3:]) +
#         self.output_gate_peephole_weights * c)
#     return c, o
#
#   def _compute_carry_and_output_fused(self, z, c_tm1):
#     z0, z1, z2, z3 = z
#     i = self.recurrent_activation(z0 +
#                                   self.input_gate_peephole_weights * c_tm1)
#     f = self.recurrent_activation(z1 +
#                                   self.forget_gate_peephole_weights * c_tm1)
#     c = f * c_tm1 + i * self.activation(z2)
#     o = self.recurrent_activation(z3 + self.output_gate_peephole_weights * c)
#     return c, o


class myLSTMCell(tensorflow.keras.layers.Layer):
    def __init__(self, units,sequence_len,emb,ne, de, n_lstm, n_out, cs, npos, lr=0.05, activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer=tensorflow.keras.initializers.RandomUniform(
                                       minval=-0.5, maxval=0.5, seed=None), **kwargs):
        self.units = units
        self.sequence_len = sequence_len
        self.emb = emb
        self.vocab = ne + 1
        self.word_in = (de * cs)
        self.window = cs
        self.emb_dim = de
        self.pos_in = (npos * cs)
        self.n_lstm = n_lstm
        self.n_out = n_out
        self.lr = lr
        self.n_in=self.word_in+self.pos_in
        self.state_size = (self.units, self.units)
        self.kernel_initializer = kernel_initializer
        self.use_bias = use_bias
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer
        super(myLSTMCell, self).__init__(**kwargs)



    def build(self, input_shape):
        input_dim = input_shape[-1]
        # print(input_shape)
        # input_dim=130
        self.W_xi = self.add_weight(shape=(self.n_in, self.n_lstm), name='W_xi', initializer=self.kernel_initializer)

        self.W_hi = self.add_weight(shape=(self.n_lstm, self.n_lstm), name='W_hi',
                                    initializer=self.recurrent_initializer)
        self.W_ci = self.add_weight(shape=(self.n_lstm, self.n_lstm), name='W_ci',
                                    initializer=self.recurrent_initializer)
        # bias to the input:
        self.b_i = self.add_weight(shape=(self.n_lstm,), name='b_i',
                                   initializer=self.bias_initializer)
        # forget gate weights:
        self.W_xf = self.add_weight(shape=(self.n_in, self.n_lstm), name='W_xf', initializer=self.kernel_initializer)
        self.W_hf = self.add_weight(shape=(self.n_lstm, self.n_lstm), name='W_hf',
                                    initializer=self.recurrent_initializer)
        self.W_cf = self.add_weight(shape=(self.n_lstm, self.n_lstm), name='W_cf',
                                    initializer=self.recurrent_initializer)
        # bias
        self.b_f = self.add_weight(shape=(self.n_lstm,), name='b_f',
                                   initializer=self.bias_initializer)
        # memory cell gate weights:
        self.W_xc = self.add_weight(shape=(self.n_in, self.n_lstm), name='W_xc', initializer=self.kernel_initializer)
        self.W_hc = self.add_weight(shape=(self.n_lstm, self.n_lstm), name='W_hc',
                                    initializer=self.recurrent_initializer)
        # bias to the memory cell:
        self.b_c = self.add_weight(shape=(self.n_lstm,), name='b_c',
                                   initializer=self.bias_initializer)
        # output gate weights:
        self.W_xo = self.add_weight(shape=(self.n_in, self.n_lstm), name='W_xo', initializer=self.kernel_initializer)
        self.W_ho = self.add_weight(shape=(self.n_lstm, self.n_lstm), name='W_ho',
                                    initializer=self.recurrent_initializer)
        self.W_co = self.add_weight(shape=(self.n_lstm, self.n_lstm), name='W_co',
                                    initializer=self.recurrent_initializer)
        # bias on output gate:
        self.b_o = self.add_weight(shape=(self.n_lstm,), name='b_o',
                                   initializer=self.bias_initializer)
        self.params = [self.W_xi, self.W_hi, self.W_ci, self.b_i,
                       self.W_xf, self.W_hf, self.W_cf, self.b_f,
                       self.W_xc, self.W_hc, self.b_c,
                       self.W_ho, self.W_co, self.W_co, self.b_o
                       ]

        self.names = ["W_xi", "W_hi", "W_ci", "b_i",
                      "W_xf", "W_hf", "W_cf", "b_f",
                      "W_xc", "W_hc", "b_c",
                      "W_xo", "W_ho", "W_co", "b_o"
                      ]




    def load_retrain_weights(self, folder):
        print('shape of weight')
        lstm_input=[]
        lstm_hidden = []
        lstm_bias = []
        output_weight=[]
        for name in self.names:

            # if name in ["W_xi",
            #           "W_xf",
            #           "W_xc"]:
            #
            #     lstm_input.append(np.load(os.path.join(folder, name + ".npy")))
            # elif name in [ "W_hi", "W_ci",
            #           "W_hf", "W_cf",
            #            "W_hc",
            #           "W_ho", "W_co" ]:
            #
            #     lstm_hidden.append(np.load(os.path.join(folder, name + ".npy")))
            # elif name in ["b_i",
            #            "b_f",
            #           "b_c",
            #            "b_o"
            #            ]:
            #
            #     lstm_bias.append(np.load(os.path.join(folder, name + ".npy")))
            # elif name in ["W_hy", "b_y"]:
            #     output_weight.append(np.load(os.path.join(folder, name + ".npy")))
            # elif name in ["embeddings"]:
            #     print("embeddings")
            #     np.load(os.path.join(folder, name + ".npy")).shape
            if name in ["W_xo"]:
                # print(self.W_xo.numpy())
                lstm_input.append(self.W_xo.numpy())

            else:


                lstm_input.append(np.load(os.path.join(folder, name + ".npy")))




        # return  np.array(lstm_input),np.array(lstm_hidden),np.array(lstm_bias),np.array(output_weight)
        return np.array(lstm_input)
    def call(self, x_t, states, training=True):
        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state
        print(h_tm1.shape)
        print(c_tm1.shape)
        print(self.W_ci.shape)

        i_t = K.sigmoid(K.dot(x_t, self.W_xi) +
                             K.dot(h_tm1, self.W_hi) +
                             K.dot(c_tm1, self.W_ci) + self.b_i)
        f_t = K.sigmoid(K.dot(x_t, self.W_xf) +
                             K.dot(h_tm1, self.W_hf) +
                             K.dot(c_tm1, self.W_cf) + self.b_f)
        c_t = f_t * c_tm1 + i_t * K.tanh(K.dot(x_t, self.W_xc) +
                                         K.dot(h_tm1, self.W_hc) +
                                         self.b_c)
        o_t = K.sigmoid(K.dot(x_t, self.W_xo) +
                             K.dot(h_tm1, self.W_ho) +
                             K.dot(c_t, self.W_co) + self.b_o)
        h_t = o_t * K.tanh(c_t)


        return h_t, [h_t, c_t]


# Let's use this cell in a RNN layer:
def create_emb():
    emb = 0.2 * np.random.uniform(-1.0, 1.0,
                                     (26, 50)).astype('Float32')

    return emb


def build_model():
    emb=create_emb()
    names=[]
    trained_model = '%03d' % 41
    folder='/AD-HOME/sfarza3/Research/deep_disfluency/deep_disfluency/experiments'+'/{0}/epoch_{1}'.format(trained_model, str(16))
    cell=myLSTMCell(50,32,emb,26, 50,  50, 22, 2, 125)
    # cell = MinimalRNNCell(32)
    # cell = PeepholeLSKMCell(50)
    word_inputs = tensorflow.keras.Input(shape=(None,2))
    pos_inputs=tensorflow.keras.Input(shape=(None,2))
    embedding = tensorflow.keras.layers.Embedding(26, 50, input_length=(None,2), weights=[emb],trainable=True)(word_inputs) # line A
    embedding = tensorflow.keras.layers.Reshape((32,100))(embedding)
    pos_inputs_window = tensorflow.keras.layers.Reshape((32, 250))(pos_inputs)
    word_pos_inputs = tensorflow.keras.layers.concatenate([embedding, pos_inputs_window], axis=2)
    lstm_layer = RNN(cell)
    # print(word_pos_inputs.shape)

    lstm = lstm_layer(word_pos_inputs)
    output = tensorflow.keras.layers.Dense(22,activation='softmax')(lstm)
    # output=tensorflow.keras.layers.Dense(22, activation='softmax')(dense)
    model = tensorflow.keras.Model(inputs=[word_inputs, pos_inputs], outputs=output)
    # lstm_input, lstm_hidden, lstm_bias, output_weight = cell.load_retrain_weights(folder)
    lstm_input = cell.load_retrain_weights(folder)

    model.get_layer('rnn').set_weights(lstm_input)

    # print(len(model.get_layer('rnn').get_weights()))
    # for  w in model.get_layer('rnn').weights:
    #     names.append(w.name)
    # for nm,elem in zip(names,model.get_layer('rnn').get_weights()):
    #     if 'W_xi' in nm:
    #         print('after')
    #         print(elem)


    model.compile(loss='categorical_crossentropy', optimizer=tensorflow.keras.optimizers.SGD(.0001), metrics=['accuracy'])
    # model.summary()

def fit(word_idx, labels, lr, indices, pos_idx=None,
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

        print('training data')
        print(word_idx[start:stop+1, :].shape)
        print(pos_idx[start:stop+1, :].shape)
        print(labels[stop].shape)




def test_train_data():
    train_dialogues_filepath=THIS_DIR+'/../data/disfluency_detection/feature_matrices/test'
    # dialogue_f='DBDePauldepaul1a.chaPAR.npy'
    dialogue_f='DBPittControlcookie0733.chaPAR.npy'

    d_matrix = np.load(train_dialogues_filepath + "/" +
                       dialogue_f)



    word_idx, pos_idx, extra, y, indices = \
        dialogue_data_and_indices_from_matrix(
            d_matrix,
            n_extra=None,
            window_size=2,
            bs=9,
            pre_seg=True
        )
    fit(word_idx,
              y,
              0.0001,
              indices,
              pos_idx=pos_idx,
              extra_features=extra)






# build_model()
test_train_data()