
from tensorflow.keras import backend as K
from tensorflow.keras.layers import RNN
import numpy as np
import tensorflow
class PeepholeLSTMCell(tensorflow.keras.layers.LSTMCell):
  """Equivalent to LSTMCell class but adds peephole connections.
  Peephole connections allow the gates to utilize the previous internal state as
  well as the previous hidden state (which is what LSTMCell is limited to).
  This allows PeepholeLSTMCell to better learn precise timings over LSTMCell.
  From [Gers et al., 2002](
    http://www.jmlr.org/papers/volume3/gers02a/gers02a.pdf):
  "We find that LSTM augmented by 'peephole connections' from its internal
  cells to its multiplicative gates can learn the fine distinction between
  sequences of spikes spaced either 50 or 49 time steps apart without the help
  of any short training exemplars."
  The peephole implementation is based on:
  [Sak et al., 2014](https://research.google.com/pubs/archive/43905.pdf)
  Example:
  ```python
  # Create 2 PeepholeLSTMCells
  peephole_lstm_cells = [PeepholeLSTMCell(size) for size in [128, 256]]
  # Create a layer composed sequentially of the peephole LSTM cells.
  layer = RNN(peephole_lstm_cells)
  input = keras.Input((timesteps, input_dim))
  output = layer(input)
  ```
  """

  def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               unit_forget_bias=True,
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               **kwargs):
    # warnings.warn('`tf.keras.experimental.PeepholeLSTMCell` is deprecated '
    #               'and will be removed in a future version. '
    #               'Please use tensorflow_addons.rnn.PeepholeLSTMCell '
    #               'instead.')
    super(PeepholeLSTMCell, self).__init__(
        units=units,
        activation=activation,
        recurrent_activation=recurrent_activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        bias_initializer=bias_initializer,
        unit_forget_bias=unit_forget_bias,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        bias_regularizer=bias_regularizer,
        kernel_constraint=kernel_constraint,
        recurrent_constraint=recurrent_constraint,
        bias_constraint=bias_constraint,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        implementation=kwargs.pop('implementation', 1),
        **kwargs)

  def build(self, input_shape):
    print(input_shape)
    super(PeepholeLSTMCell, self).build(input_shape)
    # The following are the weight matrices for the peephole connections. These
    # are multiplied with the previous internal state during the computation of
    # carry and output.
    self.input_gate_peephole_weights = self.add_weight(
        shape=(self.units,),
        name='input_gate_peephole_weights',
        initializer=self.kernel_initializer)
    self.forget_gate_peephole_weights = self.add_weight(
        shape=(self.units,),
        name='forget_gate_peephole_weights',
        initializer=self.kernel_initializer)
    self.output_gate_peephole_weights = self.add_weight(
        shape=(self.units,),
        name='output_gate_peephole_weights',
        initializer=self.kernel_initializer)

  def _compute_carry_and_output(self, x, h_tm1, c_tm1):
    x_i, x_f, x_c, x_o = x
    h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1
    i = self.recurrent_activation(
        x_i + K.dot(h_tm1_i, self.recurrent_kernel[:, :self.units]) +
        self.input_gate_peephole_weights * c_tm1)
    f = self.recurrent_activation(x_f + K.dot(
        h_tm1_f, self.recurrent_kernel[:, self.units:self.units * 2]) +
                                  self.forget_gate_peephole_weights * c_tm1)
    c = f * c_tm1 + i * self.activation(x_c + K.dot(
        h_tm1_c, self.recurrent_kernel[:, self.units * 2:self.units * 3]))
    o = self.recurrent_activation(
        x_o + K.dot(h_tm1_o, self.recurrent_kernel[:, self.units * 3:]) +
        self.output_gate_peephole_weights * c)
    return c, o

  def _compute_carry_and_output_fused(self, z, c_tm1):
    z0, z1, z2, z3 = z
    i = self.recurrent_activation(z0 +
                                  self.input_gate_peephole_weights * c_tm1)
    f = self.recurrent_activation(z1 +
                                  self.forget_gate_peephole_weights * c_tm1)
    c = f * c_tm1 + i * self.activation(z2)
    o = self.recurrent_activation(z3 + self.output_gate_peephole_weights * c)
    return c, o


    def _compute_carry_and_output_fused(self, z, c_tm1):
        z0, z1, z2, z3 = z
        i = self.recurrent_activation(z0 + self.input_gate_peephole_weights * c_tm1)
        f = self.recurrent_activation(z1 + self.forget_gate_peephole_weights * c_tm1)
        c = f * c_tm1 + i * self.activation(z2)
        o = self.recurrent_activation(z3 + self.output_gate_peephole_weights * c)
        return c, o
class MinimalRNNCell(tensorflow.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        #self.state_size = [tensorflow.TensorShape([units])]
        super(MinimalRNNCell, self).__init__(**kwargs)
    def build(self, input_shape):

        self.kernel = self.add_weight(shape=(input_shape, self.units),
                                      initializer='uniform',
                                      name='kernel')
        print(input_shape.shape)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.built = True
    def call(self, inputs, states):
        print('call')
        prev_output = states[0]
        h = K.dot(inputs[0], self.kernel)
        output = h + K.dot(prev_output, self.recurrent_kernel)
        return output, [output]

# Let's use this cell in a RNN layer:
def create_emb():
    emb = 0.2 * np.random.uniform(-1.0, 1.0,
                                     (26, 50)).astype('Float32')

    return emb


def build_model():
    emb=create_emb()
    # cell = MinimalRNNCell(32)
    cell = PeepholeLSTMCell(50)
    word_inputs = tensorflow.keras.Input(shape=(None,2))
    pos_inputs=tensorflow.keras.Input(shape=(None,30))
    embedding = tensorflow.keras.layers.Embedding(26, 50, input_length=(None,2), weights=[emb],trainable=True)(word_inputs) # line A
    embedding = tensorflow.keras.layers.Reshape((32,100))(embedding)
    word_pos_inputs = tensorflow.keras.layers.concatenate([embedding, pos_inputs], axis=2)
    lstm_layer = RNN(cell)
    print(word_pos_inputs.shape)

    lstm = lstm_layer(word_pos_inputs)
    output = tensorflow.keras.layers.Dense(22,activation='softmax')(lstm)
    # output=tensorflow.keras.layers.Dense(22, activation='softmax')(dense)
    model = tensorflow.keras.Model(inputs=[word_inputs, pos_inputs], outputs=output)
    print(len(model.get_layer('rnn').get_weights()))
    for item in model.get_layer('rnn').get_weights():
        print(len(item))
    model.compile(loss='categorical_crossentropy', optimizer=tensorflow.keras.optimizers.SGD(.0001), metrics=['accuracy'])
    # model.summary()
build_model()