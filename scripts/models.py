import tensorflow as tf

slim = tf.contrib.slim

INPUT_KEEP_PROB = 1
OUTPUT_KEEP_PROB = 1
# FC layer
def fc_layer(feed_in, name, number_of_outputs):
    with tf.variable_scope('fc'+name):
        fc = slim.layers.linear(feed_in, number_of_outputs)
    return fc

# GRU layer
# 2 layers, 128 hidden units
def recurrent_model(feed_in, hidden_units=128):
    """Adds the recurrent network on top of the spatial VGG-Face model.
    Args:
       feed_in: A `Tensor` of dimensions [batch_size, seq_length, num_features].
       hidden_units: The number of hidden units of the GRU cell.

    Returns:
       The prediction of the network.
    """
    with tf.variable_scope('GRU-layer'):
        batch_size, seq_length, num_features = feed_in.get_shape().as_list()
        cell = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(hidden_units), input_keep_prob=INPUT_KEEP_PROB, output_keep_prob=OUTPUT_KEEP_PROB),
             tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(hidden_units), input_keep_prob=INPUT_KEEP_PROB, output_keep_prob=OUTPUT_KEEP_PROB)])
        outputs, _ = tf.nn.dynamic_rnn(cell, feed_in, dtype=tf.float32)
        outputs = tf.reshape(outputs, (batch_size * seq_length, hidden_units))
    return outputs

def get_prediction(feed_in, number_of_outputs=2):
    batch_size, seq_length, num_features = feed_in.get_shape().as_list()
    rnn_output = recurrent_model(feed_in)
    prediction = fc_layer(rnn_output, 'gru', number_of_outputs)
    return tf.reshape(prediction, (batch_size, seq_length, number_of_outputs))

# LSTM network:
def lstm_cell(hidden_units):
    lstm = tf.nn.rnn_cell.LSTMCell(hidden_units,
                           use_peepholes=True,
                           # cell_clip=100,
                           state_is_tuple=True)
    lstm = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=INPUT_KEEP_PROB,
                                      output_keep_prob=OUTPUT_KEEP_PROB)
    return lstm

def lstm_model(feed_in, hidden_units=128):
    with tf.variable_scope('lstm-layer'):
        batch_size, seq_length, num_features = feed_in.get_shape().as_list()
        stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(hidden_units) for _ in range(2)], state_is_tuple=True)
        outputs, _ = tf.nn.dynamic_rnn(stacked_lstm, feed_in, dtype=tf.float32)
        outputs = tf.reshape(outputs, (batch_size * seq_length, hidden_units))
    return outputs

def get_prediction_lstm(feed_in, number_of_outputs=2):
    batch_size, seq_length, num_features = feed_in.get_shape().as_list()
    rnn_output = lstm_model(feed_in)
    prediction = fc_layer(rnn_output, 'lstm', number_of_outputs)
    return tf.reshape(prediction, (batch_size, seq_length, number_of_outputs))

# attention network:
# def atten_cell(attn_length, hidden_units):
#     gru = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(hidden_units), input_keep_prob=INPUT_KEEP_PROB,
#                                         output_keep_prob=OUTPUT_KEEP_PROB)
#     gru_atten = tf.contrib.rnn.AttentionCellWrapper(gru, attn_length, state_is_tuple=True)
#     return gru_atten
#
# # tried 3 / 30 for atten_length
# def atten_model(feed_in, attn_length=30, hidden_units=128):
#     with tf.variable_scope('attention-layer'):
#         batch_size, seq_length, num_features = feed_in.get_shape().as_list()
#         stacked_atten = tf.nn.rnn_cell.MultiRNNCell([atten_cell(attn_length, hidden_units) for _ in range(2)], state_is_tuple=True)
#         outputs, _ = tf.nn.dynamic_rnn(stacked_atten, feed_in, dtype=tf.float32)
#         outputs = tf.reshape(outputs, (batch_size * seq_length, hidden_units))
#     return outputs

def atten_model_2(feed_in, attn_length, hidden_units=128):
    with tf.variable_scope('attention-layer'):
        batch_size, seq_length, num_features = feed_in.get_shape().as_list()
        cell = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(hidden_units), input_keep_prob=INPUT_KEEP_PROB,
                                           output_keep_prob=OUTPUT_KEEP_PROB),
             tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(hidden_units), input_keep_prob=INPUT_KEEP_PROB,
                                           output_keep_prob=OUTPUT_KEEP_PROB)]
        )
        gru_atten = tf.contrib.rnn.AttentionCellWrapper(cell, attn_length, state_is_tuple=True)
        outputs, _ = tf.nn.dynamic_rnn(gru_atten, feed_in, dtype=tf.float32)
        outputs = tf.reshape(outputs, (batch_size * seq_length, hidden_units))
    return outputs

def get_prediction_atten(feed_in, attn_length, number_of_outputs=2):
    batch_size, seq_length, num_features = feed_in.get_shape().as_list()
    rnn_output = atten_model_2(feed_in, attn_length=attn_length)
    prediction = fc_layer(rnn_output, 'atten', number_of_outputs)
    return tf.reshape(prediction, (batch_size, seq_length, number_of_outputs))

def get_prediction_atten_hu(feed_in, attn_length, hidden_units, number_of_outputs=2):
    batch_size, seq_length, num_features = feed_in.get_shape().as_list()
    rnn_output = atten_model_2(feed_in, attn_length=attn_length, hidden_units=hidden_units)
    prediction = fc_layer(rnn_output, 'atten', number_of_outputs)
    return tf.reshape(prediction, (batch_size, seq_length, number_of_outputs))


def highway_cell(hidden_units, useHighway = True):
    gru = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(hidden_units), input_keep_prob=INPUT_KEEP_PROB,
                                        output_keep_prob=OUTPUT_KEEP_PROB)
    if useHighway:
        cell = tf.contrib.rnn.HighwayWrapper(gru)
    else:
        cell = tf.contrib.rnn.ResidualWrapper(gru)
    return cell

def highway_model(feed_in, useHighway = True, hidden_units=128):
    with tf.variable_scope('highway-layer'):
        batch_size, seq_length, num_features = feed_in.get_shape().as_list()
        stacked_highway = tf.nn.rnn_cell.MultiRNNCell([highway_cell(hidden_units, useHighway=useHighway) for _ in range(2)],
                                                    state_is_tuple=True)
        outputs, _ = tf.nn.dynamic_rnn(stacked_highway, feed_in, dtype=tf.float32)
        outputs = tf.reshape(outputs, (batch_size * seq_length, hidden_units))
    return outputs

def get_prediction_highway(feed_in, useHighway = True, number_of_outputs=2, hidden_units = 128):
    batch_size, seq_length, num_features = feed_in.get_shape().as_list()
    fc_in = tf.reshape(feed_in, (batch_size * seq_length, num_features))
    highway_in = fc_layer(fc_in, "-highway", hidden_units)
    highway_in = tf.reshape(highway_in, (batch_size, seq_length, hidden_units))
    rnn_output = highway_model(highway_in, useHighway=useHighway)
    prediction = fc_layer(rnn_output, "-pred", number_of_outputs)
    return tf.reshape(prediction, (batch_size, seq_length, number_of_outputs))

def indRNN_model(feed_in, hidden_units = 128):
    from ind_rnn_cell import IndRNNCell
    with tf.variable_scope('indRNN-layer'):
        batch_size, seq_length, num_features = feed_in.get_shape().as_list()
        TIME_STEPS = seq_length
        input_init = tf.random_uniform_initializer(-0.001, 0.001)
        LAST_LAYER_LOWER_BOUND = pow(0.5, 1 / TIME_STEPS)
        # Init only the last layer's recurrent weights around 1
        recurrent_init_lower_0 = 0
        recurrent_init_lower_1 = LAST_LAYER_LOWER_BOUND
        # Regulate each neuron's recurrent weight as recommended in the paper
        RECURRENT_MAX = pow(2, 1 / TIME_STEPS)

        recurrent_init_0 = tf.random_uniform_initializer(recurrent_init_lower_0, RECURRENT_MAX)
        recurrent_init_1 = tf.random_uniform_initializer(recurrent_init_lower_1, RECURRENT_MAX)

        indRnnCells = tf.contrib.rnn.MultiRNNCell([IndRNNCell(hidden_units,
                                                              recurrent_max_abs=RECURRENT_MAX,
                                                              input_kernel_initializer=input_init,
                                                              recurrent_kernel_initializer=recurrent_init_0
                                                              ),
                                                    IndRNNCell(hidden_units,
                                                               recurrent_max_abs=RECURRENT_MAX,
                                                               input_kernel_initializer=input_init,
                                                               recurrent_kernel_initializer=recurrent_init_1
                                                               )])
        outputs, _ = tf.nn.dynamic_rnn(indRnnCells, feed_in, dtype=tf.float32)
        outputs = tf.reshape(outputs, (batch_size * seq_length, hidden_units))
    return outputs

# simple version can not be trained properly
def simple_indRNN_model(feed_in, hidden_units = 128):
    from ind_rnn_cell import IndRNNCell
    with tf.variable_scope('indRNN-layer'):
        batch_size, seq_length, num_features = feed_in.get_shape().as_list()
        TIME_STEPS = seq_length
        # Regulate each neuron's recurrent weight as recommended in the paper
        RECURRENT_MAX = pow(2, 1 / TIME_STEPS)
        indRnnCells = tf.contrib.rnn.MultiRNNCell([IndRNNCell(hidden_units, recurrent_max_abs=RECURRENT_MAX),
                                                    IndRNNCell(hidden_units, recurrent_max_abs=RECURRENT_MAX)])
        outputs, _ = tf.nn.dynamic_rnn(indRnnCells, feed_in, dtype=tf.float32)
        outputs = tf.reshape(outputs, (batch_size * seq_length, hidden_units))
    return outputs

def get_prediction_indRNN(feed_in, number_of_outputs=2):
    batch_size, seq_length, num_features = feed_in.get_shape().as_list()
    rnn_output = indRNN_model(feed_in)
    prediction = fc_layer(rnn_output, "-indRNN", number_of_outputs)
    return tf.reshape(prediction, (batch_size, seq_length, number_of_outputs))