

stacked_rnn = tf.contrib.rnn.MultiRNNCell([list_of_rnn_units])
outputs, _ = tf.nn.dynamic_rnn(stacked_rnn, feed_in, dtype=tf.float32)
outputs = tf.reshape(outputs, (batch_size * seq_length, hidden_units))



GRU:

stacked_gru = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(hidden_units), tf.contrib.rnn.GRUCell(hidden_units)]


LSTM:
def lstm_cell():
    lstm = tf.nn.rnn_cell.LSTMCell(hidden_units, use_peepholes=True, state_is_tuple=True)
    return lstm
stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(hidden_units) for _ in range(2)])



Attention:
attentions = tf.contrib.rnn.AttentionCellWrapper(stacked_gru, attn_length, state_is_tuple=True)


IndRNN:
TIME_STEPS = seq_length
# Regulate each neuron's recurrent weight as recommended in the paper
RECURRENT_MAX = pow(2, 1 / TIME_STEPS)
stacked_indRnnCells = tf.contrib.rnn.MultiRNNCell([IndRNNCell(hidden_units, recurrent_max_abs=RECURRENT_MAX),
                                                    IndRNNCell(hidden_units, recurrent_max_abs=RECURRENT_MAX)])


Highway:
def highway_cell(hidden_units):
    gru = tf.contrib.rnn.GRUCell(hidden_units)
    cell = tf.contrib.rnn.HighwayWrapper(gru)
    return cell
stacked_highway = tf.nn.rnn_cell.MultiRNNCell([highway_cell(hidden_units) for _ in range(2)])



Residual:
def residual_cell(hidden_units):
    gru = tf.contrib.rnn.GRUCell(hidden_units)
    cell = tf.contrib.rnn.ResidualWrapper(gru)
    return cell
stacked_highway = tf.nn.rnn_cell.MultiRNNCell([highway_cell(hidden_units) for _ in range(2)])
