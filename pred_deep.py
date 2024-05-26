import crashes_deep
import lstm_
import bidirectional
import lstm_stateful
import rnn_
import gru_
import kernel_methods


do_specific_forecast = rnn_.RNN_1
do_specific_forecast = rnn_.RNN_2
do_specific_forecast = rnn_.RNN_3
do_specific_forecast = rnn_.RNN_4
do_specific_forecast = rnn_.RNN_5

do_specific_forecast = lstm_.LSTM_1
do_specific_forecast = lstm_.LSTM_2
do_specific_forecast = lstm_.LSTM_3

do_specific_forecast = lstm_stateful.LSTM_batch
do_specific_forecast = lstm_stateful.LSTM_copy_weights_1
do_specific_forecast = lstm_stateful.LSTM_copy_weights_2
do_specific_forecast = lstm_stateful.LSTM_online_1
do_specific_forecast = lstm_stateful.LSTM_online_2
do_specific_forecast = lstm_stateful.LSTM_online_3

do_specific_forecast = bidirectional.LSTM_Bidirectional_1
do_specific_forecast = bidirectional.LSTM_Bidirectional_2
do_specific_forecast = bidirectional.LSTM_Bidirectional_3
do_specific_forecast = bidirectional.LSTM_Bidirectional_4
do_specific_forecast = bidirectional.LSTM_Bidirectional_5
do_specific_forecast = bidirectional.LSTM_Bidirectional_6
do_specific_forecast = bidirectional.LSTM_Bidirectional_7
do_specific_forecast = bidirectional.LSTM_Bidirectional_9
do_specific_forecast = bidirectional.LSTM_Bidirectional_10
do_specific_forecast = bidirectional.LSTM_Bidirectional_11

do_specific_forecast = gru_.GRU_1
do_specific_forecast = gru_.GRU_2
do_specific_forecast = gru_.GRU_3

do_specific_forecast = kernel_methods.kernels
do_specific_forecast = bidirectional.LSTM_Bidirectional_7

crashes_deep.do_predict(do_specific_forecast, cross_validation=False, arcsined=False, full_pic=True, more_time_steps=2)

