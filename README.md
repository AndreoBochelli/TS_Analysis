# TS_Analysis
## Time Series Analysis: yesterday, today, tomorrow
The code analyses a time series constructed out of  the data from the file _.\Data\Airplane_Crashes_and_Fatalities_Since_1908.csv_ 
by means of the statistical models **_ARMA, ARIMA, SARIMA_**, recurrent neural networks, and kernel methods  _**SVR, KRR**_.
### Files/modules

- _<font color="blue">crashes_utils.py </font>_   
Contains
  - class **Crashes_load**: reading the necessary data from the file, processing improperly entered and missing data,
  - classes **Crashes_prepare_***: aggregation of the number of air crashes over an arbitrary time interval and formation of a corresponding time series,
  - class **Trigo**: trigonometric transformations of the time series,
  - graphic and other tools.
-  _<font color="blue">crashes_deep.py </font>_    
Contains
   - implementation of the **do_predict** function carrying out training and application of one of the bleeding-edge models: neural network (files **_rnn\_.py, lstm.py, lstm\_stateful.py, bidirectional.py, gru\_.py_**) or a kernel method (file _**kernel_methods.py**_),    
   - functions for bringing data to the format required by the neural networks,
   - graphing functions,
   - objects of various activation functions for  the neural networks layers, initial distributions of weights and biases, gradient descent optimizers.
-  _<font color="blue">deep_pred.py </font>_   
Call of function **crashes_deep.do_predict** with one of the models having been chosen.
-  _<font color="blue">crashes_ARIMA.py </font>_    
Application of statistical time series models considered in the article: **_ARMA, ARIMA, SARIMA_**. 
The forecast for the principal and differential versions of the time series is preceded by the stationarity analysis. Everything is accompanied by graphs. There is an option with the _arcsin_ transformation.












