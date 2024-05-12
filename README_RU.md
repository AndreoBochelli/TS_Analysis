# TS_Analysis
## Анализ временных рядов вчера, сегодня, завтра
### Файлы/модули



- _<font color="blue">crashes_utils.py </font>_   
применение
-  _<font color="blue">crashes_deep.py </font>_    
применение
-  _<font color="blue">deep_pred.py </font>_   
Содержит вызов функции _**crashes_deep.do_predict**_, осуществляющей обучение одной из bleeding-edge:)
моделей: нейронной сети (содержатся в файлах **_rnn\_.py, lstm.py, lstm\_stateful.py, bidirectional.py, gru\_.py_**) или ядерного метода (файл _**kernel\_methods.py**_) . Аргументы подробно описаны в имплементации функции
-  _<font color="blue">crashes_ARIMA.py </font>_    
Применение всех рассмотренных в статье статистических моделей временных рядов: _ARMA, ARIMA, AUTO-ARIMA, SARIMA_. 
Прогнозу в основном и дифференцированном варианте предшествует анализ данных на предмет стационарности. Всё сопровождается графиками.
Предусмотрен вариант с _arcsin_ преобразованием.











