# Power forecasting of a PV-plant with Models of Linear Regression and an LSTM Network

Implementation in Python using the librarys sklearn for Linear Models and Pytorch for the LSTM Model

# Run

See venv folder for installed librarys. Code is implemented in Pycharm enviroment and in Python Version 3.8.

# Trainings-Testdata

Download Trainings- and Testdata from the folder Daten

# Simulation

LSTM Model: files LSTMmanytomany.py, LSTMModel.py, Postprocess_STM.py, DataManagement.py are needed for running the LSTM Model
Linear Models forecast Power: files Regression.py, DataManagement.py, Postprocess.py are needed for predicting the Power with the Linear Regression Models
Linear Models forecast Clearness Index: files forecast_kt.py, DataManagement.py, Postprocess_kt.py are needed for predicting the Clearness Index with the Linear Regression Models
