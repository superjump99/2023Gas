import numpy as np

def sqrt(true):
    result = true/2
    for i in range(30):
        result = (result + (true / result)) / 2
    return result
def MAE(true, pred):
    result = 0
    for i in range(len(true)):
        result += (np.abs(true[i]-pred[i]))
    result = result/len(true)
    return result
def MSE(true, pred):
    result = 0
    for i in range(len(true)):
        result += (true[i] - pred[i]) ** 2
    result = result/len(true)
    return result
def RMSE(true, pred):
    result = 0
    for i in range(len(true)):
        result += (true[i] - pred[i]) ** 2
    result = sqrt(result / len(true))
    return result
def RMSLE(true,pred):
    log_y = np.log1p(true)
    log_pred = np.log1p(pred)
    squared_error = (log_y - log_pred)**2
    result = np.sqrt(np.mean(squared_error))
    return result
def MAPE(true, pred):
    result = 0
    for i in range(len(true)):
        result += abs((true[i] - pred[i]) / true[i])
    result = result * 100 / len(true)
    return result[0]


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
