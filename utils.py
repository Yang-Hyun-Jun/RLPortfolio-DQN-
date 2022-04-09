import numpy as np

stock_code = "005930"
train_start = ""
train_end = ""
test_start = ""
test_end = ""
Base_DIR = "/Users/mac/Desktop/OHLCV_data/ALL_OHLCV"
SAVE_DIR = "/Users/mac/Desktop/Save Results2"

def sigmoid(x):
    x = np.clip(x, -10, 10)
    return 1. / (1. + np.exp(-x))

def exp(x):
    return np.exp(x)



