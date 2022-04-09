class environment:
    PRICE_COLUMN = -1  #종가의 인덱스

    def __init__(self, chart_data=None):
        self.chart_data = chart_data
        self.observation = None
        self.idx = - 1

    def reset(self):
        self.observation = None
        self.idx = - 1

    def observe(self):
        if len(self.chart_data)-1 >= self.idx:
            self.idx += 1
            self.observation = self.chart_data[self.idx]
            self.observation_train = self.observation[:environment.PRICE_COLUMN] #Price Column 제외하고 train
            return self.observation_train.transpose()
        else:
            return None

    def get_price(self):
        if self.observation is not None:
            return self.observation[environment.PRICE_COLUMN]
        return None

if __name__ == "__main__":
    import DataManager
    import pandas as pd
    path1 = "/Users/mac/Desktop/OHLCV_data/ALL_OHLCV/005930" #삼성전자
    path2 = "/Users/mac/Desktop/OHLCV_data/ALL_OHLCV/000660" #SK하이닉스
    path3 = "/Users/mac/Desktop/OHLCV_data/ALL_OHLCV/035720" #카카오
    path4 = "/Users/mac/Desktop/OHLCV_data/ALL_OHLCV/005380" #현대차
    path5 = "/Users/mac/Desktop/OHLCV_data/ALL_OHLCV/035900" #JYP Ent.
    path6 = "/Users/mac/Desktop/OHLCV_data/ALL_OHLCV/088980" #맥쿼리인프라
    path7 = "/Users/mac/Desktop/OHLCV_data/ALL_OHLCV/003550" #LG

    path_list = [path1, path2, path3, path4, path5, path6, path7]
    train_data, test_data =DataManager.get_data_tensor(path_list,
                                            train_date_start="20090101",
                                            train_date_end="20180101",
                                            test_date_start="20180102",
                                            test_date_end=None)

    env = environment(chart_data=train_data)
    env.reset()
    state = env.observe()
    print(state.shape)
    print(env.get_price())

