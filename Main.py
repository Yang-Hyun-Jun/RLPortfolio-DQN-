import os
import argparse
import DataManager
import utils

from Learner import DQNLearner

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--stock_code", nargs="+", default= ["010140", "006280", "009830",
                                                           "011170", "010060", "034220",
                                                           "000810"])

  # ["010140", "006280", "009830",
  #  "011170", "010060", "034220",
  #  "000810"]

  #["010140", "013570", "010690",
  # "000910", "010060", "034220",
  # "009540"]

  parser.add_argument("--lr", type=float, default=1e-4)
  parser.add_argument("--tau", type=float, default=0.005)
  parser.add_argument("--discount_factor", type=float, default=0.9)
  parser.add_argument("--num_episode", type=int, default=150)
  parser.add_argument("--balance", type=int, default=14000000)
  parser.add_argument("--batch_size", type=int, default=256)
  parser.add_argument("--memory_size", type=int, default=100000)
  parser.add_argument("--train_start", default="20090101")
  parser.add_argument("--train_end", default=None)
  args = parser.parse_args()

#유틸 저장 및 경로 설정
utils.stock_code = args.stock_code
utils.train_start = args.train_start
utils.train_end = args.train_end
utils.Base_DIR = "/Users/mac/Desktop/OHLCV_data/ALL_OHLCV"
utils.SAVE_DIR = "/Users/mac/Desktop/Save Results2" + "/" + "MLPortfolio"
os.makedirs(utils.SAVE_DIR + "/Metrics", exist_ok=True)
os.makedirs(utils.SAVE_DIR + "/Models", exist_ok=True)

path_list = []
for stock_code in args.stock_code:
    path = utils.Base_DIR + "/" + stock_code
    path_list.append(path)

# 학습/테스트 데이터 준비
train_data, test_data = DataManager.get_data_tensor(path_list,
                                                    train_date_start="20090101",
                                                    train_date_end="20180101",
                                                    test_date_start="20180102",
                                                    test_date_end=None)

# # 최소/최대 투자 가격 설정
min_trading_price = 0
max_trading_price = args.balance / len(args.stock_code)

# 파라미터 설정
params = {"lr":args.lr, "tau":args.tau,
          "chart_data": train_data, "discount_factor":args.discount_factor,
          "min_trading_price": min_trading_price, "max_trading_price": max_trading_price,
          "batch_size":args.batch_size, "memory_size":args.memory_size}

# 학습/테스트 수행
learner = DQNLearner(**params)
learner.run(num_episode=args.num_episode, balance=args.balance)
learner.save_model(path=utils.SAVE_DIR + "/Models" + "/DQNPortfolio.pth")
