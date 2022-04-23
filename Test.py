import DataManager
import Visualizer
import utils
import torch
import numpy as np
import pandas as pd
from Metrics import Metrics
from ReplayMemory import ReplayMemory
from Environment import environment
from Agent import agent
from Q_network import qnet

if __name__ == "__main__":
    stock_code = ["010140", "006280", "009830",
                  "011170", "010060", "034220",
                  "000810"]

    path_list = []
    for code in stock_code:
        path = utils.Base_DIR + "/" + code
        path_list.append(path)

    #test data load
    train_data, test_data = DataManager.get_data_tensor(path_list,
                                                        train_date_start="20090101",
                                                        train_date_end="20180101",
                                                        test_date_start="20180102",
                                                        test_date_end=None)
    #dimension
    state1_dim = 5
    state2_dim = 8
    K = 7

    #Test Model load
    q_net = qnet(state1_dim=state1_dim, state2_dim=state2_dim)
    qnet_target = qnet(state1_dim=state1_dim, state2_dim=state2_dim)

    balance = 14000000
    min_trading_price = 0
    max_trading_price = balance / K

    #Agent
    environment = environment(chart_data=test_data)
    agent = agent(environment=environment,
                  qnet = q_net,
                  qnet_target=qnet_target,
                  lr = 1e-5, tau = 0.005, discount_factor=0.9,
                  min_trading_price=min_trading_price,
                  max_trading_price=max_trading_price)

    agent.epsilon = 0.0
    #Model parameter load
    model_path = utils.SAVE_DIR + "/DQNPortfolio/Models" + "/DQNPortfolio.pth"
    agent.qnet.load_state_dict(torch.load(model_path))
    agent.qnet_target.load_state_dict(agent.qnet.state_dict())

    #Test
    metrics = Metrics()
    agent.set_balance(balance)
    agent.reset()
    agent.environment.reset()
    agent.epsilon = 0
    state1 = agent.environment.observe()
    state2 = agent.portfolio.reshape(1, -1)
    steps_done = 0

    while True:
        index, actions, confidences = agent.get_action(torch.tensor(state1).float().view(1, 7, -1),
                                                       torch.tensor(state2).float())

        next_state1, next_state2, reward, done = agent.step(actions, confidences)

        steps_done += 1
        state1 = next_state1
        state2 = next_state2

        metrics.portfolio_values.append(agent.portfolio_value)
        metrics.profitlosses.append(agent.profitloss)

        if steps_done % 50 == 0:
            q_value = agent.qnet(torch.tensor(state1).float().view(1, 7, -1),
                                 torch.tensor(state2).float())
            q_value = q_value.detach().numpy()
            action_index = np.argmax(q_value, axis=1)
            action = agent.ACTIONS[action_index]
            p = agent.portfolio
            pv = agent.portfolio_value
            sv = agent.portfolio_value_static
            balance = agent.balance
            change = agent.change
            pi_vector = agent.pi_operator(change)
            epsilon = agent.epsilon
            profitloss = agent.profitloss
            np.set_printoptions(precision=4, suppress=True)
            print("------------------------------------------------------------------------------------------")
            print(f"price:{environment.get_price()}")
            print(f"action:{action}")
            print(f"portfolio:{p}")
            print(f"pi_vector:{pi_vector}")
            print(f"portfolio value:{pv}")
            print(f"static value:{sv}")
            print(f"balance:{balance}")
            print(f"epsilon:{epsilon}")
            print(f"profitloss:{profitloss}")
            print("-------------------------------------------------------------------------------------------")

        if done:
            break

    bench_profitloss = []
    agent.set_balance(14000000)
    agent.reset()
    agent.environment.reset()
    agent.epsilon = 1.0
    state1 = agent.environment.observe()
    state2 = agent.portfolio.reshape(1, -1)

    while True:
        index, actions, confidences = agent.get_action(torch.tensor(state1).float().view(1, 7, -1),
                                                       torch.tensor(state2).float())
        # actions = np.array([0]*7)
        confidences = np.array([1.0]*7)
        next_state1, next_state2, reward, done = agent.step(actions, confidences)

        steps_done += 1
        state1 = next_state1
        state2 = next_state2

        bench_profitloss.append(agent.profitloss)

        if steps_done % 50 == 0:
            q_value = agent.qnet(torch.tensor(state1).float().view(1, 7, -1),
                                 torch.tensor(state2).float())
            q_value = q_value.detach().numpy()
            action_index = np.argmax(q_value, axis=1)
            action = agent.ACTIONS[action_index]
            p = agent.portfolio
            pv = agent.portfolio_value
            sv = agent.portfolio_value_static
            balance = agent.balance
            change = agent.change
            pi_vector = agent.pi_operator(change)
            epsilon = agent.epsilon
            profitloss = agent.profitloss
            np.set_printoptions(precision=4, suppress=True)

        if done:
            break

    Vsave_path2 = utils.SAVE_DIR + "/DQNPortfolio" + "/Metrics" + "/Portfolio Value Curve_test"
    Vsave_path4 = utils.SAVE_DIR + "/DQNPortfolio" + "/Metrics" + "/Profitloss Curve_test"
    Msave_path1 = utils.SAVE_DIR + "/DQNPortfolio" + "/Metrics" + "/Portfolio Value_test"
    Msave_path3 = utils.SAVE_DIR + "/DQNPortfolio" + "/Metrics" + "/Profitloss_test"

    metrics.get_portfolio_values(save_path=Msave_path1)
    metrics.get_profitlosses(save_path=Msave_path3)

    Visualizer.get_portfolio_value_curve(metrics.portfolio_values, save_path=Vsave_path2)
    Visualizer.get_profitloss_curve(metrics.profitlosses, bench_profitloss, save_path=Vsave_path4)

