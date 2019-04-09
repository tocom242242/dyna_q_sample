import numpy as np
import copy
import ipdb

class DYNAQAgent():
    """
       qlearning エージェント
    """
    def __init__(self, alpha=0.2, policy=None, gamma=0.99, actions=None, observation=None, alpha_decay_rate=None, epsilon_decay_rate=None, nb_iter_using_model=0):
        self.alpha = alpha
        self.gamma = gamma
        self.policy = policy
        self.reward_history = []
        self.name = "dyna-q"
        self.actions = actions
        self.gamma = gamma
        self.alpha_decay_rate = alpha_decay_rate
        self.epsilon_decay_rate = epsilon_decay_rate
        self.state = str(observation)
        self.ini_state = str(observation)
        self.previous_state = str(observation)
        self.previous_action_id = None
        self.q_values = self._init_q_values()

        # 環境のモデル
        self.env_model = {} # s->a->r, s'
        self.env_model[self.state] = {}
        self.nb_iter_using_model = nb_iter_using_model  # 環境のモデルを用いて学習する回数

    def _init_q_values(self):
        """
           Q テーブルの初期化
        """
        q_values = {}
        q_values[self.state] = np.repeat(0.0, len(self.actions))
        return q_values

    def init_state(self):
        """
            状態の初期化 
        """
        self.previous_state = copy.deepcopy(self.ini_state)
        self.state = copy.deepcopy(self.ini_state)
        return self.state

    def init_policy(self, policy):
        self.policy = policy

    def act(self, q_values=None, step=0):
        action_id = self.policy.select_action(self.q_values[self.state])
        self.previous_action_id = action_id
        action = self.actions[action_id]
        return action

    def observe_state_and_reward(self, next_state, reward):
        """
            次の状態と報酬の観測 
        """
        self.observe(next_state)
        self.get_reward(reward)

    def observe(self, next_state):
        next_state = str(next_state)
        if next_state not in self.q_values: # 始めて訪れる状態であれば
            self.q_values[next_state] = np.repeat(0.0, len(self.actions))

        self.previous_state = copy.deepcopy(self.state)
        self.state = next_state

    def get_reward(self, reward, is_finish=True, step=0):
        """
            報酬の獲得とQ値の更新 
        """
        self.reward_history.append(reward)
        self.update_q_value(reward) # 直接的強化学習
        self.update_model(reward)   # 環境のモデルの学習
        self.update_q_value_using_env_model()    # 環境のモデルを用いた学習(間接的強化学習)

    def update_q_value(self, reward):
        """
            Q値の更新 (直接的強化学習)
        """
        q = self.q_values[self.previous_state][self.previous_action_id] # Q(s, a)
        max_q = max(self.q_values[self.state]) # max Q(s')
        # Q(s, a) = Q(s, a) + alpha*(r+gamma*maxQ(s')-Q(s, a))
        updated_q = q + (self.alpha * (reward + (self.gamma*max_q) - q))
        self.q_values[self.previous_state][self.previous_action_id] = copy.deepcopy(updated_q)
        return updated_q

    def update_q_value_using_env_model(self):
        """
            環境のモデルを用いた学習（間接的強化学習）
        """
        for i in range(self.nb_iter_using_model):
            for s in self.env_model.keys(): # 訪れたすべての状態に対して
                for a in self.env_model[s].keys():  # 状態sで以前とった行動に対して
                    r = self.env_model[s][a]["r"]
                    s2 = self.env_model[s][a]["s"]

                    q = self.q_values[s][a] # Q(s, a)
                    max_q = max(self.q_values[s2]) # max Q(s')
                    self.q_values[s][a] = q + (self.alpha * (r + (self.gamma*max_q) - q))

    def update_model(self, reward):
        """
            環境モデルの学習 
        """
        if self.previous_state not in self.env_model.keys():
            self.env_model[self.previous_state] = {}
        self.env_model[self.previous_state][self.previous_action_id] = {"r":reward, "s":self.state}
