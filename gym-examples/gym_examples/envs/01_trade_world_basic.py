import gym
from gym import spaces
import pygame
import numpy as np

BLACK = pygame.Color(0, 0, 0)
WHITE = pygame.Color(255, 255, 255)
RED = pygame.Color(255, 0, 0)
GREEN = pygame.Color(0, 255, 0)
BLUE = pygame.Color(0, 0, 255)
YELLOW = pygame.Color(255, 255, 0)

"""
    ### Parameter
    * df : Series trade pd.DataFrame witch consist of ohlcv columns(open, high, low, close, volume)
         : It is Sorted by time ascending order
    * obs_len : What want to observate lenth from df
    * actions : The number of discrete action space 
    * df_render : If you have preprocessing df and want to render with original df, use obs_df with original df

    ### Argument
    ...

    ### Method
    ...

    ### Action Space
    The action space is a `Discrete(actions)`
    | Num | Action                                                            |
    | --- | ----------------------------------------------------------------- |
    | 0   | Hold                                                              |
    | 1   | Buy coin and sell coin after 1 step                               |
    | 2   | Buy coin and sell coin after 2 step                               |
    ...
    | n   | Buy coin and sell coin after n step                               |


    ### Observation Space
    The observation space is a `Box(0, np.inf, (self.obs_len, 4), int)`. Each Num mean 1-dimension number
    | Num | Observation                                                  | Min    | Max    |
    |-----|--------------------------------------------------------------|--------|--------|
    | 0   | open                                                         | 0      | Inf    |
    | 1   | high                                                         | 0      | Inf    |
    | 2   | low                                                          | 0      | Inf    |
    | 3   | close                                                        | 0      | Inf    |
    | 4   | volume                                                       | 0      | Inf    |

    ### Reward
    Profit
"""

class TradeWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, df, df_render=None, render_mode=None, obs_len=30, trade_action=12):
        self.df = df
        self.lst_ohlcv = self.df_to_lst(self.df)  # convert DataFrame to list
        self.obs_len = obs_len
        self.time_step_limit = len(self.df) - self.obs_len - trade_action

        # self.df_render = df_render if df_render is not None else df
        # self.lst_ohlcv_render = self.df_to_lst(self.df_render)

        self.window_size = 512  # The size of the PyGame window

        # Observations are ohlcv data with obeservation lenth
        self.observation_space = spaces.Box(0, np.Inf, shape=(self.obs_len, 5), dtype=float)
        # Action Space
        self.action_space = spaces.Discrete(trade_action)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None


    def _action_to_trade(self, action):
        if action == 0:
            profit = 0
        else:
            # Buy close price. After action value tiem step, sell low price(conservative profit)
            buy_price = self.lst_ohlcv[self.time_step + self.obs_len - 1][3]
            sell_price = self.lst_ohlcv[self.time_step + self.obs_len - 1 + action][2]
            # STOCK: trading_fee = -1 * buy_price*(0.015_증권사수수료)*0.01 + sell_price*(0.015_증권사수수료+0.3_세금)*0.01
            # Crypto: trading_fee = -1 * buy_price*(0.05_업비트수수료)*0.01 + sell_price*(0.05_업비트수수료)*0.01
            trading_fee = buy_price*(0.05)*0.01 + sell_price*(0.05)*0.01
            trading_fee = round(trading_fee)

            # profit = -1 * buy_close_price + sell_low_price - trade_fee 
            profit = -1 * buy_price + sell_price - trading_fee
        return profit


    def _get_obs(self):
        obs_lst = self.lst_ohlcv[self.time_step:self.time_step + self.obs_len]
        return np.array(obs_lst, dtype=np.float32)

    def _get_info(self):
        return {"balance": self.balance}
        
        
    def reset(self, options=None):
        self.time_step = 0
        self.balance = 1000

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info


    def step(self, action):
        profit = self._action_to_trade(action)

        self.balance = self.balance + profit

        observation = self._get_obs()
        reward = profit
        terminated = False
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        self.time_step += 1
        return observation, reward, terminated, False, info


    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    ########################## Util ##########################
    def df_to_lst(self, df):
        lst = []
        for i in range(len(df)):
            lst.append(df.iloc[i].tolist())
        return lst