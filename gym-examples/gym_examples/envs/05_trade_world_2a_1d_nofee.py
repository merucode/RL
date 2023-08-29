import numpy as np
from typing import Optional, Union
import pygame

import gym
from gym import spaces
from gym.utils.renderer import Renderer
from gym.error import DependencyNotInstalled

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
    | Num  | Action                                                            |
    | ---- | ----------------------------------------------------------------- |
    |  0   | Hold                                                              |
    |  1   | Buy coin and sell coin after 12 step                              |


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
    metadata = {"render_modes": ["human", "rgb_array", 'single_rgb_array'], "render_fps": 30}

    def __init__(self, df, df_render=None, obs_len=288, trade_action=2, render_mode=None):
        self.df = df
        self.lst_ohlcv = self.df_to_lst(self.df)  # convert DataFrame to list
        self.obs_len = obs_len
        self.time_step_limit = len(self.df) - self.obs_len - 12 -1 # -1 is that make as idx
        self.trade_action = trade_action

        # Observations are ohlcv data with obeservation lenth
        self.observation_space = spaces.Box(0, np.Inf, shape=(self.obs_len*5,), dtype=float)
        # Action Space
        self.action_space = spaces.Discrete(self.trade_action)

        # Render
        self.df_render = df_render if df_render is not None else df
        self.lst_ohlcv_render = self.df_to_lst(self.df_render)

        self.window_size_x = 2016                      # The size of the PyGame window
        self.window_size_y = 1500
        self.candle_frame_size_x = self.window_size_x # The size of the candle frame
        self.candle_frame_size_y = 900
        self.volume_frame_size_x = self.window_size_x # The volume of the candle frame
        self.volume_frame_size_y = 500
        self.score_frame_size_x = self.window_size_x  # The score of the candle frame
        self.score_frame_size_y = 100


        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.renderer = Renderer(self.render_mode, self._render)

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
        else:   # Trade action is activate with 1 values 
            # Buy close price. After action value tiem step, sell close price
            # Use lst_ohcv_render to get original price(lst_ohlc is nomalized)
            buy_price = self.lst_ohlcv_render[self.time_step + self.obs_len][3]
            sell_price = self.lst_ohlcv_render[self.time_step + self.obs_len + 12][3]
            # STOCK: trading_fee = -1 * buy_price*(0.015_증권사수수료)*0.01 + sell_price*(0.015_증권사수수료+0.3_세금)*0.01
            # Crypto: trading_fee = -1 * buy_price*(0.05_업비트수수료)*0.01 + sell_price*(0.05_업비트수수료)*0.01
            
            # No Fee
            trading_fee = 0
            #trading_fee = buy_price*(0.05)*0.01 + sell_price*(0.05)*0.01
            #trading_fee = round(trading_fee)

            # profit = -1 * buy_close_price + sell_low_price - trade_fee 
            profit = -1 * buy_price + sell_price - trading_fee
        return profit


    def _get_obs(self):
        obs_lst = self.lst_ohlcv[self.time_step:self.time_step + self.obs_len]
        arr = np.array(obs_lst, dtype=np.float32)
        arr = self.arr_ohlcv_norm_nomalized(arr)
        return arr.flatten()

    def _get_obs_render(self):
        return self.lst_ohlcv_render[self.time_step:self.time_step + self.obs_len]

    def _get_info(self):
        return {
            "balance": self.balance,
            # "time_step": self.time_step, 
            # "time_step_limit": self.time_step_limit,
            # "len_lst": len(self.lst_ohlcv_render),
            # "idx_lst": (self.time_step + self.obs_len - 1),
        }
    
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        self.time_step = 0
        self.balance = 1000

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render()

        self.renderer.reset()
        self.renderer.render_step()

        if not return_info:
            return observation
        else:
            return observation, info


    def step(self, action):
        profit = self._action_to_trade(action)
        self.time_step = self.time_step + 1 + action

        self.balance = self.balance + profit

        observation = self._get_obs()
        reward = profit
        terminated = True if (self.balance <= 0 or self.time_step >= self.time_step_limit) else False
        info = self._get_info()

        if self.render_mode == "human":
            self._render()


        self.renderer.render_step()

        return observation, reward, terminated, False, info


    ########################## RENDER ##########################
    def render(self, mode="human"):
        if self.render_mode is not None:
            pygame.font.init()  # For display score
            return self.renderer.get_renders()
        else:
            pygame.font.init()  # For display score
            return self._render(mode)

    def _render(self, mode="human"):
        assert mode in self.metadata["render_modes"]
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.window is None:
            pygame.init()
            if mode == "human":
                pygame.display.init()
                self.window = pygame.display.set_mode(
                    (self.window_size_x, self.window_size_y)
                )
            else:  # mode in {"rgb_array", "single_rgb_array"}
                self.window = pygame.Surface((self.window_size_x, self.window_size_y))
        if self.clock is None:
            self.clock = pygame.time.Clock()     

        canvas = pygame.Surface((self.window_size_x, self.window_size_y))
        canvas.fill(WHITE)
        
        score_frame = pygame.Surface((self.score_frame_size_x, self.score_frame_size_y))
        score_frame_rect = score_frame.get_rect()
        score_frame_rect.top = 0
        score_frame_rect.left = 0
        score_frame.fill(BLACK)
        score_font = pygame.font.SysFont('arial', 80)
        score_image = score_font.render(f'{self.balance}', True, YELLOW)
        score_frame.blit(score_image, (10, 10))
        canvas.blit(score_frame, (0, 0))
        # Candle Plot
        candle_frame = self._render_candle()
        canvas.blit(candle_frame, (0, 100))
        # Bar Plot
        volume_frame = self._render_volume()
        canvas.blit(volume_frame, (0, 1000))

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
            
        elif mode in {"rgb_array", "single_rgb_array"}:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def _render_candle(self):
        candle_frame = pygame.Surface((self.candle_frame_size_x, self.candle_frame_size_y))
        candle_frame.fill(BLACK)
        candle_width = self.candle_frame_size_x // self.obs_len
        candle_line_space = candle_width // 2 
        candle_line_width = 1

        observation_render = self._get_obs_render()
        scale_ohlc_lst_render = self._get_scale_ohlc_lst_render(observation_render, new_range=self.candle_frame_size_y)
        
        for idx in range(len(scale_ohlc_lst_render)):
            candle_x = idx * candle_width
            candle_y = self.candle_frame_size_y - max(scale_ohlc_lst_render[idx][0], scale_ohlc_lst_render[idx][3])
            candle_height = abs(scale_ohlc_lst_render[idx][0] - scale_ohlc_lst_render[idx][3])

            candle_line_y = self.candle_frame_size_y - scale_ohlc_lst_render[idx][1]
            candle_line_height = scale_ohlc_lst_render[idx][1] - scale_ohlc_lst_render[idx][2]

            if scale_ohlc_lst_render[idx][0] == scale_ohlc_lst_render[idx][3]:
                candle_height = 2
                pygame.draw.rect(candle_frame, RED,
                                 pygame.Rect(candle_x , candle_y, candle_width, candle_height)) # candle
                pygame.draw.rect(candle_frame, RED,
                                 pygame.Rect(candle_x + candle_line_space, candle_line_y, candle_line_width, candle_line_height)) # candle line
            elif scale_ohlc_lst_render[idx][0] < scale_ohlc_lst_render[idx][3]:
                pygame.draw.rect(candle_frame, RED,
                                 pygame.Rect(candle_x , candle_y, candle_width, candle_height))
                pygame.draw.rect(candle_frame, RED,
                                 pygame.Rect(candle_x + candle_line_space, candle_line_y, candle_line_width, candle_line_height)) # candle line
            else:
                pygame.draw.rect(candle_frame, BLUE,
                                 pygame.Rect(candle_x , candle_y, candle_width, candle_height))
                pygame.draw.rect(candle_frame, BLUE,
                                 pygame.Rect(candle_x + candle_line_space, candle_line_y, candle_line_width, candle_line_height)) # candle line

        return candle_frame


    def _render_volume(self):
        volume_frame = pygame.Surface((self.volume_frame_size_x, self.volume_frame_size_y))
        volume_frame.fill(BLACK)

        volume_width = self.volume_frame_size_x // self.obs_len

        observation_render = self._get_obs_render()
        scale_v_lst_render = self._get_scale_v_lst_render(observation_render, new_range=self.volume_frame_size_y)
        
        for idx in range(len(scale_v_lst_render)):
            volume_x = idx * volume_width
            volume_height = scale_v_lst_render[idx][0]
            volume_y = self.volume_frame_size_y - volume_height
            
            pygame.draw.rect(volume_frame, GREEN,
                                 pygame.Rect(volume_x , volume_y, volume_width, volume_height)) # candle
        return volume_frame


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

    def _get_scale_ohlc_lst_render(self, ohlcv_lst, new_range):
        ohlc_lst = []
        for idx in range(len(ohlcv_lst)):
            ohlc_lst.append(ohlcv_lst[idx][0:4])
        ohlc_scale_lst = self._min_max_scale_lst(ohlc_lst, new_range)
        return ohlc_scale_lst

    def _get_scale_v_lst_render(self, ohlcv_lst, new_range):
        v_lst = []
        for idx in range(len(ohlcv_lst)):
            v_lst.append(ohlcv_lst[idx][-1:])
        v_scale_lst = self._min_max_scale_lst(v_lst, new_range, old_min=0)
        return v_scale_lst

    def _min_max_scale_lst(self, lst, new_range, old_min=None, old_max=None):
        old_min = min(map(min, lst)) if old_min is None else old_min # min from 2-dimension list
        old_max = max(map(max, lst)) if old_max is None else old_max # max from 2-dimension list
        new_lst = []
        new_sub_lst = []
        for sub_lst in lst:
            new_sub_lst = []
            for value in sub_lst:
                scaled_value = self._max_min_scale_value(value, old_min, old_max, new_range=new_range)
                new_sub_lst.append(scaled_value)
            new_lst.append(new_sub_lst)

        return new_lst

    def _max_min_scale_value(self,value, old_min, old_max, new_range=1):
        scaled_value = (value - old_min) / (old_max - old_min) * new_range
        scaled_value = round(scaled_value)
        return scaled_value

    def arr_ohlcv_norm_nomalized(self, arr):
        arr_ohlc, arr_v = np.hsplit(arr, [4])
        arr_ohlc_normalized = arr_ohlc/np.linalg.norm(arr_ohlc)
        arr_v_normalized = arr_v/np.linalg.norm(arr_v)
        arr = np.concatenate((arr_ohlc_normalized, arr_v_normalized), axis=1)
        return arr