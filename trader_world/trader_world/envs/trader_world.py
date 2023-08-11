import gym
from gym import spaces
import pygame
import numpy as np

"""
    ### Parameter
    * df : Series trade pd.DataFrame witch consist of ohlcv columns(open, high, low, close, volume)
         : It is Sorted by time ascending order
    * obs_len : What want to observate lenth from df

    ### Argument

    ### Method

    ### Action Space

    | Num | Action                                                            |
    | --- | ----------------------------------------------------------------- | 
    | 0   | Hold                                                              | 
    | 1   | Buy coin and sell coin after 1 step                               | 
    | 2   | Buy coin and sell coin after 2 step                               |
    ...

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
"""

class TraderWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, df, obs_len, render_mode=None):
        self.df = df
        self.lst_ohlcv = self.df_to_lst(self.df)
        self.obs_len = obs_len
        self.window_size_x = 900    # The size of the PyGame window
        self.window_size_y = 1500

        # Observations are ohlcv data with obeservation lenth
        self.observation_space = spaces.Box(0, np.Inf, shape=(self.obs_len, 5), dtype=int)

        # We have 10 actions(each action is differe)
        self.action_space = spaces.Discrete(10)
        

        # """
        # The following dictionary maps abstract actions from `self.action_space` to 
        # the direction we will walk in if that action is taken.
        # I.e. 0 corresponds to "right", 1 to "up" etc.
        # """
        # self._action_to_direction = {
        #     0: np.array([1, 0]),
        #     1: np.array([0, 1]),
        #     2: np.array([-1, 0]),
        #     3: np.array([0, -1]),
        # }

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


    def _get_obs(self):
        
        return self.lst_ohlcv[self.time_step:self.time_step + self.obs_len]

    # def _get_info(self):
    #     return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}

        
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.time_step = 0


        # Choose the agent's location uniformly at random
        #self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        # self._target_location = self._agent_location
        # while np.array_equal(self._target_location, self._agent_location):
        #     self._target_location = self.np_random.integers(
        #         0, self.size, size=2, dtype=int
        #     )

        observation = self._get_obs()
        #info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation


    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        # direction = self._action_to_direction[action]
        # # We use `np.clip` to make sure we don't leave the grid
        # self._agent_location = np.clip(
        #     self._agent_location + direction, 0, self.size - 1
        # )
        # An episode is done iff the agent has reached the target
        terminated = 0
        # = np.array_equal(self._agent_location, self._target_location)
        reward = 0
        # 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        # info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        self.time_step += 1
        return observation, reward, terminated, False, {}


    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size_x, self.window_size_y))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size_x, self.window_size_y))
        canvas.fill((255, 255, 255))
        # pix_square_size = (
        #     self.window_size / self.size
        # )  # The size of a single grid square in pixels

        # # First we draw the target
        # pygame.draw.rect(
        #     canvas,
        #     (255, 0, 0),
        #     pygame.Rect(
        #         pix_square_size * self._target_location,
        #         (pix_square_size, pix_square_size),
        #     ),
        # )
        # # Now we draw the agent
        # pygame.draw.circle(
        #     canvas,
        #     (0, 0, 255),
        #     (self._agent_location + 0.5) * pix_square_size,
        #     pix_square_size / 3,
        # )

        # # Finally, add some gridlines
        # for x in range(self.size + 1):
        #     pygame.draw.line(
        #         canvas,
        #         0,
        #         (0, pix_square_size * x),
        #         (self.window_size, pix_square_size * x),
        #         width=3,
        #     )
        #     pygame.draw.line(
        #         canvas,
        #         0,
        #         (pix_square_size * x, 0),
        #         (pix_square_size * x, self.window_size),
        #         width=3,
        #     )

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
    
    def df_to_lst(self, df):
        lst = []
        for i in range(len(df)):
            lst.append(df.iloc[i].tolist())
        return lst