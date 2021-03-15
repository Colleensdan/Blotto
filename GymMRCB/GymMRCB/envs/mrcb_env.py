import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import os
from Blotto.GameManager import GameManager
from Blotto.Strategies import *


def play_blotto(action):
    # Plays values predicted by model
    A = StaticStrategy('ComputedStrat', action)

    # Create opponent
    B = RandomStrategy('Random', [0.075, 0.075, 0.2, 0.2, 0.075, 0.075, 0.075, 0.075, 0.075,
                                  0.075])  # works well with 0.3 4th index (one indexeD) (-7000 B)

    # Set up game
    gm = GameManager(A, B, num_fields=10, num_runs=100)
    gm.run()
    # gm.plot_results()
    return gm.declare_winner()


class MRCB(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
      """
      Envinronment for Colnel blotto with a twist

      Runs game and compares against psudorandom oppononet #TODO add implementation of strategies of player B from this env

      Scores points if model scores above 5%
      Scores no points if draws (between -5 : 5%)
      Loses points if below threshhold

      Game ends when model hits upper training threshhold or max number of iterations

      There is a time component that penalises the model early on such that it has time
      to explore strategies before jumping to a solution (more likely to find optimal)

      """
      self.blotto_path = os.path.dirname(os.path.dirname(os.path.dirname((os.getcwd()))))
      os.chdir(self.blotto_path)

      self.no_troops = 100
      super(MRCB, self).__init__()

      self.action_space = spaces.Box(low=0, high=100, shape=(10,),
                                       dtype=np.int32)  # implement how 100 troops in summation
      # are required
      self.observation_space = spaces.Discrete(4)

      self.guess = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
      self.observation = 0
      self.current_step = 0
      self.max_steps = 500  # maximum amount of times model can refit vals

      self.draw_cut_off = 5
      self.max_score = 80


      self.seed()
      self.reset()

    def seed(self, seed=None):
      self.np_random, seed = seeding.np_random(seed)
      return [seed]

    def step(self, action):

        if not isinstance(action, list):
            print(action)
            print("Program is not inputting valid data to use in blotto, expected list recieved", type(action))
            quit()

        assert self.action_space.contains(action)
        score = play_blotto(action)

        reward = 0
        self.current_step += 1
        delay_modifier = self.current_step/self.max_steps

        # observations if won, lost, or draw
        if score >= self.draw_cut_off:
            self.observaton = 1
            reward = 1*delay_modifier

        elif score <= -self.draw_cut_off:
            self.observaton = 2
            reward = -1*delay_modifier

        if score > - self.draw_cut_off and score < self.draw_cut_off:
            self.observaton = 3
            reward = 0*delay_modifier

        done = False

        if score >= self.max_score or self.max_steps == self.current_step:
            done = True

        return self.observaton, reward, done, {"guess": action, "score": score}

    def reset(self):
        self.guess = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.observation = 0
        self.score = 0
        return self.observation



def render(self, mode='human', close=False):
    ...


m = MRCB()
m.step([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
