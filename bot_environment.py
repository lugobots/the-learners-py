import numpy as np
from lugo4py.rl.training_controller import TrainingController
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts


class GameEnvironment(PyEnvironment):

    def __init__(self, training_ctrl: TrainingController):
        self.num_actions = 8
        self.num_sensors = 8

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=self.num_actions - 1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self.num_sensors,), dtype=np.float32, minimum=np.zeros(self.num_sensors),
            maximum=np.ones(self.num_sensors), name='observation')
        self.training_ctrl = training_ctrl
        self._reset()

    def action_spec(self):
        print(f"action spec - called")
        return self._action_spec

    def __set_state(self, new_state):
        self._state = np.array(new_state, dtype=np.float32)

    def observation_spec(self):
        print(f"observation_spec - called")
        return self._observation_spec

    def _reset(self):
        self.total_reward = 0
        new_state = self.training_ctrl.set_environment(None)
        self.__set_state(new_state)
        self._episode_ended = False
        return ts.restart(self._state)

    def _step(self, action):
        evaluation = self.training_ctrl.update(action)
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()
        # Make sure episodes don't go on forever.
        if evaluation["done"]:
            self._episode_ended = True

        new_state = self.training_ctrl.get_state()
        self.__set_state(new_state)
        self.total_reward += evaluation["reward"]
        if self._episode_ended:
            return ts.termination(self._state, evaluation["reward"])
        else:
            return ts.transition(self._state, reward=evaluation["reward"], discount=1.0)
