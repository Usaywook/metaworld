import random

import pdb
import gym
import metaworld

class ML10Env(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        super(ML10Env, self).__init__()
        self.benchmark = metaworld.ML10()
        self.set_train_env()
        self.is_training = True

    def reset(self):
        if self.is_training:
            self.set_train_env()
        else:
            self.set_eval_env()
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self, mode='human'):
        return self.env.render(mode=mode)

    def set_train_env(self):
        envs = [(idx, name, env_cls) for idx, (name, env_cls) in enumerate(self.benchmark.train_classes.items())]
        self.num_envs = len(envs)
        self.name_idx, name, env_cls = random.choice(envs)

        tasks = [task for task in self.benchmark.train_tasks if task.env_name == name]
        self.num_tasks = len(tasks)
        self.task_idx, task = random.choice(list(zip(range(self.num_tasks), tasks)))

        self.env = env_cls()
        self.env.set_task(task)

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def set_eval_env(self):
        envs = [(idx, name, env_cls) for idx, (name, env_cls) in enumerate(self.benchmark.test_classes.items())]
        self.num_envs = len(envs)
        self.name_idx, name, env_cls = random.choice(envs)

        tasks = [task for task in self.benchmark.test_tasks if task.env_name == name]
        self.num_tasks = len(tasks)
        self.task_idx, task = random.choice(list(zip(range(self.num_tasks), tasks)))

        self.env = env_cls()
        self.env.set_task(task)

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def close(self):
        self.env.close()