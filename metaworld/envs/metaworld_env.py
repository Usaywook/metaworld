import random

import gym
import metaworld

class ML10Env(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, seed=None):
        super(ML10Env, self).__init__()
        self.benchmark = metaworld.ML10(seed=seed)

        self.is_training = True
        self.goal_observable = True
        self._scenario_idx = None

        self.set_train_env()
        if self.goal_observable:
            self.env._partially_observable = False

    def reset(self):
        self.env.close()
        if self.is_training:
            self.set_train_env()
        else:
            self.set_eval_env()
        if self.goal_observable:
            self.env._partially_observable = False

        return self.env.reset()

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        if info['success'] == True:
            done = True
        return next_state, reward, done, info

    def render(self, mode='human'):
        return self.env.render(mode=mode)

    def set_train_env(self):
        envs = [(idx, name, env_cls) for idx, (name, env_cls) in enumerate(self.benchmark.train_classes.items())]
        self.num_envs = len(envs)
        if self._scenario_idx is not None:
            self.name_idx, self.name, env_cls = envs[self._scenario_idx]
        else:
            self.name_idx, self.name, env_cls = random.choice(envs)

        tasks = [task for task in self.benchmark.train_tasks if task.env_name == self.name]
        self.num_tasks = len(tasks)
        self.task_idx, task = random.choice(list(zip(range(self.num_tasks), tasks)))

        self.env = env_cls()
        self.env.set_task(task)

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def set_eval_env(self):
        envs = [(idx, name, env_cls) for idx, (name, env_cls) in enumerate(self.benchmark.test_classes.items())]
        self.num_envs = len(envs)
        self.name_idx, self.name, env_cls = random.choice(envs)

        tasks = [task for task in self.benchmark.test_tasks if task.env_name == self.name]
        self.num_tasks = len(tasks)
        self.task_idx, task = random.choice(list(zip(range(self.num_tasks), tasks)))

        self.env = env_cls()
        self.env.set_task(task)

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def close(self):
        self.env.close()

    def eval(self):
        self.is_training = False
        super().eval()

    def train(self):
        self.is_training = True
        super().train()

    def set_scenario(self, idx):
        self._scenario_idx = idx

class ML5Env(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, seed=None):
        super(ML5Env, self).__init__()
        self.benchmark = metaworld.ML5(seed)

        self.is_training = True
        self.goal_observable = True
        self._scenario_idx = None

        self.set_train_env()
        if self.goal_observable:
            self.env._partially_observable = False

    def reset(self):
        self.env.close()
        if self.is_training:
            self.set_train_env()
        else:
            self.set_eval_env()
        if self.goal_observable:
            self.env._partially_observable = False

        return self.env.reset()

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        if info['success'] == True:
            done = True
        return next_state, reward, done, info

    def render(self, mode='human'):
        return self.env.render(mode=mode)

    def set_train_env(self):
        envs = [(idx, name, env_cls) for idx, (name, env_cls) in enumerate(self.benchmark.train_classes.items())]
        self.num_envs = len(envs)
        if self._scenario_idx is not None:
            self.name_idx, self.name, env_cls = envs[self._scenario_idx]
        else:
            self.name_idx, self.name, env_cls = random.choice(envs)

        tasks = [task for task in self.benchmark.train_tasks if task.env_name == self.name]
        self.num_tasks = len(tasks)
        self.task_idx, task = random.choice(list(zip(range(self.num_tasks), tasks)))

        self.env = env_cls()
        self.env.set_task(task)

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def set_eval_env(self):
        envs = [(idx, name, env_cls) for idx, (name, env_cls) in enumerate(self.benchmark.test_classes.items())]
        self.num_envs = len(envs)

        if self._scenario_idx is not None:
            self.name_idx, self.name, env_cls = envs[self._scenario_idx]
        else:
            self.name_idx, self.name, env_cls = random.choice(envs)

        tasks = [task for task in self.benchmark.test_tasks if task.env_name == self.name]
        self.num_tasks = len(tasks)
        self.task_idx, task = random.choice(list(zip(range(self.num_tasks), tasks)))

        self.env = env_cls()
        self.env.set_task(task)

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def close(self):
        self.env.close()

    def eval(self):
        self.is_training = False
        super().eval()

    def train(self):
        self.is_training = True
        super().train()

    def set_scenario(self, idx):
        self._scenario_idx = idx

class MT10Env(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, seed=None):
        super(MT10Env, self).__init__()
        self.benchmark = metaworld.MT10(seed=seed)

        self.goal_observable = True
        self._scenario_idx = None

        self.set_env()
        if self.goal_observable:
            self.env._partially_observable = False

    def reset(self):
        self.env.close()
        self.set_env()
        if self.goal_observable:
            self.env._partially_observable = False

        return self.env.reset()

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        if info['success'] == True:
            done = True
        return next_state, reward, done, info

    def render(self, mode='human'):
        return self.env.render(mode=mode)

    def set_env(self):
        envs = [(idx, name, env_cls) for idx, (name, env_cls) in enumerate(self.benchmark.train_classes.items())]
        self.num_envs = len(envs)

        if self._scenario_idx is not None:
            self.name_idx, self.name, env_cls = envs[self._scenario_idx]
        else:
            self.name_idx, self.name, env_cls = random.choice(envs)

        tasks = [task for task in self.benchmark.train_tasks if task.env_name == self.name]
        self.num_tasks = len(tasks)
        self.task_idx, task = random.choice(list(zip(range(self.num_tasks), tasks)))

        self.env = env_cls()
        self.env.set_task(task)

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def close(self):
        self.env.close()

    def eval(self):
        super().eval()

    def train(self):
        super().train()

    def set_scenario(self, idx):
        self._scenario_idx = idx