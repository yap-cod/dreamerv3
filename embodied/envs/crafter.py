import json

import crafter
import elements
import embodied
import numpy as np


class Crafter(embodied.Env):

  def __init__(self, task, size=(64, 64), logs=False, logdir=None, seed=None, epsilon=0.0):
    assert task in ('reward', 'noreward')
    self._env = crafter.Env(size=size, reward=(task == 'reward'), seed=seed)
    self._logs = logs
    self._logdir = logdir and elements.Path(logdir)
    self._logdir and self._logdir.mkdir()
    self._episode = 0
    self._length = None
    self._reward = None
    self._achievements = crafter.constants.achievements.copy()
    self._done = True
    self._epsilon = float(epsilon)
    self._prev_wood = 0
    self._prev_hp = 9  # Crafter max health
    self._prev_ach = 0

  @property
  def obs_space(self):
    spaces = {
        'image': elements.Space(np.uint8, self._env.observation_space.shape),
        'reward': elements.Space(np.float32, (3,)),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
        'log/reward': elements.Space(np.float32),
        'log/wood_collected': elements.Space(np.int32),
        'log/health_level': elements.Space(np.int32),
        'log/energy_level': elements.Space(np.int32),
        'log/achievements_unlocked': elements.Space(np.int32),
    }
    if self._logs:
      spaces.update({
          f'log/achievement_{k}': elements.Space(np.int32)
          for k in self._achievements})
    return spaces

  @property
  def act_space(self):
    return {
        'action': elements.Space(np.int32, (), 0, self._env.action_space.n),
        'reset': elements.Space(bool),
    }

  def step(self, action):
    if action['reset'] or self._done:
      self._episode += 1
      self._length = 0
      self._reward = 0
      self._done = False
      self._prev_wood = 0
      self._prev_hp = 9
      self._prev_ach = 0
      image = self._env.reset()
      vec_reward = np.zeros(3, dtype=np.float32)
      return self._obs(image, vec_reward, {}, is_first=True)
    image, reward, self._done, info = self._env.step(action['action'])
    
    current_wood = info['inventory']['wood'] if info and 'inventory' in info else 0
    if current_wood > self._prev_wood:
      reward += self._epsilon
    self._prev_wood = current_wood
    
    current_hp = info['inventory']['health'] if info and 'inventory' in info else 9
    hp_delta = current_hp - self._prev_hp
    self._prev_hp = current_hp
    
    current_ach = sum(info['achievements'].values()) if info and 'achievements' in info else 0
    ach_delta = current_ach - self._prev_ach
    self._prev_ach = current_ach
    
    vec_reward = np.array([float(reward), float(hp_delta), float(ach_delta)], dtype=np.float32)
    
    self._reward += reward
    self._length += 1
    if self._done and self._logdir:
      self._write_stats(self._length, self._reward, info)
    return self._obs(
        image, vec_reward, info,
        is_last=self._done,
        is_terminal=info['discount'] == 0)

  def _obs(
      self, image, reward, info,
      is_first=False, is_last=False, is_terminal=False):
    obs = dict(
        image=image,
        reward=np.asarray(reward, dtype=np.float32),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
        **{'log/reward': np.float32(info['reward'] if info else 0.0)},
        **{'log/wood_collected': np.int32(info['inventory']['wood'] if info and 'inventory' in info else 0)},
        **{'log/health_level': np.int32(info['inventory']['health'] if info and 'inventory' in info else 0)},
        **{'log/energy_level': np.int32(info['inventory']['energy'] if info and 'inventory' in info else 0)},
        **{'log/achievements_unlocked': np.int32(sum(info['achievements'].values()) if info and 'achievements' in info else 0)},
    )
    if self._logs:
      log_achievements = {
          f'log/achievement_{k}': info['achievements'][k] if info else 0
          for k in self._achievements}
      obs.update({k: np.int32(v) for k, v in log_achievements.items()})
    return obs

  def _write_stats(self, length, reward, info):
    stats = {
        'episode': self._episode,
        'length': length,
        'reward': round(reward, 1),
        **{f'achievement_{k}': v for k, v in info['achievements'].items()},
    }
    filename = self._logdir / 'stats.jsonl'
    lines = filename.read() if filename.exists() else ''
    lines += json.dumps(stats) + '\n'
    filename.write(lines, mode='w')
    print(f'Wrote stats: {filename}')
