from torch import nn as nn
import torch
import math
import numpy as np
import pygame
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.base_class import BaseAlgorithm

from environment.agent import *
from typing import Optional, Type, List, Tuple
from functools import partial

# ----------------------------- AGENT CLASSES -----------------------------

class SB3Agent(Agent):
    def __init__(self, sb3_class: Optional[Type[BaseAlgorithm]] = PPO, file_path: Optional[str] = None):
        self.sb3_class = sb3_class
        super().__init__(file_path)

    def _initialize(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.file_path is None:
            self.model = self.sb3_class(
                "MlpPolicy",
                self.env,
                verbose=1,
                n_steps=2048,
                batch_size=256,
                n_epochs=10,
                learning_rate=3e-4,
                gamma=0.99,
                ent_coef=0.01,
                clip_range=0.2,
                device=device,
            )
            del self.env
        else:
            self.model = self.sb3_class.load(self.file_path, device=device)

    def _gdown(self) -> str:
        return

    def predict(self, obs):
        action, _ = self.model.predict(obs, deterministic=False)
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path, include=['num_timesteps'])

    def learn(self, env, total_timesteps, log_interval: int = 1, verbose=0):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)


class RecurrentPPOAgent(Agent):
    def __init__(self, file_path: Optional[str] = None):
        super().__init__(file_path)
        self.lstm_states = None
        self.episode_starts = np.ones((1,), dtype=bool)

    def _initialize(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.file_path is None:
            policy_kwargs = {
                'activation_fn': nn.ReLU,
                'lstm_hidden_size': 256,
                'net_arch': [dict(pi=[1024, 512, 512, 1024], vf=[1024, 512, 512, 1024])],
                'shared_lstm': False,
                'enable_critic_lstm': True,
                'share_features_extractor': True,
            }
            self.model = RecurrentPPO(
                "MlpLstmPolicy",
                self.env,
                verbose=1,
                n_steps=2048,
                batch_size=256,
                n_epochs=10,
                gamma=0.99,
                ent_coef=0.01,
                learning_rate=3e-4,
                clip_range=0.2,
                policy_kwargs=policy_kwargs,
                device=device
            )
        else:
            self.model = RecurrentPPO.load(self.file_path, device=device)

    def reset(self) -> None:
        self.episode_starts = True

    def predict(self, obs):
        action, self.lstm_states = self.model.predict(obs, state=self.lstm_states, episode_start=self.episode_starts, deterministic=False)
        if self.episode_starts:
            self.episode_starts = False
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path)

    def learn(self, env, total_timesteps, log_interval: int = 2, verbose=1):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)


class EMARecurrentPPOAgent(RecurrentPPOAgent):
    def __init__(self, file_path: Optional[str] = None):
        super().__init__(file_path)
        self.reward_ema_alpha = 0.15
        self.ema_reward = 0.0

    def _update_ema(self, value):
        self.ema_reward = (1 - self.reward_ema_alpha) * self.ema_reward + self.reward_ema_alpha * value
        return self.ema_reward

    def _initialize(self) -> None:
        super()._initialize()


class WoLFEMARecurrentPPOAgent(RecurrentPPOAgent):
    def __init__(self, file_path: Optional[str] = None):
        super().__init__(file_path)
        self.fast_lr = 5e-4
        self.slow_lr = 1e-4
        self.avg_reward = 0.0
        self.episode_count = 0
        self.current_reward = 0.0

    def _get_learning_rate(self, progress_remaining):
        if self.current_reward < self.avg_reward:
            return self.fast_lr
        return self.slow_lr

    def update_episode_reward(self, r):
        self.episode_count += 1
        self.current_reward = r
        self.avg_reward += (r - self.avg_reward) / self.episode_count

    def _initialize(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.file_path is None:
            policy_kwargs = {
                'activation_fn': nn.ReLU,
                'lstm_hidden_size': 256,
                'net_arch': [dict(pi=[1024, 512, 512, 1024], vf=[1024, 512, 512, 1024])],
                'shared_lstm': False,
                'enable_critic_lstm': True,
                'share_features_extractor': True,
            }
            self.model = RecurrentPPO(
                "MlpLstmPolicy",
                self.env,
                verbose=1,
                n_steps=2048,
                batch_size=256,
                n_epochs=10,
                gamma=0.99,
                ent_coef=0.01,
                learning_rate=self._get_learning_rate,
                clip_range=0.2,
                policy_kwargs=policy_kwargs,
                device=device
            )
        else:
            self.model = RecurrentPPO.load(self.file_path, device=device)

# ----------------------------- SIMPLE AGENTS -----------------------------

class BasedAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time = 0

    def predict(self, obs):
        self.time += 1
        pos = self.obs_helper.get_section(obs, 'player_pos')
        opp_pos = self.obs_helper.get_section(obs, 'opponent_pos')
        opp_KO = self.obs_helper.get_section(obs, 'opponent_state') in [5, 11]
        action = self.act_helper.zeros()
        edge = 10.67 / 2
        if pos[0] > edge:
            action = self.act_helper.press_keys(['a'])
        elif pos[0] < -edge:
            action = self.act_helper.press_keys(['d'])
        elif not opp_KO:
            if (opp_pos[0] > pos[0]):
                action = self.act_helper.press_keys(['d'])
            else:
                action = self.act_helper.press_keys(['a'])
        if (pos[1] > 1.6 or pos[1] > opp_pos[1]) and self.time % 2 == 0:
            action = self.act_helper.press_keys(['space'], action)
        if (pos[0] - opp_pos[0])**2 + (pos[1] - opp_pos[1])**2 < 4.0:
            action = self.act_helper.press_keys(['j'], action)
        return action

class UserInputAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self, obs):
        action = self.act_helper.zeros()
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            action = self.act_helper.press_keys(['w'], action)
        if keys[pygame.K_a]:
            action = self.act_helper.press_keys(['a'], action)
        if keys[pygame.K_s]:
            action = self.act_helper.press_keys(['s'], action)
        if keys[pygame.K_d]:
            action = self.act_helper.press_keys(['d'], action)
        if keys[pygame.K_SPACE]:
            action = self.act_helper.press_keys(['space'], action)
        if keys[pygame.K_j]:
            action = self.act_helper.press_keys(['j'], action)
        if keys[pygame.K_k]:
            action = self.act_helper.press_keys(['k'], action)
        if keys[pygame.K_l]:
            action = self.act_helper.press_keys(['l'], action)
        if keys[pygame.K_g]:
            action = self.act_helper.press_keys(['g'], action)
        return action

class ClockworkAgent(Agent):
    def __init__(self, action_sheet: Optional[List[Tuple[int, List[str]]]] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.steps = 0
        self.current_action_end = 0
        self.current_action_data = None
        self.action_index = 0
        if action_sheet is None:
            self.action_sheet = [
                (1, ['a']),
                (1, ['l']),
                (2, ['d']),
                (1, ['a', 'j']),
                (15, ['space']),
            ]
        else:
            self.action_sheet = action_sheet

    def predict(self, obs):
        if self.steps >= self.current_action_end and self.action_index < len(self.action_sheet):
            hold_time, action_data = self.action_sheet[self.action_index]
            self.current_action_data = action_data
            self.current_action_end = self.steps + hold_time
            self.action_index += 1
        action = self.act_helper.press_keys(self.current_action_data)
        self.steps += 1
        return action

# ----------------------------- REWARD FUNCTIONS -----------------------------

def base_height_l2(env: WarehouseBrawl, target_height: float, obj_name: str = 'player') -> float:
    obj: GameObject = env.objects[obj_name]
    return -((obj.body.position.y - target_height)**2) * 0.01

class RewardMode(Enum):
    ASYMMETRIC_OFFENSIVE = 0
    SYMMETRIC = 1
    ASYMMETRIC_DEFENSIVE = 2

def damage_interaction_reward(env: WarehouseBrawl, mode: RewardMode = RewardMode.SYMMETRIC, alpha: float = 0.2) -> float:
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    damage_taken = getattr(player, 'damage_taken_this_frame', 0.0)
    damage_dealt = getattr(opponent, 'damage_taken_this_frame', 0.0)
    if mode == RewardMode.ASYMMETRIC_OFFENSIVE:
        damage_diff = damage_dealt * 1.0
    elif mode == RewardMode.SYMMETRIC:
        damage_diff = damage_dealt - damage_taken
    else:
        damage_diff = -damage_taken
    return damage_diff * 0.5

def going_to_spawner_award(env: WarehouseBrawl) -> float:
    player: Player = env.objects["player"]
    spawners = env.get_spawner_info()
    if not spawners or player.weapon != "Punch":
        return 0.0
    ppos = np.array([player.body.position.x, player.body.position.y])
    best = None
    best_d = float('inf')
    for (wtype, wpos) in spawners:
        d = np.linalg.norm(ppos - np.array(wpos))
        if d < best_d:
            best = (wtype, wpos); best_d = d
    if best is None:
        return 0.0
    target = np.array(best[1])
    vel = np.array([player.body.velocity.x, player.body.velocity.y]) * env.dt
    dir_to = target - ppos
    nd = np.linalg.norm(dir_to)
    nv = np.linalg.norm(vel)
    if nd < 1e-6 or nv < 1e-6:
        return 0.0
    return float(np.dot(vel / nv, dir_to / nd)) * 0.5

def danger_zone_reward(env: WarehouseBrawl, zone_penalty: int = 1, zone_height: float = 4.2) -> float:
    player: Player = env.objects["player"]
    reward = -zone_penalty if player.body.position.y >= zone_height else 0.0
    return reward * env.dt * 0.5

def in_state_reward(env: WarehouseBrawl, desired_state: Type[PlayerObjectState]=BackDashState) -> float:
    player: Player = env.objects["player"]
    reward = 1.0 if isinstance(player.state, desired_state) else 0.0
    return reward * env.dt * 0.2

def edge_penalty_reward(env: WarehouseBrawl, edge_penalty: int = 1, edge_boundary: float = 10.73 / 2) -> float:
    player: Player = env.objects["player"]
    pos_x = player.body.position.x
    is_off_edge = pos_x > edge_boundary or pos_x < -edge_boundary
    reward = -edge_penalty if is_off_edge else 0.0
    return reward * env.dt * 2.0

def recovery_positioning_reward(env: WarehouseBrawl) -> float:
    player: Player = env.objects["player"]
    pos_x = player.body.position.x
    pos_y = player.body.position.y
    edge_boundary = 10.73 / 2
    danger_height = 4.2
    is_in_danger = (abs(pos_x) > edge_boundary * 0.7) or (pos_y > danger_height * 0.7)
    if not is_in_danger:
        return 0.0
    vel_x = player.body.velocity.x
    vel_y = player.body.velocity.y
    direction_x = -np.sign(pos_x) if abs(pos_x) > 0.1 else 0
    direction_y = -1 if pos_y > 0 else 0
    recovery_alignment = vel_x * direction_x + vel_y * direction_y
    try:
        stock_multiplier = 2.0 if player.stocks <= 1 else 1.0
    except Exception:
        stock_multiplier = 1.0
    return max(0.0, recovery_alignment) * env.dt * stock_multiplier * 1.5

def move_to_opponent_reward(env: WarehouseBrawl) -> float:
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    vel = np.array([player.body.velocity.x, player.body.velocity.y]) * env.dt
    dir_to = np.array([opponent.body.position.x - player.body.position.x, opponent.body.position.y - player.body.position.y])
    nd = np.linalg.norm(dir_to)
    nv = np.linalg.norm(vel)
    if nd < 1e-6 or nv < 1e-6:
        return 0.0
    mult = 1.0 if player.weapon != "Punch" else -0.2
    return float(np.dot(vel / nv, dir_to / nd)) * mult * 0.5

def holding_more_than_3_keys(env: WarehouseBrawl) -> float:
    player: Player = env.objects["player"]
    a = getattr(player, 'cur_action', None)
    if a is None:
        return 0.0
    if (a > 0.5).sum() > 3:
        return -0.2 * env.dt
    return 0.0

def having_weapon_reward(env: WarehouseBrawl) -> float:
    player: Player = env.objects["player"]
    return (10.0 * env.dt) if player.weapon != "Punch" else (-5.0 * env.dt)

def on_win_reward(env: WarehouseBrawl, agent: str) -> float:
    return 50.0 if agent == 'player' else -50.0

def on_knockout_reward(env: WarehouseBrawl, agent: str) -> float:
    return 15.0 if agent == 'opponent' else -15.0

def on_equip_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == "player":
        if env.objects["player"].weapon == "Hammer":
            return 8.0
        elif env.objects["player"].weapon == "Spear":
            return 5.0
    return 0.0

def on_drop_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == "player":
        if env.objects["player"].weapon == "Punch":
            return -10.0
    return 0.0

def on_combo_reward(env: WarehouseBrawl, agent: str) -> float:
    return 5.0 if agent == 'opponent' else -5.0

def on_attack_reward(env:WarehouseBrawl, agent:str) -> float:
    if agent != 'player':
        return 0.0
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    dif = np.array([opponent.body.position.x - player.body.position.x, opponent.body.position.y - player.body.position.y])
    dist = np.linalg.norm(dif)
    if dist <= 1.0:
        val = 8.0
    elif dist <= 1.6:
        val = 1.0
    else:
        val = -2.0
    if player.weapon == "Punch":
        val -= 1.0
    return val

def on_dodge_reward(env:WarehouseBrawl, agent:str) -> float:
    if agent != 'player':
        return 0.0
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    dif = np.array([opponent.body.position.x - player.body.position.x, opponent.body.position.y - player.body.position.y])
    dist = np.linalg.norm(dif)
    if getattr(opponent, 'state', None) and opponent.state == opponent.states_types.get('attack'):
        if dist <= 1.0:
            return 8.0
        elif dist <= 1.6:
            return 3.0
    return 0.0

# ----------------------------- REWARD MANAGER -----------------------------

def gen_reward_manager():
    reward_functions = {
        'target_height_reward': RewTerm(func=base_height_l2, weight=0.2, params={'target_height': -4, 'obj_name': 'player'}),
        'danger_zone_reward': RewTerm(func=danger_zone_reward, weight=0.6),
        'damage_interaction_reward': RewTerm(func=damage_interaction_reward, weight=2.0),
        'having_weapon_reward' : RewTerm(func=having_weapon_reward, weight=0.5),
        'move_to_opponent_reward': RewTerm(func=move_to_opponent_reward, weight=0.6),
        'holding_more_than_3_keys': RewTerm(func=holding_more_than_3_keys, weight=1.0),
        'edge_penalty_reward' : RewTerm(func=edge_penalty_reward, weight = 0.8),
        'recovery_positioning_reward' : RewTerm(func=recovery_positioning_reward, weight =0.6),
        'going_to_spawner_award' : RewTerm(func=going_to_spawner_award, weight=0.6),
    }
    signal_subscriptions = {
        'on_win_reward': ('win_signal', RewTerm(func=on_win_reward, weight=1.0)),
        'on_knockout_reward': ('knockout_signal', RewTerm(func=on_knockout_reward, weight=1.0)),
        'on_combo_reward': ('hit_during_stun', RewTerm(func=on_combo_reward, weight=0.8)),
        'on_equip_reward': ('weapon_equip_signal', RewTerm(func=on_equip_reward, weight=0.8)),
        'on_drop_reward': ('weapon_drop_signal', RewTerm(func=on_drop_reward, weight=1.0)),
        'on_attack_reward':('attacked_signal', RewTerm(func = on_attack_reward , weight=0.9)),
        'on_dodge_reward':('dodged_signal', RewTerm(func = on_dodge_reward , weight=0.8)),
    }
    return RewardManager(reward_functions, signal_subscriptions)

# ----------------------------- MAIN FUNCTION -----------------------------

if __name__ == '__main__':
    mode = input("single or multiple? >>>")
    assert mode in ["single","multiple"]

    if mode == "single":
        my_agent = RecurrentPPOAgent()
        reward_manager = gen_reward_manager()
        selfplay_handler = SelfPlayRandom(partial(type(my_agent)))
        save_handler = SaveHandler(
            agent=my_agent,
            save_freq=100_000,
            max_saved=1000,
            save_path='checkpoints',
            run_name='training_improved',
            mode=SaveHandlerMode.FORCE
        )
        opponent_specification = {
            'self_play': (6, selfplay_handler),
            'constant_agent': (10, partial(ConstantAgent)),
            'based_agent': (3, partial(BasedAgent)),
            'clockwise_agent': (4, partial(ClockworkAgent)),
        }
        opponent_cfg = OpponentsCfg(opponents=opponent_specification)

        train(my_agent,
              reward_manager,
              save_handler,
              opponent_cfg,
              CameraResolution.LOW,
              train_timesteps=2_000_000,
              train_logging=TrainLogging.PLOT,
        )
    else:
        multiple_agent_classes= [RecurrentPPOAgent, EMARecurrentPPOAgent, WoLFEMARecurrentPPOAgent]
        for agentclass in multiple_agent_classes:
            try:
                my_agent = agentclass()
                reward_manager = gen_reward_manager()
                selfplay_handler = SelfPlayRandom(partial(type(my_agent)))
                save_handler = SaveHandler(
                    agent=my_agent,
                    save_freq=100_000,
                    max_saved=1000,
                    save_path='checkpoints',
                    run_name=f'{agentclass.__name__}_{__import__("datetime.datetime").today().strftime("%Y-%m-%d %H:%M:%S")}',
                    mode=SaveHandlerMode.FORCE
                )
                opponent_specification = {
                    'self_play': (6, selfplay_handler),
                    'constant_agent': (10, partial(ConstantAgent)),
                    'Based_agent': (4, partial(BasedAgent)),
                    'ClockworkAgent': (3, partial(ClockworkAgent)),
                }
                opponent_cfg = OpponentsCfg(opponents=opponent_specification)

                train(my_agent,
                      reward_manager,
                      save_handler,
                      opponent_cfg,
                      CameraResolution.LOW,
                      train_timesteps=2_000_000,
                      train_logging=TrainLogging.PLOT,
                )
            except Exception as e:
                print("Error occurred:", e)
                with open(f"errors/{agentclass.__name__}_{__import__('datetime.datetime').today().strftime('%Y-%m-%d %H:%M:%S')}", 'w') as file:
                    file.write(str(e))
