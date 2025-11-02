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

class ImprovedBasedAgent(Agent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import random
        random.seed(1234)
        self.random = random
        self.time = 0

        # combo state
        self.combo_stage = 0
        self.combo_cooldown = 0
        self.combo_timer = 0

        # defensive timers
        self.last_dodge = -999
        self.dodge_cooldown = 30  # frames

        # movement smoothing, avoid spam
        self.last_dash = -999
        self.dash_cooldown = 60

    def _reset_combo(self):
        self.combo_stage = 0
        self.combo_cooldown = 0
        self.combo_timer = 0

    def _start_combo(self):
        self.combo_stage = 1
        self.combo_timer = 6
        self.combo_cooldown = 25

    def _should_dodge(self, opp_state, dist):
        """
        Basit kural: rakip saldırı içindeyse ve yakınsa dodge/geri çekil
        """
        attacking_states = [self.obs_helper.get_state_code('attack'), self.obs_helper.get_state_code('charge')] if hasattr(self.obs_helper, 'get_state_code') else []
        # fallback: check numeric states commonly used earlier (attack might be present)
        if getattr(opp_state, 'name', None) and 'attack' in opp_state.name.lower():
            return True
        # numeric fallback: treat some state ints as attack (best-effort)


    def predict(self, obs):
        self.time += 1
        if self.combo_cooldown > 0:
            self.combo_cooldown -= 1
        if self.combo_timer > 0:
            self.combo_timer -= 1
            if self.combo_timer == 0:
                self.combo_stage = 2  # progress combo

        pos = self.obs_helper.get_section(obs, 'player_pos')
        opp_pos = self.obs_helper.get_section(obs, 'opponent_pos')
        opp_state = self.obs_helper.get_section(obs, 'opponent_state')
        # Some envs return int for opponent_state; we keep existing checks too
        opp_KO = opp_state in [5, 11]

        action = self.act_helper.zeros()
        dx = opp_pos[0] - pos[0]
        dy = opp_pos[1] - pos[1]
        dist = (dx*dx + dy*dy) ** 0.5

        edge = 10.67 / 2
        edge_margin = 1.2  # ekstra güvenli alan

        # 1) Eğer edge'e çok yakınsak, hemen içeri dön (kenardan kurtul)
        if pos[0] > edge - edge_margin:
            # sola git
            return self.act_helper.press_keys(['a'])
        if pos[0] < -edge + edge_margin:
            return self.act_helper.press_keys(['d'])

        # 2) Eğer rakip KO ise pozisyon kur (güvenli uzaklık / yaklaş)
        if opp_KO:
            # rakip yerdeyse pozisyon al ve gereksiz saldırı yapma
            if abs(pos[0]) > 0.6:
                return self.act_helper.press_keys(['a' if pos[0] > 0 else 'd'])
            else:
                # yaklaşıp j ile finish (şanslı bir bitiriş)
                if dist < 1.0:
                    return self.act_helper.press_keys(['j'])
                return self.act_helper.press_keys([])  # bekle



        if dist > 3.5:
            # close the gap with cautious movement; occasionally dash+jump to close quickly
            if self.random.random() < 0.15 and (self.time - self.last_dash) > self.dash_cooldown:
                self.last_dash = self.time
                # dash + jump to close
                if dx > 0:
                    action = self.act_helper.press_keys(['d'])
                else:
                    action = self.act_helper.press_keys(['a'])
                if self.random.random() < 0.6:
                    action = self.act_helper.press_keys(['space'], action)
                return action
            else:
                return self.act_helper.press_keys(['d'] if dx > 0 else ['a'])

        if 1.5 < dist <= 3.5:
            if self.random.random() < 0.6:
                action = self.act_helper.press_keys(['space'])
                if self.random.random() < 0.4:
                    action = self.act_helper.press_keys(['j'], action)
                return action
            else:
                # approach carefully
                return self.act_helper.press_keys(['d'] if dx > 0 else ['a'])

        if dist <= 1.5:
            # start combo if ready
            if self.combo_stage == 0 and self.combo_cooldown == 0 and self.random.random() < 0.9:
                self._start_combo()
                return self.act_helper.press_keys(['j'])
            elif self.combo_stage == 1:
                if dx > 0:
                    return self.act_helper.press_keys(['d', 'j'])
                else:
                    return self.act_helper.press_keys(['a', 'j'])
            elif self.combo_stage == 2:
                # finish
                self._reset_combo()
                # chance to space + heavy
                if self.random.random() < 0.4:
                    return self.act_helper.press_keys(['space', 'j'])
                return self.act_helper.press_keys(['j'])

        # 8) If slightly above opponent, try aerial punish occasionally
        if pos[1] - opp_pos[1] > 0.6 and dist < 2.0 and self.time % 3 == 0:
            action = self.act_helper.press_keys(['space'])
            if self.random.random() < 0.5:
                action = self.act_helper.press_keys(['j'], action)
            return action

        # 9) fallback: face opponent and approach (or step back if too close)
        if dist < 0.8:
            # too close: step back and maybe light attack
            if self.random.random() < 0.4:
                return self.act_helper.press_keys(['a' if dx > 0 else 'd'])
            else:
                return self.act_helper.press_keys(['j'])
        else:
            return self.act_helper.press_keys(['d'] if dx > 0 else ['a'])
"""player_pos: 2
player_vel: 2
player_facing: 1
player_grounded: 1
player_aerial: 1
player_jumps_left: 1
player_state: 1
player_recoveries_left: 1
player_dodge_timer: 1
player_stun_frames: 1
player_damage: 1
player_stocks: 1
player_move_type: 1
player_weapon_type: 1
player_spawner_1: 3
player_spawner_2: 3
player_spawner_3: 3
player_spawner_4: 3
player_moving_platform_pos: 2
player_moving_platform_vel: 2
opponent_pos: 2
opponent_vel: 2
opponent_facing: 1
opponent_grounded: 1
opponent_aerial: 1
opponent_jumps_left: 1
opponent_state: 1
opponent_recoveries_left: 1
opponent_dodge_timer: 1
opponent_stun_frames: 1
opponent_damage: 1
opponent_stocks: 1
opponent_move_type: 1
opponent_weapon_type: 1
opponent_spawner_1: 3
opponent_spawner_2: 3
opponent_spawner_3: 3
opponent_spawner_4: 3
opponent_moving_platform_pos: 2
opponent_moving_platform_vel: 2"""

class StallingAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.time = 0
        self.spawnpos=None
        self.starter = 0
        self.boundries = 0.15
        self.yboundary = 0
        self.pressedh = False
        self.spacecd = 0
        self.spearseen = False

    def predict(self, obs):
        pos = self.obs_helper.get_section(obs, 'player_pos')
        if self.time ==0:
            self.spawnpos = pos
            if pos[0]<0:
                self.yboundary=3
            else:
                self.yboundary=0.7
            self.starter = self.obs_helper.get_section(obs, 'player_weapon_type')

            # print(self.obs_helper.print_all_sections())

        self.time += 1
        keys_pressed = []
        boundries =  self.boundries = 0.18
        player_vel = self.obs_helper.get_section(obs, 'player_vel')
        opp_pos = self.obs_helper.get_section(obs, 'opponent_pos')
        opp_KO = self.obs_helper.get_section(obs, 'opponent_state') in [5, 11]
        action = self.act_helper.zeros()
        edge = 10.67 / 2
        sideways_move = False
        heavy_attack = False


        if self.spacecd >0:
            self.spacecd -=1
        
        if pos[1]>self.yboundary:
            if self.spacecd == 0:
                keys_pressed.append('space')
                self.spacecd =15
            
        if pos[0] > self.spawnpos[0]+boundries:
            keys_pressed.append('a')
            sideways_move = True
        elif pos[0] < self.spawnpos[0]-boundries:
            keys_pressed.append('d')
            sideways_move = True
        elif not opp_KO:
            if (opp_pos[0] > pos[0]) and 0==self.obs_helper.get_section(obs,"player_facing"): #player.facing == Facing.RIGHT
                keys_pressed.append('d')
                sideways_move = True
            elif (opp_pos[0] < pos[0]) and 1==self.obs_helper.get_section(obs,"player_facing"):
                keys_pressed.append('a')
                sideways_move = True
        else:
            pass
            # action = self.act_helper.press_keys(['g']) # h is pickup

        # if not sideways_move and pos[0]-boundries< self.spawnpos[0] <pos[0]+boundries and opp_pos[0]-boundries< self.spawnpos[0] <opp_pos[0]+boundries:

        #     if opp_pos[1]>pos[1]:
        #         keys_pressed.append('s')
        #         keys_pressed.append('k')
        #     else:
        #         keys_pressed.append('k')
        #     heavy_attack = True
        # print("Variable content:", self.obs_helper.get_section(obs, 'player_weapon_type'))
        # print("Type:", type(self.obs_helper.get_section(obs, 'player_weapon_type')))
        # print("Dtype:", self.obs_helper.get_section(obs, 'player_weapon_type').dtype)
        # print("Shape:", self.obs_helper.get_section(obs, 'player_weapon_type').shape)
        # print('player_weapon_type',-0.0001 <float( self.obs_helper.get_section(obs, 'player_weapon_type')[0]) - 1 < 0.0001 )
        # a = np.array([1.], dtype=np.float64) # spear
        if not self.spearseen and self.obs_helper.get_section(obs, 'player_weapon_type') == self.starter or -0.0001 <float( self.obs_helper.get_section(obs, 'player_weapon_type')[0]) - 1 < 0.0001 :
           if not self.pressedh:
                keys_pressed.append('h')
            
           self.spearseen =-0.0001 <float( self.obs_helper.get_section(obs, 'player_weapon_type')[0]) - 1 < 0.0001
           
           self.pressedh = not self.pressedh

        if not sideways_move and not heavy_attack and (pos[0] - opp_pos[0])**2 + (pos[1] - opp_pos[1])**2 < 3.9:
            keys_pressed.append('j')
        # print(keys_pressed)
        action=self.act_helper.press_keys(keys_pressed,action)
        return action

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
                n_steps=4086*4,
                batch_size=512,
                n_epochs=20,
                learning_rate=3e-4,
                gamma=0.97,
                ent_coef=0.01,
                clip_range=0.20,
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
                'lstm_hidden_size': 512,
                'net_arch': [dict(pi=[512,512,512], vf=[512,512,512])],
                'shared_lstm': False,
                'enable_critic_lstm': True,
                'share_features_extractor': True,
            }
            self.model = RecurrentPPO(
                "MlpLstmPolicy",
                self.env,
                verbose=1,
                n_steps=4086*16,
                batch_size=128,
                n_epochs=30,
                gamma=0.97,
                ent_coef=0.02,
                learning_rate=4e-4,
                clip_range=0.20,
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


class WoLFEMARecurrentPPOAgent(RecurrentPPOAgent): # Trainings w this sucked, Im switching over to EMA
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
                'lstm_hidden_size': 512,
                'net_arch':  [dict(pi=[512,1024,512], vf=[512,1024, 512])],
                'shared_lstm': False,
                'enable_critic_lstm': True,
                'share_features_extractor': True,
            }
            self.model = RecurrentPPO(
                "MlpLstmPolicy",
                self.env,
                verbose=1,
                n_steps=4086,
                batch_size=1024,
                n_epochs=30,
                gamma=0.97,
                ent_coef=0.02,
                learning_rate=self._get_learning_rate,
                clip_range=0.20,
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


def staying_alive_reward(env:WarehouseBrawl) -> float:
    return 1*env.dt

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
        damage_diff = damage_dealt * 1.3
    elif mode == RewardMode.SYMMETRIC:
        damage_diff = damage_dealt - damage_taken
    else:
        damage_diff = -damage_taken
    return damage_diff * 0.2

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
    danger_height = 3.2
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

def recovery_actions_award(env: WarehouseBrawl) -> float:
    """This reward function applies a penalty if the agent tries to attack, or do somehting else during faling."""
    player: Player = env.objects["player"]
    opponent : Player = env.objects["opponent"]
    pos_x = player.body.position.x
    pos_y = player.body.position.y
    edge_boundary = 10.73 / 2
    danger_height = 3.2
    is_in_danger = (abs(pos_x) > edge_boundary * 0.7) or (pos_y > danger_height * 0.7)

    if not is_in_danger:return 0

    if player.state in [player.states_types.get("dodge"),player.states_types.get("attack"),player.states_types.get("dash"),player.states_types.get("taunt")]:
        return -30 * env.dt
    else:
        return 10 * env.dt

def move_to_opponent_reward(env: WarehouseBrawl) -> float:
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    vel = np.array([player.body.velocity.x, player.body.velocity.y]) * env.dt
    dir_to = np.array([opponent.body.position.x - player.body.position.x, opponent.body.position.y - player.body.position.y])
    nd = np.linalg.norm(dir_to)
    nv = np.linalg.norm(vel)
    if nd < 1e-6 or nv < 1e-6:
        return 0.0
    mult = 10.0 if player.weapon != "Punch" else 5
    return float(np.dot(vel / nv, dir_to / nd)) * mult *  env.dt

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
    return (10.0 * env.dt) if player.weapon != "Punch" else (-3.0 * env.dt)

def downward_velocity_reward(env: WarehouseBrawl) -> float:
    """
    Reward the agent for having a downward velocity magnitude exceeding 0.7.
    Encourages aggressive dropping or fast descent when beneficial.

    Returns:
        float: Positive reward proportional to downward velocity over threshold, scaled by env.dt.
    """
    player: Player = env.objects["player"]
    vy = player.body.velocity.y
    
    # Only consider downward velocity (positive y)
    if vy > 1:
        reward = (-vy - 1) * 5.0  # Scale factor 10; tune as necessary
        return reward * env.dt
    else:
        return 0.0


def platform_reward(env:WarehouseBrawl)-> float:
    player: Player = env.objects["player"]
    if abs(player.body.position.x-env.objects['platform1'].body.position[0])<0.8 and player.body.position.x< env.objects['platform1'].body.position[1]+1:# (x,y)
        return 15 * env.dt
    elif abs(player.body.position.x-env.objects['platform1'].body.position[0])>0.8 and 1.5> player.body.position.x  and player.body.position.x >-1.5 and player.body.position.x< env.objects['platform1'].body.position[1]+1:
        return -10 * env.dt
    elif  1.5> player.body.position.x  and player.body.position.x >-1.5 and player.body.position.x> env.objects['platform1'].body.position[1]+1:
        return -50 * env.dt
    else:
        return 0



def on_win_reward(env: WarehouseBrawl, agent: str) -> float:
    return 1000.0 if agent == 'player' else -1000.0

def on_knockout_reward(env: WarehouseBrawl, agent: str) -> float:
    return 400.0 if agent == 'opponent' else -400.0

def on_equip_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == "player":
        if env.objects["player"].weapon == "Hammer":
            return 8.0
        elif env.objects["player"].weapon == "Spear":
            return 5.0
    return 0.0

def on_drop_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == "player":
        # if env.objects["player"].weapon == "Punch":
        return -40.0
    return 0

def on_combo_reward(env: WarehouseBrawl, agent: str) -> float:
    return 50.0 if agent == 'opponent' else -10.0

def on_attack_reward(env:WarehouseBrawl, agent:str) -> float:
    if agent != 'player':
        return 0.0
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    dif = np.array([opponent.body.position.x - player.body.position.x, opponent.body.position.y - player.body.position.y])
    dist = np.linalg.norm(dif)
    if dist <= 1.0:
        return (-((dist - 0.5) * 3.2)**3 + 2)
    elif dist <= 1.6:
        val = 0
    else:
        return -1
    # Vector from player to opponent
    vec_to_opponent = np.array([
        opponent.body.position.x - player.body.position.x,
        opponent.body.position.y - player.body.position.y
    ])

    # Check if opponent is to the right or left relative to player
    opponent_is_right = vec_to_opponent[0] > 0

    # Player facing value (Facing.RIGHT = 1, Facing.LEFT = -1)
    player_facing_right = (player.facing == Facing.RIGHT)

    # Reward 1.0 if player faces opponent correctly, else -0.5
    if (opponent_is_right and player_facing_right) or (not opponent_is_right and not player_facing_right):
        val+=10
    else:
        val-=5

    val -=5

    if getattr(player, 'state', None) and player.state == player.states_types.get('in_air'):
        val = -5
    if player.weapon == "Punch":
        val -= 3.0
    else:
        val +=10
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
            return -((dist - 0.5) * 3.2)**3 + 4
        elif dist <= 1.6:
            return 0
    return -math.log(dist-1.5)

def on_taunt_reward(env:WarehouseBrawl, agent:str) -> float:
    """penaltize for taunting, give out a big reward if taunted during opponent knockout"""
    if agent == "player":
        if env.objects["opponent"].state == env.objects["opponent"].states_types.get("KO"):
            return 10
        return -50
    return 0

def on_dash_award(env:WarehouseBrawl, agent:str )-> float:
    """we are penaltizing dash because the AI 90% jumps off the map when it dashes."""
    if agent == "player":
        return -4
    return 0


def head_to_middle_reward(
    env: WarehouseBrawl,
) -> float:
    """
    Applies a penalty for every time frame player surpases a certain height threshold in the environment.

    Args:
        env (WarehouseBrawl): The game environment.
        zone_penalty (int): The penalty applied when the player is in the danger zone.
        zone_height (float): The height threshold defining the danger zone.

    Returns:
        float: The computed penalty as a tensor.
    """
    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is in the danger zone
    multiplier = -1 if player.body.position.x > 0 else 1
    reward = multiplier * (player.body.position.x - player.prev_x)

    return reward


#TODO: reward for not jumping too high
#TODO: reward for jumping while falling off the map (I dont think I can do that but yeah.)

# ----------------------------- REWARD MANAGER -----------------------------

def gen_reward_manager():
    reward_functions = {
        'target_height_reward': RewTerm(func=base_height_l2, weight=0.5, params={'target_height': -4, 'obj_name': 'player'}),
        'danger_zone_reward': RewTerm(func=danger_zone_reward, weight=15),
        'damage_interaction_reward': RewTerm(func=damage_interaction_reward, weight=4.0),
        'having_weapon_reward' : RewTerm(func=having_weapon_reward, weight=3),
        'move_to_opponent_reward': RewTerm(func=move_to_opponent_reward, weight=10),
        'holding_more_than_3_keys': RewTerm(func=holding_more_than_3_keys, weight=0.6),
        'edge_penalty_reward' : RewTerm(func=edge_penalty_reward, weight = 10),
        'recovery_positioning_reward' : RewTerm(func=recovery_positioning_reward, weight =3),
        'going_to_spawner_award' : RewTerm(func=going_to_spawner_award, weight=6),
        'staying_alive_reward' : RewTerm (func = staying_alive_reward, weight = 6),
        'recovery_actions_award' : RewTerm(func=recovery_actions_award, weight=2),
        'downward_velocity_reward' : RewTerm(func=downward_velocity_reward, weight=3),
        'platform_reward' : RewTerm(func = platform_reward, weight =4),
        'head_to_middle_reward' : RewTerm(func=head_to_middle_reward, weight = 2)
        
}
    signal_subscriptions = {
        'on_win_reward': ('win_signal', RewTerm(func=on_win_reward, weight=1.4)),
        'on_knockout_reward': ('knockout_signal', RewTerm(func=on_knockout_reward, weight=1.4)),
        'on_combo_reward': ('hit_during_stun', RewTerm(func=on_combo_reward, weight=1.5)),
        'on_equip_reward': ('weapon_equip_signal', RewTerm(func=on_equip_reward, weight=3)),
        'on_drop_reward': ('weapon_drop_signal', RewTerm(func=on_drop_reward, weight=5)),
        'on_attack_reward':('attacked_signal', RewTerm(func = on_attack_reward , weight=6)),
        'on_dodge_reward':('dodged_signal', RewTerm(func = on_dodge_reward , weight=2)),
        'on_taunt_reward': ('taunted_signal', RewTerm(func = on_taunt_reward, weight=5)),
        'on_dash_award' : ('dashed_signal', RewTerm(func=on_dash_award, weight=5))
    }
    return RewardManager(reward_functions, signal_subscriptions)

# ----------------------------- MAIN FUNCTION -----------------------------

if __name__ == '__main__':
    mode = input("single or multiple? >>>")
    assert mode in ["single","multiple"]

    if mode == "single":
        my_agent = EMARecurrentPPOAgent(r"checkpoints\EMA_MORE_STEPS_2025-11-01-03-25-23\rl_model_7975872_steps.zip".replace("\\","/"))
        reward_manager = gen_reward_manager()
        selfplay_handler = SelfPlayRandom(partial(type(my_agent)))
        save_handler = SaveHandler(
            agent=my_agent,
            save_freq=100_000,
            max_saved=1000,
            save_path='checkpoints',
            run_name=f'{"EMA_MORE_STEPS"}_{__import__("datetime").datetime.today().strftime("%Y-%m-%d-%H-%M-%S")}',
            mode=SaveHandlerMode.FORCE
        )
        opponent_specification = {
            'self_play': (1, selfplay_handler),
            'constant_agent': (15, partial(ConstantAgent)),
            'based_agent': (3, partial(BasedAgent)),
            'clockwise_agent': (1, partial(ClockworkAgent)),
        }
        opponent_cfg = OpponentsCfg(opponents=opponent_specification)

        train(my_agent,
              reward_manager,
              save_handler,
              opponent_cfg,
              CameraResolution.LOW,
              train_timesteps=10_000_000,
              train_logging=TrainLogging.PLOT,
        )
    else:
        multiple_agent_classes= [WoLFEMARecurrentPPOAgent, EMARecurrentPPOAgent, RecurrentPPOAgent ]
        for agentclass in multiple_agent_classes:
            try:
                my_agent = agentclass()
                reward_manager = gen_reward_manager()
                selfplay_handler = SelfPlayRandom(partial(type(my_agent)))
                save_handler = SaveHandler(
                    agent=my_agent,
                    save_freq=25_000,
                    max_saved=1000,
                    save_path='checkpoints',
                    run_name=f'{agentclass.__name__}_{__import__("datetime").datetime.today().strftime("%Y-%m-%d-%H-%M-%S")}',
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
                      train_timesteps=1_000_000,
                      train_logging=TrainLogging.PLOT,
                )
            except Exception as e:
                print("Error occurred:", e)
                with open(f"errors/{agentclass.__name__}_{__import__('datetime').datetime.today().strftime('%Y-%m-%d-%H-%M-%S')}", 'w') as file:
                    file.write(str(e))
