'''
TRAINING: AGENT

This file contains all the types of Agent classes, the Reward Function API, and the built-in train function from our multi-agent RL API for self-play training.
- All of these Agent classes are each described below. 

Running this file will initiate the training function, and will:
a) Start training from scratch
b) Continue training from a specific timestep given an input `file_path`
'''

# -------------------------------------------------------------------
# ----------------------------- IMPORTS -----------------------------
# -------------------------------------------------------------------

from torch import nn as nn
import numpy as np
import pygame
from stable_baselines3 import A2C, PPO, SAC, DQN, DDPG, TD3, HER 
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from environment.agent import *
from typing import Optional, Type, List, Tuple

# -------------------------------------------------------------------------
# ----------------------------- AGENT CLASSES -----------------------------
# -------------------------------------------------------------------------

class SB3Agent(Agent):
    '''
    SB3Agent:
    - Defines an AI Agent that takes an SB3 class input for specific SB3 algorithm (e.g. PPO, SAC)
    Note:
    - For all SB3 classes, if you'd like to define your own neural network policy you can modify the `policy_kwargs` parameter in `self.sb3_class()` or make a custom SB3 `BaseFeaturesExtractor`
    You can refer to this for Custom Policy: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
    '''
    def __init__(
            self,
            sb3_class: Optional[Type[BaseAlgorithm]] = PPO,
            file_path: Optional[str] = None
    ):
        self.sb3_class = sb3_class
        super().__init__(file_path)

    def _initialize(self) -> None:
        if self.file_path is None:
            self.model = self.sb3_class(
                "MlpPolicy", 
                self.env, 
                verbose=0, 
                n_steps=30*90*3, 
                batch_size=128, 
                ent_coef=0.01,
                device="cuda"
            )
            del self.env
        else:
            self.model = self.sb3_class.load(self.file_path, device="cuda")  # Add device here too


    def _gdown(self) -> str:
        # Call gdown to your link
        return

    #def set_ignore_grad(self) -> None:
        #self.model.set_ignore_act_grad(True)

    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path, include=['num_timesteps'])

    def learn(self, env, total_timesteps, log_interval: int = 1, verbose=0):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
        )

class EMARewardWrapper: # Thanks Perplexity, much kudos
    """Applies exponential moving average smoothing to rewards"""
    def __init__(self, alpha=0.1):
        self.alpha = alpha  # EMA coefficient (0 < alpha < 1)
        self.ema_reward = 0.0
        
    def update(self, reward):
        self.ema_reward = (1 - self.alpha) * self.ema_reward + self.alpha * reward
        return self.ema_reward

class WoLFRewardTracker:# Thanks Perplexity, much kudos 2
    """Tracks win/loss status for adaptive learning"""
    def __init__(self, fast_lr=0.001, slow_lr=0.0003):
        self.fast_lr = fast_lr
        self.slow_lr = slow_lr
        self.avg_reward = 0.0
        self.current_reward = 0.0
        self.episode_count = 0
        
    def update(self, episode_reward):
        self.episode_count += 1
        self.current_reward = episode_reward
        self.avg_reward += (episode_reward - self.avg_reward) / self.episode_count
        
    def get_learning_rate(self,other):
        # print(other, "debug get_learning_rate second argument")
        # Learn fast when losing (below average)
        if self.current_reward < self.avg_reward:
            return self.fast_lr
        # Learn slow when winning (above average)
        return self.slow_lr

class RecurrentPPOAgent(Agent):
    '''
    RecurrentPPOAgent:
    - Defines an RL Agent that uses the Recurrent PPO (LSTM+PPO) algorithm
    '''
    def __init__(
            self,
            file_path: Optional[str] = None
    ):
        super().__init__(file_path)
        self.lstm_states = None
        self.episode_starts = np.ones((1,), dtype=bool)
        # self.reward_ema = EMARewardWrapper(alpha=0.15)

    def _initialize(self) -> None:
        if self.file_path is None:
            policy_kwargs = {
                'activation_fn': nn.ReLU,
                'lstm_hidden_size': 1024,
                'net_arch': [dict(pi=[256, 128, 64], vf=[512, 256])],
                'shared_lstm': False,
                'enable_critic_lstm': True,
                'share_features_extractor': True,
            }
            self.model = RecurrentPPO(
                "MlpLstmPolicy",
                self.env,
                verbose=1,
                n_steps=30*90*20*30,      # More steps per rollout
                batch_size=2048,           # Larger batches
                n_epochs=80,               # More training epochs
                gamma=0.94,                # Slightly higher discount
                ent_coef=0.02,            # Lower entropy for exploitation
                learning_rate=2.5e-4,     # Standard PPO learning rate
                clip_range=0.2,           # PPO clip range
                policy_kwargs=policy_kwargs,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
        else:
            self.model = RecurrentPPO.load(self.file_path, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))  # Add device here too
        # print(f"initialized with {self.device}")
    def reset(self) -> None:
        self.episode_starts = True

    def predict(self, obs):
        action, self.lstm_states = self.model.predict(obs, state=self.lstm_states, episode_start=self.episode_starts, deterministic=False)
        if self.episode_starts: self.episode_starts = False
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path)

    def learn(self, env, total_timesteps, log_interval: int = 2, verbose=1):
        print(f"timestamps: {total_timesteps}")
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)

class EMARecurrentPPOAgent(Agent):
    '''
    RecurrentPPOAgent:
    - Defines an RL Agent that uses the Recurrent PPO (LSTM+PPO) algorithm
    '''
    def __init__(
            self,
            file_path: Optional[str] = None
    ):
        super().__init__(file_path)
        self.lstm_states = None
        self.episode_starts = np.ones((1,), dtype=bool)
        self.reward_ema = EMARewardWrapper(alpha=0.15)

    def _initialize(self) -> None:
        if self.file_path is None:
            policy_kwargs = {
                'activation_fn': nn.ReLU,
                'lstm_hidden_size': 1024,
                'net_arch': [dict(pi=[256, 128, 64], vf=[512, 256])],
                'shared_lstm': False,
                'enable_critic_lstm': True,
                'share_features_extractor': True,
            }
            self.model = RecurrentPPO(
                "MlpLstmPolicy",
                self.env,
                verbose=1,
                n_steps=30*90*20*30,      # More steps per rollout
                batch_size=2048,           # Larger batches
                n_epochs=80,               # More training epochs
                gamma=0.97,                # Slightly higher discount
                ent_coef=0.02,            # Lower entropy for exploitation
                learning_rate=2.5e-4,     # Standard PPO learning rate
                clip_range=0.2,           # PPO clip range
                policy_kwargs=policy_kwargs,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
        else:
            self.model = RecurrentPPO.load(self.file_path, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))  # Add device here too
        # print(f"initialized with {self.device}")
    def reset(self) -> None:
        self.episode_starts = True

    def predict(self, obs):
        action, self.lstm_states = self.model.predict(obs, state=self.lstm_states, episode_start=self.episode_starts, deterministic=False)
        if self.episode_starts: self.episode_starts = False
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path)

    def learn(self, env, total_timesteps, log_interval: int = 2, verbose=1):
        print(f"timestamps: {total_timesteps}")
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)

class WoLFEMARecurrentPPOAgent(Agent):
    '''
    RecurrentPPOAgent:
    - Defines an RL Agent that uses the Recurrent PPO (LSTM+PPO) algorithm
    '''
    def __init__(
            self,
            file_path: Optional[str] = None

    ):

        super().__init__(file_path)
        self.lstm_states = None
        self.episode_starts = np.ones((1,), dtype=bool)
        self.reward_ema = EMARewardWrapper(alpha=0.11) # Note that it still has 

    def _initialize(self) -> None:
        # print(self.env.__dict__)
        if self.file_path is None:
            self.wolf_tracker = WoLFRewardTracker(fast_lr=0.0005, slow_lr=0.0001)
            policy_kwargs = {
                'activation_fn': nn.ReLU,
                'lstm_hidden_size': 1024,
                'net_arch': [dict(pi=[256, 128, 64], vf=[512, 256])],
                'shared_lstm': False,
                'enable_critic_lstm': True,
                'share_features_extractor': True,
            }
            self.model = RecurrentPPO(
                "MlpLstmPolicy",
                self.env,
                verbose=1,
                n_steps=30*90*20*30,      # More steps per rollout
                batch_size=2048,           # Larger batches
                n_epochs=80,               # More training epochs
                gamma=0.97,                # Slightly higher discount
                ent_coef=0.02,            # Lower entropy for exploitation
                learning_rate=self.wolf_tracker.get_learning_rate,
                clip_range=0.2,           # PPO clip range
                policy_kwargs=policy_kwargs,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
        else:
            self.model = RecurrentPPO.load(self.file_path, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))  # Add device here too
        # print(f"initialized with {self.device}")
    
    def reset(self) -> None:
        self.episode_starts = True

    def predict(self, obs):
        # print(env.player_state, "player_state debug")
        action, self.lstm_states = self.model.predict(obs, state=self.lstm_states, episode_start=self.episode_starts, deterministic=False)
        if self.episode_starts: self.episode_starts = False
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path)

    def learn(self, env, total_timesteps, log_interval: int = 2, verbose=1):
        print(f"timestamps: {total_timesteps}")
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)

class BasedAgent(Agent):
    '''
    BasedAgent:
    - Defines a hard-coded Agent that predicts actions based on if-statements. Interesting behaviour can be achieved here.
    - The if-statement algorithm can be developed within the `predict` method below.
    '''
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.time = 0

    def predict(self, obs):
        self.time += 1
        pos = self.obs_helper.get_section(obs, 'player_pos')
        opp_pos = self.obs_helper.get_section(obs, 'opponent_pos')
        opp_KO = self.obs_helper.get_section(obs, 'opponent_state') in [5, 11]
        action = self.act_helper.zeros()

        # If off the edge, come back
        if pos[0] > 10.67/2:
            action = self.act_helper.press_keys(['a'])
        elif pos[0] < -10.67/2:
            action = self.act_helper.press_keys(['d'])
        elif not opp_KO:
            # Head toward opponent
            if (opp_pos[0] > pos[0]):
                action = self.act_helper.press_keys(['d'])
            else:
                action = self.act_helper.press_keys(['a'])

        # Note: Passing in partial action
        # Jump if below map or opponent is above you
        if (pos[1] > 1.6 or pos[1] > opp_pos[1]) and self.time % 2 == 0:
            action = self.act_helper.press_keys(['space'], action)

        # Attack if near
        if (pos[0] - opp_pos[0])**2 + (pos[1] - opp_pos[1])**2 < 4.0:
            action = self.act_helper.press_keys(['j'], action)
        return action

class UserInputAgent(Agent):
    '''
    UserInputAgent:
    - Defines an Agent that performs actions entirely via real-time player input
    '''
    def __init__(
            self,
            *args,
            **kwargs
    ):
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
        # h j k l
        if keys[pygame.K_h]:
            # action = self.act_helper.press_keys(['h'], action) emotes
            pass 
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
    '''
    ClockworkAgent:
    - Defines an Agent that performs sequential steps of [duration, action]
    '''
    def __init__(
            self,
            action_sheet: Optional[List[Tuple[int, List[str]]]] = None,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.steps = 0
        self.current_action_end = 0  # Tracks when the current action should stop
        self.current_action_data = None  # Stores the active action
        self.action_index = 0  # Index in the action sheet

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
        """
        Returns an action vector based on the predefined action sheet.
        """
        # Check if the current action has expired
        if self.steps >= self.current_action_end and self.action_index < len(self.action_sheet):
            hold_time, action_data = self.action_sheet[self.action_index]
            self.current_action_data = action_data  # Store the action
            self.current_action_end = self.steps + hold_time  # Set duration
            self.action_index += 1  # Move to the next action

        # Apply the currently active action
        action = self.act_helper.press_keys(self.current_action_data)
        self.steps += 1  # Increment step counter
        return action
    
# --------------------------------------------------------------------------------
# ----------------------------- REWARD FUNCTIONS API -----------------------------
# --------------------------------------------------------------------------------

'''
Example Reward Functions:
- Find more [here](https://colab.research.google.com/drive/1qMs336DclBwdn6JBASa5ioDIfvenW8Ha?usp=sharing#scrollTo=-XAOXXMPTiHJ).
'''

def base_height_l2(
    env: WarehouseBrawl,
    target_height: float,
    obj_name: str = 'player'
) -> float:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # Extract the used quantities (to enable type-hinting)
    obj: GameObject = env.objects[obj_name]

    # Compute the L2 squared penalty
    return (obj.body.position.y - target_height)**2

class RewardMode(Enum):
    ASYMMETRIC_OFFENSIVE = 0
    SYMMETRIC = 1
    ASYMMETRIC_DEFENSIVE = 2


def damage_interaction_reward(
    env: WarehouseBrawl,
    mode: RewardMode = RewardMode.SYMMETRIC,
    alpha: float = 0.2,  # Momentum coefficient for EMA smoothing
) -> float:
    """
    Damage reward with momentum smoothing to reduce variance.
    
    Combines damage interaction modes with exponential moving average smoothing
    to create more stable reward signals during training.
    
    Modes:
    - ASYMMETRIC_OFFENSIVE (0): Reward based only on damage dealt to opponent
    - SYMMETRIC (1): Reward based on both dealing damage and avoiding damage
    - ASYMMETRIC_DEFENSIVE (2): Reward based only on avoiding damage
    
    Args:
        env (WarehouseBrawl): The game environment
        mode (RewardMode): Reward mode for damage calculation
        alpha (float): EMA smoothing coefficient (0 < alpha < 1)
                      Higher values = less smoothing, more responsive
                      Lower values = more smoothing, more stable
    
    Returns:
        float: The smoothed reward value
    """
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    
    # Initialize momentum storage if not exists
    if not hasattr(player, '_damage_momentum'):
        player._damage_momentum = 0.0
        opponent._damage_momentum = 0.0
    
    # Get current frame damage values
    damage_taken = player.damage_taken_this_frame
    damage_dealt = opponent.damage_taken_this_frame
    
    # Compute raw damage differential based on mode
    if mode == RewardMode.ASYMMETRIC_OFFENSIVE:
        damage_diff = damage_dealt * 1.5
    elif mode == RewardMode.SYMMETRIC:
        damage_diff = damage_dealt - damage_taken
    elif mode == RewardMode.ASYMMETRIC_DEFENSIVE:
        damage_diff = -damage_taken
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    # Apply EMA smoothing to reduce variance
    smoothed_reward = (
        alpha * damage_diff + 
        (1 - alpha) * player._damage_momentum
    )
    
    # Update momentum for next frame
    player._damage_momentum = smoothed_reward
    
    # Scale reward to reasonable range
    return (smoothed_reward) / 70

def going_to_spawner_award(env):
    """
    Computes the reward based on whether the agent is moving toward the pickup spawner.
    The reward is calculated by taking the dot product of the agent's normalized velocity
    with the normalized direction vector toward the closest spawner.

    Args:
        env (WarehouseBrawl): The game environment

    Returns:
        float: The computed reward
    """
    # Getting agent and opponent from the enviornment
    player: Player = env.objects["player"]
    spawners = env.get_spawner_info()

    # Extracting player velocity and position from environment
    if spawners ==[] or env.objects["player"].weapon != "Punch":return

    closest_spawner = None
    closest_distance = 9999999999
    for i in spawners:
        if closest_distance > np.linalg.norm(player.body.location - i):
            closest_spawner = i

    player_position_dif = np.array([player.body.velocity.x * env.dt, player.body.velocity.y * env.dt])

    direction_to_spawner = np.array([closest_spawner[0] - player.body.position.x,
                                      closest_spawner[1] - player.body.position.y])

    # Prevent division by zero or extremely small values
    direc_to_spawner_norm = np.linalg.norm(direction_to_spawner)
    player_pos_dif_norm = np.linalg.norm(player_position_dif)

    if direc_to_spawner_norm < 1e-6 or player_pos_dif_norm < 1e-6:
        return 0.0

    # Compute the dot product of the normalized vectors to figure out how much
    # current movement (aka velocity) is in alignment with the direction they need to go in
    reward = np.dot(player_position_dif / direc_to_spawner_norm, direction_to_spawner / direc_to_spawner_norm)

    return reward if reward is not None else 0

def danger_zone_reward(
    env: WarehouseBrawl,
    zone_penalty: int = 1,
    zone_height: float = 4.2
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
    reward = -zone_penalty  if player.body.position.y >= zone_height else 0.0

    return reward * env.dt  if reward is not None else 0

def in_state_reward(
    env: WarehouseBrawl,
    desired_state: Type[PlayerObjectState]=BackDashState,
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
    reward = 1 if isinstance(player.state, desired_state) else 0.0

    return reward * env.dt  if reward is not None else 0


def edge_penalty_reward( # gemini helped reward function
    env: WarehouseBrawl,
    edge_penalty: int = 1,
    edge_boundary: float = 10.73 / 2
) -> float:
    """
    Applies a penalty for every time frame the player is beyond the 
    specified horizontal boundaries (off the edge).

    This reward function is based on the logic:
    if pos[0] > 10.67/2:
        # Player is too far right
    elif pos[0] < -10.67/2:
        # Player is too far left

    Args:
        env (WarehouseBrawl): The game environment.
        edge_penalty (int): The penalty applied when the player is off the edge.
        edge_boundary (float): The horizontal position threshold defining the edge.
                               Defaults to 10.67 / 2.

    Returns:
        float: The computed penalty, scaled by env.dt.
    """
    # Get player object from the environment
    player: Player = env.objects["player"]

    # Get player's horizontal position (pos[0] in the example)
    pos_x = player.body.position.x

    # Apply penalty if the player is off the edge
    is_off_edge = pos_x > edge_boundary or pos_x < -edge_boundary
    reward = -edge_penalty * 20 if is_off_edge else 1 # idk if well need to adjust for the middle edge. 

    return reward * env.dt  if reward is not None else 0

def recovery_positioning_reward( # kudos perplexity
    env: WarehouseBrawl,
) -> float:
    """
    Rewards player for moving toward stage center when off-stage or in danger.
    Encourages survival behavior when stocks are low.
    """
    player: Player = env.objects["player"]
    
    # Check if player is in danger (high or off-stage)
    pos_x = player.body.position.x
    pos_y = player.body.position.y
    
    edge_boundary = 10.73 / 2
    danger_height = 4.2
    
    is_in_danger = (abs(pos_x) > edge_boundary * 0.7) or (pos_y > danger_height * 0.7)
    
    if not is_in_danger:
        return 0.0
    
    # Reward moving toward center (0, 0)
    vel_x = player.body.velocity.x
    vel_y = player.body.velocity.y
    
    # Direction to center
    direction_x = -np.sign(pos_x) if abs(pos_x) > 0.1 else 0
    direction_y = -1 if pos_y > 0 else 0  # Move down toward stage
    
    # Reward velocity aligned with recovery direction
    recovery_alignment = vel_x * direction_x + vel_y * direction_y
    
    # Scale by stock disadvantage
    try:
        stock_multiplier = 2.0 if player.stocks <= 1 else 1.0
    except:
        stock_multiplier = 1.0
    
    return max(0, recovery_alignment) * env.dt * stock_multiplier * 5.0  if recovery_alignment is not None  and stock_multiplier is not None else 0


def head_to_middle_reward( # we don't need thisshi
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

    return reward * 0   if reward is not None else 0#remove if you want the middle award back

def having_weapon_reward(env) -> float:
    player: Player = env.objects["player"]
    if env.objects["player"].weapon == "Punch":
        return -25 * env.dt
    else: 
        return 10 * env.dt

player_prev_position = None
def move_to_opponent_reward(
    env: WarehouseBrawl,
) -> float:
    """
    Computes the reward based on whether the agent is moving toward the opponent.
    The reward is calculated by taking the dot product of the agent's normalized velocity
    with the normalized direction vector toward the opponent.

    Args:
        env (WarehouseBrawl): The game environment

    Returns:
        float: The computed reward
    """

    #variable names in this function is kinda hard to understand but I hope the math works. 

    # Getting agent and opponent from the enviornment
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    # Extracting player velocity and position from environment
    player_position_dif = np.array([player.body.velocity.x * env.dt, player.body.velocity.y * env.dt])

    direction_to_opponent = np.array([opponent.body.position.x - player.body.position.x,
                                      opponent.body.position.y - player.body.position.y])

    # Prevent division by zero or extremely small values
    direc_to_opp_norm = np.linalg.norm(direction_to_opponent)
    player_pos_dif_norm = np.linalg.norm(player_position_dif)

    if direc_to_opp_norm < 1e-6 or player_pos_dif_norm < 1e-6:
        return 0.0

    # Compute the dot product of the normalized vectors to figure out how much
    # current movement (aka velocity) is in alignment with the direction they need to go in
    if env.objects["player"].weapon != "Punch": # This if statement ensures that the the agent will escape the enemy and will chase if it has a weapon.
        reward = np.dot(player_position_dif / direc_to_opp_norm, direction_to_opponent / direc_to_opp_norm)
    else:
        reward = -0.1*np.dot(player_position_dif / direc_to_opp_norm, direction_to_opponent / direc_to_opp_norm)

    return reward  if reward is not None else 0

def holding_more_than_3_keys(
    env: WarehouseBrawl,
) -> float:

    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is holding more than 3 keys
    a = player.cur_action
    # print(a," debug of action list")
    if (a > 0.5).sum() > 3:
        return env.dt * 7
    return 0 



def on_win_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'player':
        return 600
    else:
        return -1000

def on_knockout_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'player':
        return -200
    else:
        return 150

def on_equip_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == "player":
        if env.objects["player"].weapon == "Hammer":
            return 40.0
        elif env.objects["player"].weapon == "Spear":
            return 30.0
    return 0.0

def on_drop_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == "player":
        if env.objects["player"].weapon == "Punch":
            return -300
    return 0.0

def on_combo_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'player':
        return -25.0
    else:
        return 100.0

def on_attack_reward(env:WarehouseBrawl, agent:str):
    """This functions gets called if the player attacks, it assets whether the player is close to the enemy while attacking so it isn't swinging punches to thin air and stuff. """
    if agent != 'player': return 0

    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    difvector = np.array([opponent.body.position.x - player.body.position.x,
                                        opponent.body.position.y - player.body.position.y])

    # Prevent division by zero or extremely small values
    distance = np.linalg.norm(difvector)
    #calculate the distance between the player and the opp, then 
    if distance <= 1:
        val = ((distance-0.5)*6)**3 + 25
        val *= (1/opponent.stocks)*30
    elif 1< distance <= 1.6:
        val = 0 
    else:
        val = -math.log(distance-0.6)*5
    
    if env.objects["player"].weapon == "Punch":
        val -= 5
    
    return val   if val is not None else 0



def on_dodge_reward(env:WarehouseBrawl, agent:str):
    """This function gets called when the opponent attacks, if the agent manages to time a dodge right, they get a big reward. However, this also makes it so the agent doesn't just dodge when its far away."""
    if agent != 'opponent':return 0

    
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    

    difvector = np.array([opponent.body.position.x - player.body.position.x,
                                        opponent.body.position.y - player.body.position.y])

    # Prevent division by zero or extremely small values
    distance = np.linalg.norm(difvector)
    #calculate the distance between the player and the opp, then 
    if env.objects["opponent"].state == env.objects["opponent"].states_types['attack']:
        if distance <= 1:
            val = ((distance-0.5)*6)**3 + 35
        elif 1< distance <= 1.6:
            val = 10
    else: 
        if distance <= 1:
            val = (((distance-0.5)*6)**3 + 25)/3
        else:
            val = 0

        
    return val * (1/player.stocks)*30  if val is not None else 0

#TODO: a functinon that rewards for edgeguarding and platform camping. 
'''
Add your dictionary of RewardFunctions here using RewTerms
'''
def gen_reward_manager(): 
    reward_functions = {
        'target_height_reward': RewTerm(func=base_height_l2, weight=0.1, params={'target_height': -4, 'obj_name': 'player'}),
        'danger_zone_reward': RewTerm(func=danger_zone_reward, weight=80),
        'damage_interaction_reward': RewTerm(func=damage_interaction_reward, weight=3),
        'having_weapon_reward' : RewTerm(func = having_weapon_reward, weight = 2),
        #'head_to_middle_reward': RewTerm(func=head_to_middle_reward, weight=0.01),
        'move_to_opponent_reward': RewTerm(func=move_to_opponent_reward, weight=0.7),
        'holding_more_than_3_keys': RewTerm(func=holding_more_than_3_keys, weight=-0.1),
        'edge_penalty_reward' : RewTerm(func=edge_penalty_reward, weight = 1),
        'recovery_positioning_reward' : RewTerm(func=recovery_positioning_reward, weight =0.4)
        #'taunt_reward': RewTerm(func=in_state_reward, weight=0.2, params={'desired_state': TauntState}), #TODO: add emote after knowckout of opponent if safe etc etc
    }
    signal_subscriptions = {
        'on_win_reward': ('win_signal', RewTerm(func=on_win_reward, weight=10)),
        'on_knockout_reward': ('knockout_signal', RewTerm(func=on_knockout_reward, weight=10)),
        'on_combo_reward': ('hit_during_stun', RewTerm(func=on_combo_reward, weight=4)),
        'on_equip_reward': ('weapon_equip_signal', RewTerm(func=on_equip_reward, weight=6)),
        'on_drop_reward': ('weapon_drop_signal', RewTerm(func=on_drop_reward, weight=5)),
        'on_attack_reward':('attacked_signal', RewTerm(func = on_attack_reward , weight=10)),
        'on_dodge_reward':('dodged_signal', RewTerm(func = on_dodge_reward , weight=6)),
    }
    return RewardManager(reward_functions, signal_subscriptions)

# -------------------------------------------------------------------------
# ----------------------------- MAIN FUNCTION 
# -------------------------------------------------------------------------
'''
The main function runs training. You can change configurations such as the Agent type or opponent specifications here.
'''
if __name__ == '__main__':

    mode = input("single or multiple? >>>")
    assert mode in ["single","multiple"]

    if mode == "single":
        # Create agent
        # Start here if you want to train from scratch. e.g:
        my_agent = WoLFEMARecurrentPPOAgent()

        # Start here if you want to train from a specific timestep. e.g:
        # my_agent = RecurrentPPOAgent(file_path=r'checkpoints/training_1_26thlater/rl_model_13226977_steps.zip')

        # Reward manager
        reward_manager = gen_reward_manager()
        # Self-play settings
        selfplay_handler = SelfPlayRandom(
            partial(type(my_agent)), # Agent class and its keyword arguments
                                    # type(my_agent) = Agent class
        )

        # Set save settings here:
        save_handler = SaveHandler(
            agent=my_agent, # Agent to save
            save_freq=100_000, # Save frequency
            max_saved=100, # Maximum number of saved models
            save_path='checkpoints', # Save path
            run_name='training_28th',
            mode=SaveHandlerMode.FORCE # Save mode, FORCE or RESUME
        )

        # Set opponent settings here:
        opponent_specification = {
                        'self_play': (6, selfplay_handler),
                        'constant_agent': (10, partial(ConstantAgent)),
                        'based_agent': (3, partial(BasedAgent)),
                        'clockwise_agent': (4, partial(BasedAgent)),
                    }
        opponent_cfg = OpponentsCfg(opponents=opponent_specification)

        train(my_agent,
            reward_manager,
            save_handler,
            opponent_cfg,
            CameraResolution.LOW,
            train_timesteps=1_000_000_000,
            train_logging=TrainLogging.PLOT,
        )
    else:
        multiple_agent_classes= [RecurrentPPOAgent, EMARecurrentPPOAgent, WoLFEMARecurrentPPOAgent]
        for agentclass in multiple_agent_classes:
            try:
                        # Create agent
                # Start here if you want to train from scratch. e.g:
                my_agent = agentclass()

                # Start here if you want to train from a specific timestep. e.g:
                # my_agent = RecurrentPPOAgent(file_path=r'checkpoints/training_1_26thlater/rl_model_13226977_steps.zip')

                # Reward manager
                reward_manager = gen_reward_manager()
                # Self-play settings
                selfplay_handler = SelfPlayRandom(
                    partial(type(my_agent)), # Agent class and its keyword arguments
                                            # type(my_agent) = Agent class
                )

                # Set save settings here:
                save_handler = SaveHandler(
                    agent=my_agent, # Agent to save
                    save_freq=100_000, # Save frequency
                    max_saved=1000, # Maximum number of saved models
                    save_path='checkpoints', # Save path
                    run_name=f'{agentclass.__name__}_{__import__("datetime.datetime").today().strftime("%Y-%m-%d %H:%M:%S")}',
                    mode=SaveHandlerMode.FORCE # Save mode, FORCE or RESUME
                )

                # Set opponent settings here:
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
                    train_timesteps=2_000_000_000,
                    train_logging=TrainLogging.PLOT,
                )
            except Exception as e:
                print("Error occurred:", e)
                with open(f"errors/f'{agentclass.__name__}_{__import__('datetime.datetime').today().strftime('%Y-%m-%d %H:%M:%S')}",w) as file:
                    file.write(str(e))
