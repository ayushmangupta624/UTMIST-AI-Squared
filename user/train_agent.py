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

    def _initialize(self) -> None:
        if self.file_path is None:
            policy_kwargs = {
                'activation_fn': nn.ReLU,
                'lstm_hidden_size': 1024,
                'net_arch': [dict(pi=[32, 32], vf=[32, 32])],
                'shared_lstm': False,
                'enable_critic_lstm': True,
                'share_features_extractor': True,
            }
            self.model = RecurrentPPO(
                "MlpLstmPolicy",
                self.env,
                verbose=1,
                n_steps= 30*90*20*30,
                batch_size=1024,
                n_epochs=60,
                gamma=0.90,
                ent_coef=0.03,
                policy_kwargs=policy_kwargs,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Add this line
                
            )
            del self.env
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
                (10, ['a']),
                (1, ['l']),
                (20, ['a']),
                (3, ['a', 'j']),
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
) -> float:
    """
    Computes the reward based on damage interactions between players.

    Modes:
    - ASYMMETRIC_OFFENSIVE (0): Reward is based only on damage dealt to the opponent
    - SYMMETRIC (1): Reward is based on both dealing damage to the opponent and avoiding damage
    - ASYMMETRIC_DEFENSIVE (2): Reward is based only on avoiding damage

    Args:
        env (WarehouseBrawl): The game environment
        mode (DamageRewardMode): Reward mode, one of DamageRewardMode

    Returns:
        float: The computed reward.
    """
    # Getting player and opponent from the enviornment
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    # Reward dependent on the mode
    damage_taken = player.damage_taken_this_frame
    damage_dealt = opponent.damage_taken_this_frame

    if mode == RewardMode.ASYMMETRIC_OFFENSIVE:
        reward = damage_dealt * 1.5
    elif mode == RewardMode.SYMMETRIC:
        reward = damage_dealt - damage_taken
    elif mode == RewardMode.ASYMMETRIC_DEFENSIVE:
        reward = -damage_taken
    else:
        raise ValueError(f"Invalid mode: {mode}")
    reward *= 50
    return reward / 140


# In[ ]:


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
    reward = -zone_penalty if player.body.position.y >= zone_height else 0.0

    return reward * env.dt * 3

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

    return reward * env.dt


def edge_penalty_reward( # gemini helped reward function
    env: WarehouseBrawl,
    edge_penalty: int = 10,
    edge_boundary: float = 10.67 / 2
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
    reward = -edge_penalty if is_off_edge else 0.0

    return reward * env.dt

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

def head_to_opponent(
    env: WarehouseBrawl,
) -> float:

    # Get player object from the environment
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    # Apply penalty if the player is in the danger zone
    multiplier = -2 if player.body.position.x > opponent.body.position.x else 2
    reward = multiplier * (player.body.position.x - player.prev_x)

    return reward * 10

def holding_more_than_3_keys(
    env: WarehouseBrawl,
) -> float:

    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is holding more than 3 keys
    a = player.cur_action
    if (a > 0.5).sum() > 3:
        return env.dt * 3
    return 0

def on_win_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'player':
        return 300
    else:
        return -200

def on_knockout_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'player':
        return -100
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
            return -50
    return 0.0

def on_combo_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'player':
        return -50.0
    else:
        return 100.0

'''
Add your dictionary of RewardFunctions here using RewTerms
'''
def gen_reward_manager(): 
    reward_functions = {
        #'target_height_reward': RewTerm(func=base_height_l2, weight=0.0, params={'target_height': -4, 'obj_name': 'player'}),
        'danger_zone_reward': RewTerm(func=danger_zone_reward, weight=0.8),
        'damage_interaction_reward': RewTerm(func=damage_interaction_reward, weight=3),
        #'head_to_middle_reward': RewTerm(func=head_to_middle_reward, weight=0.01),
        'head_to_opponent': RewTerm(func=head_to_opponent, weight=0.7),
        'penalize_attack_reward': RewTerm(func=in_state_reward, weight=-0.04, params={'desired_state': AttackState}),
        'holding_more_than_3_keys': RewTerm(func=holding_more_than_3_keys, weight=-0.03),
        #'taunt_reward': RewTerm(func=in_state_reward, weight=0.2, params={'desired_state': TauntState}), #TODO: add emote after knowckout of opponent if safe etc etc
    }
    signal_subscriptions = {
        'on_win_reward': ('win_signal', RewTerm(func=on_win_reward, weight=60)),
        'on_knockout_reward': ('knockout_signal', RewTerm(func=on_knockout_reward, weight=10)),
        'on_combo_reward': ('hit_during_stun', RewTerm(func=on_combo_reward, weight=5)),
        'on_equip_reward': ('weapon_equip_signal', RewTerm(func=on_equip_reward, weight=15)),
        'on_drop_reward': ('weapon_drop_signal', RewTerm(func=on_drop_reward, weight=20))
    }
    return RewardManager(reward_functions, signal_subscriptions)

# -------------------------------------------------------------------------
# ----------------------------- MAIN FUNCTION 
# -------------------------------------------------------------------------
'''
The main function runs training. You can change configurations such as the Agent type or opponent specifications here.
'''
if __name__ == '__main__':
    # Create agent
    # Start here if you want to train from scratch. e.g:
    my_agent = RecurrentPPOAgent()

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
                    'self_play': (10, selfplay_handler),
                    'constant_agent': (6, partial(ConstantAgent)),
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