# # SUBMISSION: Agent
# This will be the Agent class we run in the 1v1. We've started you off with a functioning RL agent (`SB3Agent(Agent)`) and if-statement agent (`BasedAgent(Agent)`). Feel free to copy either to `SubmittedAgent(Agent)` then begin modifying.
# 
# Requirements:
# - Your submission **MUST** be of type `SubmittedAgent(Agent)`
# - Any instantiated classes **MUST** be defined within and below this code block.
# 
# Remember, your agent can be either machine learning, OR if-statement based. I've seen many successful agents arising purely from if-statements - give them a shot as well, if ML is too complicated at first!!
# 
# Also PLEASE ask us questions in the Discord server if any of the API is confusing. We'd be more than happy to clarify and get the team on the right track.
# Requirements:
# - **DO NOT** import any modules beyond the following code block. They will not be parsed and may cause your submission to fail validation.
# - Only write imports that have not been used above this code block
# - Only write imports that are from libraries listed here
# We're using PPO by default, but feel free to experiment with other Stable-Baselines 3 algorithms!

import os
import gdown
from typing import Optional
from environment.agent import Agent
from stable_baselines3 import PPO, A2C # Sample RL Algo imports
from sb3_contrib import RecurrentPPO # Importing an LSTM

class SubmittedAgent(Agent):
    '''
    Input the **file_path** to your agent here for submission!
    '''
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