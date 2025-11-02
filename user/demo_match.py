from environment.environment import RenderMode, CameraResolution
from environment.agent import run_match, run_real_time_match
from user.train_agent import UserInputAgent,StallingAgent, ImprovedBasedAgent,BasedAgent, ConstantAgent, ClockworkAgent, SB3Agent, RecurrentPPOAgent #add anymore custom Agents (from train_agent.py) here as needed
from user.my_agent import SubmittedAgent
import pygame



pygame.init()

my_agent= BasedAgent()

#Input your file path here in SubmittedAgent if you are loading a model:
opponent = UserInputAgent()#SubmittedAgent(r"checkpoints\EMA_MORE_STEPS_2025-11-01-12-11-09\rl_model_9975892_steps.zip".replace('\\','/'))

match_time = 999999

# Run a single real-time match
run_real_time_match(    
    agent_1=opponent,
    agent_2=my_agent,
    max_timesteps=30 * match_time,  # Match time in frames (adjust as needed)
    resolution=CameraResolution.LOW,
    # video_path='tt_agent.mp4',
    
)