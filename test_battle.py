## Static Imports
import os
import importlib
import gym
import gym_everglades
import pdb
import sys

import numpy as np

from everglades_server import server
#from everglades-server import generate_map

## Input Variables
# Agent files must include a class of the same name with a 'get_action' function
# Do not include './' in file path
if len(sys.argv) > 1:
    agent0_file = 'agents/' + sys.argv[1]
else:
    agent0_file = 'agents/random_actions'
<<<<<<< HEAD
    
=======

>>>>>>> cc0a606dc9d28927d2d03363b69585e3a25fea87
if len(sys.argv) > 2:
    agent1_file = 'agents/' + sys.argv[2]
else:
    agent1_file = 'agents/random_actions'

map_name = "DemoMap.json"
<<<<<<< HEAD
    
config_dir = './config/'  
=======

config_dir = './config/'
>>>>>>> cc0a606dc9d28927d2d03363b69585e3a25fea87
map_file = config_dir + map_name
setup_file = config_dir + 'GameSetup.json'
unit_file = config_dir + 'UnitDefinitions.json'
output_dir = './game_telemetry/'

debug = 1

## Specific Imports
agent0_name, agent0_extension = os.path.splitext(agent0_file)
<<<<<<< HEAD
agent0_mod = importlib.import_module(agent0_name.replace('/','.'))
agent0_class = getattr(agent0_mod, os.path.basename(agent0_name))

agent1_name, agent1_extension = os.path.splitext(agent1_file)
agent1_mod = importlib.import_module(agent1_name.replace('/','.'))
=======
agent0_mod = importlib.import_module(agent0_name.replace('/', '.'))
agent0_class = getattr(agent0_mod, os.path.basename(agent0_name))

agent1_name, agent1_extension = os.path.splitext(agent1_file)
agent1_mod = importlib.import_module(agent1_name.replace('/', '.'))
>>>>>>> cc0a606dc9d28927d2d03363b69585e3a25fea87
agent1_class = getattr(agent1_mod, os.path.basename(agent1_name))

## Main Script
env = gym.make('everglades-v0')
players = {}
names = {}

players[0] = agent0_class(env.num_actions_per_turn, 0, map_name)
names[0] = agent0_class.__name__
players[1] = agent1_class(env.num_actions_per_turn, 1, map_name)
names[1] = agent1_class.__name__

<<<<<<< HEAD
observations = env.reset(
        players=players,
        config_dir = config_dir,
        map_file = map_file,
        unit_file = unit_file,
        output_dir = output_dir,
        pnames = names,
        debug = debug
)
=======
observations = env.reset(players=players,
                         config_dir=config_dir,
                         map_file=map_file,
                         unit_file=unit_file,
                         output_dir=output_dir,
                         pnames=names,
                         debug=debug)
>>>>>>> cc0a606dc9d28927d2d03363b69585e3a25fea87

actions = {}

## Game Loop
done = 0
while not done:
    if debug:
        env.game.debug_state()

    for pid in players:
<<<<<<< HEAD
        actions[pid] = players[pid].get_action( observations[pid] )
=======
        actions[pid] = players[pid].get_action(observations[pid])
>>>>>>> cc0a606dc9d28927d2d03363b69585e3a25fea87

    observations, reward, done, info = env.step(actions)
    pdb.set_trace()

print(reward)
