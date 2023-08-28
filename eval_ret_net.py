# -*- coding: utf-8 -*-
"""
Created on Thu May 17 17:20:17 2018

@author: gabys
"""

p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-266')

winner_net = neat.nn.FeedForwardNetwork.create(winner, config)