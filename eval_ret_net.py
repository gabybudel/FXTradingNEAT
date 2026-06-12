# -*- coding: utf-8 -*-
"""Restore a NEAT checkpoint and rebuild the winner network (snippet).

NOT a standalone script: ``winner`` and ``config`` must already exist in the
session (e.g. from a ``neat_trader_*`` run), and the checkpoint filename
refers to a specific local training run.
"""

import neat

p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-266')

winner_net = neat.nn.FeedForwardNetwork.create(winner, config)