"""NEAT currency trading strategy trained on Sharpe-ratio fitness.

Evolves a feed-forward network (see ``config-feedforward``) that maps 15
scaled per-currency features to a single activation. Each training month the
currencies with complete data are ranked by activation; the top ``pf_size``
fraction is held long and the bottom fraction short. Fitness is the
annualized Sharpe ratio of the resulting monthly long-short excess returns
(signals lagged one month).

Inputs are read from ``Scaled/scaled_full.xlsx`` (produced from ``data.py``);
the winning network is pickled to ``winner_net.pckl``.

Usage::

    python neat_trader_sharpe.py
"""

from __future__ import print_function
import numpy as np
import pandas as pd
import os
import neat
import pickle
import visualize


# Fraction of the available cross-section held long (and short) each month
pf_size = 0.2

# Penalty fitness assigned when a genome produces no valid returns
FITNESS_PENALTY = -999.0


f = pd.ExcelFile('Scaled/scaled_full.xlsx')
lagged1_ret_sc = f.parse('Lagged1_Ret_Sc', index_col = 0, parse_dates = True)
lagged2_ret_sc = f.parse('Lagged2_Ret_Sc', index_col = 0, parse_dates = True)
lagged3_ret_sc = f.parse('Lagged3_Ret_Sc', index_col = 0, parse_dates = True)
lagged1_ret_ex_sc = f.parse('Lagged1_Ret_Ex_Sc', index_col = 0, parse_dates = True)
lagged2_ret_ex_sc = f.parse('Lagged2_Ret_Ex_Sc', index_col = 0, parse_dates = True)
lagged3_ret_ex_sc = f.parse('Lagged3_Ret_Ex_Sc', index_col = 0, parse_dates = True)
curr_vol_sc = f.parse('Curr_Vol_Sc', index_col = 0, parse_dates = True)
lagged1_vol_sc = f.parse('Lagged1_Vol_Sc', index_col = 0, parse_dates = True)
lagged2_vol_sc = f.parse('Lagged2_Vol_Sc', index_col = 0, parse_dates = True)
lagged3_vol_sc = f.parse('Lagged3_Vol_Sc', index_col = 0, parse_dates = True)
curr_imp_vol_sc = f.parse('Imp_Vol_Sc', index_col = 0, parse_dates = True)
curr_vrp_sc = f.parse('Curr_Vrp_Sc', index_col = 0, parse_dates = True)
curr_fw_prem_sc = f.parse('Curr_Fw_Prem_Sc', index_col = 0, parse_dates = True)
curr_spread_sc = f.parse('Curr_Spread_Sc', index_col = 0, parse_dates = True)
nber_sc = f.parse('NBER', index_col = 0, parse_dates = True)
ret_ex_long = f.parse('Ret_Ex_Long', index_col = 0, parse_dates = True)
ret_ex_short = f.parse('Ret_Ex_Short', index_col = 0, parse_dates = True)
split_date = lagged1_ret_sc.index[-60]  # last 5 years of data are the test set

# A (date, currency) cell is "complete" iff all 14 currency-level features are
# non-missing: NaN propagates through the sum, and NaN != NaN.
feature_sum = lagged1_ret_sc + lagged2_ret_sc + lagged3_ret_sc + \
lagged1_ret_ex_sc + lagged2_ret_ex_sc + lagged3_ret_ex_sc + curr_vol_sc + \
lagged1_vol_sc + lagged2_vol_sc + lagged3_vol_sc + curr_imp_vol_sc + \
curr_vrp_sc + curr_fw_prem_sc + curr_spread_sc
complete_test = (feature_sum == feature_sum)

train_dates = lagged1_ret_sc.index[lagged1_ret_sc.index < split_date]


def network_inputs(d, c):
    """Return the 15-element feature tuple for currency ``c`` at date ``d``.

    Order must match the ``node_names`` mapping in ``run`` and the
    ``num_inputs`` setting in ``config-feedforward``.
    """
    return (lagged1_ret_sc.loc[d,c], lagged2_ret_sc.loc[d,c],
            lagged3_ret_sc.loc[d,c], lagged1_ret_ex_sc.loc[d,c],
            lagged2_ret_ex_sc.loc[d,c], lagged3_ret_ex_sc.loc[d,c],
            curr_vol_sc.loc[d,c], lagged1_vol_sc.loc[d,c],
            lagged2_vol_sc.loc[d,c], lagged3_vol_sc.loc[d,c],
            curr_imp_vol_sc.loc[d,c], curr_vrp_sc.loc[d,c],
            curr_fw_prem_sc.loc[d,c], curr_spread_sc.loc[d,c],
            nber_sc.loc[d,'RECESS'])


def eval_genomes(genomes, config):
    """Assign each genome the annualized Sharpe ratio of its strategy.

    For every training month with at least 3 complete currencies, the genome's
    network scores each currency; the top/bottom ``pf_size`` fraction become
    long/short signals. Signals are lagged one month and combined with
    transaction-cost-adjusted excess returns (``ret_ex_long``/``ret_ex_short``).
    Genomes with no valid return observations receive ``FITNESS_PENALTY``.
    """
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        ml_signals = pd.DataFrame(0,
                                  index = train_dates,
                                  columns = lagged1_ret_sc.columns)
        for d in train_dates:
            complete_crncy = complete_test.columns[complete_test.loc[d,:]]
            if len(complete_crncy) >= 3:
                activations = pd.DataFrame(index = complete_crncy,
                                           columns = ['Activation'])
                for c in activations.index:
                    output = net.activate(network_inputs(d, c))
                    activations.loc[c,'Activation'] = output[0]

                sorted_act = np.sort(activations['Activation'])  # index 0 is smallest
                n_invest = np.int64(np.round(len(activations.index)*pf_size))
                lb = sorted_act[n_invest-1]
                ub = sorted_act[-n_invest]
                long = activations.index[activations['Activation'] >= ub]
                short = activations.index[activations['Activation'] <= lb]
                ml_signals.loc[d,[(c in long) for c in ml_signals.columns]] = 1
                ml_signals.loc[d,[(c in short) for c in ml_signals.columns]] = -1

        # Monthly long-short excess returns with a one-month signal lag
        returns = [(np.mean(ret_ex_long.loc[d,ml_signals.shift(1).loc[d,:] == 1]) -
                    np.mean(ret_ex_short.loc[d,ml_signals.shift(1).loc[d,:] == -1]))
                   for d in train_dates]
        returns = [x for x in returns if not np.isnan(x)]
        if len(returns) == 0:
            genome.fitness = FITNESS_PENALTY
            continue
        # Annualized Sharpe ratio (population std, ddof=0)
        sharpe = (12*np.mean(returns))/(np.sqrt(12)*np.std(returns))
        genome.fitness = FITNESS_PENALTY if np.isnan(sharpe) else sharpe


def run(config_file):
    """Run NEAT for up to 300 generations and return the winning network."""
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 300)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    node_names = {-1: 'Lagged1_Ret_Sc', -2: 'Lagged2_Ret_Sc',
                  -3: 'Lagged3_Ret_Sc', -4: 'Lagged1_Ret_Ex_Sc',
                  -5: 'Lagged2_Ret_Ex_Sc', -6: 'Lagged3_Ret_Ex_Sc',
                  -7: 'Curr_Vol_Sc', -8: 'Lagged1_Vol_Sc',
                  -9: 'Lagged2_Vol_Sc', -10: 'Lagged3_Vol_Sc',
                  -11: 'Imp_Vol_Sc', -12: 'Curr_Vrp_Sc',
                  -13: 'Curr_Fw_Prem_Sc', -14: 'Curr_Spread_Sc',
                  -15: 'NBER', 0: 'Invest'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    return winner_net


if __name__ == '__main__':
    # Resolve the config file relative to this script so the run does not
    # depend on the current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    best_net = run(config_path)
    with open('winner_net.pckl', 'wb') as out:
        pickle.dump(best_net, out)
