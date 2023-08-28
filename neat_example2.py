"""
NEAT Python Currency Trading Strategy
"""

from __future__ import print_function
import numpy as np
import pandas as pd
import os
import neat
import sys
import pickle
sys.path.append('C:\Python\External')
import visualize

fpath = 'C:/Users/gabys/Documents/Erasmus/MScFI Thesis/Data'
os.chdir(fpath)

pf_size = 0.2


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
split_date = lagged1_ret_sc.index[-60]  # last 5 years of data

temp = lagged1_ret_sc + lagged2_ret_sc + lagged3_ret_sc + \
lagged1_ret_ex_sc + lagged2_ret_ex_sc + lagged3_ret_ex_sc + curr_vol_sc + \
lagged1_vol_sc + lagged2_vol_sc + lagged3_vol_sc + curr_imp_vol_sc + \
curr_vrp_sc + curr_fw_prem_sc + curr_spread_sc
complete_test = (temp == temp)

inputs = tuple([(lagged1_ret_sc.loc[d,c],lagged2_ret_sc.loc[d,c],\
                 lagged3_ret_sc.loc[d,c]) \
for d in lagged1_ret_sc.index[lagged1_ret_sc.index < split_date] \
for c in lagged1_ret_sc.columns if complete_test.loc[d,c]])
train_dates = lagged1_ret_sc.index[lagged1_ret_sc.index < split_date]


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 0.0
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
                    xi = (lagged1_ret_sc.loc[d,c],lagged2_ret_sc.loc[d,c],\
                          lagged3_ret_sc.loc[d,c],lagged1_ret_ex_sc.loc[d,c],\
                          lagged2_ret_ex_sc.loc[d,c],lagged3_ret_ex_sc.loc[d,c],\
                          curr_vol_sc.loc[d,c],lagged1_vol_sc.loc[d,c],\
                          lagged2_vol_sc.loc[d,c],lagged3_vol_sc.loc[d,c],\
                          curr_imp_vol_sc.loc[d,c],curr_vrp_sc.loc[d,c],\
                          curr_fw_prem_sc.loc[d,c],curr_spread_sc.loc[d,c],nber_sc.loc[d,'RECESS'])
                    output = net.activate(xi)
                    activations.loc[c,'Activation'] = output[0]
                    
                sorted_act = np.sort(activations['Activation'])  # index 0 is smallest
                n_invest = np.int64(np.round(len(activations.index)*pf_size))
                lb = sorted_act[n_invest-1]
                ub = sorted_act[-n_invest]
                long = activations.index[activations['Activation'] >= ub]
                short = activations.index[activations['Activation'] <= lb]
                ml_signals.loc[d,[(c in long) for c in ml_signals.columns]] = 1
                ml_signals.loc[d,[(c in short) for c in ml_signals.columns]] = -1
            
        # Calculate return
        returns = [(np.mean(ret_ex_long.loc[d,ml_signals.shift(1).loc[d,:] == 1]) - \
         np.mean(ret_ex_short.loc[d,ml_signals.shift(1).loc[d,:] == -1])) for d in train_dates]
        returns = [x for x in returns if str(x) != 'nan']
        sharpe = ((12*np.mean(returns))/(np.sqrt(12)*np.std(returns)))
        if len(returns) == 0:
            print('Error: ' + str(sharpe))
        if str(sharpe) == 'nan':
            sharpe = -999
        genome.fitness += sharpe
#test = {c: lagged1_ret_sc.loc[lagged1_ret_sc.index[190],c] for c in lagged1_ret_sc.columns}

def run(config_file):
    # Load configuration.
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

    # Show output of the most fit genome against training data.
#    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
#    for xi, xo in zip(xor_inputs, xor_outputs):
#        output = winner_net.activate(xi)
#        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))
#
    node_names = {-1: 'Lagged1_Ret_Sc', -2: 'Lagged2_Ret_Sc', 
                  -3: 'Lagged3_Ret_Sc', -4: 'Lagged1_Ret_Ex_Sc', 
                  -5: 'Lagged2_Ret_Ex_Sc', -6: 'Lagged3_Ret_Ex_Sc', 
                  -7: 'Curr_Vol_Sc', -8: 'Lagged1_Vol_Sc', 
                  -9: 'Lagged2_Vol_Sc', -10: 'Lagged3_Vol_Sc', 
                  -11: 'Imp_Vol_Sc', -12: 'Curr_Vrp_Sc', 
                  -13: 'Curr_Fw_Prem_Sc', -14: 'Curr_Spread_Sc', 
                  -15: 'NBER', 0:'Invest'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

#    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
#    p.run(eval_genomes, 10)
    return winner_net

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    global best_net
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    best_net = run(config_path)
    f = open('winner_net.pckl', 'wb')
    pickle.dump(best_net, f)
    f.close()
    