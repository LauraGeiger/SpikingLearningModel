from neuron import h#, gui
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, CheckButtons, RadioButtons
import matplotlib.gridspec as gridspec
import numpy as np
import time
import random
from openpyxl import Workbook
from datetime import datetime
from itertools import groupby
from operator import itemgetter
# --- TODO --------------------------------------------------#
# determine dopamine level from rel SNc rate
# --- TODO --------------------------------------------------#


h.load_file("stdrun.hoc")
h.cvode_active(1)


#--- Global Variables ------------------------------------------------------------------------------------------------------------------------------------------------#
'''
cell_types_numbers = {'SNc': 5, 'MSNd': 5, 'MSNi': 5, 'GPe': 5, 'GPi': 5, 'Thal': 5}
cell_types = list(cell_types_numbers.keys())

# Tonic stimulation for all cells
stim_intervals = {
    'SNc'       : 1000 / 5,   # Hz
    'MSNd'      : 1000 / 7.4, # Hz (tonic baseline)
    'MSNi'      : 1000 / 3.5, # Hz (tonic baseline)
    'GPe'       : 1000 / 48,  # Hz
    'GPi'       : 1000 / 69,  # Hz
    'Thal'      : 1000 / 14,  # Hz
    'SNc_burst' : 1000 / 50,  # Hz
    'Cor'       : 1000 / 40   # Hz (cortical input stimulation)
}

stim_weights = {
    'SNc'       : 2,
    'MSNd'      : 2,
    'MSNi'      : 2,
    'GPe'       : 2,
    'GPi'       : 2,
    'Thal'      : 4,
    'SNc_burst' : 2,
    'Cor'       : 1.8
}
stims, syns, ncs = {}, {}, {}

# Define connection specifications
connection_specs = [# pre_group, post_group, label, e_rev, weight, tau, delay
    ('SNc', 'MSNd', 'SNc_to_MSNd',   0, 0,    10, 1),   # excitatory
    ('SNc', 'MSNi', 'SNc_to_MSNi', -85, 0,    10, 1),   # inhibitory
    ('MSNd', 'GPi', 'MSNd_to_GPi', -85, 0.4,  10, 1),   # inhibitory
    ('MSNi', 'GPe', 'MSNi_to_GPe', -85, 0.2,  10, 1),   # inhibitory
    ('GPe',  'GPi',  'GPe_to_GPi', -85, 0.04, 10, 1),   # inhibitory
    ('GPi', 'Thal', 'GPi_to_Thal', -85, 0.8,  10, 1)    # inhibitory
]

N_actions = 3
paused = False
target_actions = []
noise = 0
selection_threshold = 0.8
n_spikes_SNc_burst = 5
learning_rate = 0.05
reward_times = []
activation_times = []
expected_reward_over_time = {}
activation_over_time = {}
target_activation_over_time = {}
cortical_input_dur_rel = [1] * N_actions
expected_reward_value = 0.2
reward_over_time = {}
dopamine_over_time = {}
weight_cell_types = ['MSNd', 'MSNi']
weight_times = []
weights_over_time = {(ct, a, i): [] 
                     for ct in weight_cell_types
                     for a in range(N_actions) 
                     for i in range(cell_types_numbers[ct])
                     }
apc_refs = []


# Plotting setup
plot_interval = 1000  # ms
bin_width_firing_rate = plot_interval / 10  # ms
simulation_stop_time = 20000 # ms

buttons = {}
rates = {}
rates_rel = {}
'''

output = False # Set to True (print values in the terminal) or False (no printing)

#--- BasalGanglia ------------------------------------------------------------------------------------------------------------------------------------------------#

class BasalGanglia:

    def __init__(self, name, N_actions=3):
        self.name = name
        self.N_actions = N_actions

        self.cell_types_numbers = {'SNc': 5, 'MSNd': 5, 'MSNi': 5, 'GPe': 5, 'GPi': 5, 'Thal': 5}
        self.cell_types = list(self.cell_types_numbers.keys())

        self.stim_intervals = {
            'SNc'       : 1000 / 5,   # Hz
            'MSNd'      : 1000 / 7.4, # Hz (tonic baseline)
            'MSNi'      : 1000 / 3.5, # Hz (tonic baseline)
            'GPe'       : 1000 / 48,  # Hz
            'GPi'       : 1000 / 69,  # Hz
            'Thal'      : 1000 / 14,  # Hz
            'SNc_burst' : 1000 / 50,  # Hz
            'Cor'       : 1000 / 40   # Hz (cortical input stimulation)
        }
        self.stim_weights = {
            'SNc'       : 2,
            'MSNd'      : 2,
            'MSNi'      : 2,
            'GPe'       : 2,
            'GPi'       : 2,
            'Thal'      : 4,
            'SNc_burst' : 2,
            'Cor'       : 1.8
        }

        self.stims = {}
        self.syns = {}
        self.ncs = {}

        # Define connection specifications
        self.connection_specs = [# pre_group, post_group, label, e_rev, weight, tau, delay
            ('SNc', 'MSNd', 'SNc_to_MSNd',   0, 0,    10, 1),   # excitatory
            ('SNc', 'MSNi', 'SNc_to_MSNi', -85, 0,    10, 1),   # inhibitory
            ('MSNd', 'GPi', 'MSNd_to_GPi', -85, 0.4,  10, 1),   # inhibitory
            ('MSNi', 'GPe', 'MSNi_to_GPe', -85, 0.2,  10, 1),   # inhibitory
            ('GPe',  'GPi',  'GPe_to_GPi', -85, 0.04, 10, 1),   # inhibitory
            ('GPi', 'Thal', 'GPi_to_Thal', -85, 0.8,  10, 1)    # inhibitory
        ]

        self.paused = False
        self.target_actions = []
        self.noise = 0
        self.selection_threshold = 0.8
        self.n_spikes_SNc_burst = 5
        self.learning_rate = 0.05
        self.reward_times = []
        self.activation_times = []
        self.expected_reward_over_time = {}
        self.activation_over_time = {}
        self.target_activation_over_time = {}
        self.cortical_input_dur_rel = [1] * self.N_actions
        self.expected_reward_value = 0.2
        self.reward_over_time = {}
        self.dopamine_over_time = {}
        self.weight_cell_types = ['MSNd', 'MSNi']
        self.weight_times = []
        self.weights_over_time = {(ct, a, i): [] 
                            for ct in self.weight_cell_types
                            for a in range(self.N_actions) 
                            for i in range(self.cell_types_numbers[ct])
                            }
        self.apc_refs = []

        self.plot_interval = 1000  # ms
        self.bin_width_firing_rate = self.plot_interval / 10  # ms
        self.simulation_stop_time = 20000 # ms

        self.buttons = {}
        self.rates = {}
        self.rates_rel = {}

        self._init_cells()
        self._init_spike_detectors()
        self._init_stimuli()
        self._init_connections()
        self._init_connection_stimuli()
        self._init_reward()
        self._init_activation()
        self._init_recording()
        self._init_plotting()

    def _init_cells(self):
        # Create neuron populations
        self.cells = {
            cell_type: [
                [create_cell(f'{cell_type}_{a}_{i}') for i in range(self.cell_types_numbers[cell_type])]
                for a in range(self.N_actions)
            ] 
            for cell_type in self.cell_types
        }
    
    def _init_spike_detectors(self):
        # Spike detectors and vectors
        self.spike_times = {
            cell_type: [
                [h.Vector() for _ in range(self.cell_types_numbers[cell_type])] 
                for _ in range(self.N_actions)
            ] for cell_type in self.cell_types
        }

        for cell_type in self.cell_types:
            for a in range(self.N_actions):
                for i, cell in enumerate(self.cells[cell_type][a]):
                    apc = h.APCount(cell(0.5))
                    apc.record(self.spike_times[cell_type][a][i])
                    self.apc_refs.append(apc)

    def _init_stimuli(self):
        for cell_type in self.cell_types:
            self.stims[cell_type], self.syns[cell_type], self.ncs[cell_type] = [], [], []
            for a in range(self.N_actions):
                for i, cell in enumerate(self.cells[cell_type][a]):
                    if 'SNc' in str(cell):
                        offset = self.stim_intervals['SNc']/2
                    else:
                        offset = i*self.stim_intervals[cell_type]/self.cell_types_numbers[cell_type]
                    stim, syn, nc = create_stim(cell, start=offset, interval=self.stim_intervals[cell_type], weight=self.stim_weights[cell_type], noise=self.noise)
                    self.stims[cell_type].append(stim)
                    self.syns[cell_type].append(syn)
                    self.ncs[cell_type].append(nc)

    def _init_connections(self):
        # Create connections based on the specification
        for pre_group, post_group, label, e_rev, weight, tau, delay in self.connection_specs:
            self.ncs.update({label: []}) # Additional connections dict to store NetCons
            self.syns.update({label: []}) # Additional connections dict to store ExpSyns
            for a in range(self.N_actions):
                for pre_cell in self.cells[pre_group][a]:
                    for post_cell in self.cells[post_group][a]:
                        syn = h.ExpSyn(post_cell(0.5))
                        syn.e = e_rev
                        syn.tau = tau
                        nc = h.NetCon(pre_cell(0.5)._ref_v, syn, sec=pre_cell)
                        nc.weight[0] = random.uniform(0.9*weight, 1.1*weight)
                        #nc.weight[0] = weight
                        nc.delay = delay
                        self.syns[label].append(syn)
                        self.ncs[label].append(nc)

    def _init_connection_stimuli(self):
        # Additional stimuli for dopamine bursts
        self.stims.update({'SNc_burst': []})
        self.syns.update({'SNc_burst': []})
        self.ncs.update({'SNc_burst': []})

        for a in range(self.N_actions):
            for cell in self.cells['SNc'][a]:
                stim, syn, nc = create_stim(cell, start=0, interval=self.stim_intervals['SNc_burst'], weight=self.stim_weights['SNc_burst'], noise=self.noise)
                self.stims['SNc_burst'].append(stim)
                self.syns['SNc_burst'].append(syn)
                self.ncs['SNc_burst'].append(nc)
        for nc in self.ncs['SNc_burst']:
            nc.active(False)

        # Additional stimuli for cortical input
        self.weight_times.append(0)
        for ct in self.weight_cell_types:
            self.stims.update({f'Cor_{ct}': []})
            self.syns.update({f'Cor_{ct}': []})
            self.ncs.update({f'Cor_{ct}': []})
            for a in range(self.N_actions):
                for k, cell in enumerate(self.cells[ct][a]):
                    stim, syn, nc = create_stim(cell, start=0, interval=self.stim_intervals['Cor'], weight=self.stim_weights['Cor'], noise=self.noise)
                    self.stims[f'Cor_{ct}'].append(stim)
                    self.syns[f'Cor_{ct}'].append(syn)
                    self.ncs[f'Cor_{ct}'].append(nc)
                    self.weights_over_time[(ct, a, k)].append(self.stim_weights['Cor'])
            for nc in self.ncs[f'Cor_{ct}']:
                nc.active(False)

    def _init_reward(self):
        # Reward initialization
        self.reward_times.append(0)
        for action in range(self.N_actions):
            for target in [True, False]:
                input_key = f"{action}{target}"
                self.expected_reward_over_time[input_key] = [self.expected_reward_value]
            self.reward_over_time[action] = [0]
            self.dopamine_over_time[action] = [0]

    def _init_activation(self):
        # Activation initialization
        self.activation_times.append(0)
        for action in range(self.N_actions):
            self.activation_over_time[action] = [0]
            self.target_activation_over_time[action] = [0]

    def _init_recording(self):
        # Recording
        self.recordings = {ct: [[h.Vector().record(cell(0.5)._ref_v) for cell in self.cells[ct][a]] for a in range(self.N_actions)] for ct in self.cell_types}
        self.t_vec = h.Vector().record(h._ref_t)

    def _init_plotting(self):
        #global fig
        plt.ion()
        self.fig = plt.figure(figsize=(13, 8))
        self.rows = 5
        self.gs = gridspec.GridSpec(self.rows, 1 + self.N_actions, height_ratios = [2] + [4] * (self.rows - 1), width_ratios = [0.3] + [1] * self.N_actions)
        self.axs = [[self.fig.add_subplot(self.gs[i, j]) for j in range(1 + self.N_actions)] for i in range(self.rows)]
        [self.row_control_upper, self.row_potential, self.row_spike, self.row_weights, self.row_reward] = list(range(self.rows))
        self.col_left_control = 0
        self.col_potential = 1
        self.col_spike = 1
        self.col_weights = 1
        self.col_reward = 1
        

        # Deactivate axis for non-plot axes
        for row in range(self.rows):
            if row == 0: # first row are control panels
                for ax in self.axs[row]:
                    ax.set_axis_off() # deactivate axis
            else:
                self.axs[row][0].set_axis_off() # left column is also control panel
        
        self._init_membrane_potential_plot()
        self._init_spike_plot()
        self._init_weight_plot()
        self._init_reward_plot()
        self._init_left_control_panel()
        self._init_upper_control_panel()

        plt.show()
        plt.tight_layout()

    def _init_membrane_potential_plot(self):
        # Membrane potential plot
        self.axs[self.row_potential][self.col_potential].set_ylabel('Membrane potential (mV)')
        self.mem_lines = {ct: [] for ct in self.cell_types}

        for i, ch in enumerate(range(self.N_actions)):
            for j, ct in enumerate(self.cell_types):
                avg_line, = self.axs[self.row_potential][self.col_potential+i].plot([], [], f'C{j}', label=ct)
                self.mem_lines[ct].append(avg_line)

            self.axs[self.row_potential][self.col_potential+i].set_title(f'Action {ch}')
            #axs[row_potential][col_potential+i].legend(loc='upper right')
            self.axs[self.row_potential][self.col_potential+i].set_xlim(0, self.plot_interval)
            self.axs[self.row_potential][self.col_potential+i].set_ylim(-85, 65)
        self.axs[self.row_potential][-1].legend(loc='upper right')

    def _init_spike_plot(self):
        # Spike raster plot and rate lines
        self.raster_lines = {ct: [[] for _ in range(self.N_actions)] for ct in self.cell_types}
        self.rate_lines = {ct: [] for ct in self.cell_types}
        self.rate_lines.update({'Threshold': []})
        self.axs[self.row_spike][self.col_spike].set_ylabel('Spike raster')

        for i, ch in enumerate(range(self.N_actions)):
            for j, ct in enumerate(self.cell_types):
                self.raster_lines[ct][i] = []
                for k in range(self.cell_types_numbers[ct]):
                    line, = self.axs[self.row_spike][self.col_spike+i].plot([], [], f'C{j}.', markersize=3)
                    self.raster_lines[ct][i].append(line)

                rate_line, = self.axs[self.row_spike][self.col_spike+i].step([], [], f'C{j}')#, label=f'{ct} rate')
                self.rate_lines[ct].append(rate_line)
            self.axs[self.row_spike][self.col_spike+i].plot([], [], color='black', label=f'Relative rate')

            self.total_cells = sum(self.cell_types_numbers[ct] for ct in self.cell_types)
            
            y_base_thal = self.cell_types_numbers['Thal'] * self.selection_threshold
            self.axs[self.row_spike][self.col_spike+i].axhline(y=y_base_thal, color='black', linestyle='dotted', label=f'Activation threshold')
            
            y_max = self.total_cells + 1.5
            self.axs[self.row_spike][self.col_spike+i].set_ylim(0.5, y_max)
            yticks = []
            cumulative = 0
            for ct in self.cell_types:
                mid = y_max - (cumulative + (self.cell_types_numbers[ct]+1) / 2)
                yticks.append(mid)
                cumulative += self.cell_types_numbers[ct] 
            self.axs[self.row_spike][self.col_spike+i].set_yticks(yticks)
            self.axs[self.row_spike][self.col_spike+i].set_yticklabels(self.cell_types)
            self.axs[self.row_spike][self.col_spike+i].set_xlim(0, self.plot_interval)
            #self.axs[row_spike][col_spike+i].legend(loc='upper right')
        self.axs[self.row_spike][-1].legend(loc='upper right')

    def _init_weight_plot(self):
        # Weight plot
        self.axs[self.row_weights][self.col_weights].set_ylabel('Cortical input weight')
        self.weight_lines = {ct: [[] for _ in range(self.N_actions)] for ct in self.weight_cell_types}

        for i, ch in enumerate(range(self.N_actions)):
            for j, ct in enumerate(self.cell_types):
                if ct in self.weight_cell_types:
                    self.weight_lines[ct][i] = []
                    for k in range(self.cell_types_numbers[ct]):
                        label = ct if k == 0 else None  # Only label the first line
                        line, = self.axs[self.row_weights][self.col_weights+i].step([], [], f'C{j}', label=label)
                        self.weight_lines[ct][i].append(line)

            #axs[row_weights][col_weights+i].legend(loc='upper right')
            self.axs[self.row_weights][self.col_weights+i].set_xlim(0, self.plot_interval)
            self.axs[self.row_weights][self.col_weights+i].set_ylim(0, 3)
        self.axs[self.row_weights][-1].legend(loc='upper right')

    def _init_reward_plot(self):
        # Reward plot
        self.axs[self.row_reward][self.col_reward].set_ylabel('Dopamine')
        self.expected_reward_lines = [[] for _ in range(self.N_actions)]
        self.reward_lines = [[] for _ in range(self.N_actions)]
        self.dopamine_lines = [[] for _ in range(self.N_actions)]
        self.activation_lines = [[] for _ in range(self.N_actions)]
        self.target_activation_lines = [[] for _ in range(self.N_actions)]

        for i, ch in enumerate(range(self.N_actions)):
            expected_reward_line, = self.axs[self.row_reward][self.col_reward+i].plot([], [], 'C9', label='Expected reward')
            reward_line, = self.axs[self.row_reward][self.col_reward+i].step([], [], 'C8', label='Reward')
            dopamine_line, = self.axs[self.row_reward][self.col_reward+i].plot([], [], 'C6', label='Dopamine')
            activation_line, = self.axs[self.row_reward][self.col_reward+i].step([], [], 'C7', linestyle='dashed', label='Activation time')
            target_activation_line,  = self.axs[self.row_reward][self.col_reward+i].step([], [], color='blue', linestyle='dotted', label=f'Target act. time')
            self.expected_reward_lines[i].append(expected_reward_line)
            self.reward_lines[i].append(reward_line)
            self.dopamine_lines[i].append(dopamine_line)
            self.activation_lines[i].append(activation_line)
            self.target_activation_lines[i].append(target_activation_line)
            
            #self.axs[row_reward][col_reward+i].legend(loc='upper right')
            self.axs[self.row_reward][self.col_reward+i].set_xlabel('Simulation time (ms)')
            self.axs[self.row_reward][self.col_reward+i].set_xlim(0, self.plot_interval)
            self.axs[self.row_reward][self.col_reward+i].set_ylim(-1.1, 1.1)
        self.axs[self.row_reward][-1].legend(loc='upper right')

        for a in range(self.N_actions):
            self.target_activation_lines[a][0].set_visible(False)

    def _init_left_control_panel(self):
        #--- Left control panel ---#
        i = 0
        ax_pause = self.axs[i][self.col_left_control].inset_axes([0,0,1,1])
        self.buttons['pause'] = Button(ax_pause, 'Pause')
        self.buttons['pause'].on_clicked(self.toggle_pause)
        
        i += 1
        ax_noise = self.axs[i][self.col_left_control].inset_axes([0.5,0,0.5,0.45])
        self.buttons['noise_slider'] = Slider(ax_noise, 'Noise', 0, 1, valinit=self.noise, valstep=0.1)
        self.buttons['noise_slider'].on_changed(self.update_stim)

    def _init_upper_control_panel(self):
        #--- Upper control panel ---#
        for a in range(self.N_actions):
            # Target button
            ax_target = self.axs[self.row_control_upper][self.col_potential+a].inset_axes([0.0,0,0.3,1])
            self.buttons[f'target_{a}'] = Button(ax_target, 'Set as\nTarget')
            self.buttons[f'target_{a}'].on_clicked(lambda event, a=a: self.toggle_target_action(event=event, action=a))
            ax_cor_dur = self.axs[self.row_control_upper][self.col_potential+a].inset_axes([0.7,0,0.3,0.3])
            self.buttons[f'cor_dur_slider{a}'] = Slider(ax_cor_dur, 'Input Dur', 0.2, 1, valinit=self.cortical_input_dur_rel[a], valstep=0.2)
            self.buttons[f'cor_dur_slider{a}'].on_changed(lambda val, a=a: self.update_cor_dur(val=val, action=a))

    def update_stimulus_activation(self, cell, stimulus, actions=None, active=True):
        if actions == None:
            actions = list(range(self.N_actions))
        i = 0
        for a in range(self.N_actions):
            for _ in self.cells[cell][a]:
                if a in actions:
                    self.ncs[stimulus][i].active(active)
                i += 1

    def SNc_dip(self, event=None, actions=None):
        self.update_stimulus_activation(cell='SNc', stimulus='SNc', actions=actions, active=False) # stop SNc tonic stimulus
        h.cvode.event(h.t + self.stim_intervals['SNc'], lambda actions=actions: self.update_stimulus_activation(cell='SNc', stimulus='SNc', actions=actions, active=True))  # start SNc tonic stimulus

    def SNc_burst(self, event=None, actions=None, n_spikes=None):
        self.update_stimulus_activation(cell='SNc', stimulus='SNc', actions=actions, active=False) # stop SNc tonic stimulus
        self.update_stimulus_activation(cell='SNc', stimulus='SNc_burst', actions=actions, active=True) # start SNc burst stimulus
        if n_spikes == None:
            n_spikes = self.n_spikes_SNc_burst
        delay = self.stim_intervals['SNc_burst'] * n_spikes
        h.cvode.event(h.t + delay, lambda actions=actions: self.update_stimulus_activation(cell='SNc', stimulus='SNc_burst', actions=actions, active=False))  # stop SNc burst stimulus
        h.cvode.event(h.t + delay, lambda actions=actions: self.update_stimulus_activation(cell='SNc', stimulus='SNc', actions=actions, active=True))  # start SNc tonic stimulus

    def analyse_firing_rate(self, cell, window=None, average=True):
        """Returns a list of firing rates (Hz) for each action's cell population."""
        current_time = h.t
        rates_avg = []
        rates = []
        if window == None:
            window = self.bin_width_firing_rate
        for a in range(self.N_actions):
            spikes_avg = 0
            spikes = []
            for i in range(self.cell_types_numbers[cell]):
                spike_vec = self.spike_times[cell][a][i]
                # Count spikes in the last `window` ms
                recent_spikes = [t for t in spike_vec if current_time - window <= t <= current_time]
                if average:
                    spikes_avg += len(recent_spikes)
                else:
                    spikes.append(len(recent_spikes))
            if average:
                rate_avg = spikes_avg / (self.cell_types_numbers[cell] * (window / 1000.0))  # spikes/sec per neuron
                rates_avg.append(rate_avg)
            else:
                rate = [spike / (window / 1000.0) for spike in spikes]
                rates.append(rate)

        max_rate = 1000.0 / self.stim_intervals[cell]
        
        if average:
            rates_rel_avg = [rate_avg / max_rate for rate_avg in rates_avg]
            return rates_avg, rates_rel_avg
        else:
            rates_rel = [[r / max_rate for r in rate] for rate in rates]
            return rates, rates_rel

    def toggle_pause(self, event=None):
        self.paused = not self.paused
        self.buttons['pause'].label.set_text('Continue' if self.paused else 'Pause')
        if not self.paused:
            self.buttons['pause'].color = '0.85'
        else:
            self.buttons['pause'].color = 'c'
        self.fig.canvas.draw_idle()

    def toggle_target_action(self, event=None, action=None):
        if action != None:
            if action in self.target_actions:
                self.target_actions.remove(action)
                self.buttons[f'target_{action}'].label.set_text('Set as\nTarget')
                self.buttons[f'target_{action}'].color = '0.85'

                #target_activation_lines[action][0].set_visible(False)

                h.cvode.event(h.t + 1, lambda: self.update_stimulus_activation(cell='MSNd', stimulus=f'Cor_MSNd', actions=[action], active=False)) # stop cortical input stimulus for that action
                h.cvode.event(h.t + 1, lambda: self.update_stimulus_activation(cell='MSNi', stimulus=f'Cor_MSNi', actions=[action], active=False)) # stop cortical input stimulus for that action
                
            else:
                self.target_actions.append(action)
                self.buttons[f'target_{action}'].label.set_text('Target')
                self.buttons[f'target_{action}'].color = 'y'

                self.target_activation_lines[action][0].set_visible(True)
            self.fig.canvas.draw_idle()

    def update_stim(self, val):
        self.noise = val
        for ct in self.cell_types:
            for stim in self.stims[ct]:
                stim.noise = self.noise

    def update_cor_dur(self, val, action):
        self.cortical_input_dur_rel
        self.cortical_input_dur_rel[action] = val

#--- Functions ------------------------------------------------------------------------------------------------------------------------------------------------#

def create_cell(name):
    sec = h.Section(name=name)
    sec.insert('hh')
    return sec

# Stimulus helper
def create_stim(cell, start=0, number=1e9, interval=10, weight=2, noise=0):
    stim = h.NetStim()
    stim.start = start
    stim.number = number
    stim.interval = interval
    stim.noise = noise
    syn = h.ExpSyn(cell(0.5))
    syn.e = 0
    nc = h.NetCon(stim, syn)
    nc.weight[0] = weight
    nc.delay = 1
    return stim, syn, nc
'''
def SNc_dip(event=None, actions=None):
    update_stimulus_activation(cell='SNc', stimulus='SNc', actions=actions, active=False) # stop SNc tonic stimulus
    h.cvode.event(h.t + stim_intervals['SNc'], lambda actions=actions: update_stimulus_activation(cell='SNc', stimulus='SNc', actions=actions, active=True))  # start SNc tonic stimulus

def update_stimulus_activation(cell, stimulus, actions=None, active=True):
    if actions == None:
        actions = list(range(N_actions))
    i = 0
    for a in range(N_actions):
        for _ in cells[cell][a]:
            if a in actions:
                ncs[stimulus][i].active(active)
            i += 1

def SNc_burst(event=None, actions=None, n_spikes=None):
        update_stimulus_activation(cell='SNc', stimulus='SNc', actions=actions, active=False) # stop SNc tonic stimulus
        update_stimulus_activation(cell='SNc', stimulus='SNc_burst', actions=actions, active=True) # start SNc burst stimulus
        if n_spikes == None:
            n_spikes = n_spikes_SNc_burst
        delay = stim_intervals['SNc_burst'] * n_spikes
        h.cvode.event(h.t + delay, lambda actions=actions: update_stimulus_activation(cell='SNc', stimulus='SNc_burst', actions=actions, active=False))  # stop SNc burst stimulus
        h.cvode.event(h.t + delay, lambda actions=actions: update_stimulus_activation(cell='SNc', stimulus='SNc', actions=actions, active=True))  # start SNc tonic stimulus

def analyse_firing_rate(cell, window=None, average=True):
    """Returns a list of firing rates (Hz) for each action's cell population."""
    current_time = h.t
    rates_avg = []
    rates = []
    if window == None:
        window = bin_width_firing_rate
    for a in range(N_actions):
        spikes_avg = 0
        spikes = []
        for i in range(cell_types_numbers[cell]):
            spike_vec = spike_times[cell][a][i]
            # Count spikes in the last `window` ms
            recent_spikes = [t for t in spike_vec if current_time - window <= t <= current_time]
            if average:
                spikes_avg += len(recent_spikes)
            else:
                spikes.append(len(recent_spikes))
        if average:
            rate_avg = spikes_avg / (cell_types_numbers[cell] * (window / 1000.0))  # spikes/sec per neuron
            rates_avg.append(rate_avg)
        else:
            rate = [spike / (window / 1000.0) for spike in spikes]
            rates.append(rate)

    max_rate = 1000.0 / stim_intervals[cell]
    
    if average:
        rates_rel_avg = [rate_avg / max_rate for rate_avg in rates_avg]
        return rates_avg, rates_rel_avg
    else:
        rates_rel = [[r / max_rate for r in rate] for rate in rates]
        return rates, rates_rel

def toggle_pause(event=None):
    global paused
    paused = not paused
    buttons['pause'].label.set_text('Continue' if paused else 'Pause')
    if not paused:
        buttons['pause'].color = '0.85'
    else:
        buttons['pause'].color = 'c'
    fig.canvas.draw_idle()

def toggle_target_action(event=None, action=None):
    if action != None:
        if action in target_actions:
            target_actions.remove(action)
            buttons[f'target_{action}'].label.set_text('Set as\nTarget')
            buttons[f'target_{action}'].color = '0.85'

            #target_activation_lines[action][0].set_visible(False)

            h.cvode.event(h.t + 1, lambda: update_stimulus_activation(cell='MSNd', stimulus=f'Cor_MSNd', actions=[action], active=False)) # stop cortical input stimulus for that action
            h.cvode.event(h.t + 1, lambda: update_stimulus_activation(cell='MSNi', stimulus=f'Cor_MSNi', actions=[action], active=False)) # stop cortical input stimulus for that action
            
        else:
            target_actions.append(action)
            buttons[f'target_{action}'].label.set_text('Target')
            buttons[f'target_{action}'].color = 'y'

            target_activation_lines[action][0].set_visible(True)
        fig.canvas.draw_idle()

def update_stim(val):
    global noise
    noise = val
    for ct in cell_types:
        for stim in stims[ct]:
            stim.noise = noise

def update_cor_dur(val, action):
    global cortical_input_dur_rel
    cortical_input_dur_rel[action] = val
'''
#--- Network ------------------------------------------------------------------------------------------------------------------------------------------#
'''
# Create neuron populations
cells = {
    cell_type: [
        [create_cell(f'{cell_type}_{a}_{i}') for i in range(cell_types_numbers[cell_type])]
        for a in range(N_actions)
    ] 
    for cell_type in cell_types
}

# Spike detectors and vectors
spike_times = {
    cell_type: [
        [h.Vector() for _ in range(cell_types_numbers[cell_type])] 
        for _ in range(N_actions)
    ] for cell_type in cell_types
}

for cell_type in cell_types:
    for a in range(N_actions):
        for i, cell in enumerate(cells[cell_type][a]):
            apc = h.APCount(cell(0.5))
            apc.record(spike_times[cell_type][a][i])
            apc_refs.append(apc)

for cell_type in cell_types:
    stims[cell_type], syns[cell_type], ncs[cell_type] = [], [], []
    for a in range(N_actions):
        for i, cell in enumerate(cells[cell_type][a]):
            if 'SNc' in str(cell):
                offset = stim_intervals['SNc']/2
            else:
                offset = i*stim_intervals[cell_type]/cell_types_numbers[cell_type]
            stim, syn, nc = create_stim(cell, start=offset, interval=stim_intervals[cell_type], weight=stim_weights[cell_type])
            stims[cell_type].append(stim)
            syns[cell_type].append(syn)
            ncs[cell_type].append(nc)

# Create connections based on the specification
for pre_group, post_group, label, e_rev, weight, tau, delay in connection_specs:
    ncs.update({label: []}) # Additional connections dict to store NetCons
    syns.update({label: []}) # Additional connections dict to store ExpSyns
    for a in range(N_actions):
        for pre_cell in cells[pre_group][a]:
            for post_cell in cells[post_group][a]:
                syn = h.ExpSyn(post_cell(0.5))
                syn.e = e_rev
                syn.tau = tau
                nc = h.NetCon(pre_cell(0.5)._ref_v, syn, sec=pre_cell)
                nc.weight[0] = random.uniform(0.9*weight, 1.1*weight)
                #nc.weight[0] = weight
                nc.delay = delay
                syns[label].append(syn)
                ncs[label].append(nc)

# Additional stimuli for dopamine bursts
stims.update({'SNc_burst': []})
syns.update({'SNc_burst': []})
ncs.update({'SNc_burst': []})

for a in range(N_actions):
    for cell in cells['SNc'][a]:
        stim, syn, nc = create_stim(cell, start=0, interval=stim_intervals['SNc_burst'], weight=stim_weights['SNc_burst'])
        stims['SNc_burst'].append(stim)
        syns['SNc_burst'].append(syn)
        ncs['SNc_burst'].append(nc)
for nc in ncs['SNc_burst']:
    nc.active(False)

# Additional stimuli for cortical input
weight_times.append(0)
for ct in weight_cell_types:
    stims.update({f'Cor_{ct}': []})
    syns.update({f'Cor_{ct}': []})
    ncs.update({f'Cor_{ct}': []})
    for a in range(N_actions):
        for k, cell in enumerate(cells[ct][a]):
            stim, syn, nc = create_stim(cell, start=0, interval=stim_intervals['Cor'], weight=stim_weights['Cor'])
            stims[f'Cor_{ct}'].append(stim)
            syns[f'Cor_{ct}'].append(syn)
            ncs[f'Cor_{ct}'].append(nc)
            weights_over_time[(ct, a, k)].append(stim_weights['Cor'])
    for nc in ncs[f'Cor_{ct}']:
        nc.active(False)

# Reward initialization
reward_times.append(0)
for action in range(N_actions):
    for target in [True, False]:
        input_key = f"{action}{target}"
        expected_reward_over_time[input_key] = [expected_reward_value]
    reward_over_time[action] = [0]
    dopamine_over_time[action] = [0]

# Activation initialization
activation_times.append(0)
for action in range(N_actions):
    activation_over_time[action] = [0]
    target_activation_over_time[action] = [0]

# Recording
recordings = {ct: [[h.Vector().record(cell(0.5)._ref_v) for cell in cells[ct][a]] for a in range(N_actions)] for ct in cell_types}
t_vec = h.Vector().record(h._ref_t)
'''
'''
#--- Plotting ------------------------------------------------------------------------------------------------------------------------------------------#

plt.ion()
fig = plt.figure(figsize=(13, 8))
rows = 5
gs = gridspec.GridSpec(rows, 1+N_actions, height_ratios=[2]+[4]*(rows-1), width_ratios=[0.3]+[1]*N_actions)
axs = [[fig.add_subplot(gs[i, j]) for j in range(1+N_actions)] for i in range(rows)]
[row_control_upper, row_potential, row_spike, row_weights, row_reward] = list(range(rows))
col_potential = 1
col_spike = 1
col_weights = 1
col_reward = 1

# Deactivate axis for non-plot axes
for row in range(rows):
    if row == 0: # first row are control panels
        for ax in axs[row]:
            ax.set_axis_off() # deactivate axis
    else:
        axs[row][0].set_axis_off() # left column is also control panel

# Membrane potential plot
axs[row_potential][col_potential].set_ylabel('Membrane potential (mV)')
mem_lines = {ct: [] for ct in cell_types}

for i, ch in enumerate(range(N_actions)):
    for j, ct in enumerate(cell_types):
        avg_line, = axs[row_potential][col_potential+i].plot([], [], f'C{j}', label=ct)
        mem_lines[ct].append(avg_line)

    axs[row_potential][col_potential+i].set_title(f'Action {ch}')
    #axs[row_potential][col_potential+i].legend(loc='upper right')
    axs[row_potential][col_potential+i].set_xlim(0, plot_interval)
    axs[row_potential][col_potential+i].set_ylim(-85, 65)
axs[row_potential][-1].legend(loc='upper right')

# Spike raster plot and rate lines
raster_lines = {ct: [[] for _ in range(N_actions)] for ct in cell_types}
rate_lines = {ct: [] for ct in cell_types}
rate_lines.update({'Threshold': []})
axs[row_spike][col_spike].set_ylabel('Spike raster')

for i, ch in enumerate(range(N_actions)):
    for j, ct in enumerate(cell_types):
        raster_lines[ct][i] = []
        for k in range(cell_types_numbers[ct]):
            line, = axs[row_spike][col_spike+i].plot([], [], f'C{j}.', markersize=3)
            raster_lines[ct][i].append(line)

        rate_line, = axs[row_spike][col_spike+i].step([], [], f'C{j}')#, label=f'{ct} rate')
        rate_lines[ct].append(rate_line)
    axs[row_spike][col_spike+i].plot([], [], color='black', label=f'Relative rate')

    total_cells = sum(cell_types_numbers[ct] for ct in cell_types)
    
    y_base_thal = cell_types_numbers['Thal'] * selection_threshold
    axs[row_spike][col_spike+i].axhline(y=y_base_thal, color='black', linestyle='dotted', label=f'Activation threshold')
    

    y_max = total_cells + 1.5
    axs[row_spike][col_spike+i].set_ylim(0.5, y_max)
    yticks = []
    cumulative = 0
    for ct in cell_types:
        mid = y_max - (cumulative + (cell_types_numbers[ct]+1) / 2)
        yticks.append(mid)
        cumulative += cell_types_numbers[ct] 
    axs[row_spike][col_spike+i].set_yticks(yticks)
    axs[row_spike][col_spike+i].set_yticklabels(cell_types)
    axs[row_spike][col_spike+i].set_xlim(0, plot_interval)
    #axs[row_spike][col_spike+i].legend(loc='upper right')
axs[row_spike][-1].legend(loc='upper right')

# Weight plot
axs[row_weights][col_weights].set_ylabel('Cortical input weight')
weight_lines = {ct: [[] for _ in range(N_actions)] for ct in weight_cell_types}

for i, ch in enumerate(range(N_actions)):
    for j, ct in enumerate(cell_types):
        if ct in weight_cell_types:
            weight_lines[ct][i] = []
            for k in range(cell_types_numbers[ct]):
                label = ct if k == 0 else None  # Only label the first line
                line, = axs[row_weights][col_weights+i].step([], [], f'C{j}', label=label)
                weight_lines[ct][i].append(line)

    #axs[row_weights][col_weights+i].legend(loc='upper right')
    axs[row_weights][col_weights+i].set_xlim(0, plot_interval)
    axs[row_weights][col_weights+i].set_ylim(0, 3)
axs[row_weights][-1].legend(loc='upper right')

# Reward plot
axs[row_reward][col_reward].set_ylabel('Dopamine')
expected_reward_lines = [[] for _ in range(N_actions)]
reward_lines = [[] for _ in range(N_actions)]
dopamine_lines = [[] for _ in range(N_actions)]
activation_lines = [[] for _ in range(N_actions)]
target_activation_lines = [[] for _ in range(N_actions)]

for i, ch in enumerate(range(N_actions)):
    expected_reward_line, = axs[row_reward][col_reward+i].plot([], [], 'C9', label='Expected reward')
    reward_line, = axs[row_reward][col_reward+i].step([], [], 'C8', label='Reward')
    dopamine_line, = axs[row_reward][col_reward+i].plot([], [], 'C6', label='Dopamine')
    activation_line, = axs[row_reward][col_reward+i].step([], [], 'C7', linestyle='dashed', label='Activation time')
    target_activation_line,  = axs[row_reward][col_reward+i].step([], [], color='blue', linestyle='dotted', label=f'Target act. time')
    expected_reward_lines[i].append(expected_reward_line)
    reward_lines[i].append(reward_line)
    dopamine_lines[i].append(dopamine_line)
    activation_lines[i].append(activation_line)
    target_activation_lines[i].append(target_activation_line)
    
    #axs[row_reward][col_reward+i].legend(loc='upper right')
    axs[row_reward][col_reward+i].set_xlabel('Simulation time (ms)')
    axs[row_reward][col_reward+i].set_xlim(0, plot_interval)
    axs[row_reward][col_reward+i].set_ylim(-1.1, 1.1)
axs[row_reward][-1].legend(loc='upper right')

for a in range(N_actions):
    target_activation_lines[a][0].set_visible(False)

#--- Left control panel ---#
ax_pause = axs[row_control_upper][0].inset_axes([0,0.5,1,0.5])
buttons['pause'] = Button(ax_pause, 'Pause')
buttons['pause'].on_clicked(toggle_pause)

ax_noise = axs[row_control_upper][0].inset_axes([0.5,0,0.5,0.45])
buttons['noise_slider'] = Slider(ax_noise, 'Noise', 0, 1, valinit=noise, valstep=0.1)
buttons['noise_slider'].on_changed(update_stim)


#--- Upper control panel ---#
for a in range(N_actions):
    # Target button
    ax_target = axs[row_control_upper][col_potential+a].inset_axes([0.0,0,0.3,1])
    buttons[f'target_{a}'] = Button(ax_target, 'Set as\nTarget')
    buttons[f'target_{a}'].on_clicked(lambda event, a=a: toggle_target_action(event=event, action=a))
    ax_cor_dur = axs[row_control_upper][col_potential+a].inset_axes([0.7,0,0.3,0.3])
    buttons[f'cor_dur_slider{a}'] = Slider(ax_cor_dur, 'Input Dur', 0.2, 1, valinit=cortical_input_dur_rel[a], valstep=0.2)
    buttons[f'cor_dur_slider{a}'].on_changed(lambda val, a=a: update_cor_dur(val=val, action=a))


plt.show()
plt.tight_layout()
'''
    
# Basal Ganglia
bg_m = BasalGanglia('MotorLoop')

#--- Simulation ---------------------------------------------------------------------------------------------------------------------------------------------------#
h.cvode.active(0)  # turn off variable time step
h.dt = 1#0.1
h.finitialize()

# Define cortical input stimuli
for action in bg_m.target_actions:
    h.cvode.event(h.t + 1, lambda: bg_m.update_stimulus_activation(cell='MSNd', stimulus=f'Cor_MSNd', actions=[action], active=True))  # start cortical input stimulus for that action
    h.cvode.event(h.t + bg_m.cortical_input_dur_rel[action] * bg_m.plot_interval/2,  lambda: bg_m.update_stimulus_activation(cell='MSNd', stimulus=f'Cor_MSNd', actions=[action], active=False)) # stop cortical input stimulus for that action
    h.cvode.event(h.t + 1, lambda: bg_m.update_stimulus_activation(cell='MSNi', stimulus=f'Cor_MSNi', actions=[action], active=True))  # start cortical input stimulus for that action
    h.cvode.event(h.t + bg_m.cortical_input_dur_rel[action] * bg_m.plot_interval/2,  lambda: bg_m.update_stimulus_activation(cell='MSNi', stimulus=f'Cor_MSNi', actions=[action], active=False)) # stop cortical input stimulus for that action

state = 0  
try:
    while True:
        if bg_m.paused or len(bg_m.target_actions) == 0:
            # Simulation paused
            time.sleep(0.1)
            bg_m.fig.canvas.draw_idle()   
            bg_m.fig.canvas.flush_events()
            plt.pause(0.1)
            continue

        # Run simulation for half of the interval
        h.continuerun(h.t + bg_m.plot_interval // 2)
        t_array = np.array(bg_m.t_vec)

        # --- Action selection and SNc dip/burst trigger ---#
        if state == 0: # executed after half time of plot_interval
            bg_m.activation_times.append(int(h.t))

            for i, ch in enumerate(range(bg_m.N_actions)):
                for ct in bg_m.cell_types:
                    all_spikes = []
                    for k in range(bg_m.cell_types_numbers[ct]):
                        spikes = np.array(bg_m.spike_times[ct][ch][k].to_python())
                        all_spikes.extend(spikes)
                    bins = np.arange(0, t_array[-1], bg_m.bin_width_firing_rate)
                    hist, edges = np.histogram(all_spikes, bins=bins)
                    if np.any(hist):  # Only proceed if there's at least one spike
                        rate = hist / (bg_m.cell_types_numbers[ct] * bg_m.bin_width_firing_rate / 1000.0)
                        bin_centers = (edges[:-1] + edges[1:]) / 2
                        
                        if ct == 'Thal':
                            window_start = t_array[-1] - bg_m.plot_interval/2
                            # Get the indices of bins in the last window
                            bin_indices = np.where(bin_centers > window_start)[0]
                            rate_window = rate[bin_indices]
                            bin_centers_window = bin_centers[bin_indices]
                            edges_window = edges[bin_indices[0] : bin_indices[-1] + 2]  # include right edge of last bin

                            indices = np.where(rate_window > bg_m.selection_threshold * 1000.0 / bg_m.stim_intervals[ct])[0]
                            
                            # Group into continuous chunks
                            longest_duration = 0
                            longest_start = None
                            longest_end = None
                            if len(indices) > 0:
                                for k, g in groupby(enumerate(indices), lambda x: x[0] - x[1]):
                                    group = list(map(itemgetter(1), g))
                                    start = edges_window[group[0]] - bg_m.bin_width_firing_rate/2
                                    end = edges_window[group[-1] + 1] + bg_m.bin_width_firing_rate/2
                                    duration = end - start
                                    print(f"{i}: start={start}, end={end}, duration={duration}")
                                    if duration > longest_duration:
                                        longest_duration = duration
                                        longest_start = start
                                        longest_end = end
                            bg_m.activation_over_time[i].append(longest_duration/(bg_m.plot_interval/2))
                            bg_m.target_activation_over_time[i].append(bg_m.cortical_input_dur_rel[i] if i in bg_m.target_actions else 0)
                            
            # Select actions
            bg_m.rates['Thal'], bg_m.rates_rel['Thal'] = bg_m.analyse_firing_rate('Thal', window=bg_m.plot_interval/2)
            #selected_actions = [i for i, rate_rel in enumerate(rates_rel['Thal']) if rate_rel > selection_threshold]
            selected_actions = [i for i, activations in bg_m.activation_over_time.items() if activations[-1] > 0]
            if output: print(f"{int(h.t)} ms: Target Actions = {bg_m.target_actions}, Selected Actions = {selected_actions}, Rates Thal = {bg_m.rates['Thal']}, Rates Thal relative = {bg_m.rates_rel['Thal']}")
            
            correct_actions = list(set(bg_m.target_actions) & set(selected_actions))
            incorrect_actions = list(set(bg_m.target_actions) ^ set(selected_actions))
            bg_m.reward_times.append(int(h.t))
            for action in range(bg_m.N_actions):
                # Determine reward
                #if  not ((action in target_actions) ^ (action in selected_actions)): #XNOR
                if (action in bg_m.target_actions) and (action in selected_actions): 
                    bg_m.reward_over_time[action].append(1)
                else:
                    bg_m.reward_over_time[action].append(0)

                # Trigger SNc dips or bursts based on difference between actual reward and expected reward
                input_key = f"{action}{action in bg_m.target_actions}"
                current_expected_reward = bg_m.expected_reward_over_time[input_key][-1]
                bg_m.dopamine_over_time[action].append(round(bg_m.reward_over_time[action][-1] - current_expected_reward, 4)) # TODO: determine dopamine from relative rate of SNc
                if bg_m.reward_over_time[action][-1] - current_expected_reward > 0:
                    bg_m.SNc_burst(event=None, actions=[action])
                elif bg_m.reward_over_time[action][-1] - current_expected_reward < 0:
                    bg_m.SNc_dip(event=None, actions=[action])

                # Update expected reward based on actual reward
                bg_m.expected_reward_over_time[input_key].append(round(current_expected_reward + 0.1 * (bg_m.reward_over_time[action][-1] - current_expected_reward), 4))

                # Repeat latest expected reward value when input is not triggered
                alternative_input_key = f"{action}{action not in bg_m.target_actions}"
                bg_m.expected_reward_over_time[alternative_input_key].append(bg_m.expected_reward_over_time[alternative_input_key][-1])

        # --- Weight and plot update ---#  
        else: # executed after full time of plot_interval

            # Define cortical input stimuli
            h.cvode.event(h.t + 1, lambda: bg_m.update_stimulus_activation(cell='MSNd', stimulus=f'Cor_MSNd', actions=bg_m.target_actions, active=True))  # start cortical input stimulus for that action
            h.cvode.event(h.t + 1, lambda: bg_m.update_stimulus_activation(cell='MSNi', stimulus=f'Cor_MSNi', actions=bg_m.target_actions, active=True))  # start cortical input stimulus for that action
            for action in bg_m.target_actions:
                h.cvode.event(h.t + bg_m.cortical_input_dur_rel[action] * bg_m.plot_interval/2, lambda action=action: bg_m.update_stimulus_activation(cell='MSNd', stimulus=f'Cor_MSNd', actions=[action], active=False)) # stop cortical input stimulus for that action
                h.cvode.event(h.t + bg_m.cortical_input_dur_rel[action] * bg_m.plot_interval/2, lambda action=action: bg_m.update_stimulus_activation(cell='MSNi', stimulus=f'Cor_MSNi', actions=[action], active=False)) # stop cortical input stimulus for that action
            
            # Analyse firing rates
            bg_m.rates['SNc'], bg_m.rates_rel['SNc'] = bg_m.analyse_firing_rate('SNc', window=bg_m.plot_interval)
            bg_m.rates['MSNd'], bg_m.rates_rel['MSNd'] = bg_m.analyse_firing_rate('MSNd', window=bg_m.plot_interval, average=False)
            bg_m.rates['MSNi'], bg_m.rates_rel['MSNi'] = bg_m.analyse_firing_rate('MSNi', window=bg_m.plot_interval, average=False)

            # TODO: set dopamine value based on relative SNc rate (lenth of dip and burst to be adapted)
            #print(f"dopamine over time: {dopamine_over_time}")
            #print(f"rel SNc rate: {rates_rel['SNc']}")

            # Update weights
            bg_m.weight_times.append(int(h.t))
            for a in range(bg_m.N_actions):
                for ct in bg_m.weight_cell_types:
                    for k in range(bg_m.cell_types_numbers[ct]):
                        delta_w = 0
                        if ct == 'MSNd':
                            # dopamine facilitates active MSNd and inhibits less active MSNd
                            delta_w = bg_m.learning_rate * bg_m.rates_rel[ct][a][k] * bg_m.dopamine_over_time[a][-1] # rel_rate = 1 corresponds to tonic baseline activity
                        elif ct == 'MSNi':
                            # high dopamine increases inhibition, low dopamine suppresses inhibition
                            delta_w = - bg_m.learning_rate * bg_m.dopamine_over_time[a][-1]
                        idx = a * bg_m.cell_types_numbers[ct] + k
                        new_weight = max(0, bg_m.weights_over_time[(ct, a, k)][-1] + delta_w) # Update weight ensure weight is non-zero
                        bg_m.weights_over_time[(ct, a, k)].append(round(new_weight, 4))
                        bg_m.ncs[f'Cor_{ct}'][idx].weight[0] = new_weight # update weight of cortical input stimulation
                        bg_m.ncs[f'{ct}'][idx].weight[0] = new_weight # update weight of tonical stimulation 
                if output: print(f"{bg_m.weight_times[-1]} ms: Action {a}: rel rate MSNd = {[f'{rate_rel:.2f}' for rate_rel in bg_m.rates_rel['MSNd'][a]]}, rel rate SNc = {bg_m.rates_rel['SNc'][a]:.2f}, Exp. Reward = {bg_m.expected_reward_over_time[f'{a}{a in bg_m.target_actions}'][-1]:.2f}, DA = {bg_m.dopamine_over_time[a][-1]}, Cor-MSNd-Weights = {[f'{nc.weight[0]:.2f}' for nc in bg_m.ncs['Cor_MSNd'][a*bg_m.cell_types_numbers['MSNd']:(a+1)*bg_m.cell_types_numbers['MSNd']]]}, Cor-MSNi-Weights = {[f'{nc.weight[0]:.2f}' for nc in bg_m.ncs['Cor_MSNi'][a*bg_m.cell_types_numbers['MSNi']:(a+1)*bg_m.cell_types_numbers['MSNi']]]}")               
                        
            # Update plots
            #t_array = np.array(t_vec)

            #activation_times.append(int(h.t))

            for i, ch in enumerate(range(bg_m.N_actions)):
                # Membrane potential plot
                for ct in bg_m.cell_types:
                    voltages = np.array([np.array(bg_m.recordings[ct][ch][j]) for j in range(bg_m.cell_types_numbers[ct])])
                    avg_voltage = np.mean(voltages, axis=0)
                    bg_m.mem_lines[ct][i].set_data(t_array, avg_voltage)
                    bg_m.axs[bg_m.row_potential][bg_m.col_potential+i].set_xlim(max(0, int(h.t) - bg_m.plot_interval), max(bg_m.plot_interval, int(h.t)))

                # Spike raster plot
                y_base = bg_m.total_cells
                for ct in bg_m.cell_types:
                    all_spikes = []
                    for k in range(bg_m.cell_types_numbers[ct]):
                        spikes = np.array(bg_m.spike_times[ct][ch][k].to_python())
                        y_val = y_base - k
                        y_vals = np.ones_like(spikes) * y_val
                        bg_m.raster_lines[ct][i][k].set_data(spikes, y_vals)
                        all_spikes.extend(spikes)
                    # Rate lines
                    if len(all_spikes) > 0:
                        bins = np.arange(0, t_array[-1], bg_m.bin_width_firing_rate)
                        hist, edges = np.histogram(all_spikes, bins=bins)
                        if np.any(hist):  # Only proceed if there's at least one spike
                            rate = hist / (bg_m.cell_types_numbers[ct] * bg_m.bin_width_firing_rate / 1000.0)
                            bin_centers = (edges[:-1] + edges[1:]) / 2
                            #offset = y_base + cell_types_numbers[ct] / 2
                            if ct == 'SNc':
                                spike_rate_max = 1000.0 / bg_m.stim_intervals['SNc_burst'] # Hz
                            elif ct == 'MSNd' or ct == 'MSNi':
                                spike_rate_max = 1000.0 / bg_m.stim_intervals['Cor'] # Hz
                            else:
                                spike_rate_max = 1000.0 / bg_m.stim_intervals[ct] # Hz
                            rate_scaled = (rate) / (spike_rate_max + 1e-9)
                            rate_scaled = rate_scaled * (bg_m.cell_types_numbers[ct] - 1) + y_base - bg_m.cell_types_numbers[ct] + 1
                            bg_m.rate_lines[ct][i].set_data(bin_centers, rate_scaled)

                            if ct == 'Thal':
                                window_start = t_array[-1] - bg_m.plot_interval
                                # Get the indices of bins in the last window
                                bin_indices = np.where(bin_centers > window_start)[0]
                                rate_window = rate[bin_indices]
                                bin_centers_window = bin_centers[bin_indices]
                                edges_window = edges[bin_indices[0] : bin_indices[-1] + 2]  # include right edge of last bin

                                indices = np.where(rate_window > bg_m.selection_threshold * spike_rate_max)[0]
                                
                                # Group into continuous chunks

                                longest_duration = 0
                                longest_start = None
                                longest_end = None
                                if len(indices) > 0:
                                    for k, g in groupby(enumerate(indices), lambda x: x[0] - x[1]):
                                        group = list(map(itemgetter(1), g))
                                        start = edges_window[group[0]]
                                        end = edges_window[group[-1] + 1]
                                        duration = end - start
                                        if duration > longest_duration:
                                            longest_duration = duration
                                            longest_start = start
                                            longest_end = end
                                #activation_over_time[i].append(longest_duration/(plot_interval/2))
                        else:
                            bg_m.rate_lines[ct][i].set_data([], [])
                    y_base -= bg_m.cell_types_numbers[ct]
                bg_m.axs[bg_m.row_spike][bg_m.col_spike+i].set_xlim(max(0, int(h.t) - bg_m.plot_interval), max(bg_m.plot_interval, int(h.t)))
                
                # Weight plot
                for ct in bg_m.weight_cell_types:
                    for k in range(bg_m.cell_types_numbers[ct]):
                        bg_m.weight_lines[ct][i][k].set_data(bg_m.weight_times, bg_m.weights_over_time[(ct, i, k)])
                bg_m.axs[bg_m.row_weights][bg_m.col_weights+i].set_xlim(0, max(bg_m.plot_interval, int(h.t)))
                all_weights = [w for lst in bg_m.weights_over_time.values() for w in lst if lst]  # flatten and exclude empty lists
                ymin, ymax = min(all_weights), max(all_weights)
                bg_m.axs[bg_m.row_weights][bg_m.col_weights+i].set_ylim(ymin*0.9, ymax*1.1)

                # Reward plot
                input_key = f"{i}{i in bg_m.target_actions}"
                bg_m.expected_reward_lines[i][0].set_data(bg_m.reward_times, bg_m.expected_reward_over_time[input_key])
                bg_m.reward_lines[i][0].set_data(bg_m.reward_times, bg_m.reward_over_time[i])
                bg_m.dopamine_lines[i][0].set_data(bg_m.reward_times, bg_m.dopamine_over_time[i])
                bg_m.activation_lines[i][0].set_data(bg_m.activation_times, bg_m.activation_over_time[i])
                bg_m.target_activation_lines[i][0].set_data(bg_m.activation_times, bg_m.target_activation_over_time[i])
                bg_m.axs[bg_m.row_reward][bg_m.col_reward+i].set_xlim(0, max(bg_m.plot_interval, int(h.t)))

            bg_m.fig.canvas.draw_idle()   
            bg_m.fig.canvas.flush_events() 
            plt.pause(0.001)

        state = 1 - state # toggle state

        # Pause simulation
        if int(h.t) % bg_m.simulation_stop_time == 0:
            bg_m.toggle_pause()
    

except KeyboardInterrupt:
    print("\nCtrl-C pressed. Storing data...")
    plt.close()

finally:
    # Workbook
    wb = Workbook()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = f"Data\{timestamp}"

    # Worksheet for General Details
    ws_globals = wb.active
    ws_globals.title = "GlobalVariables"

    row = 1

    # --- Dictionaries ---
    def write_dict(name, data):
        global row
        ws_globals.cell(row=row, column=1, value=name)
        row += 1
        for k, v in data.items():
            ws_globals.cell(row=row, column=1, value=str(k))
            ws_globals.cell(row=row, column=2, value=v)
            row += 1
        row += 1

    write_dict("cell_types_numbers", bg_m.cell_types_numbers)
    write_dict("stim_intervals", bg_m.stim_intervals)
    write_dict("stim_weights", bg_m.stim_weights)

    # --- Lists ---
    def write_list(name, lst):
        global row
        ws_globals.cell(row=row, column=1, value=name)
        for i, val in enumerate(lst):
            ws_globals.cell(row=row + 1 + i, column=1, value=val)
        row += len(lst) + 2

    write_list("target_actions", bg_m.target_actions)
    write_list("cortical_input_dur_rel", bg_m.cortical_input_dur_rel)

    # --- Tuples/List of Tuples ---
    def write_tuples(name, tuples_list):
        global row
        ws_globals.cell(row=row, column=1, value=name)
        for i, tup in enumerate(tuples_list):
            for j, val in enumerate(tup):
                ws_globals.cell(row=row + 1 + i, column=1 + j, value=val)
        row += len(tuples_list) + 2

    write_tuples("connection_specs", bg_m.connection_specs)

    # --- Scalars ---
    scalars = {
        "N_actions": bg_m.N_actions,
        "plot_interval": bg_m.plot_interval,
        "bin_width_firing_rate": bg_m.bin_width_firing_rate,
        "n_spikes_SNc_burst": bg_m.n_spikes_SNc_burst,
        "selection_threshold": bg_m.selection_threshold,
        "learning_rate": bg_m.learning_rate,
        "expected_reward_value": bg_m.expected_reward_value,
        "noise": bg_m.noise,
        "simulation_stop_time": bg_m.simulation_stop_time

    }
    write_dict("Scalars", scalars)

    # Worksheet for Weights
    ws_weights = wb.create_sheet(title="WeightsOverTime")

    # Header
    header = ['time']
    keys = sorted(bg_m.weights_over_time.keys())
    header.extend(f"{ct}_a{a}_n{i}" for ct, a, i in keys)
    ws_weights.append(header)

    # Rows
    max_len = len(bg_m.weight_times)
    for t_idx in range(max_len):
        row = [bg_m.weight_times[t_idx]]
        for key in keys:
            val = bg_m.weights_over_time[key][t_idx] if t_idx < len(bg_m.weights_over_time[key]) else None
            row.append(val)
        ws_weights.append(row)

    # Worksheets
    data_list = [
        ("ExpectedRewardOverTime", bg_m.expected_reward_over_time), 
        ("RewardOverTime", bg_m.reward_over_time), 
        ("DopamineOverTime", bg_m.dopamine_over_time)
        ]
    ws_list = []
    for name, data in data_list:
        ws = wb.create_sheet(title=f"{name}")
        
        # Header
        header = ['time']
        keys = sorted(data.keys())
        header.extend(f"{key}" for key in keys)
        ws.append(header)

        # Rows
        max_len = len(bg_m.reward_times)
        for t_idx in range(max_len):
            row = [bg_m.reward_times[t_idx]]
            for key in keys:
                val = data[key][t_idx] if t_idx < len(data[key]) else None
                row.append(val)
            ws.append(row)

        ws_list.append(ws)

    # Save
    wb.save(f"{path}.xlsx") # Excel
    bg_m.fig.savefig(f"{path}.png", dpi=300, bbox_inches='tight') # GUI Screenshot

    print(f"Data saved as {path}")


