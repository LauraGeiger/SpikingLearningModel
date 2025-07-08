from neuron import h#, gui
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, CheckButtons, RadioButtons
import matplotlib.gridspec as gridspec
import numpy as np
import time
import threading
import random
from openpyxl import Workbook
from datetime import datetime
from itertools import groupby
from itertools import product
from operator import itemgetter
# --- TODO --------------------------------------------------#
# determine dopamine level from rel SNc rate
# --- TODO --------------------------------------------------#


h.load_file("stdrun.hoc")

output = False # Set to True (print values in the terminal) or False (no printing)

#--- BasalGanglia ------------------------------------------------------------------------------------------------------------------------------------------------#

class BasalGanglia:

    def __init__(self, name, input, output, child_loop=None, binary_input=False, single_goal=False, goal_action_table=None, actions_to_plot=None):
        self.name = name
        self.goals = input
        self.N_actions = 2**len(output)
        self.actions = [''.join(bits) for bits in product('01', repeat=len(output))] # binary combination of all outputs
        if actions_to_plot is not None:
            self.actions_to_plot = self.N_actions if self.N_actions <= actions_to_plot else actions_to_plot
        self.child_loop = child_loop
        self.binary_input = binary_input
        self.single_goal = single_goal
        self.goal_action_table = goal_action_table

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
            'SNc'       : 0.5,
            'MSNd'      : 0.35,
            'MSNi'      : 0.35,
            'GPe'       : 0.5,
            'GPi'       : 0.5,
            'Thal'      : 0.5,
            'SNc_burst' : 0.5,
            'Cor'       : 0.5
        }

        self.stims = {}
        self.syns = {}
        self.ncs = {}

        # Define connection specifications
        self.connection_specs = [# pre_group, post_group, label, e_rev, weight, tau, delay
            ('SNc', 'MSNd', 'SNc_to_MSNd',   0, 0,    10, 1),   # excitatory
            ('SNc', 'MSNi', 'SNc_to_MSNi', -85, 0,    10, 1),   # inhibitory
            ('MSNd', 'GPi', 'MSNd_to_GPi', -85, 0.6,  10, 1),   # inhibitory
            ('MSNi', 'GPe', 'MSNi_to_GPe', -85, 0.9,  10, 1),   # inhibitory
            ('GPe',  'GPi',  'GPe_to_GPi', -85, 0.14, 10, 1),   # inhibitory
            ('GPi', 'Thal', 'GPi_to_Thal', -85, 0.7,  10, 1)    # inhibitory
        ]

        self.paused = False
        #self.target_actions = []
        self.noise = 0
        self.selection_threshold = 0.8
        self.n_spikes_SNc_burst = 5
        self.learning_rate = 0.05
        self.reward_times = []
        self.activation_times = []
        self.expected_reward_over_time = {}
        self.activation_over_time = {}
        #self.target_activation_over_time = {}
        self.cortical_input_dur_rel = [0] * len(self.goals)
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
        self.simulation_stop_time = 100000 # ms

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
                        #nc.weight[0] = random.uniform(0.9*weight, 1.1*weight)
                        nc.weight[0] = weight
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
            #self.target_activation_over_time[action] = [0]

    def _init_recording(self):
        # Recording
        self.recordings = {ct: [[h.Vector().record(cell(0.5)._ref_v) for cell in self.cells[ct][a]] for a in range(self.N_actions)] for ct in self.cell_types}
        self.t_vec = h.Vector().record(h._ref_t)

    def _init_plotting(self):
        plt.ion()
        self.fig = plt.figure(figsize=(13, 8))
        self.fig.canvas.manager.set_window_title(self.name)
        self.rows = 4
        self.gs = gridspec.GridSpec(2, 1, figure=self.fig, height_ratios=[1, 4 * self.rows])
        self.gs_control = self.gs[0].subgridspec(1, 1 + len(self.goals))
        self.gs_plot = self.gs[1].subgridspec(self.rows, self.actions_to_plot)
        self.axs_control = [self.fig.add_subplot(self.gs_control[i]) for i in range(1 + len(self.goals))]
        self.axs_plot = [[self.fig.add_subplot(self.gs_plot[i, j]) for j in range(self.actions_to_plot)] for i in range(self.rows)]
        [self.row_potential, self.row_spike, self.row_weights, self.row_reward] = list(range(self.rows))
        
        # Deactivate axis for control axes
        for ax in self.axs_control:
            ax.set_axis_off() # deactivate axis
        
        self._init_membrane_potential_plot()
        self._init_spike_plot()
        self._init_weight_plot()
        self._init_reward_plot()
        self._init_control_panel()

        plt.show()
        plt.tight_layout()

    def _init_membrane_potential_plot(self):
        # Membrane potential plot
        self.axs_plot[self.row_potential][0].set_ylabel('Membrane potential (mV)')
        self.mem_lines = {ct: [] for ct in self.cell_types}

        for i, ch in enumerate(range(self.actions_to_plot)):
            for j, ct in enumerate(self.cell_types):
                avg_line, = self.axs_plot[self.row_potential][i].plot([], [], f'C{j}', label=ct)
                self.mem_lines[ct].append(avg_line)

            self.axs_plot[self.row_potential][i].set_title(f'{self.actions[ch]}')
            self.axs_plot[self.row_potential][i].set_xlim(0, self.plot_interval)
            self.axs_plot[self.row_potential][i].set_ylim(-85, 65)
        self.axs_plot[self.row_potential][-1].legend(loc='upper right')

    def _init_spike_plot(self):
        # Spike raster plot and rate lines
        self.raster_lines = {ct: [[] for _ in range(self.actions_to_plot)] for ct in self.cell_types}
        self.rate_lines = {ct: [] for ct in self.cell_types}
        self.rate_lines.update({'Threshold': []})
        self.axs_plot[self.row_spike][0].set_ylabel('Spike raster')

        for i, ch in enumerate(range(self.actions_to_plot)):
            for j, ct in enumerate(self.cell_types):
                self.raster_lines[ct][i] = []
                for k in range(self.cell_types_numbers[ct]):
                    line, = self.axs_plot[self.row_spike][i].plot([], [], f'C{j}.', markersize=3)
                    self.raster_lines[ct][i].append(line)

                rate_line, = self.axs_plot[self.row_spike][i].step([], [], f'C{j}')
                self.rate_lines[ct].append(rate_line)
            self.axs_plot[self.row_spike][i].plot([], [], color='black', label=f'Relative rate')

            self.total_cells = sum(self.cell_types_numbers[ct] for ct in self.cell_types)
            
            y_base_thal = self.cell_types_numbers['Thal'] * self.selection_threshold
            self.axs_plot[self.row_spike][i].axhline(y=y_base_thal, color='black', linestyle='dotted', label=f'Activation threshold')
            
            y_max = self.total_cells + 1.5
            self.axs_plot[self.row_spike][i].set_ylim(0.5, y_max)
            yticks = []
            cumulative = 0
            for ct in self.cell_types:
                mid = y_max - (cumulative + (self.cell_types_numbers[ct]+1) / 2)
                yticks.append(mid)
                cumulative += self.cell_types_numbers[ct] 
            self.axs_plot[self.row_spike][i].set_yticks(yticks)
            self.axs_plot[self.row_spike][i].set_yticklabels(self.cell_types)
            self.axs_plot[self.row_spike][i].set_xlim(0, self.plot_interval)
        self.axs_plot[self.row_spike][-1].legend(loc='upper right')

    def _init_weight_plot(self):
        # Weight plot
        self.axs_plot[self.row_weights][0].set_ylabel('Cortical input weight')
        self.weight_lines = {ct: [[] for _ in range(self.actions_to_plot)] for ct in self.weight_cell_types}

        for i, ch in enumerate(range(self.actions_to_plot)):
            for j, ct in enumerate(self.cell_types):
                if ct in self.weight_cell_types:
                    self.weight_lines[ct][i] = []
                    for k in range(self.cell_types_numbers[ct]):
                        label = ct if k == 0 else None  # Only label the first line
                        line, = self.axs_plot[self.row_weights][i].step([], [], f'C{j}', label=label)
                        self.weight_lines[ct][i].append(line)

            self.axs_plot[self.row_weights][i].set_xlim(0, self.plot_interval)
            self.axs_plot[self.row_weights][i].set_ylim(0, 3)
        self.axs_plot[self.row_weights][-1].legend(loc='upper right')

    def _init_reward_plot(self):
        # Reward plot
        self.axs_plot[self.row_reward][0].set_ylabel('Dopamine')
        self.expected_reward_lines = [[] for _ in range(self.N_actions)]
        self.reward_lines = [[] for _ in range(self.N_actions)]
        self.dopamine_lines = [[] for _ in range(self.N_actions)]
        self.activation_lines = [[] for _ in range(self.N_actions)]
        self.target_activation_lines = [[] for _ in range(self.N_actions)]

        for i, ch in enumerate(range(self.actions_to_plot)):
            expected_reward_line, = self.axs_plot[self.row_reward][i].plot([], [], 'C9', label='Expected reward')
            reward_line, = self.axs_plot[self.row_reward][i].step([], [], 'C8', label='Reward')
            dopamine_line, = self.axs_plot[self.row_reward][i].plot([], [], 'C6', label='Dopamine')
            activation_line, = self.axs_plot[self.row_reward][i].step([], [], 'C7', linestyle='dashed', label='Activation time')
            target_activation_line,  = self.axs_plot[self.row_reward][i].step([], [], color='blue', linestyle='dotted', label=f'Target act. time')
            self.expected_reward_lines[i].append(expected_reward_line)
            self.reward_lines[i].append(reward_line)
            self.dopamine_lines[i].append(dopamine_line)
            self.activation_lines[i].append(activation_line)
            self.target_activation_lines[i].append(target_activation_line)
            
            self.axs_plot[self.row_reward][i].set_xlabel('Simulation time (ms)')
            self.axs_plot[self.row_reward][i].set_xlim(0, self.plot_interval)
            self.axs_plot[self.row_reward][i].set_ylim(-1.1, 1.1)
        self.axs_plot[self.row_reward][-1].legend(loc='upper right')

        for a in range(self.actions_to_plot):
            self.target_activation_lines[a][0].set_visible(False)

    def _init_control_panel(self):
        #--- Upper control panel ---#
        #print(f"{self.name} width={self.axs_control[0].get_position().width}, height={self.axs_control[0].get_position().height}")
        ax_pause = self.axs_control[0].inset_axes([0,0.5,1,0.5]) #[x0, y0, width, height]
        self.buttons['pause'] = Button(ax_pause, 'Pause')
        self.buttons['pause'].on_clicked(self.toggle_pause)
        
        ax_noise = self.axs_control[0].inset_axes([0.4,0,0.5,0.45]) #[x0, y0, width, height]
        self.buttons['noise_slider'] = Slider(ax_noise, 'Noise', 0, 1, valinit=self.noise, valstep=0.1)
        self.buttons['noise_slider'].on_changed(self.update_stim)

        for i, goal in enumerate(self.goals):
            # Target button
            #ax_target = self.axs_control[1+i].inset_axes([0,0.5,1,0.5]) #[x0, y0, width, height]
            #self.buttons[f'target_{i}'] = Button(ax_target, f'{goal}')#'Set as\nTarget')
            #self.buttons[f'target_{i}'].on_clicked(lambda event, a=i: self.toggle_target_action(event=event, action=a))
            ax_cor_dur = self.axs_control[1+i].inset_axes([0,0,0.9,0.45]) #[x0, y0, width, height]
            ax_cor_dur.set_title(goal)
            self.buttons[f'cor_dur_slider{i}'] = Slider(ax_cor_dur, '', 0, 1, valinit=self.cortical_input_dur_rel[i], valstep=1 if self.binary_input else 0.2)
            self.buttons[f'cor_dur_slider{i}'].on_changed(lambda val, i=i: self.update_cor_dur(val=val, goal_idx=i))

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

    def analyze_firing_rate(self, cell, window=None, average=True):
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

    '''
    def toggle_target_action(self, event=None, action=None):
        if action != None:
            if action in self.target_actions:
                self.target_actions.remove(action)
                self.buttons[f'target_{action}'].color = '0.85'

                h.cvode.event(h.t + 1, lambda: self.update_stimulus_activation(cell='MSNd', stimulus=f'Cor_MSNd', actions=[action], active=False)) # stop cortical input stimulus for that action
                h.cvode.event(h.t + 1, lambda: self.update_stimulus_activation(cell='MSNi', stimulus=f'Cor_MSNi', actions=[action], active=False)) # stop cortical input stimulus for that action
                
            else:
                self.target_actions.append(action)
                #self.buttons[f'target_{action}'].label.set_text('Target')
                self.buttons[f'target_{action}'].color = 'y'

                self.target_activation_lines[action][0].set_visible(True)
            self.fig.canvas.draw_idle()
    '''
    def update_stim(self, val):
        self.noise = val
        for ct in self.cell_types:
            for stim in self.stims[ct]:
                stim.noise = self.noise

    def update_cor_dur(self, val, goal_idx):
        self.cortical_input_dur_rel
        if self.single_goal: # reset all durations to zero
            self.cortical_input_dur_rel = [0 for _ in self.cortical_input_dur_rel]
        self.cortical_input_dur_rel[goal_idx] = val

    def select_action_with_highest_differential_weight(self):
        best_action = None
        max_avg_diff = float('-inf')

        for a in range(self.N_actions):
            # Sum weights for MSNd
            msnd_weights = [
                self.ncs['Cor_MSNd'][a * self.cell_types_numbers['MSNd'] + k].weight[0]
                for k in range(self.cell_types_numbers['MSNd'])
            ]
            msni_weights = [
                self.ncs['Cor_MSNi'][a * self.cell_types_numbers['MSNi'] + k].weight[0]
                for k in range(self.cell_types_numbers['MSNi'])
            ]

            # Compute averages
            avg_msnd = sum(msnd_weights) / self.cell_types_numbers['MSNd']
            avg_msni = sum(msni_weights) / self.cell_types_numbers['MSNi']

            avg_diff = avg_msnd - avg_msni

            # Track best action
            if avg_diff > max_avg_diff:
                max_avg_diff = avg_diff
                best_action = a
        
        return best_action

    def cortical_input_stimuli(self, current_time):
        target_actions = [i for i, val in enumerate(self.cortical_input_dur_rel) if val != 0]

        best_action = self.select_action_with_highest_differential_weight()
        print(f"best action = {best_action}")

        h.cvode.event(current_time + 1, lambda: self.update_stimulus_activation(cell='MSNd', stimulus=f'Cor_MSNd', actions=target_actions, active=True))  # start cortical input stimulus for that action
        h.cvode.event(current_time + 1, lambda: self.update_stimulus_activation(cell='MSNi', stimulus=f'Cor_MSNi', actions=target_actions, active=True))  # start cortical input stimulus for that action
        for action in target_actions:
            h.cvode.event(current_time + self.cortical_input_dur_rel[action] * self.plot_interval/2, lambda action=best_action: self.update_stimulus_activation(cell='MSNd', stimulus=f'Cor_MSNd', actions=[action], active=False)) # stop cortical input stimulus for that action
            h.cvode.event(current_time + self.cortical_input_dur_rel[action] * self.plot_interval/2, lambda action=best_action: self.update_stimulus_activation(cell='MSNi', stimulus=f'Cor_MSNi', actions=[action], active=False)) # stop cortical input stimulus for that action
            
    def analyze_thalamus_activation_time(self, current_time):
        self.activation_times.append(int(current_time))

        for i, ch in enumerate(range(self.N_actions)):
            for ct in self.cell_types:
                all_spikes = []
                for k in range(self.cell_types_numbers[ct]):
                    spikes = np.array(self.spike_times[ct][ch][k].to_python())
                    all_spikes.extend(spikes)
                bins = np.arange(0, np.array(self.t_vec)[-1], self.bin_width_firing_rate)
                hist, edges = np.histogram(all_spikes, bins=bins)
                if np.any(hist):  # Only proceed if there's at least one spike
                    rate = hist / (self.cell_types_numbers[ct] * self.bin_width_firing_rate / 1000.0)
                    bin_ends = edges[1:]
                    
                    if ct == 'Thal':
                        window_start = np.array(self.t_vec)[-1] - self.plot_interval/2
                        # Get the indices of bins in the last window
                        bin_indices = np.where(bin_ends > window_start)[0]
                        rate_window = rate[bin_indices]
                        edges_window = edges[bin_indices[0] : bin_indices[-1] + 2]  # include right edge of last bin

                        indices = np.where(rate_window > self.selection_threshold * 1000.0 / self.stim_intervals[ct])[0]
                        
                        # Group into continuous chunks
                        longest_duration = 0
                        if len(indices) > 0:
                            for k, g in groupby(enumerate(indices), lambda x: x[0] - x[1]):
                                group = list(map(itemgetter(1), g))
                                start = edges_window[group[0]] 
                                end = edges_window[group[-1] + 1] + self.bin_width_firing_rate
                                duration = end - start
                                #print(f"{i}: start={start}, end={end}, duration={duration}")
                                if duration > longest_duration:
                                    longest_duration = duration
                        self.activation_over_time[i].append(longest_duration/(self.plot_interval/2))
                        #self.target_activation_over_time[i].append(self.cortical_input_dur_rel[i] if self.cortical_input_dur_rel[i] != 0 else 0)
  
    def select_actions(self, current_time):
        if output: 
            self.rates['Thal'], self.rates_rel['Thal'] = self.analyze_firing_rate('Thal', window=self.plot_interval/2)
            print(f"{int(current_time)} ms: Target Actions = {[i for i, val in enumerate(self.cortical_input_dur_rel) if val != 0]}, Selected Actions = {self.selected_actions}, Rates Thal = {self.rates['Thal']}, Rates Thal relative = {self.rates_rel['Thal']}")
        self.selected_actions = [i for i, activations in self.activation_over_time.items() if activations[-1] > 0]
        selected_actions_and_activation_times = {i: activations[-1] for i, activations in self.activation_over_time.items() if i in self.selected_actions}
        if self.child_loop is not None and len(self.selected_actions) > 0:
            for action in self.selected_actions:
                self.child_loop.buttons[f'cor_dur_slider{action}'].set_val(selected_actions_and_activation_times[action])

    def determine_reward(self, current_time):
        self.reward_times.append(int(current_time))
        for action in range(self.N_actions):
            # Determine reward
            #if  not ((action in target_actions) ^ (action in selected_actions)): #XNOR
            if action in self.selected_actions:# and self.cortical_input_dur_rel[action] != 0: 
                self.reward_over_time[action].append(1)
            else:
                self.reward_over_time[action].append(0)

            # Trigger SNc dips or bursts based on difference between actual reward and expected reward
            input_key = f"{action}{self.cortical_input_dur_rel}"
            current_expected_reward = self.expected_reward_over_time[input_key][-1]
            self.dopamine_over_time[action].append(round(self.reward_over_time[action][-1] - current_expected_reward, 4)) # TODO: determine dopamine from relative rate of SNc
            if self.reward_over_time[action][-1] - current_expected_reward > 0:
                self.SNc_burst(event=None, actions=[action])
            elif self.reward_over_time[action][-1] - current_expected_reward < 0:
                self.SNc_dip(event=None, actions=[action])

            # Update expected reward based on actual reward
            self.expected_reward_over_time[input_key].append(round(current_expected_reward + 0.1 * (self.reward_over_time[action][-1] - current_expected_reward), 4))

            # Repeat latest expected reward value when input is not triggered
            alternative_input_key = f"{action}{self.cortical_input_dur_rel[action] == 0}"
            self.expected_reward_over_time[alternative_input_key].append(self.expected_reward_over_time[alternative_input_key][-1])

    def update_weights(self, current_time):
        # Analyze firing rates
        self.rates['SNc'], self.rates_rel['SNc'] = self.analyze_firing_rate('SNc', window=self.plot_interval)
        self.rates['MSNd'], self.rates_rel['MSNd'] = self.analyze_firing_rate('MSNd', window=self.plot_interval, average=False)
        self.rates['MSNi'], self.rates_rel['MSNi'] = self.analyze_firing_rate('MSNi', window=self.plot_interval, average=False)

        # TODO: set dopamine value based on relative SNc rate (lenth of dip and burst to be adapted)
        #print(f"dopamine over time: {dopamine_over_time}")
        #print(f"rel SNc rate: {rates_rel['SNc']}")

        # Update weights
        self.weight_times.append(int(current_time))
        for a in range(self.N_actions):
            for ct in self.weight_cell_types:
                for k in range(self.cell_types_numbers[ct]):
                    delta_w = 0
                    if ct == 'MSNd':
                        # dopamine facilitates active MSNd and inhibits less active MSNd
                        delta_w = self.learning_rate * self.rates_rel[ct][a][k] * self.dopamine_over_time[a][-1] # rel_rate = 1 corresponds to tonic baseline activity
                    elif ct == 'MSNi':
                        # high dopamine increases inhibition, low dopamine suppresses inhibition
                        delta_w = - self.learning_rate * self.dopamine_over_time[a][-1]
                    idx = a * self.cell_types_numbers[ct] + k
                    new_weight = max(0, self.weights_over_time[(ct, a, k)][-1] + delta_w) # Update weight ensure weight is non-zero
                    self.weights_over_time[(ct, a, k)].append(round(new_weight, 4))
                    self.ncs[f'Cor_{ct}'][idx].weight[0] = new_weight # update weight of cortical input stimulation
                    #self.ncs[f'{ct}'][idx].weight[0] = new_weight # update weight of tonical stimulation 
            if output: print(f"{self.weight_times[-1]} ms: Action {a}: rel rate MSNd = {[f'{rate_rel:.2f}' for rate_rel in self.rates_rel['MSNd'][a]]}, rel rate SNc = {self.rates_rel['SNc'][a]:.2f}, Exp. Reward = {self.expected_reward_over_time[f'{a}{self.cortical_input_dur_rel[a] != 0}'][-1]:.2f}, DA = {self.dopamine_over_time[a][-1]}, Cor-MSNd-Weights = {[f'{nc.weight[0]:.2f}' for nc in self.ncs['Cor_MSNd'][a*self.cell_types_numbers['MSNd']:(a+1)*self.cell_types_numbers['MSNd']]]}, Cor-MSNi-Weights = {[f'{nc.weight[0]:.2f}' for nc in self.ncs['Cor_MSNi'][a*self.cell_types_numbers['MSNi']:(a+1)*self.cell_types_numbers['MSNi']]]}")               
                 
    def update_plots(self, current_time):
        # Update plots
        for i, ch in enumerate(range(self.N_actions)):
            # Membrane potential plot
            for ct in self.cell_types:
                voltages = np.array([np.array(self.recordings[ct][ch][j]) for j in range(self.cell_types_numbers[ct])])
                avg_voltage = np.mean(voltages, axis=0)
                self.mem_lines[ct][i].set_data(np.array(self.t_vec), avg_voltage)
                self.axs_plot[self.row_potential][i].set_xlim(max(0, int(current_time) - self.plot_interval), max(self.plot_interval, int(current_time)))

            # Spike raster plot
            y_base = self.total_cells
            for ct in self.cell_types:
                all_spikes = []
                for k in range(self.cell_types_numbers[ct]):
                    spikes = np.array(self.spike_times[ct][ch][k].to_python())
                    y_val = y_base - k
                    y_vals = np.ones_like(spikes) * y_val
                    self.raster_lines[ct][i][k].set_data(spikes, y_vals)
                    all_spikes.extend(spikes)
                # Rate lines
                if len(all_spikes) > 0:
                    bins = np.arange(0, np.array(self.t_vec)[-1], self.bin_width_firing_rate)
                    hist, edges = np.histogram(all_spikes, bins=bins)
                    if np.any(hist):  # Only proceed if there's at least one spike
                        rate = hist / (self.cell_types_numbers[ct] * self.bin_width_firing_rate / 1000.0)
                        bin_ends = edges[1:]
                        if ct == 'SNc':
                            spike_rate_max = 1000.0 / self.stim_intervals['SNc_burst'] # Hz
                        elif ct == 'MSNd' or ct == 'MSNi':
                            spike_rate_max = 1000.0 / self.stim_intervals['Cor'] # Hz
                        else:
                            spike_rate_max = 1000.0 / self.stim_intervals[ct] # Hz
                        rate_scaled = (rate) / (spike_rate_max + 1e-9)
                        rate_scaled = rate_scaled * (self.cell_types_numbers[ct] - 1) + y_base - self.cell_types_numbers[ct] + 1
                        self.rate_lines[ct][i].set_data(bin_ends, rate_scaled)

                        if ct == 'Thal':
                            window_start = np.array(self.t_vec)[-1] - self.plot_interval
                            # Get the indices of bins in the last window
                            bin_indices = np.where(bin_ends > window_start)[0]
                            rate_window = rate[bin_indices]
                            edges_window = edges[bin_indices[0] : bin_indices[-1] + 2]  # include right edge of last bin

                            indices = np.where(rate_window > self.selection_threshold * spike_rate_max)[0]
                            
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
                    else:
                        self.rate_lines[ct][i].set_data([], [])
                y_base -= self.cell_types_numbers[ct]
            self.axs_plot[self.row_spike][i].set_xlim(max(0, int(current_time) - self.plot_interval), max(self.plot_interval, int(current_time)))
            
            # Weight plot
            for ct in self.weight_cell_types:
                for k in range(self.cell_types_numbers[ct]):
                    self.weight_lines[ct][i][k].set_data(self.weight_times, self.weights_over_time[(ct, i, k)])
            self.axs_plot[self.row_weights][i].set_xlim(0, max(self.plot_interval, int(current_time)))
            all_weights = [w for lst in self.weights_over_time.values() for w in lst if lst]  # flatten and exclude empty lists
            ymin, ymax = min(all_weights), max(all_weights)
            self.axs_plot[self.row_weights][i].set_ylim(ymin*0.9, ymax*1.1)

            # Reward plot
            loop.cortical_input_dur_rel
            input_key = f"{i}{i in [i for i, val in enumerate(self.cortical_input_dur_rel) if val != 0]}"
            self.expected_reward_lines[i][0].set_data(self.reward_times, self.expected_reward_over_time[input_key])
            self.reward_lines[i][0].set_data(self.reward_times, self.reward_over_time[i])
            self.dopamine_lines[i][0].set_data(self.reward_times, self.dopamine_over_time[i])
            self.activation_lines[i][0].set_data(self.activation_times, self.activation_over_time[i])
            #self.target_activation_lines[i][0].set_data(self.activation_times, self.target_activation_over_time[i])
            self.axs_plot[self.row_reward][i].set_xlim(0, max(self.plot_interval, int(current_time)))
    
    def save_data(self, path):
        # Workbook
        wb = Workbook()
        path_extented = f"{path}_{self.name}"

        # Worksheet for General Details
        ws_globals = wb.active
        ws_globals.title = "GlobalVariables"

        row = 1

        # --- Dictionaries ---
        def write_dict(name, data, row):
            ws_globals.cell(row=row, column=1, value=name)
            row += 1
            for k, v in data.items():
                ws_globals.cell(row=row, column=1, value=str(k))
                ws_globals.cell(row=row, column=2, value=v)
                row += 1
            row += 1
            return row

        row = write_dict("cell_types_numbers", self.cell_types_numbers, row)
        row = write_dict("stim_intervals", self.stim_intervals, row)
        row = write_dict("stim_weights", self.stim_weights, row)

        # --- Lists ---
        def write_list(name, lst, row):
            ws_globals.cell(row=row, column=1, value=name)
            for i, val in enumerate(lst):
                ws_globals.cell(row=row + 1 + i, column=1, value=val)
            row += len(lst) + 2
            return row

        #row = write_list("target_actions", self.target_actions, row)
        row = write_list("cortical_input_dur_rel", self.cortical_input_dur_rel, row)

        # --- Tuples/List of Tuples ---
        def write_tuples(name, tuples_list, row):
            ws_globals.cell(row=row, column=1, value=name)
            for i, tup in enumerate(tuples_list):
                for j, val in enumerate(tup):
                    ws_globals.cell(row=row + 1 + i, column=1 + j, value=val)
            row += len(tuples_list) + 2
            return row

        row = write_tuples("connection_specs", self.connection_specs, row)

        # --- Scalars ---
        scalars = {
            "N_actions": self.N_actions,
            "plot_interval": self.plot_interval,
            "bin_width_firing_rate": self.bin_width_firing_rate,
            "n_spikes_SNc_burst": self.n_spikes_SNc_burst,
            "selection_threshold": self.selection_threshold,
            "learning_rate": self.learning_rate,
            "expected_reward_value": self.expected_reward_value,
            "noise": self.noise,
            "simulation_stop_time": self.simulation_stop_time

        }
        row = write_dict("Scalars", scalars, row)

        # Worksheet for Weights
        ws_weights = wb.create_sheet(title="WeightsOverTime")

        # Header
        header = ['time']
        keys = sorted(self.weights_over_time.keys())
        header.extend(f"{ct}_a{a}_n{i}" for ct, a, i in keys)
        ws_weights.append(header)

        # Weights
        max_len = len(self.weight_times)
        for t_idx in range(max_len):
            row = [self.weight_times[t_idx]]
            for key in keys:
                val = self.weights_over_time[key][t_idx] if t_idx < len(self.weights_over_time[key]) else None
                row.append(val)
            ws_weights.append(row)

        # Worksheets
        data_list = [
            ("ExpectedRewardOverTime", self.expected_reward_over_time), 
            ("RewardOverTime", self.reward_over_time), 
            ("DopamineOverTime", self.dopamine_over_time)
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
            max_len = len(self.reward_times)
            for t_idx in range(max_len):
                row = [self.reward_times[t_idx]]
                for key in keys:
                    val = data[key][t_idx] if t_idx < len(data[key]) else None
                    row.append(val)
                ws.append(row)

            ws_list.append(ws)

        # Save
        wb.save(f"{path_extented}.xlsx") # Excel
        print(path_extented)

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


# Basal Ganglia Loops
#grasp_types_dict = {"Precision pinch": [1, 1, 1, 0, 0, 0],
#                    "Power grasp":     [1, 1, 1, 1, 1, 1]}
grasp_type_goal_action_table = {"Precision pinch": "10",
                    "Power grasp":     "01"}
grasp_types = list(grasp_type_goal_action_table.keys())
joints = ["Thumb opposition", "Thumb flexion"]
#joints = ["Thumb opposition", "Thumb flexion", "Index finger flexion", "Middle finger flexion", "Ring finger flexion", "Pinky finger flexion"]
#actuators = ["Thumb oppositor", "Thumb abductor", "Thumb flexor", "Thumb extensor", "Index finger flexor", "Index finger extensor", "Middle finger flexor", "Middle finger extensor", "Ring finger flexor", "Ring finger extensor", "Pinky finger flexor", "Pinky finger extensor"]
#actuators = ["Thumb oppositor", "Thumb flexor", "Index finger flexor", "Middle finger flexor", "Ring finger flexor", "Pinky finger flexor"]
actuators = ["Thumb oppositor", "Thumb flexor"]

bg_m = BasalGanglia('MotorLoop', input=joints, output=actuators, actions_to_plot=6)
bg_p = BasalGanglia('PrefrontalLoop', input=grasp_types, output=joints, child_loop=bg_m, binary_input=True, single_goal=True, goal_action_table=grasp_type_goal_action_table, actions_to_plot=6)
bg = [bg_m, bg_p]

#--- Simulation ---------------------------------------------------------------------------------------------------------------------------------------------------#
h.dt = 1
h.finitialize()

state = 0  

try:
    while True:
        any_paused = any(loop.paused for loop in bg)
        all_inputs_zero = all(all(val == 0 for val in loop.cortical_input_dur_rel) for loop in bg)
        if any_paused or all_inputs_zero:
            # Simulation paused
            time.sleep(0.1)
            for loop in bg:
                loop.fig.canvas.draw_idle()   
                loop.fig.canvas.flush_events()
            plt.pause(0.1)
            continue

        # Run simulation for half of the interval
        h.continuerun(h.t + loop.plot_interval // 2)

        # --- Action selection and reward update ---#
        if state == 0: # executed after half time of plot_interval
            for loop in bg:
                loop.analyze_thalamus_activation_time(current_time=h.t)
                loop.select_actions(current_time=h.t)
                #loop.determine_reward(current_time=h.t)

        # --- Weight and plot update ---#  
        else: # executed after full time of plot_interval
            for loop in bg:
                loop.cortical_input_stimuli(current_time=h.t)
                #loop.update_weights(current_time=h.t)
                loop.update_plots(current_time=h.t)

                loop.fig.canvas.draw_idle()   
                loop.fig.canvas.flush_events() 
            plt.pause(0.001)

        state = 1 - state # toggle state

        # Pause simulation
        for loop in bg:
            if int(h.t) % loop.simulation_stop_time == 0:
                loop.toggle_pause()
    
except KeyboardInterrupt:
        print("\nCtrl-C pressed. Storing data...")
        plt.close()

finally:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = f"Data\{timestamp}"

    for loop in bg:
        loop.save_data(path) # Excel file
        loop.fig.savefig(f"{path}.png", dpi=300, bbox_inches='tight') # GUI Screenshot

    print(f"{path}")
