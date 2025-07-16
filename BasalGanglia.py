from neuron import h#, gui
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, TextBox
from matplotlib import rcParams
import matplotlib.gridspec as gridspec
import numpy as np
import time
from openpyxl import Workbook
from datetime import datetime
from itertools import product
from matplotlib.ticker import FuncFormatter
from matplotlib.animation import FuncAnimation

# --- TODO --------------------------------------------------#
# determine dopamine level from rel SNc rate
# --- TODO --------------------------------------------------#


h.load_file("stdrun.hoc")

output = False # Set to True (print values in the terminal) or False (no printing)

#--- BasalGanglia ------------------------------------------------------------------------------------------------------------------------------------------------#

class BasalGanglia:

    def __init__(self, loops):
        self.loops = loops
        self.buttons = {}
        self.paused = False

        self._init_child_loops()
        self._init_plotting()
        self._init_simulation()

        self.run_simulation()
    
    def _init_child_loops(self):
        for idx, loop in enumerate(self.loops):
            if idx > 0:
                loop.set_child_loop(self.loops[idx-1])

    def _init_plotting(self):
        plt.ion()
        self.fig = plt.figure(figsize=(13, 8))
        self.fig.canvas.manager.set_window_title('BasalGangliaOverview')
        self.gs = gridspec.GridSpec(2, 1, figure=self.fig, height_ratios=[1,10])  
        self.gs_control = self.gs[0].subgridspec(1, 4)
        self.gs_plot = self.gs[1].subgridspec(1, 4, width_ratios=[1,1,10,1])     
        self.gs_loops = self.gs_plot[0].subgridspec(len(self.loops), 1)
        self.gs_names = self.gs_plot[1].subgridspec(len(self.loops) + 1, 1)
        self.gs_selections = self.gs_plot[2].subgridspec(len(self.loops) + 1, 1)
        self.gs_probabilities = self.gs_plot[3].subgridspec(len(self.loops) + 1, 1)

        self.axs_control = {}
        self.axs_selections = {}
        self.axs_loops = {}
        self.axs_names = {}
        self.axs_probabilities = {}

        # Pause button
        self.axs_control[0] = self.fig.add_subplot(self.gs_control[0])
        self.axs_control[0].set_axis_off() 
        ax_pause = self.axs_control[0].inset_axes([0,0,1,1]) #[x0, y0, width, height]
        self.buttons['pause'] = Button(ax_pause, 'Pause')
        self.buttons['pause'].on_clicked(self.toggle_pause)

        for idx, loop in enumerate(self.loops):
            # Plot loop name
            self.axs_loops[idx] = self.fig.add_subplot(self.gs_loops[idx])
            self.axs_loops[idx].set_axis_off() 
            ax_loops = self.axs_loops[idx].inset_axes([0,0,1,1]) #[x0, y0, width, height]
            ax_loops.text(0.5, 0.5, f'{loop.name}', rotation=90, ha='center', va='center', transform=ax_loops.transAxes)
            self.buttons[f'{loop.name}'] = TextBox(ax_loops, label='', textalignment='center')

            row = len(self.loops) - idx
            if idx == 0:
                # Plot output group name
                self.axs_names[idx] = self.fig.add_subplot(self.gs_names[row])
                self.axs_names[idx].set_axis_off() 
                ax_names = self.axs_names[idx].inset_axes([0,0,1,1]) #[x0, y0, width, height]
                ax_names.set_axis_off()
                output_group = 'Actuators'
                ax_names.text(0.5, 0.5, s=output_group, ha='center', va='center', transform=ax_names.transAxes)

                # Plot outputs of loop
                col = len(loop.actions_names)
                sgs = self.gs_selections[row].subgridspec(1, col)
                self.axs_selections[idx] = [self.fig.add_subplot(sgs[i]) for i in range(col)]
                for i, name in enumerate(loop.actions_names):
                    self.axs_selections[idx][i].set_axis_off() 
                    ax_selections = self.axs_selections[idx][i].inset_axes([0,0,1,1]) #[x0, y0, width, height]
                    self.buttons[f'{name}'] = TextBox(ax_selections, label='', label_pad=0.05, initial=f'{name}', color='None', textalignment='center')
                
                # Init probabilities
                self.axs_probabilities[idx] = self.fig.add_subplot(self.gs_probabilities[row])
                self.axs_probabilities[idx].set_axis_off()
                self.axs_probabilities[idx].set_xlim(0, 1)
                self.axs_probabilities[idx].set_ylim(0, 1)
                # Plot a horizontal bar with width=prob
                self.buttons[f'probability_bar_{idx}'] = self.axs_probabilities[idx].bar(
                    x=0.5, width=0.8, height=0, color='tab:cyan')[0]  # Get BarContainer object
                ax_probabilities = self.axs_probabilities[idx].inset_axes([0,0,1,1]) #[x0, y0, width, height]
                ax_probabilities.set_axis_off()
                self.buttons[f'probability_{idx}'] = ax_probabilities.text(0.5, 0, s='', ha='center', va='center', transform=ax_probabilities.transAxes)

            # Plot input group name
            self.axs_names[idx+1] = self.fig.add_subplot(self.gs_names[row-1])
            self.axs_names[idx+1].set_axis_off() 
            ax_names = self.axs_names[idx+1].inset_axes([0,0,1,1]) #[x0, y0, width, height]
            ax_names.set_axis_off()
            input_group = 'Joints' if idx == 0 else 'Grasp type'
            ax_names.text(0.5, 0.5, s=input_group, ha='center', va='center', transform=ax_names.transAxes)

            # Plot inputs of loop  
            col = len(loop.goals_names)
            sgs = self.gs_selections[row-1].subgridspec(1, col)
            self.axs_selections[idx+1] = [self.fig.add_subplot(sgs[i]) for i in range(col)]
            for i, name in enumerate(loop.goals_names):
                    self.axs_selections[idx+1][i].set_axis_off() 
                    ax_selections = self.axs_selections[idx+1][i].inset_axes([0,0,1,1]) #[x0, y0, width, height]
                    self.buttons[f'{name}'] = Button(ax_selections, label=f'{name}', color='None', hovercolor='lightgray')
                    self.buttons[f'{name}'].on_clicked(lambda event, loop=loop, i=i: self.update_goals(loop, i))
            
            if idx == 1:
                # Init probabilities
                self.axs_probabilities[idx] = self.fig.add_subplot(self.gs_probabilities[row])
                self.axs_probabilities[idx].set_axis_off() 
                self.axs_probabilities[idx].set_xlim(0, 1)
                self.axs_probabilities[idx].set_ylim(0, 1)
                # Plot a horizontal bar with width=prob
                self.buttons[f'probability_bar_{idx}'] = self.axs_probabilities[idx].bar(
                    x=0.5, width=0.8, height=0, color='tab:cyan')[0]  # Get BarContainer object
                ax_probabilities = self.axs_probabilities[idx].inset_axes([0,0,1,1]) #[x0, y0, width, height]
                ax_probabilities.set_axis_off()
                self.buttons[f'probability_{idx}'] = ax_probabilities.text(0.5, 0, s='', ha='center', va='center', transform=ax_probabilities.transAxes)
            
    def _init_simulation(self):
        h.dt = 1
        h.finitialize()

    def toggle_pause(self, event=None):
        self.paused = not self.paused
        self.buttons['pause'].label.set_text('Continue' if self.paused else 'Pause')
        if not self.paused:
            self.buttons['pause'].ax.set_facecolor(rcParams['axes.facecolor'])
            self.buttons['pause'].color = '0.85'
        else:
            self.buttons['pause'].ax.set_facecolor('r')
            self.buttons['pause'].color = 'r'
        self.fig.canvas.draw_idle()

    def update_selections(self, frame=None):
        for idx, loop in enumerate(self.loops):
            if loop.selected_goal:
                
                visibility = False if set(loop.selected_goal) == {'0'} else True
                self.update_probability(loop_id=idx, probability=loop.expected_reward_over_time[loop.selected_goal][-1], visibility=visibility)

            if idx == 0:
                for i, name in enumerate(loop.actions_names):
                    if loop.selected_action:
                        char = loop.selected_action[0][i]
                        selected = char == '1'
                        reward = loop.reward_over_time[loop.selected_goal][-1]
                        self.highlight_textbox(name, selected, reward)
            for i, name in enumerate(loop.goals_names):
                    if loop.selected_goal:
                        char = loop.selected_goal[i]
                        selected = char == '1'
                        reward = None
                        if idx < len(self.loops) - 1:
                            if self.loops[idx+1].selected_goal:
                                reward = self.loops[idx+1].reward_over_time[self.loops[idx+1].selected_goal][-1]
                        self.highlight_textbox(name, selected, reward)
    
    def highlight_textbox(self, name, selected, reward):
        if selected:
            if reward:
                self.buttons[f'{name}'].ax.set_facecolor('tab:olive')
                self.buttons[f'{name}'].color = 'tab:olive'
            else:
                self.buttons[f'{name}'].ax.set_facecolor('gold')
                self.buttons[f'{name}'].color = 'gold'
        else:
            self.buttons[f'{name}'].ax.set_facecolor(rcParams['axes.facecolor'])
            self.buttons[f'{name}'].color = 'None'

    def update_goals(self, loop, goal_idx):
        if loop.selected_goal:
            value = loop.selected_goal[goal_idx]=='0'
        else:
            value = 1
        loop.buttons[f'cor_dur_slider{goal_idx}'].set_val(value)
        loop.update_cor_dur(val=value, goal_idx=goal_idx)

    def update_probability(self, loop_id, probability, visibility):
        if visibility:
            self.buttons[f'probability_{loop_id}'].set_text(f'{probability:.1%}')
            self.buttons[f'probability_{loop_id}'].set_position((0.5, probability + 0.03))
            self.buttons[f'probability_bar_{loop_id}'].set_height(probability)
        else:
            self.buttons[f'probability_{loop_id}'].set_text(f'')
            self.buttons[f'probability_bar_{loop_id}'].set_height(0)

    def run_simulation(self):
        state = 0  

        try:
            while True:
                if self.paused or all(all(val == 0 for val in loop.cortical_input_dur_rel) for loop in self.loops): 
                    # Simulation paused
                    time.sleep(0.1)
                    for loop in self.loops:
                        loop.fig.canvas.draw_idle()   
                        loop.fig.canvas.flush_events()
                    self.update_selections()
                    continue
                
                # Update selections in basal ganglia overview plot
                self.update_selections()

                # Run simulation for half of the interval
                h.continuerun(h.t + self.loops[0].plot_interval // 2)

                # --- Action selection and reward update ---#
                if state == 0: # executed after half time of plot_interval
                    for loop in self.loops:
                        loop.analyze_thalamus_activation_time(current_time=h.t)
                        loop.select_action(current_time=h.t)
                        loop.determine_reward(current_time=h.t)

                # --- Weight and plot update ---#  
                else: # executed after full time of plot_interval
                    for loop in self.loops:
                        loop.cortical_input_stimuli(current_time=h.t)
                        loop.update_weights(current_time=h.t)
                        loop.update_plots(current_time=h.t)

                        loop.fig.canvas.draw_idle()   
                        loop.fig.canvas.flush_events() 
                    plt.pause(0.001)

                state = 1 - state # toggle state

                # Pause simulation
                for loop in self.loops:
                    if int(h.t) % loop.simulation_stop_time == 0:
                        self.toggle_pause()
            
        except KeyboardInterrupt:
                print("\nCtrl-C pressed. Storing data...")
                plt.close()

        finally:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            path = f"Data\{timestamp}"

            for loop in self.loops:
                loop.save_data(path) # Excel file
                loop.fig.savefig(f"{path}_{loop.name}.png", dpi=300, bbox_inches='tight') # GUI Screenshot
                        

#--- BasalGangliaLoop ------------------------------------------------------------------------------------------------------------------------------------------------#

class BasalGangliaLoop:

    def __init__(self, name, input, output, binary_input=False, single_goal=False, goal_action_table=None, actions_to_plot=None):
        self.name = name
        self.goals_names = input
        self.goals = [''.join(bits) for bits in product('01', repeat=len(input))] # binary combination of all inputs
        self.selected_goal = None
        self.actions_names = output
        self.actions = [''.join(bits) for bits in product('01', repeat=len(output)) if any(bit == '1' for bit in bits)] # binary combination of all outputs
        self.selected_action = None
        if actions_to_plot is not None:
            self.actions_to_plot = len(self.actions) if len(self.actions) <= actions_to_plot else actions_to_plot
        self.binary_input = binary_input
        self.single_goal = single_goal
        self.goal_action_table = goal_action_table
        self.child_loop = None

        self.cell_types_numbers = {'Cor':  [len(self.goals), 5], 
                                   'SNc':  [len(self.actions), 5], 
                                   'MSNd': [len(self.actions), 5], 
                                   'MSNi': [len(self.actions), 5], 
                                   'GPe':  [len(self.actions), 5], 
                                   'GPi':  [len(self.actions), 5], 
                                   'Thal': [len(self.actions), 5]}
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
            'MSNd'      : 0.5,
            'MSNi'      : 0.5,
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
        self.connection_specs_cor = [# pre_group, post_group, label, e_rev, weight, tau, delay
            ('Cor', 'MSNd', 'Cor_to_MSNd',   0, 0.15,   5, 1),   # excitatory
            ('Cor', 'MSNi', 'Cor_to_MSNi',   0, 0.15,   5, 1)    # excitatory
        ]

        # Define connection specifications
        self.connection_specs = [# pre_group, post_group, label, e_rev, weight, tau, delay
            ('SNc', 'MSNd', 'SNc_to_MSNd',   0, 0,     10, 1),   # excitatory
            ('SNc', 'MSNi', 'SNc_to_MSNi', -85, 0,     10, 1),   # inhibitory
            ('MSNd', 'GPi', 'MSNd_to_GPi', -85, 0.6,   10, 1),   # inhibitory
            ('MSNi', 'GPe', 'MSNi_to_GPe', -85, 1.0,   10, 1),   # inhibitory
            ('GPe',  'GPi',  'GPe_to_GPi', -85, 0.3,   10, 1),   # inhibitory
            ('GPi', 'Thal', 'GPi_to_Thal', -85, 1.0,   10, 1)    # inhibitory
        ]
        self._is_updating_programmatically = False
        #self.paused = False
        self.noise = 0
        self.n_spikes_SNc_burst = 5
        self.learning_rate = 0.1
        self.reward_times = []
        self.activation_times = []
        self.expected_reward_over_time = {}
        self.activation_over_time = {}
        self.cortical_input_dur_rel = [0] * len(self.goals_names)
        self.expected_reward_value = 0.5
        self.reward_over_time = {}
        self.dopamine_over_time = {}
        self.weight_cell_types = ['MSNd', 'MSNi']
        self.weight_times = []
        self.weights_over_time = {(ct, action_id, msn_id, goal_id, cor_id): [] 
                            for ct in self.weight_cell_types
                            for action_id, action in enumerate(self.actions)
                            for msn_id in range(self.cell_types_numbers[ct][1])
                            for goal_id, goal in enumerate(self.goals)
                            for cor_id in range(self.cell_types_numbers['Cor'][1])
                            }
        self.cor_nc_index_map = {} 
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
        self._init_connections(self.connection_specs)
        self._init_connections(self.connection_specs_cor, cor=True)
        self._init_connection_stimuli()
        self._init_reward()
        self._init_activation()
        self._init_recording()
        self._init_plotting()

    def _init_cells(self):
        # Create neuron populations
        self.cells = {
            cell_type: [
                [create_cell(f'{cell_type}_{population}_{index}') for index in range(self.cell_types_numbers[cell_type][1])]
                for population in range(self.cell_types_numbers[cell_type][0])
            ] 
            for cell_type in self.cell_types
        }
    
    def _init_spike_detectors(self):
        # Spike detectors and vectors
        self.spike_times = {
            cell_type: [
                [h.Vector() for index in range(self.cell_types_numbers[cell_type][1])]
                for population in range(self.cell_types_numbers[cell_type][0])
            ] 
            for cell_type in self.cell_types
        }

        for cell_type in self.cell_types:
            for population in range(self.cell_types_numbers[cell_type][0]):
                for index, cell in enumerate(self.cells[cell_type][population]):
                    apc = h.APCount(cell(0.5))
                    apc.record(self.spike_times[cell_type][population][index])
                    self.apc_refs.append(apc)

    def _init_stimuli(self):
        for cell_type in self.cell_types:
            self.stims[cell_type], self.syns[cell_type], self.ncs[cell_type] = [], [], []
            for population in range(self.cell_types_numbers[cell_type][0]):
                for i, cell in enumerate(self.cells[cell_type][population]):
                    if 'SNc' in str(cell):
                        offset = self.stim_intervals['SNc']/2
                    elif 'Cor' in str(cell):
                        offset = 0
                    else:
                        offset = i*self.stim_intervals[cell_type]/self.cell_types_numbers[cell_type][1]
                    stim, syn, nc = create_stim(cell, start=offset, interval=self.stim_intervals[cell_type], weight=self.stim_weights[cell_type], noise=self.noise)
                    self.stims[cell_type].append(stim)
                    self.syns[cell_type].append(syn)
                    self.ncs[cell_type].append(nc)

    def _init_connections(self, connection_specs, cor=False):
        if cor: 
            index = 0

        # Create connections based on the specification
        for pre_group, post_group, label, e_rev, weight, tau, delay in connection_specs:
            self.ncs.update({label: []}) # Additional connections dict to store NetCons
            self.syns.update({label: []}) # Additional connections dict to store ExpSyns

            if cor:
                # Only do index mapping for cortical to striatal connections
                self.cor_nc_index_map[label] = {}
                index = 0
                for population_pre_cell in range(self.cell_types_numbers[pre_group][0]):
                    for pre_cell_id, pre_cell in enumerate(self.cells[pre_group][population_pre_cell]):
                        for population_post_cell in range(self.cell_types_numbers[post_group][0]):
                            for post_cell_id, post_cell in enumerate(self.cells[post_group][population_post_cell]):
                                syn = h.ExpSyn(post_cell(0.5))
                                syn.e = e_rev
                                syn.tau = tau
                                nc = h.NetCon(pre_cell(0.5)._ref_v, syn, sec=pre_cell)
                                nc.weight[0] = weight
                                nc.delay = delay
                                self.syns[label].append(syn)
                                self.ncs[label].append(nc)

                                # Save index mapping if cortical to striatal
                                if label.startswith('Cor_to'):
                                    key = (post_group, population_post_cell, post_cell_id, population_pre_cell, pre_cell_id)
                                    self.cor_nc_index_map[label][key] = index
                                index += 1
            else:
                for action_id, action in enumerate(self.actions):
                    for pre_cell in self.cells[pre_group][action_id]:
                        for post_cell in self.cells[post_group][action_id]:
                            syn = h.ExpSyn(post_cell(0.5))
                            syn.e = e_rev
                            syn.tau = tau
                            nc = h.NetCon(pre_cell(0.5)._ref_v, syn, sec=pre_cell)
                            nc.weight[0] = weight
                            nc.delay = delay
                            self.syns[label].append(syn)
                            self.ncs[label].append(nc)

    def _init_connection_stimuli(self):
        # Additional stimuli for dopamine bursts
        self.stims.update({'SNc_burst': []})
        self.syns.update({'SNc_burst': []})
        self.ncs.update({'SNc_burst': []})

        for a, _ in enumerate(self.actions):
            for cell in self.cells['SNc'][a]:
                stim, syn, nc = create_stim(cell, start=0, interval=self.stim_intervals['SNc_burst'], weight=self.stim_weights['SNc_burst'], noise=self.noise)
                self.stims['SNc_burst'].append(stim)
                self.syns['SNc_burst'].append(syn)
                self.ncs['SNc_burst'].append(nc)
        for nc in self.ncs['SNc_burst']:
            nc.active(False)

        # Initialize weights over time
        self.weight_times.append(0)
        for ct in self.weight_cell_types:
            for action_id, action in enumerate(self.actions):
                for msn_id in range(self.cell_types_numbers[ct][1]):
                    for goal_id, goal in enumerate(self.goals):
                        for cor_id in range(self.cell_types_numbers['Cor'][1]):
                            for _, post_group, _, _, weight, _, _ in self.connection_specs_cor:
                                if ct == post_group:
                                    self.weights_over_time[(ct, action_id, msn_id, goal_id, cor_id)].append(weight)
        for nc in self.ncs[f'Cor']:
            nc.active(False)

    def _init_reward(self):
        # Reward initialization
        self.reward_times.append(0)
        for goal in self.goals:
            #for goal_id, _ in enumerate(self.goals):
            #    input_key = f"{goal_id}-{action}"
            #    self.expected_reward_over_time[input_key] = [self.expected_reward_value]
            self.expected_reward_over_time[goal] = [self.expected_reward_value]
            self.reward_over_time[goal] = [0]
            self.dopamine_over_time[goal] = [0]
        
    def _init_activation(self):
        # Activation initialization
        self.activation_times.append(0)
        for action in self.actions:
            self.activation_over_time[action] = [0]

    def _init_recording(self):
        # Recording
        self.recordings = {ct: [[h.Vector().record(cell(0.5)._ref_v) for cell in self.cells[ct][population]] for population in range(self.cell_types_numbers[ct][0])] for ct in self.cell_types}
        self.t_vec = h.Vector().record(h._ref_t)

    def _init_plotting(self):
        plt.ion()
        self.fig = plt.figure(figsize=(13, 8))
        self.fig.canvas.manager.set_window_title(self.name)
        self.rows = 3
        self.gs = gridspec.GridSpec(2, 1, figure=self.fig, height_ratios=[1, 2 * self.rows])
        self.gs_control = self.gs[0].subgridspec(1, 2 + len(self.goals_names), width_ratios=[1] + [1] * len(self.goals_names) + [2])
        self.gs_plot = self.gs[1].subgridspec(self.rows, self.actions_to_plot)
        self.axs_control = [self.fig.add_subplot(self.gs_control[i]) for i in range(2 + len(self.goals_names))]
        #self.axs_plot = [[self.fig.add_subplot(self.gs_plot[i, j]) for j in range(self.actions_to_plot)] for i in range(self.rows)]
        self.axs_plot = []
        for i in range(self.rows):
            row = []
            for j in range(self.actions_to_plot):
                if j == 0:
                    ax = self.fig.add_subplot(self.gs_plot[i, j])
                else:
                    ax = self.fig.add_subplot(self.gs_plot[i, j], sharey=row[0])
                row.append(ax)
            self.axs_plot.append(row)
        for i in range(self.rows):
            for j in range(1, self.actions_to_plot):  # skip first in row
                plt.setp(self.axs_plot[i][j].get_yticklabels(), visible=False)

        [self.row_potential, self.row_spike, self.row_weights] = list(range(self.rows))
        
        # Deactivate axis for control axes
        for ax in self.axs_control:
            ax.set_axis_off() # deactivate axis
        self.axs_control[-1].set_axis_on() # activate axis for reward plot
        
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
            self.axs_plot[self.row_potential][i].xaxis.set_major_formatter(ms_to_s)
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
                for k in range(self.cell_types_numbers[ct][1]):
                    line, = self.axs_plot[self.row_spike][i].plot([], [], f'C{j}.', markersize=3)
                    self.raster_lines[ct][i].append(line)

                rate_line, = self.axs_plot[self.row_spike][i].step([], [], f'C{j}')
                self.rate_lines[ct].append(rate_line)
            self.axs_plot[self.row_spike][i].plot([], [], color='black', linestyle='dotted', label=f'Spikes')
            self.axs_plot[self.row_spike][i].plot([], [], color='black', label=f'Relative rate')

            self.total_cells = sum(self.cell_types_numbers[ct][1] for ct in self.cell_types)
            
            y_max = self.total_cells + 1.5
            self.axs_plot[self.row_spike][i].set_ylim(0.5, y_max)
            yticks = []
            cumulative = 0
            for ct in self.cell_types:
                mid = y_max - (cumulative + (self.cell_types_numbers[ct][1]+1) / 2)
                yticks.append(mid)
                cumulative += self.cell_types_numbers[ct][1] 
            self.axs_plot[self.row_spike][i].set_xlim(0, self.plot_interval)
            self.axs_plot[self.row_spike][i].set_yticks(yticks)
            self.axs_plot[self.row_spike][i].set_yticklabels(self.cell_types)
            self.axs_plot[self.row_spike][i].xaxis.set_major_formatter(ms_to_s)
        self.axs_plot[self.row_spike][-1].legend(loc='upper right')

    def _init_weight_plot(self):
        # Weight plot
        self.axs_plot[self.row_weights][0].set_ylabel('Cortical input weight')
        self.weight_lines = {ct: [[] for _ in range(self.actions_to_plot)] for ct in self.weight_cell_types}
        self.activation_lines = []

        for i, ch in enumerate(range(self.actions_to_plot)):
            for j, ct in enumerate(self.cell_types):
                if ct in self.weight_cell_types:
                    self.weight_lines[ct][i] = []
                    for k in range(self.cell_types_numbers[ct][1]):
                        label = ct if k == 0 else None  # Only label the first line
                        line, = self.axs_plot[self.row_weights][i].step([], [], f'C{j}', label=label, where='post')
                        self.weight_lines[ct][i].append(line)

            activation_line, = self.axs_plot[self.row_weights][i].step([], [], 'C7', linestyle='dashed', label='Activation', where='post')
            self.activation_lines.append(activation_line)

            self.axs_plot[self.row_weights][i].set_xlim(0, self.plot_interval)
            self.axs_plot[self.row_weights][i].set_ylim(-0.1, 1.1)
            self.axs_plot[self.row_weights][i].xaxis.set_major_formatter(ms_to_s)
            self.axs_plot[self.row_weights][i].set_xlabel('Simulation time (s)')
        self.axs_plot[self.row_weights][-1].legend(loc='upper right')

    def _init_reward_plot(self):
        # Reward plot
        self.expected_reward_lines = []
        self.reward_lines = []
        self.dopamine_lines = []

        self.expected_reward_lines, = self.axs_control[-1].plot([], [], 'C9', label='Expected reward')
        self.reward_lines, = self.axs_control[-1].step([], [], 'C8', label='Reward', where='post')
        self.dopamine_lines, = self.axs_control[-1].plot([], [], 'C6', label='Dopamine')

        self.axs_control[-1].xaxis.set_major_formatter(ms_to_s)
        self.axs_control[-1].set_xlabel('Simulation time (s)')
        self.axs_control[-1].set_xlim(0, self.plot_interval)
        self.axs_control[-1].set_ylim(-1.1, 1.1)
        self.axs_control[-1].legend(loc='upper left')

    def _init_control_panel(self):
        #--- Upper control panel ---#
        #print(f"{self.name} width={self.axs_control[0].get_position().width}, height={self.axs_control[0].get_position().height}")
        #ax_pause = self.axs_control[0].inset_axes([0,0.5,1,0.5]) #[x0, y0, width, height]
        #self.buttons['pause'] = Button(ax_pause, 'Pause')
        #self.buttons['pause'].on_clicked(self.toggle_pause)
        
        ax_noise = self.axs_control[0].inset_axes([0.4,0,0.5,0.45]) #[x0, y0, width, height]
        self.buttons['noise_slider'] = Slider(ax_noise, 'Noise', 0, 1, valinit=self.noise, valstep=0.1)
        self.buttons['noise_slider'].on_changed(self.update_stim)

        for i, goal_name in enumerate(self.goals_names):
            ax_cor_dur = self.axs_control[1+i].inset_axes([0,0,0.9,0.45]) #[x0, y0, width, height]
            ax_cor_dur.set_title(goal_name)
            self.buttons[f'cor_dur_slider{i}'] = Slider(ax_cor_dur, '', 0, 1, valinit=self.cortical_input_dur_rel[i], valstep=1 if self.binary_input else 0.2)
            self.buttons[f'cor_dur_slider{i}'].on_changed(lambda val, i=i: self.update_cor_dur(val=val, goal_idx=i))

    def set_child_loop(self, child_loop):
        self.child_loop = child_loop
        #self.child_loop.buttons['pause'].ax.set_visible(False)

    def update_stimulus_activation(self, cell, stimulus, index, active=True):
        i = 0
        for population in range(self.cell_types_numbers[cell][0]):
            for _ in self.cells[cell][population]:
                if population == index:
                    self.ncs[stimulus][i].active(active)
                i += 1

    def SNc_dip(self, event=None, action=None):
        self.update_stimulus_activation(cell='SNc', stimulus='SNc', index=self.actions.index(action), active=False) # stop SNc tonic stimulus
        h.cvode.event(h.t + self.stim_intervals['SNc'], lambda action=action: self.update_stimulus_activation(cell='SNc', stimulus='SNc', index=self.actions.index(action), active=True))  # start SNc tonic stimulus

    def SNc_burst(self, event=None, action=None, n_spikes=None):
        self.update_stimulus_activation(cell='SNc', stimulus='SNc', index=self.actions.index(action), active=False) # stop SNc tonic stimulus
        self.update_stimulus_activation(cell='SNc', stimulus='SNc_burst', index=self.actions.index(action), active=True) # start SNc burst stimulus
        if n_spikes == None:
            n_spikes = self.n_spikes_SNc_burst
        delay = self.stim_intervals['SNc_burst'] * n_spikes
        h.cvode.event(h.t + delay, lambda action=action: self.update_stimulus_activation(cell='SNc', stimulus='SNc_burst', index=self.actions.index(action), active=False))  # stop SNc burst stimulus
        h.cvode.event(h.t + delay, lambda action=action: self.update_stimulus_activation(cell='SNc', stimulus='SNc', index=self.actions.index(action), active=True))  # start SNc tonic stimulus

    def analyze_firing_rate(self, cell, window=None, average=True):
        """Returns a list of firing rates (Hz) for each action's cell population."""
        current_time = h.t
        rates_avg = []
        rates = []
        if window == None:
            window = self.bin_width_firing_rate
        for a, _ in enumerate(self.actions):
            spikes_avg = 0
            spikes = []
            for i in range(self.cell_types_numbers[cell][1]):
                spike_vec = self.spike_times[cell][a][i]
                # Count spikes in the last `window` ms
                recent_spikes = [t for t in spike_vec if current_time - window <= t <= current_time]
                if average:
                    spikes_avg += len(recent_spikes)
                else:
                    spikes.append(len(recent_spikes))
            if average:
                rate_avg = spikes_avg / (self.cell_types_numbers[cell][1] * (window / 1000.0))  # spikes/sec per neuron
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
    '''
    def toggle_pause(self, event=None):
        self.paused = not self.paused
        self.buttons['pause'].label.set_text('Continue' if self.paused else 'Pause')
        if not self.paused:
            self.buttons['pause'].color = '0.85'
        else:
            self.buttons['pause'].color = 'c'
        self.fig.canvas.draw_idle()
    '''
    def update_stim(self, val):
        self.noise = val
        for ct in self.cell_types:
            for stim in self.stims[ct]:
                stim.noise = self.noise

    def update_cor_dur(self, val, goal_idx):
        if self._is_updating_programmatically:
            return
        self.cortical_input_dur_rel[goal_idx] = val
        if self.single_goal:
            self.reset_other_durations(goal_idx)

        self.update_selected_goal()

    def reset_other_durations(self, goal_idx):
        self._is_updating_programmatically = True
        for i, _ in enumerate(self.goals_names):
            if i is not goal_idx:
                self.cortical_input_dur_rel[i] = 0
                self.buttons[f'cor_dur_slider{i}'].set_val(0)
        self._is_updating_programmatically = False

    def update_selected_goal(self):
        self.selected_goal = ''.join('1' if val != 0 else '0' for val in self.cortical_input_dur_rel)

    def cortical_input_stimuli(self, current_time):
        selected_goal_index = self.goals.index(self.selected_goal)

        for idx, _ in enumerate(self.goals):
            h.cvode.event(current_time + 1, lambda: self.update_stimulus_activation(cell='Cor', stimulus=f'Cor', index=idx, active=False)) # stop all cortical input stimuli
        
        if set(self.selected_goal) != {'0'}: # Only stimulate cortex if a goal is set
            h.cvode.event(current_time + 1, lambda: self.update_stimulus_activation(cell='Cor', stimulus=f'Cor', index=selected_goal_index, active=True)) # start particular cortical input stimulus
            h.cvode.event(current_time + self.plot_interval/2, lambda: self.update_stimulus_activation(cell='Cor', stimulus=f'Cor', index=selected_goal_index, active=False)) # stop particular cortical input stimulus

    def analyze_thalamus_activation_time(self, current_time):
        self.activation_times.append(int(current_time))

        max_rates = {}
        window_start = np.array(self.t_vec.to_python())[-1] - self.plot_interval / 2
        window_end = np.array(self.t_vec.to_python())[-1]
        window_duration_sec = (window_end - window_start) / 1000.0  # Convert ms to seconds

        for action_id, action in enumerate(self.actions):
            # Collect all thalamic spike times for this action
            all_spikes = []
            for k in range(self.cell_types_numbers['Thal'][1]):
                spikes = np.array(self.spike_times['Thal'][action_id][k].to_python())
                all_spikes.extend(spikes)

            # Filter spikes within the last window
            spikes_in_window = [spk for spk in all_spikes if window_start < spk <= window_end]

            # Compute firing rate: total spikes / (number of cells * window duration)
            num_cells = self.cell_types_numbers['Thal'][1]
            rate = len(spikes_in_window) / (num_cells * window_duration_sec) if window_duration_sec > 0 else 0

            max_rates[action] = rate

        # Find the action with the highest thalamus firing rate
        best_action = max(max_rates, key=max_rates.get, default=None)
        #print(f"{self.name} action = {best_action} max_rate = {max_rates[best_action]}")

        # Store 1 for the best action, 0 for the rest
        for action in self.actions:
            self.activation_over_time[action].append(1 if action == best_action else 0)
  
    def select_action(self, current_time):
        active_actions = [(i, activations[-1]) for i, activations in self.activation_over_time.items() if activations[-1] > 0]
        if active_actions and self.selected_goal and set(self.selected_goal) != {'0'}:
            # Pick the action with the maximum last activation value
            self.selected_action = max(active_actions, key=lambda x: x[1])
        else:
            # No active actions found
            self.selected_action = []

        if self.child_loop and self.selected_action and set(self.selected_goal) != {'0'}:
            # Set input of child loop based on output of current loop
            for id, state in enumerate(list(map(int, self.selected_action[0]))):
                self.child_loop.buttons[f'cor_dur_slider{id}'].set_val(0 if state == 0 else self.selected_action[1])
            self.child_loop.update_selected_goal()

    def determine_reward(self, current_time):
        self.reward_times.append(int(current_time))
        
        goal_state = tuple((goal_name, dur != 0) for goal_name, dur in zip(self.goals_names, self.cortical_input_dur_rel))
        target_actions = self.goal_action_table.get(goal_state, {})
        target_action_indices = ''.join('1' if v else '0' for v in target_actions.values())
        
        for goal in self.goals:
            if goal == self.selected_goal and self.selected_action and self.selected_action[0] == target_action_indices:
                self.reward_over_time[goal].append(1)
            else:
                self.reward_over_time[goal].append(0)
                
            current_expected_reward = self.expected_reward_over_time[goal][-1]
            self.dopamine_over_time[goal].append(round(self.reward_over_time[goal][-1] - current_expected_reward, 4)) # TODO: determine dopamine from relative rate of SNc
            for action in self.actions:
                if self.reward_over_time[goal][-1] - current_expected_reward > 0:
                    self.SNc_burst(event=None, action=action)
                elif self.reward_over_time[goal][-1] - current_expected_reward < 0:
                    self.SNc_dip(event=None, action=action)

            # Update expected reward based on actual reward
            self.expected_reward_over_time[goal].append(round(current_expected_reward + 0.1 * (self.reward_over_time[goal][-1] - current_expected_reward), 4))

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
        for goal_id, goal in enumerate(self.goals):
            for cor_id in range(self.cell_types_numbers['Cor'][1]):
                for action_id, action in enumerate(self.actions):
                    for ct in self.weight_cell_types:
                        for msn_id in range(self.cell_types_numbers[ct][1]):
                            delta_w = 0
                            if goal == self.selected_goal and self.selected_action and action == self.selected_action[0]:
                                delta_w = self.learning_rate * (self.rates_rel[ct][action_id][msn_id] - 1) * self.dopamine_over_time[goal][-1] # rel_rate = 1 corresponds to tonic baseline activity
                            if ct == 'MSNd':
                                # dopamine facilitates active MSNd and inhibits less active MSNd
                                delta_w = delta_w 
                            elif ct == 'MSNi':
                                # high dopamine increases inhibition, low dopamine suppresses inhibition
                                delta_w = - delta_w
                            key = (ct, action_id, msn_id, goal_id, cor_id)
                            new_weight = max(0, self.weights_over_time[key][-1] + delta_w) # Update weight ensure weight is non-zero
                            self.weights_over_time[key].append(round(new_weight, 4))
                            
                            #idx = goal_id * self.cell_types_numbers['Cor'][1] * self.cell_types_numbers[ct][0] * self.cell_types_numbers[ct][1] + action_id * self.cell_types_numbers[ct][1] + k
                            #key = (goal_id, cor_id, action_id, msn_id)
                            idx = self.cor_nc_index_map[f'Cor_to_{ct}'][key]
                            self.ncs[f'Cor_to_{ct}'][idx].weight[0] = new_weight # update weight of cortical input stimulation
                    if output: print(f"{self.weight_times[-1]} ms: Action {action}: Goal {goal} rel rate MSNd = {[f'{rate_rel:.2f}' for rate_rel in self.rates_rel['MSNd'][action_id]]}, rel rate SNc = {self.rates_rel['SNc'][action_id]:.2f}, Exp. Reward = {self.expected_reward_over_time[f'{goal}{self.cortical_input_dur_rel[goal] != 0}'][-1]:.2f}, DA = {self.dopamine_over_time[goal][-1]}, Cor-MSNd-Weights = {[f'{nc.weight[0]:.2f}' for nc in self.ncs['Cor_to_MSNd'][action_id*self.cell_types_numbers['MSNd'][1]:(action_id+1)*self.cell_types_numbers['MSNd'][1]]]}, Cor-MSNi-Weights = {[f'{nc.weight[0]:.2f}' for nc in self.ncs['Cor_to_MSNi'][action_id*self.cell_types_numbers['MSNi'][1]:(action_id+1)*self.cell_types_numbers['MSNi'][1]]]}")               
            
    def update_plots(self, current_time):
        selected_goal_index = self.goals.index(self.selected_goal)
        
        # Update plots
        for action_id in range(self.actions_to_plot):
            action = self.actions[action_id]
            # Membrane potential plot
            for ct in self.cell_types:
                if ct == 'Cor':
                    voltages = np.array([np.array(self.recordings[ct][selected_goal_index][j]) for j in range(self.cell_types_numbers[ct][1])])
                else:
                    voltages = np.array([np.array(self.recordings[ct][action_id][j]) for j in range(self.cell_types_numbers[ct][1])])
                avg_voltage = np.mean(voltages, axis=0)
                self.mem_lines[ct][action_id].set_data(np.array(self.t_vec), avg_voltage)
                self.axs_plot[self.row_potential][action_id].set_xlim(max(0, int(current_time) - self.plot_interval), max(self.plot_interval, int(current_time)))

            # Spike raster plot
            y_base = self.total_cells
            for ct in self.cell_types:
                all_spikes = []
                for k in range(self.cell_types_numbers[ct][1]):
                    if ct == 'Cor':
                        spikes = np.array(self.spike_times[ct][selected_goal_index][k].to_python())
                    else:
                        spikes = np.array(self.spike_times[ct][action_id][k].to_python())
                    y_val = y_base - k
                    y_vals = np.ones_like(spikes) * y_val
                    self.raster_lines[ct][action_id][k].set_data(spikes, y_vals)
                    all_spikes.extend(spikes)
                # Rate lines
                if len(all_spikes) > 0:
                    bins = np.arange(0, np.array(self.t_vec)[-1], self.bin_width_firing_rate)
                    hist, edges = np.histogram(all_spikes, bins=bins)
                    if np.any(hist):  # Only proceed if there's at least one spike
                        rate = hist / (self.cell_types_numbers[ct][1] * self.bin_width_firing_rate / 1000.0)
                        bin_ends = edges[1:]
                        if ct == 'SNc':
                            spike_rate_max = 1000.0 / self.stim_intervals['SNc_burst'] # Hz
                        elif ct == 'MSNd' or ct == 'MSNi':
                            spike_rate_max = 1000.0 / self.stim_intervals['Cor'] # Hz
                        else:
                            spike_rate_max = 1000.0 / self.stim_intervals[ct] # Hz
                        rate_scaled = (rate) / (spike_rate_max + 1e-9)
                        rate_scaled = rate_scaled * (self.cell_types_numbers[ct][1] - 1) + y_base - self.cell_types_numbers[ct][1] + 1
                        self.rate_lines[ct][action_id].set_data(bin_ends, rate_scaled)
                    else:
                        self.rate_lines[ct][action_id].set_data([], [])
                y_base -= self.cell_types_numbers[ct][1]
            self.axs_plot[self.row_spike][action_id].set_xlim(max(0, int(current_time) - self.plot_interval), max(self.plot_interval, int(current_time)))

            # Weight plot
            for ct in self.weight_cell_types:
                for msn_id in range(self.cell_types_numbers[ct][1]):
                    for cor_id in range(self.cell_types_numbers['Cor'][1]):
                        self.weight_lines[ct][action_id][msn_id].set_data(self.weight_times, self.weights_over_time[(ct, action_id, msn_id, selected_goal_index, cor_id)])
            self.activation_lines[action_id].set_data(self.activation_times, self.activation_over_time[action])
            self.axs_plot[self.row_weights][action_id].set_xlim(0, max(self.plot_interval, int(current_time)))
            all_weights = [w for lst in self.weights_over_time.values() for w in lst if lst]  # flatten and exclude empty lists
            ymin, ymax = min(all_weights), max(all_weights)
            self.axs_plot[self.row_weights][action_id].set_ylim(-0.1, max(1.1, ymax*1.1))

        # Reward plot
        self.expected_reward_lines.set_data(self.reward_times, self.expected_reward_over_time[self.selected_goal])
        self.reward_lines.set_data(self.reward_times, self.reward_over_time[self.selected_goal])
        self.dopamine_lines.set_data(self.reward_times, self.dopamine_over_time[self.selected_goal])
        self.axs_control[-1].set_xlim(0, max(self.plot_interval, int(current_time)))
    
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

        #row = write_dict("cell_types_numbers", self.cell_types_numbers, row)
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
            #"N_actions": self.N_actions,
            "plot_interval": self.plot_interval,
            "bin_width_firing_rate": self.bin_width_firing_rate,
            "n_spikes_SNc_burst": self.n_spikes_SNc_burst,
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
        header.extend(f"{ct}_a{action_id}_msn{msn_id}_g{goal_id}_cor{cor_id}" for ct, action_id, msn_id, goal_id, cor_id in keys)
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

def create_goal_action_table(mapping, goals, actions):

    goal_action_table = {}

    all_goal_combinations = [''.join(bits) for bits in product("01", repeat=len(goals))]

    for goal_combo in all_goal_combinations:
        # Get the corresponding action string, or default to "0" * len(actions)
        action_combo = mapping.get(goal_combo, "0" * len(actions))
        
        # Build the key: a tuple of (goal, True/False) tuples
        key = tuple((goal, bit == "1") for goal, bit in zip(goals, goal_combo))
        
        # Build the value: a dict of {action: True/False}
        value = {action: bit == "1" for action, bit in zip(actions, action_combo)}
        
        # Add to final mapping
        goal_action_table[key] = value

    return goal_action_table

# Formatter function: 1000 ms → 1 s
ms_to_s = FuncFormatter(lambda x, _: f'{int(x/1000)}' if x % 1000 == 0 else '')


# Basal Ganglia Loops
#grasp_types_dict = {"Precision pinch": [1, 1, 1, 0, 0, 0],
#                    "Power grasp":     [1, 1, 1, 1, 1, 1]}
#joints = ["Thumb opposition", "Thumb flexion", "Index finger flexion", "Middle finger flexion", "Ring finger flexion", "Pinky finger flexion"]
#actuators = ["Thumb oppositor", "Thumb abductor", "Thumb flexor", "Thumb extensor", "Index finger flexor", "Index finger extensor", "Middle finger flexor", "Middle finger extensor", "Ring finger flexor", "Ring finger extensor", "Pinky finger flexor", "Pinky finger extensor"]
#actuators = ["Thumb oppositor", "Thumb flexor", "Index finger flexor", "Middle finger flexor", "Ring finger flexor", "Pinky finger flexor"]

grasp_types = ["Precision pinch", "Power grasp"]
joints = ["Thumb flexion", "Index finger flexion", "Middle finger flexion"]
actuators = ["Thumb flexor", "Index finger flexor", "Middle finger flexor"]
grasp_type_joint_mapping = {"10": "110", # Precision pinch
                            "01": "111"} # Power grasp
grasp_type_joint_table = create_goal_action_table(mapping=grasp_type_joint_mapping, goals=grasp_types, actions=joints)
#print(grasp_type_joint_table)

joint_actuator_mapping = {"100": "100", 
                          "010": "010",
                          "001": "001",
                          "110": "110",
                          "101": "101",
                          "011": "011",
                          "111": "111"
                          }
joint_actuator_table = create_goal_action_table(mapping=joint_actuator_mapping, goals=joints, actions=actuators)





#--- Basal Ganglia ---------------------------------------------------------------------------------------------------------------------------------------------------#
bg_m = BasalGangliaLoop('MotorLoop', input=joints, output=actuators, goal_action_table=joint_actuator_table, actions_to_plot=7)
bg_p = BasalGangliaLoop('PrefrontalLoop', input=grasp_types, output=joints, binary_input=True, single_goal=True, goal_action_table=grasp_type_joint_table, actions_to_plot=7)
#bg = [bg_m, bg_p]
bg = BasalGanglia(loops=[bg_m, bg_p]) # loops ordered from low-level to high-level



