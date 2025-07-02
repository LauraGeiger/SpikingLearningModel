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
selection_threshold = 0.7
n_spikes_SNc_burst = 5
learning_rate = 0.05
reward_times = []
activation_times = []
expected_reward_over_time = {}
activation_over_time = {}
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
channels_to_plot = [i for i in range(N_actions)]
bin_width_firing_rate = plot_interval / 10  # ms
cortical_input_dur = [round(1/2 * plot_interval)] * N_actions
simulation_stop_time = 20000 # ms

buttons = {}
rates = {}
rates_rel = {}

output = False # Set to True (print values in the terminal) or False (no printing)

#--- Functions ------------------------------------------------------------------------------------------------------------------------------------------------#

def create_cell(name):
    sec = h.Section(name=name)
    sec.insert('hh')
    return sec

# Stimulus helper
def create_stim(cell, start=0, number=1e9, interval=10, weight=2):
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

            target_activation_lines[action][0].set_visible(False)

            h.cvode.event(h.t + 1,  lambda: update_stimulus_activation(cell='MSNd', stimulus=f'Cor_MSNd', actions=[action], active=False)) # stop cortical input stimulus for that action
            h.cvode.event(h.t + 1,  lambda: update_stimulus_activation(cell='MSNi', stimulus=f'Cor_MSNi', actions=[action], active=False)) # stop cortical input stimulus for that action
            
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
    global cortical_input_dur
    cortical_input_dur[action] = round(val * 1/2 * plot_interval)
    target_activation_lines[action][0].set_ydata([val, val])

#--- Network ------------------------------------------------------------------------------------------------------------------------------------------#

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
            #random_post_cell = random.sample(cells[post_group][a], k=round((cell_types_numbers[post_group])/2.0)) # sparse connection: connect pre group randomly to half of post group
            #for post_cell in random_post_cell:
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

# Recording
recordings = {ct: [[h.Vector().record(cell(0.5)._ref_v) for cell in cells[ct][a]] for a in range(N_actions)] for ct in cell_types}
t_vec = h.Vector().record(h._ref_t)

#--- Plotting ------------------------------------------------------------------------------------------------------------------------------------------#

plt.ion()
fig = plt.figure(figsize=(13, 8))
rows = 5
gs = gridspec.GridSpec(rows, 1+len(channels_to_plot), height_ratios=[2]+[4]*(rows-1), width_ratios=[0.3]+[1]*len(channels_to_plot))
axs = [[fig.add_subplot(gs[i, j]) for j in range(1+len(channels_to_plot))] for i in range(rows)]
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

for i, ch in enumerate(channels_to_plot):
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

for i, ch in enumerate(channels_to_plot):
    for j, ct in enumerate(cell_types):
        raster_lines[ct][i] = []
        for k in range(cell_types_numbers[ct]):
            line, = axs[row_spike][col_spike+i].plot([], [], f'C{j}.', markersize=3)
            raster_lines[ct][i].append(line)

        rate_line, = axs[row_spike][col_spike+i].plot([], [], f'C{j}')#, label=f'{ct} rate')
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

for i, ch in enumerate(channels_to_plot):
    for j, ct in enumerate(cell_types):
        if ct in weight_cell_types:
            weight_lines[ct][i] = []
            for k in range(cell_types_numbers[ct]):
                label = ct if k == 0 else None  # Only label the first line
                line, = axs[row_weights][col_weights+i].plot([], [], f'C{j}', label=label)
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

for i, ch in enumerate(channels_to_plot):
    expected_reward_line, = axs[row_reward][col_reward+i].plot([], [], 'C9', label='Expected reward')
    reward_line, = axs[row_reward][col_reward+i].plot([], [], 'C8', label='Reward')
    dopamine_line, = axs[row_reward][col_reward+i].plot([], [], 'C6', label='Dopamine')
    activation_line, = axs[row_reward][col_reward+i].plot([], [], 'C7', linestyle='dotted', label='Activation time')
    target_activation_line = axs[row_reward][col_reward+i].axhline(y=cortical_input_dur[i] / (1/2 * plot_interval), color='blue', linestyle='dashed', label=f'Target act. time')
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
    buttons[f'cor_dur_slider{a}'] = Slider(ax_cor_dur, 'Input Dur', 0.2, 1, valinit=cortical_input_dur[a] / (1/2 * plot_interval), valstep=0.2)
    buttons[f'cor_dur_slider{a}'].on_changed(lambda val, a=a: update_cor_dur(val=val, action=a))


plt.show()
plt.tight_layout()
    
#--- Simulation ---------------------------------------------------------------------------------------------------------------------------------------------------#
h.dt = 0.1
h.finitialize()

# Define cortical input stimuli
for action in target_actions:
    h.cvode.event(h.t + 1, lambda: update_stimulus_activation(cell='MSNd', stimulus=f'Cor_MSNd', actions=[action], active=True))  # start cortical input stimulus for that action
    h.cvode.event(h.t + cortical_input_dur[action],  lambda: update_stimulus_activation(cell='MSNd', stimulus=f'Cor_MSNd', actions=[action], active=False)) # stop cortical input stimulus for that action
    h.cvode.event(h.t + 1, lambda: update_stimulus_activation(cell='MSNi', stimulus=f'Cor_MSNi', actions=[action], active=True))  # start cortical input stimulus for that action
    h.cvode.event(h.t + cortical_input_dur[action],  lambda: update_stimulus_activation(cell='MSNi', stimulus=f'Cor_MSNi', actions=[action], active=False)) # stop cortical input stimulus for that action

state = 0  
try:
    while True:
        if paused or len(target_actions) == 0:
            # Simulation paused
            time.sleep(0.1)
            fig.canvas.draw_idle()   
            fig.canvas.flush_events()
            plt.pause(0.1)
            continue

        # Run simulation for half of the interval
        h.continuerun(h.t + plot_interval // 2)

        t_array = np.array(t_vec)

        

        # --- Action selection and SNc dip/burst trigger ---#
        if state == 0: # executed after half time of plot_interval
            activation_times.append(int(h.t))

            for i, ch in enumerate(range(N_actions)):
                for ct in cell_types:
                    all_spikes = []
                    for k in range(cell_types_numbers[ct]):
                        spikes = np.array(spike_times[ct][ch][k].to_python())
                        all_spikes.extend(spikes)
                    bins = np.arange(0, t_array[-1], bin_width_firing_rate)
                    hist, edges = np.histogram(all_spikes, bins=bins)
                    if np.any(hist):  # Only proceed if there's at least one spike
                        rate = hist / (cell_types_numbers[ct] * bin_width_firing_rate / 1000.0)
                        bin_centers = (edges[:-1] + edges[1:]) / 2
                        
                        if ct == 'Thal':
                            window_start = t_array[-1] - plot_interval/2
                            # Get the indices of bins in the last window
                            bin_indices = np.where(bin_centers > window_start)[0]
                            rate_window = rate[bin_indices]
                            bin_centers_window = bin_centers[bin_indices]
                            edges_window = edges[bin_indices[0] : bin_indices[-1] + 2]  # include right edge of last bin

                            indices = np.where(rate_window > selection_threshold * 1000.0 / stim_intervals[ct])[0]
                            
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
                            activation_over_time[i].append(longest_duration/(plot_interval/2))
                            
            # Select actions
            rates['Thal'], rates_rel['Thal'] = analyse_firing_rate('Thal', window=plot_interval/2)
            #selected_actions = [i for i, rate_rel in enumerate(rates_rel['Thal']) if rate_rel > selection_threshold]
            selected_actions = [i for i, activations in activation_over_time.items() if activations[-1] > 0]
            if output: print(f"{int(h.t)} ms: Target Actions = {target_actions}, Selected Actions = {selected_actions}, Rates Thal = {rates['Thal']}, Rates Thal relative = {rates_rel['Thal']}")
            
            correct_actions = list(set(target_actions) & set(selected_actions))
            incorrect_actions = list(set(target_actions) ^ set(selected_actions))
            reward_times.append(int(h.t))
            for action in range(N_actions):
                # Determine reward
                #if  not ((action in target_actions) ^ (action in selected_actions)): #XNOR
                if (action in target_actions) and (action in selected_actions): 
                    reward_over_time[action].append(1)
                else:
                    reward_over_time[action].append(0)

                # Trigger SNc dips or bursts based on difference between actual reward and expected reward
                input_key = f"{action}{action in target_actions}"
                current_expected_reward = expected_reward_over_time[input_key][-1]
                dopamine_over_time[action].append(round(reward_over_time[action][-1] - current_expected_reward, 4)) # TODO: determine dopamine from relative rate of SNc
                if reward_over_time[action][-1] - current_expected_reward > 0:
                    SNc_burst(event=None, actions=[action])
                elif reward_over_time[action][-1] - current_expected_reward < 0:
                    SNc_dip(event=None, actions=[action])

                # Update expected reward based on actual reward
                expected_reward_over_time[input_key].append(round(current_expected_reward + 0.1 * (reward_over_time[action][-1] - current_expected_reward), 4))

                # Repeat latest expected reward value when input is not triggered
                alternative_input_key = f"{action}{action not in target_actions}"
                expected_reward_over_time[alternative_input_key].append(expected_reward_over_time[alternative_input_key][-1])

        # --- Weight and plot update ---#  
        else: # executed after full time of plot_interval

            # Define cortical input stimuli
            h.cvode.event(h.t + 1, lambda: update_stimulus_activation(cell='MSNd', stimulus=f'Cor_MSNd', actions=target_actions, active=True))  # start cortical input stimulus for that action
            h.cvode.event(h.t + 1, lambda: update_stimulus_activation(cell='MSNi', stimulus=f'Cor_MSNi', actions=target_actions, active=True))  # start cortical input stimulus for that action
            for action in target_actions:
                h.cvode.event(h.t + cortical_input_dur[action], lambda action=action: update_stimulus_activation(cell='MSNd', stimulus=f'Cor_MSNd', actions=[action], active=False)) # stop cortical input stimulus for that action
                h.cvode.event(h.t + cortical_input_dur[action], lambda action=action: update_stimulus_activation(cell='MSNi', stimulus=f'Cor_MSNi', actions=[action], active=False)) # stop cortical input stimulus for that action
            
            # Analyse firing rates
            rates['SNc'], rates_rel['SNc'] = analyse_firing_rate('SNc', window=plot_interval)
            rates['MSNd'], rates_rel['MSNd'] = analyse_firing_rate('MSNd', window=plot_interval, average=False)
            rates['MSNi'], rates_rel['MSNi'] = analyse_firing_rate('MSNi', window=plot_interval, average=False)

            # TODO: set dopamine value based on relative SNc rate (lenth of dip and burst to be adapted)
            #print(f"dopamine over time: {dopamine_over_time}")
            #print(f"rel SNc rate: {rates_rel['SNc']}")

            # Update weights
            weight_times.append(int(h.t))
            for a in range(N_actions):
                for ct in weight_cell_types:
                    for k in range(cell_types_numbers[ct]):
                        delta_w = 0
                        if ct == 'MSNd':
                            # dopamine facilitates active MSNd and inhibits less active MSNd
                            delta_w = learning_rate * rates_rel[ct][a][k] * dopamine_over_time[a][-1] # rel_rate = 1 corresponds to tonic baseline activity
                        elif ct == 'MSNi':
                            # high dopamine increases inhibition, low dopamine suppresses inhibition
                            delta_w = - learning_rate * dopamine_over_time[a][-1]
                        idx = a * cell_types_numbers[ct] + k
                        new_weight = max(0, weights_over_time[(ct, a, k)][-1] + delta_w) # Update weight ensure weight is non-zero
                        weights_over_time[(ct, a, k)].append(round(new_weight, 4))
                        ncs[f'Cor_{ct}'][idx].weight[0] = new_weight # update weight of cortical input stimulation
                        ncs[f'{ct}'][idx].weight[0] = new_weight # update weight of tonical stimulation 
                if output: print(f"{weight_times[-1]} ms: Action {a}: rel rate MSNd = {[f'{rate_rel:.2f}' for rate_rel in rates_rel['MSNd'][a]]}, rel rate SNc = {rates_rel['SNc'][a]:.2f}, Exp. Reward = {expected_reward_over_time[f'{a}{a in target_actions}'][-1]:.2f}, DA = {dopamine_over_time[a][-1]}, Cor-MSNd-Weights = {[f'{nc.weight[0]:.2f}' for nc in ncs['Cor_MSNd'][a*cell_types_numbers['MSNd']:(a+1)*cell_types_numbers['MSNd']]]}, Cor-MSNi-Weights = {[f'{nc.weight[0]:.2f}' for nc in ncs['Cor_MSNi'][a*cell_types_numbers['MSNi']:(a+1)*cell_types_numbers['MSNi']]]}")               
                        
            # Update plots
            #t_array = np.array(t_vec)

            #activation_times.append(int(h.t))

            for i, ch in enumerate(channels_to_plot):
                # Membrane potential plot
                for ct in cell_types:
                    voltages = np.array([np.array(recordings[ct][ch][j]) for j in range(cell_types_numbers[ct])])
                    avg_voltage = np.mean(voltages, axis=0)
                    mem_lines[ct][i].set_data(t_array, avg_voltage)
                    axs[row_potential][col_potential+i].set_xlim(max(0, int(h.t) - plot_interval), max(plot_interval, int(h.t)))

                # Spike raster plot
                y_base = total_cells
                for ct in cell_types:
                    all_spikes = []
                    for k in range(cell_types_numbers[ct]):
                        spikes = np.array(spike_times[ct][ch][k].to_python())
                        y_val = y_base - k
                        y_vals = np.ones_like(spikes) * y_val
                        raster_lines[ct][i][k].set_data(spikes, y_vals)
                        all_spikes.extend(spikes)
                    # Rate lines
                    if len(all_spikes) > 0:
                        bins = np.arange(0, t_array[-1], bin_width_firing_rate)
                        hist, edges = np.histogram(all_spikes, bins=bins)
                        if np.any(hist):  # Only proceed if there's at least one spike
                            rate = hist / (cell_types_numbers[ct] * bin_width_firing_rate / 1000.0)
                            bin_centers = (edges[:-1] + edges[1:]) / 2
                            #offset = y_base + cell_types_numbers[ct] / 2
                            if ct == 'SNc':
                                spike_rate_max = 1000.0 / stim_intervals['SNc_burst'] # Hz
                            elif ct == 'MSNd' or ct == 'MSNi':
                                spike_rate_max = 1000.0 / stim_intervals['Cor'] # Hz
                            else:
                                spike_rate_max = 1000.0 / stim_intervals[ct] # Hz
                            rate_scaled = (rate) / (spike_rate_max + 1e-9)
                            rate_scaled = rate_scaled * (cell_types_numbers[ct] - 1) + y_base - cell_types_numbers[ct] + 1
                            rate_lines[ct][i].set_data(bin_centers, rate_scaled)

                            if ct == 'Thal':
                                window_start = t_array[-1] - plot_interval
                                # Get the indices of bins in the last window
                                bin_indices = np.where(bin_centers > window_start)[0]
                                rate_window = rate[bin_indices]
                                bin_centers_window = bin_centers[bin_indices]
                                edges_window = edges[bin_indices[0] : bin_indices[-1] + 2]  # include right edge of last bin

                                indices = np.where(rate_window > selection_threshold * spike_rate_max)[0]
                                
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
                            rate_lines[ct][i].set_data([], [])
                    y_base -= cell_types_numbers[ct]
                axs[row_spike][col_spike+i].set_xlim(max(0, int(h.t) - plot_interval), max(plot_interval, int(h.t)))
                
                # Weight plot
                for ct in weight_cell_types:
                    for k in range(cell_types_numbers[ct]):
                        weight_lines[ct][i][k].set_data(weight_times, weights_over_time[(ct, i, k)])
                axs[row_weights][col_weights+i].set_xlim(0, max(plot_interval, int(h.t)))
                all_weights = [w for lst in weights_over_time.values() for w in lst if lst]  # flatten and exclude empty lists
                ymin, ymax = min(all_weights), max(all_weights)
                axs[row_weights][col_weights+i].set_ylim(ymin*0.9, ymax*1.1)

                # Reward plot
                input_key = f"{i}{i in target_actions}"
                expected_reward_lines[i][0].set_data(reward_times, expected_reward_over_time[input_key])
                reward_lines[i][0].set_data(reward_times, reward_over_time[i])
                dopamine_lines[i][0].set_data(reward_times, dopamine_over_time[i])
                activation_lines[i][0].set_data(activation_times, activation_over_time[i])
                axs[row_reward][col_reward+i].set_xlim(0, max(plot_interval, int(h.t)))

            #fig.canvas.draw_idle()   
            #fig.canvas.flush_events() 
            plt.pause(0.001)

        state = 1 - state # toggle state

        # Pause simulation
        if int(h.t) % simulation_stop_time == 0:
            toggle_pause()

except KeyboardInterrupt:
    print("\nCtrl-C pressed. Storing data...")

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

    write_dict("cell_types_numbers", cell_types_numbers)
    write_dict("stim_intervals", stim_intervals)
    write_dict("stim_weights", stim_weights)

    # --- Lists ---
    def write_list(name, lst):
        global row
        ws_globals.cell(row=row, column=1, value=name)
        for i, val in enumerate(lst):
            ws_globals.cell(row=row + 1 + i, column=1, value=val)
        row += len(lst) + 2

    write_list("target_actions", target_actions)
    write_list("cortical_input_dur", cortical_input_dur)

    # --- Tuples/List of Tuples ---
    def write_tuples(name, tuples_list):
        global row
        ws_globals.cell(row=row, column=1, value=name)
        for i, tup in enumerate(tuples_list):
            for j, val in enumerate(tup):
                ws_globals.cell(row=row + 1 + i, column=1 + j, value=val)
        row += len(tuples_list) + 2

    write_tuples("connection_specs", connection_specs)

    # --- Scalars ---
    scalars = {
        "N_actions": N_actions,
        "plot_interval": plot_interval,
        "bin_width_firing_rate": bin_width_firing_rate,
        "n_spikes_SNc_burst": n_spikes_SNc_burst,
        "selection_threshold": selection_threshold,
        "learning_rate": learning_rate,
        "expected_reward_value": expected_reward_value,
        "noise": noise,
        "simulation_stop_time": simulation_stop_time

    }
    write_dict("Scalars", scalars)

    # Worksheet for Weights
    ws_weights = wb.create_sheet(title="WeightsOverTime")

    # Header
    header = ['time']
    keys = sorted(weights_over_time.keys())
    header.extend(f"{ct}_a{a}_n{i}" for ct, a, i in keys)
    ws_weights.append(header)

    # Rows
    max_len = len(weight_times)
    for t_idx in range(max_len):
        row = [weight_times[t_idx]]
        for key in keys:
            val = weights_over_time[key][t_idx] if t_idx < len(weights_over_time[key]) else None
            row.append(val)
        ws_weights.append(row)

    # Worksheets
    data_list = [
        ("ExpectedRewardOverTime", expected_reward_over_time), 
        ("RewardOverTime", reward_over_time), 
        ("DopamineOverTime", dopamine_over_time)
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
        max_len = len(reward_times)
        for t_idx in range(max_len):
            row = [reward_times[t_idx]]
            for key in keys:
                val = data[key][t_idx] if t_idx < len(data[key]) else None
                row.append(val)
            ws.append(row)

        ws_list.append(ws)

    # Save
    wb.save(f"{path}.xlsx") # Excel
    fig.savefig(f"{path}.png", dpi=300, bbox_inches='tight') # GUI Screenshot

    print(f"Data saved as {path}")
