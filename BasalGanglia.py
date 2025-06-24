from neuron import h#, gui
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, Slider, RadioButtons
import matplotlib.gridspec as gridspec
import numpy as np
import time
import random

h.load_file("stdrun.hoc")
h.cvode_active(1)

#--- Global Variables ------------------------------------------------------------------------------------------------------------------------------------------------#
cell_types = ['SNc', 'MSNd', 'MSNi', 'GPe', 'GPi', 'Thal']
cell_numbers = {'SNc': 5, 'MSNd': 5, 'MSNi': 5, 'GPe': 5, 'GPi': 5, 'Thal': 5}
connection_types = ['SNc_to_MSNd', 'SNc_to_MSNi', 'MSNd_to_GPi', 'MSNi_to_GPe', 'GPe_to_GPi', 'GPi_to_Thal']

# Tonic stimulation for all cells
stim_intervals = {
    'SNc'       : 1000 / 5,  #  5 Hz
    'MSNd'      : 1000 / 5,  #  5 Hz (tonic baseline)
    'MSNi'      : 1000 / 5,  #  5 Hz (tonic baseline)
    'GPe'       : 25,#12.5,  # 80 Hz
    'GPi'       : 25,#10,    # 100 Hz
    'Thal'      : 1000 / 20, # 20 Hz
    'SNc_burst' : 1000 / 50, # 50 Hz
    'Cor'       : 1000 / 40  # 40 Hz
}

stim_weights = {
    'SNc' : 2,
    'MSNd'  : 2,
    'MSNi'  : 2,
    'GPe' : 2,
    'GPi' : 2,
    'Thal': 2,
    'SNc_burst': 2,
    'Cor': 1.8
}
stims, syns, ncs = {}, {}, {}

# Define connection specifications
connection_specs = [# pre_group, post_group, label, e_rev, weight, tau, delay
    ('SNc', 'MSNd', 'SNc_to_MSNd',   0, 0,    10, 1),   # excitatory
    ('SNc', 'MSNi', 'SNc_to_MSNi', -85, 0,    10, 1),   # inhibitory
    ('MSNd', 'GPi', 'MSNd_to_GPi', -85, 0.1,  10, 1),   # inhibitory
    ('MSNi', 'GPe', 'MSNi_to_GPe', -85, 0.1,  10, 1),   # inhibitory
    ('GPe',  'GPi',  'GPe_to_GPi', -85, 0.05, 10, 1),   # inhibitory
    ('GPi', 'Thal', 'GPi_to_Thal', -85, 0.1,  10, 1)    # inhibitory
]

N_actions = 3
paused = False
target_actions = []
expected_reward = {}
DA_signal = {}
weight_cell_types = ['MSNd', 'MSNi']
weights_over_time = {(ct, a, i): [] 
                     for ct in weight_cell_types
                     for a in range(N_actions) 
                     for i in range(cell_numbers[ct])
                     } # track weights over time

apc_refs = []

# Plotting setup
plot_interval = 200  # ms
plot_time_range = 400  # ms
last_plot_update = h.t
channels_to_plot = [i for i in range(N_actions)]

buttons = {}
rates = {}
rates_rel = {}

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
    stim.noise = 0
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

def SNc_burst(event=None, actions=None):
    update_stimulus_activation(cell='SNc', stimulus='SNc', actions=actions, active=False) # stop SNc tonic stimulus
    update_stimulus_activation(cell='SNc', stimulus='SNc_burst', actions=actions, active=True) # start SNc burst stimulus
    n_spikes = 5
    delay = stim_intervals['SNc_burst'] * n_spikes
    h.cvode.event(h.t + delay, lambda actions=actions: update_stimulus_activation(cell='SNc', stimulus='SNc_burst', actions=actions, active=False))  # stop SNc burst stimulus
    h.cvode.event(h.t + delay, lambda actions=actions: update_stimulus_activation(cell='SNc', stimulus='SNc', actions=actions, active=True))  # start SNc tonic stimulus

def analyse_firing_rate(cell, window=100, average=True):
    """Returns a list of firing rates (Hz) for each action's cell population."""
    current_time = h.t
    rates_avg = []
    rates = []
    for a in range(N_actions):
        spikes_avg = 0
        spikes = []
        for i in range(cell_numbers[cell]):
            spike_vec = spike_times[cell][a][i]
            # Count spikes in the last `window` ms
            recent_spikes = [t for t in spike_vec if current_time - window <= t <= current_time]
            if average:
                spikes_avg += len(recent_spikes)
            else:
                spikes.append(len(recent_spikes))
        if average:
            rate_avg = spikes_avg / (cell_numbers[cell] * (window / 1000.0))  # spikes/sec per neuron
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
        for a in range(N_actions):
            buttons[f'selected_{a}'].label.set_text('Action not\nselected')
            buttons[f'selected_{a}'].color = '0.85'
    else:
        buttons['pause'].color = 'c'
    fig.canvas.draw_idle()

def toggle_target_action(event=None, action=None):
    if action != None:
        if action in target_actions:
            target_actions.remove(action)
            buttons[f'target_{action}'].label.set_text('Set as\nTarget')
            buttons[f'target_{action}'].color = '0.85'
            
        else:
            target_actions.append(action)
            buttons[f'target_{action}'].label.set_text('Target')
            buttons[f'target_{action}'].color = 'y'
        fig.canvas.draw_idle()

def update_stim(noise):
    for ct in cell_types:
        for stim in stims[ct]:
            stim.noise = noise

#--- Network ------------------------------------------------------------------------------------------------------------------------------------------#

# Create neuron populations
cells = {
    cell_type: [
        [create_cell(f'{cell_type}_{a}_{i}') for i in range(cell_numbers[cell_type])]
        for a in range(N_actions)
    ] 
    for cell_type in cell_types
}

# Spike detectors and vectors
spike_times = {
    cell_type: [
        [h.Vector() for _ in range(cell_numbers[cell_type])] 
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
                offset = i*stim_intervals[cell_type]/cell_numbers[cell_type]
            stim, syn, nc = create_stim(cell, start=offset, interval=stim_intervals[cell_type], weight=stim_weights[cell_type])
            stims[cell_type].append(stim)
            syns[cell_type].append(syn)
            ncs[cell_type].append(nc)

# Additional connections dict to store NetCons and ExpSyns
ncs.update({conn: [] for conn in connection_types})
syns.update({conn: [] for conn in connection_types})

# Create connections based on the specification
for pre_group, post_group, label, e_rev, weight, tau, delay in connection_specs:
    for a in range(N_actions):
        for pre_cell in cells[pre_group][a]:
            for post_cell in cells[post_group][a]:
                syn = h.ExpSyn(post_cell(0.5))
                syn.e = e_rev
                syn.tau = tau
                nc = h.NetCon(pre_cell(0.5)._ref_v, syn, sec=pre_cell)
                nc.weight[0] = weight
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
stims.update({'Cor_MSNd': []})
stims.update({'Cor_MSNi': []})
syns.update({'Cor_MSNd': []})
syns.update({'Cor_MSNi': []})
ncs.update({'Cor_MSNd': []})
ncs.update({'Cor_MSNi': []})

for a in range(N_actions):
    for cell in cells['MSNd'][a]:
        stim, syn, nc = create_stim(cell, start=0, interval=stim_intervals['Cor'], weight=stim_weights['Cor'])
        stims['Cor_MSNd'].append(stim)
        syns['Cor_MSNd'].append(syn)
        ncs['Cor_MSNd'].append(nc)
    for cell in cells['MSNi'][a]:
        stim, syn, nc = create_stim(cell, start=0, interval=stim_intervals['Cor'], weight=stim_weights['Cor'])
        stims['Cor_MSNi'].append(stim)
        syns['Cor_MSNi'].append(syn)
        ncs['Cor_MSNi'].append(nc)
for nc in ncs['Cor_MSNd']:
    nc.active(False)
for nc in ncs['Cor_MSNi']:
    nc.active(False)

# Recording
recordings = {ct: [[h.Vector().record(cell(0.5)._ref_v) for cell in cells[ct][a]] for a in range(N_actions)] for ct in cell_types}
t_vec = h.Vector().record(h._ref_t)

#--- Plotting ------------------------------------------------------------------------------------------------------------------------------------------#

plt.ion()
fig = plt.figure(figsize=(13, 8))
rows = 5
gs = gridspec.GridSpec(rows, 1+len(channels_to_plot), height_ratios=[2]+[4]*(rows-2)+[1], width_ratios=[0.3]+[1]*len(channels_to_plot))
axs = [[fig.add_subplot(gs[i, j]) for j in range(1+len(channels_to_plot))] for i in range(rows)]
[row_control_upper, row_potential, row_spike, row_weights, row_control_lower] = list(range(rows))
col_potential = 1
col_spike = 1
col_weights = 1

# Deactivate axis for non-plot axes
for row in range(rows):
    if row == row_potential or row == row_spike or row == row_weights:
        axs[row][0].set_axis_off()
    else:
        for ax in axs[row]:
            ax.set_axis_off()

# Membrane potential plot
axs[row_potential][col_potential].set_ylabel('Membrane potential (mV)')
mem_lines = {ct: [] for ct in cell_types}

for i, ch in enumerate(channels_to_plot):
    for j, ct in enumerate(cell_types):
        avg_line, = axs[row_potential][col_potential+i].plot([], [], f'C{j}', label=ct)
        mem_lines[ct].append(avg_line)

    axs[row_potential][col_potential+i].set_title(f'Action {ch}')
    axs[row_potential][col_potential+i].legend(loc='upper right')
    axs[row_potential][col_potential+i].set_xlim(0, plot_time_range)
    axs[row_potential][col_potential+i].set_ylim(-85, 65)

# Spike raster plot and rate lines
raster_lines = {ct: [[] for _ in range(N_actions)] for ct in cell_types}
rate_lines = {ct: [] for ct in cell_types}
axs[row_spike][col_spike].set_ylabel('Spike raster')

bin_width = 100  # ms for firing rate bins

for i, ch in enumerate(channels_to_plot):
    for j, ct in enumerate(cell_types):
        raster_lines[ct][i] = []
        for k in range(cell_numbers[ct]):
            line, = axs[row_spike][col_spike+i].plot([], [], f'C{j}.', markersize=3)
            raster_lines[ct][i].append(line)

        rate_line, = axs[row_spike][col_spike+i].plot([], [], f'C{j}', label=f'{ct} rate')
        rate_lines[ct].append(rate_line)

    total_cells = sum(cell_numbers[ct] for ct in cell_types)

    y_max = total_cells + 1.5
    axs[row_spike][col_spike+i].set_ylim(0.5, y_max)
    yticks = []
    cumulative = 0
    for ct in cell_types:
        mid = y_max - (cumulative + (cell_numbers[ct]+1) / 2)
        yticks.append(mid)
        cumulative += cell_numbers[ct] 
    axs[row_spike][col_spike+i].set_yticks(yticks)
    axs[row_spike][col_spike+i].set_yticklabels(cell_types)
    axs[row_spike][col_spike+i].set_xlim(0, plot_time_range)
    axs[row_spike][col_spike+i].set_xlabel('Simulation time (ms)')
    axs[row_spike][col_spike+i].legend(loc='upper right')

# Weight plot
axs[row_weights][col_weights].set_ylabel('Cortical Input Weight')
weight_lines = {ct: [[] for _ in range(N_actions)] for ct in weight_cell_types}

for i, ch in enumerate(channels_to_plot):
    for j, ct in enumerate(cell_types):
        if ct in weight_cell_types:
            weight_lines[ct][i] = []
            for k in range(cell_numbers[ct]):
                label = ct if k == 0 else None  # Only label the first line
                line, = axs[row_weights][col_weights+i].plot([], [], f'C{j}', label=label)
                weight_lines[ct][i].append(line)

    axs[row_weights][col_weights+i].set_title(f'Action {ch}')
    axs[row_weights][col_weights+i].legend(loc='upper right')
    axs[row_weights][col_weights+i].set_xlim(0, plot_time_range)
    axs[row_weights][col_weights+i].set_ylim(0, 3)

plt.show()
plt.tight_layout()

#--- Left control panel ---#
ax_pause = axs[row_control_upper][0].inset_axes([0,0.5,1,0.45])
buttons['pause'] = Button(ax_pause, 'Pause')
buttons['pause'].on_clicked(toggle_pause)

ax_noise = axs[row_control_upper][0].inset_axes([0.5,0,0.5,0.5])
noise_slider = Slider(ax_noise, 'Noise', 0, 1, valinit=0, valstep=0.1)
noise_slider.on_changed(update_stim)

#--- Upper control panel ---#
for a in range(N_actions):
    # Target button
    ax_target = axs[row_control_upper][col_potential+a].inset_axes([0.4,0,0.3,1])
    buttons[f'target_{a}'] = Button(ax_target, 'Set as\nTarget')
    buttons[f'target_{a}'].on_clicked(lambda event, a=a: toggle_target_action(event=event, action=a))

#--- Lower control panel ---#
for a in range(N_actions):
    ax_selected = axs[row_control_lower][col_potential+a].inset_axes([0.4,0,0.3,1])
    buttons[f'selected_{a}'] = Button(ax_selected, 'Action not\nselected')
    
#--- Simulation ---------------------------------------------------------------------------------------------------------------------------------------------------#
h.dt = 0.1
h.finitialize()

last_action_selection_time = 0
last_weight_update_time = 0

while True:
    if paused or len(target_actions) == 0:
        time.sleep(0.1)
        fig.canvas.draw_idle()   
        fig.canvas.flush_events()
        plt.pause(0.1)
        continue

    if int(h.t) % plot_time_range == 0:
        h.cvode.event(h.t + 100, lambda: update_stimulus_activation(cell='MSNd', stimulus='Cor_MSNd', actions=target_actions, active=True))  # start cortical input stimulus for that action
        h.cvode.event(h.t + 100, lambda: update_stimulus_activation(cell='MSNi', stimulus='Cor_MSNi', actions=target_actions, active=True))  # start cortical input stimulus for that action
        h.cvode.event(h.t + 300, lambda: update_stimulus_activation(cell='MSNd', stimulus='Cor_MSNd', actions=target_actions, active=False))  # stop cortical input stimulus for that action
        h.cvode.event(h.t + 300, lambda: update_stimulus_activation(cell='MSNi', stimulus='Cor_MSNi', actions=target_actions, active=False))  # stop cortical input stimulus for that action

    if int(h.t) != last_action_selection_time and int(h.t) % plot_time_range == plot_time_range/2:
        last_action_selection_time = int(h.t)
        rates['Thal'], rates_rel['Thal'] = analyse_firing_rate('Thal')
        selected_actions = [i for i, rate_rel in enumerate(rates_rel['Thal']) if rate_rel > 0.5]

        for action in selected_actions:
            buttons[f'selected_{action}'].label.set_text('Action\nselected')
            buttons[f'selected_{action}'].color = 'y'

        print(f"{last_action_selection_time} ms: Target Actions = {target_actions}, Selected Actions = {selected_actions}, Rates Thal = {rates['Thal']}, Rates Thal relative = {rates_rel['Thal']}")
        correct_actions = list(set(target_actions) & set(selected_actions))
        incorrect_actions = list(set(target_actions) ^ set(selected_actions))
        
        for action in range(N_actions):
            if  not ((action in target_actions) ^ (action in selected_actions)): #XNOR
                reward = 1
                buttons[f'selected_{action}'].color = 'g'
            else:
                reward = 0
                buttons[f'selected_{action}'].color = 'r'
            
            input_key = f"{action}{action in target_actions}"
            if input_key not in expected_reward:
                expected_reward[input_key] = 0.2
            expected_reward[input_key] += 0.1 * (reward - expected_reward[input_key])
            DA_signal[action] = round(reward - expected_reward[input_key], 2) # round to 2 digits after comma
            if DA_signal[action] > 0:
                SNc_burst(event=None, actions=[action])
            elif DA_signal[action] < 0:
                SNc_dip(event=None, actions=[action])
        fig.canvas.draw_idle()
        plt.pause(0.5)

    elif int(h.t) != last_weight_update_time and int(h.t) % plot_time_range == 0:
        last_weight_update_time = int(h.t)
        #toggle_pause()
        rates['SNc'], rates_rel['SNc'] = analyse_firing_rate('SNc', window=plot_time_range)
        rates['MSNd'], rates_rel['MSNd'] = analyse_firing_rate('MSNd', window=plot_time_range, average=False)
        rates['MSNi'], rates_rel['MSNi'] = analyse_firing_rate('MSNi', window=plot_time_range, average=False)

        learning_rate = 0.05
        i, j = 0, 0
        for a in range(N_actions):
            print(f"{last_weight_update_time} ms: Action {a}: rel rate MSNd = {rates_rel['MSNd'][a]}, rel rate SNc = {rates_rel['SNc'][a]}, Expected Reward = {expected_reward[f'{a}{a in target_actions}']:.2f}, DA = {DA_signal[a]}, Cor-MSNd-Weights = {[f'{nc.weight[0]:.2f}' for nc in ncs['Cor_MSNd'][a*cell_numbers['MSNd']:(a+1)*cell_numbers['MSNd']]]}, Cor-MSNi-Weights = {[f'{nc.weight[0]:.2f}' for nc in ncs['Cor_MSNi'][a*cell_numbers['MSNi']:(a+1)*cell_numbers['MSNi']]]}")
            for n,_ in enumerate(cells['MSNd'][a]):
                # Update weight: DA facilitates active MSNd and inhibits less active MSNd
                #delta_w = (rates_rel['MSNd'][a][n] - 1) * DA_signal[a] #(rates_rel['SNc'][a] - 1) # baseline tonic firing rate corresponds to relative rate of 1
                delta_w = learning_rate * (rates_rel['MSNd'][a][n]) * DA_signal[a] 
                ncs['Cor_MSNd'][i].weight[0] += delta_w
                ncs['Cor_MSNd'][i].weight[0] = max(0, ncs['Cor_MSNd'][i].weight[0]) # ensure weights are non-zero
                
                i += 1
            for m,_ in enumerate(cells['MSNi'][a]):
                # Update weight: high DA increases inhibition, low DA suppresses inhibition
                delta_w = learning_rate * (- DA_signal[a]) # (1 - rates_rel['SNc'][a]) # baseline tonic firing rate of SNc corresponds to relative rate of 1
                ncs['Cor_MSNi'][j].weight[0] += delta_w
                ncs['Cor_MSNi'][j].weight[0] = max(0, ncs['Cor_MSNi'][j].weight[0]) # ensure weights are non-zero
                j += 1

        continue
        
    h.continuerun(h.t + plot_interval)

    last_plot_update = int(h.t)
    t_array = np.array(t_vec)

    for i, ch in enumerate(channels_to_plot):
        for ct in cell_types:
            voltages = np.array([np.array(recordings[ct][ch][j]) for j in range(cell_numbers[ct])])
            avg_voltage = np.mean(voltages, axis=0)
            mem_lines[ct][i].set_data(t_array, avg_voltage)
            axs[row_potential][col_potential+i].set_xlim(max(0, last_plot_update - plot_time_range), max(plot_time_range, last_plot_update))

        y_base = total_cells
        for ct in cell_types:
            all_spikes = []
            for k in range(cell_numbers[ct]):
                spikes = np.array(spike_times[ct][ch][k].to_python())
                y_val = y_base - k
                y_vals = np.ones_like(spikes) * y_val
                raster_lines[ct][i][k].set_data(spikes, y_vals)
                all_spikes.extend(spikes)

            if len(all_spikes) > 0:
                bins = np.arange(0, t_array[-1], bin_width)
                hist, edges = np.histogram(all_spikes, bins=bins)
                if np.any(hist):  # Only proceed if there's at least one spike
                    rate = hist / (cell_numbers[ct] * bin_width / 1000.0)
                    bin_centers = (edges[:-1] + edges[1:]) / 2
                    offset = y_base + cell_numbers[ct] / 2
                    spike_rate_max = 100 #1000.0 / stim_intervals[ct] # Hz
                    rate_scaled = (rate) / (spike_rate_max + 1e-9)
                    rate_scaled = rate_scaled * (cell_numbers[ct] - 1) + y_base - cell_numbers[ct] + 1
                    rate_lines[ct][i].set_data(bin_centers, rate_scaled)
                else:
                    rate_lines[ct][i].set_data([], [])

            y_base -= cell_numbers[ct]
        axs[row_spike][col_spike+i].set_xlim(max(0, last_plot_update - plot_time_range), max(plot_time_range, last_plot_update))
        
        
        for ct in weight_cell_types:
            for k in range(cell_numbers[ct]):
                min_weight = weights_over_time[(ct, i, k)]
                max_weight = weights_over_time[(ct, i, k)]
                while len(weights_over_time[(ct, i, k)]) < len(t_array):
                    weights_over_time[(ct, i, k)].append(ncs[f'Cor_{ct}'][i*cell_numbers[ct]+k].weight[0])
                weight_lines[ct][i][k].set_data(t_array, weights_over_time[(ct, i, k)])
        axs[row_weights][col_weights+i].set_xlim(0, max(plot_time_range, last_plot_update))
        all_weights = [w for lst in weights_over_time.values() for w in lst if lst]  # flatten and exclude empty lists
        ymin, ymax = min(all_weights), max(all_weights)
        axs[row_weights][col_weights+i].set_ylim(ymin*0.9, ymax*1.1)

    fig.canvas.draw_idle()   
    fig.canvas.flush_events() 
    plt.pause(0.001)
