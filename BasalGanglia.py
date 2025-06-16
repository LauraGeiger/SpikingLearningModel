from neuron import h#, gui
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, Slider, RadioButtons
import matplotlib.gridspec as gridspec
import numpy as np
import time
import random

h.load_file("stdrun.hoc")
h.cvode_active(1)

N_actions = 3
paused = False
actions = []

def create_cell(name):
    sec = h.Section(name=name)
    sec.insert('hh')
    return sec

# Create neuron populations
cell_types = ['SNc', 'D1', 'D2', 'GPe', 'GPi', 'Thal']
cell_numbers = {'SNc': 1, 'D1': 5, 'D2': 5, 'GPe': 5, 'GPi': 5, 'Thal': 5}
connection_types = ['SNc_to_D1', 'SNc_to_D2', 'D1_to_GPi', 'D2_to_GPe', 'GPe_to_GPi', 'GPi_to_Thal']

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
apc_refs = []

for cell_type in cell_types:
    for a in range(N_actions):
        for i, cell in enumerate(cells[cell_type][a]):
            apc = h.APCount(cell(0.5))
            apc.record(spike_times[cell_type][a][i])
            apc_refs.append(apc)

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

# Tonic stimulation for all cells
stim_intervals = {
    'SNc' : 200, #  5 Hz
    'D1'  : 25,#20,     # 50 Hz (if enabled)
    'D2'  : 25,#20,     # 50 Hz (if enabled)
    'GPe' : 25,#12.5,  # 80 Hz
    'GPi' : 25,#10,    # 100 Hz
    'Thal': 25,    # 20 Hz
    'SNc_burst':20
}

stim_weights = {
    'SNc' : 2,
    'D1'  : 0,
    'D2'  : 0,
    'GPe' : 2,
    'GPi' : 2,
    'Thal': 2,
    'SNc_burst': 0
}
stims, syns, ncs = {}, {}, {}

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

# Define connection specifications
connection_specs = [# pre_group, post_group, label, e_rev, weight, tau, delay
    ('SNc', 'D1',   'SNc_to_D1',      0, 0,    10, 1), # excitatory
    ('SNc', 'D2',   'SNc_to_D2',    -85, 0,    10, 1), # inhibitory
    ('D1',  'GPi',  'D1_to_GPi',    -85, 0.1,  10, 1), # inhibitory
    ('D2',  'GPe',  'D2_to_GPe',    -85, 0.1,  10, 1), # inhibitory
    ('GPe', 'GPi',  'GPe_to_GPi',   -85, 0.05, 10, 1), # inhibitory
    ('GPi', 'Thal', 'GPi_to_Thal',  -85, 0.1,  10, 1)  # inhibitory
]

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

# Recording
recordings = {ct: [[h.Vector().record(cell(0.5)._ref_v) for cell in cells[ct][a]] for a in range(N_actions)] for ct in cell_types}
t_vec = h.Vector().record(h._ref_t)

# Plotting setup
plot_interval = 10  # ms
plot_time_range = 400  # ms
last_plot_update = h.t
channels_to_plot = [i for i in range(N_actions)]

plt.ion()
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, len(channels_to_plot)+1, width_ratios=[1]*len(channels_to_plot) + [0.3])
axs = [[fig.add_subplot(gs[i, j]) for j in range(len(channels_to_plot))] for i in range(2)]

# Membrane potential plot
axs[0][0].set_ylabel('Membrane potential (mV)')
mem_lines = {ct: [] for ct in cell_types}

for i, ch in enumerate(channels_to_plot):
    for j, ct in enumerate(cell_types):
        avg_line, = axs[0][i].plot([], [], f'C{j}', label=ct)
        mem_lines[ct].append(avg_line)

    axs[0][i].set_title(f'Action {ch}')
    axs[0][i].legend(loc='upper right')
    axs[0][i].set_xlim(0, plot_time_range)
    axs[0][i].set_ylim(-85, 65)

# Spike raster plot and rate lines
raster_lines = {ct: [[] for _ in range(N_actions)] for ct in cell_types}
rate_lines = {ct: [] for ct in cell_types}
axs[1][0].set_ylabel('Spike raster')

bin_width = 100  # ms for firing rate bins

for i, ch in enumerate(channels_to_plot):
    for j, ct in enumerate(cell_types):
        raster_lines[ct][i] = []
        for k in range(cell_numbers[ct]):
            line, = axs[1][i].plot([], [], f'C{j}.', markersize=3)
            raster_lines[ct][i].append(line)

        rate_line, = axs[1][i].plot([], [], f'C{j}', label=f'{ct} rate')
        rate_lines[ct].append(rate_line)

    total_cells = sum(cell_numbers[ct] for ct in cell_types)

    y_max = total_cells + 1.5
    axs[1][i].set_ylim(0.5, y_max)
    yticks = []
    cumulative = 0
    for ct in cell_types:
        mid = y_max - (cumulative + (cell_numbers[ct]+1) / 2)
        yticks.append(mid)
        cumulative += cell_numbers[ct] 
    axs[1][i].set_yticks(yticks)
    axs[1][i].set_yticklabels(cell_types)
    axs[1][i].set_xlim(0, plot_time_range)
    axs[1][i].legend(loc='upper right')

plt.show()


def analyse_firing_rate(cell, window=100):
    """Returns a list of firing rates (Hz) for each action's cell population."""
    current_time = h.t
    rates = []
    for a in range(N_actions):
        spikes = 0
        for i in range(cell_numbers[cell]):
            spike_vec = spike_times[cell][a][i]
            # Count spikes in the last `window` ms
            recent_spikes = [t for t in spike_vec if current_time - window <= t <= current_time]
            spikes += len(recent_spikes)
        rate = spikes / (cell_numbers[cell] * (window / 1000.0))  # spikes/sec per neuron
        rates.append(rate)
        max_rate = 1000.0 / stim_intervals[cell]
        rates_rel = [rate / max_rate for rate in rates]
    return rates, rates_rel


# GUI Controls
control_panel_x = 0.85
control_panel_y = 1.0
control_panel_y_gap = 0.05

buttons = {}

width = 0.12
height = 0.05
control_panel_y -= height + control_panel_y_gap
ax_pause = plt.axes([control_panel_x, control_panel_y, width, height])
buttons['pause'] = Button(ax_pause, 'Pause')
def toggle_pause(event=None):
    global paused
    paused = not paused
    buttons['pause'].label.set_text('Continue' if paused else 'Pause')
    if not paused:
        buttons['error'].color = '0.85'
        buttons['reward'].color = '0.85'
buttons['pause'].on_clicked(toggle_pause)

width = 0.12
height = 0.1
control_panel_y -= height + control_panel_y_gap
ax_action = plt.axes([control_panel_x, control_panel_y, width, height], frameon=False)
action_button = CheckButtons(ax_action, [f'Action {i}' for i in range(N_actions)], [i in actions for i in range(N_actions)])
ax_action.set_title('Goal')
def toggle_action(label):
    index = int(label.split()[-1])
    if index in actions:
        actions.remove(index)
    else:
        actions.append(index)
action_button.on_clicked(toggle_action)

width = 0.05
height = 0.05
gap_x = 0.02
control_panel_y -= height + control_panel_y_gap
def mute_tonic_SNc():
    global ncs
    for nc in ncs['SNc']:
        nc.weight[0] = 0
def restore_tonic_SNc():
    global ncs
    for nc in ncs['SNc']:
        nc.weight[0] = stim_weights['SNc']
ax_error = plt.axes([control_panel_x, control_panel_y, width, height])
buttons['error'] = Button(ax_error, 'Error')
def SNc_dip(event=None):
    buttons['error'].color = 'r'
    mute_tonic_SNc()
    h.cvode.event(h.t + stim_intervals['SNc'] + 1, restore_tonic_SNc)  # Restore after 200 ms
buttons['error'].on_clicked(SNc_dip)
ax_reward = plt.axes([control_panel_x + width + gap_x, control_panel_y, width, height])
buttons['reward'] = Button(ax_reward, 'Reward')
def start_burst_SNc():
    for nc in ncs['SNc_burst']:
        nc.weight[0] = 2
def mute_burst_SNc():
    for nc in ncs['SNc_burst']:
        nc.weight[0] = 0
def SNc_burst(event=None):
    buttons['reward'].color = 'g'
    mute_tonic_SNc()
    start_burst_SNc()
    n_spikes = 5
    delay = stim_intervals['SNc_burst'] * n_spikes
    h.cvode.event(h.t + delay, mute_burst_SNc)  
    h.cvode.event(h.t + delay, restore_tonic_SNc)  
buttons['reward'].on_clicked(SNc_burst)



width = 0.12
height = 0.03
control_panel_y -= height + control_panel_y_gap
ax_noise = plt.axes([control_panel_x, control_panel_y, width, height])
ax_noise.set_title('Noise')
noise_slider = Slider(ax_noise, '', 0, 1, valinit=0, valstep=0.1)
def update_stim(noise):
    global stims
    for ct in cell_types:
        for stim in stims[ct]:
            stim.noise = noise

noise_slider.on_changed(update_stim)

width = 0.12
height = 0.03
control_panel_y -= height + control_panel_y_gap
ax_DA = plt.axes([control_panel_x, control_panel_y, width, height])
ax_DA.set_title('SNc->D1/2 weight')
DA_slider = Slider(ax_DA, '', 0, 0.1, valinit=0, valstep=0.01)
def update_DA(weight):
    for nc in ncs['SNc_to_D1']:
        nc.weight[0] = weight
    for nc in ncs['SNc_to_D2']:
        nc.weight[0] = weight

DA_slider.on_changed(update_DA)

# States: OFF=0, ON=2, RAND=random(0-2)
radio_buttons = {'D1': [], 'D2': []}
d1_modes = ['OFF'] * N_actions 
d2_modes = ['OFF'] * N_actions

# Positioning variables for radio buttons
width = 0.04
height = 0.05
gap_x = 0.05
gap_y = 0.01
control_panel_y -= height + control_panel_y_gap
for a in range(N_actions):
    # Create axis for D1 radio buttons
    ax_d1 = plt.axes([control_panel_x + a * gap_x, control_panel_y, width, height])
    r_d1 = RadioButtons(ax_d1, ('OFF', 'ON', 'RAND'))
    r_d1.set_active(0)
    radio_buttons['D1'].append(r_d1)
    ax_d1.set_title(f'Action {a}', fontsize=10)
    if a==0: ax_d1.text(-0.1, 0, 'D1', verticalalignment='center')

control_panel_y -= height + gap_y
for a in range(N_actions):
    # Create axis for D2 radio buttons
    ax_d2 = plt.axes([control_panel_x + a * gap_x, control_panel_y, width, height])
    r_d2 = RadioButtons(ax_d2, ('OFF', 'ON', 'RAND'))
    r_d2.set_active(0)
    radio_buttons['D2'].append(r_d2)
    if a==0: ax_d2.text(-0.1, 0, 'D2', verticalalignment='center')

for a in range(N_actions):
    # Bind with action index via lambda default arg trick
    radio_buttons['D1'][a].on_clicked(lambda label, a=a: update_d1_mode(label, a))
    radio_buttons['D2'][a].on_clicked(lambda label, a=a: update_d2_mode(label, a))
    

def update_d1_mode(label, action_idx):
    d1_modes[action_idx] = label
    update_d1_weights(action_idx)

def update_d2_mode(label, action_idx):
    d2_modes[action_idx] = label
    update_d2_weights(action_idx)

def update_d1_weights(action_idx):
    start = action_idx * cell_numbers['D1']
    end = start + cell_numbers['D1']
    mode = d1_modes[action_idx]
    for nc in ncs['D1'][start:end]:
        if mode == 'OFF':
            nc.weight[0] = 0
        elif mode == 'ON':
            nc.weight[0] = 2
        elif mode == 'RAND':
            nc.weight[0] = random.uniform(0, 2)

def update_d2_weights(action_idx):
    start = action_idx * cell_numbers['D2']
    end = start + cell_numbers['D2']
    mode = d2_modes[action_idx]

    for nc in ncs['D2'][start:end]:
        if mode == 'OFF':
            nc.weight[0] = 0
        elif mode == 'ON':
            nc.weight[0] = 2
        elif mode == 'RAND':
            nc.weight[0] = random.uniform(0, 2)



# Simulation loop
h.dt = 0.1
h.finitialize()

last_action_selection_time = 0
last_weight_update_time = 0

while True:
    if paused or len(actions) == 0:
        time.sleep(0.1)
        plt.pause(0.1)
        continue

    elif int(h.t) != last_action_selection_time and int(h.t) % plot_time_range == plot_time_range/2:
        last_action_selection_time = h.t
        rates, rates_rel = analyse_firing_rate('Thal')
        selected_actions = [i for i,rate_rel in enumerate(rates_rel) if rate_rel > 0.5]
        print(f'Time {h.t:.1f} ms: Selected Action = {selected_actions}, Rates = {rates}, Rates relative = {rates_rel}')
        print(f"Goal {actions}")
        if set(selected_actions) == set(actions):
            print("Reward")
            SNc_burst()
        else:
            print("Error")
            SNc_dip()

    elif int(h.t) != last_weight_update_time and int(h.t) % plot_time_range == 0:
        last_weight_update_time = h.t
        toggle_pause()
        print("Learning")
        rates, rates_rel = analyse_firing_rate('SNc', window=plot_time_range)
        print(f'Time {h.t:.1f} ms: Rates = {rates}, Rates relative = {rates_rel}')
        continue
        
        

    while (h.t - last_plot_update) < plot_interval:
        h.fadvance()

    last_plot_update = h.t
    t_array = np.array(t_vec)

    for i, ch in enumerate(channels_to_plot):
        for ct in cell_types:
            voltages = np.array([np.array(recordings[ct][ch][j]) for j in range(cell_numbers[ct])])
            avg_voltage = np.mean(voltages, axis=0)
            mem_lines[ct][i].set_data(t_array, avg_voltage)
            axs[0][i].set_xlim(max(0, last_plot_update - plot_time_range), max(plot_time_range, last_plot_update))

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
                    spike_rate_max = 1000.0 / stim_intervals[ct] # Hz
                    rate_scaled = (rate) / (spike_rate_max + 1e-9)
                    rate_scaled = rate_scaled * (cell_numbers[ct] - 1) + y_base - cell_numbers[ct] + 1
                    rate_lines[ct][i].set_data(bin_centers, rate_scaled)
                else:
                    rate_lines[ct][i].set_data([], [])

            y_base -= cell_numbers[ct]

        axs[1][i].set_xlim(max(0, last_plot_update - plot_time_range), max(plot_time_range, last_plot_update))

    plt.pause(0.01)
