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
cell_types = ['snc', 'd1', 'd2', 'gpe', 'gpi', 'thal']
cell_numbers = {'snc': 5, 'd1': 5, 'd2': 5, 'gpe': 5, 'gpi': 5, 'thal': 5}
connection_types = ['snc_to_d1', 'snc_to_d2', 'd1_to_gpi', 'd2_to_gpe', 'gpe_to_gpi', 'gpi_to_thal']

cells = {
    cell_type: [
        [create_cell(f'{cell_type.upper()}_{a}_{i}') for i in range(cell_numbers[cell_type])]
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
    'snc' : 25,#100,   # 10 Hz (if enabled)
    'd1'  : 25,#20,     # 50 Hz (if enabled)
    'd2'  : 25,#20,     # 50 Hz (if enabled)
    'gpe' : 25,#12.5,  # 80 Hz
    'gpi' : 25,#10,    # 100 Hz
    'thal': 25    # 20 Hz
}

stim_weights = {
    'snc' : 0,
    'd1'  : 0,
    'd2'  : 0,
    'gpe' : 2,
    'gpi' : 2,
    'thal': 2
}
stims, syns, ncs = {}, {}, {}


for cell_type in cell_types:
    stims[cell_type], syns[cell_type], ncs[cell_type] = [], [], []
    for a in range(N_actions):
        for i, cell in enumerate(cells[cell_type][a]):
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
    ('d1',  'gpi',  'd1_to_gpi',    -85, 0.1,  10, 1), # inhibitory
    ('d2',  'gpe',  'd2_to_gpe',    -85, 0.1,  10, 1), # inhibitory
    ('gpe', 'gpi',  'gpe_to_gpi',   -85, 0.05, 10, 1), # inhibitory
    ('gpi', 'thal', 'gpi_to_thal',  -85, 0.1,  10, 1), # inhibitory
    ('snc', 'd1',   'snc_to_d1',      0, 0.1,  10, 1), # excitatory
    ('snc', 'd2',   'snc_to_d2',    -85, 0.1,  10, 1), # inhibitory
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

# Recording
recordings = {ct: [[h.Vector().record(cell(0.5)._ref_v) for cell in cells[ct][a]] for a in range(N_actions)] for ct in cell_types}
t_vec = h.Vector().record(h._ref_t)

# Plotting setup
plot_interval = 10  # ms
plot_time_range = 200  # ms
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
        avg_line, = axs[0][i].plot([], [], f'C{j}', label=ct.upper())
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

        rate_line, = axs[1][i].plot([], [], f'C{j}', label=f'{ct.upper()} rate')
        rate_lines[ct].append(rate_line)

    total_cells = sum(cell_numbers[ct] for ct in cell_types)
    axs[1][i].set_ylim(0.5, total_cells + 1.5)
    yticks = []
    cumulative = 0
    for ct in cell_types:
        mid = cumulative + (cell_numbers[ct]+1) / 2
        yticks.append(mid)
        cumulative += cell_numbers[ct] 
    ylabels = list(reversed([ct.upper() for ct in cell_types]))
    axs[1][i].set_yticks(yticks)
    axs[1][i].set_yticklabels(ylabels)
    axs[1][i].set_xlim(0, plot_time_range)
    axs[1][i].legend(loc='upper right')

plt.show()

# GUI Controls
control_panel_x = 0.85
control_panel_y = 1.0
control_panel_y_gap = 0.05

width = 0.12
height = 0.05
control_panel_y -= height + control_panel_y_gap
ax_pause = plt.axes([control_panel_x, control_panel_y, width, height])
pause_button = Button(ax_pause, 'Pause')
def toggle_pause(event):
    global paused
    paused = not paused
    pause_button.label.set_text('Continue' if paused else 'Pause')
pause_button.on_clicked(toggle_pause)

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
ax_error = plt.axes([control_panel_x, control_panel_y, width, height])
error_button = Button(ax_error, 'Error')
def DA_dip(event):
    None
error_button.on_clicked(DA_dip)
ax_reward = plt.axes([control_panel_x + width + gap_x, control_panel_y, width, height])
reward_button = Button(ax_reward, 'Reward')
def DA_burst(event):
    None
reward_button.on_clicked(DA_burst)

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

# States: OFF=0, ON=2, RAND=random(0-2)
radio_buttons = {'d1': [], 'd2': []}
d1_modes = ['OFF'] * N_actions 
d2_modes = ['OFF'] * N_actions

# Positioning variables for radio buttons
#radio_start_x = 0.83
#radio_start_x_d2 = 0.9
#radio_start_y_d1 = 0.55
#radio_start_y_d2 = 0.49
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
    radio_buttons['d1'].append(r_d1)
    ax_d1.set_title(f'Action {a}', fontsize=10)
    if a==0: ax_d1.text(-0.1, 0, 'D1', verticalalignment='center')

control_panel_y -= height + gap_y
for a in range(N_actions):
    # Create axis for D2 radio buttons
    ax_d2 = plt.axes([control_panel_x + a * gap_x, control_panel_y, width, height])
    r_d2 = RadioButtons(ax_d2, ('OFF', 'ON', 'RAND'))
    r_d2.set_active(0)
    radio_buttons['d2'].append(r_d2)
    if a==0: ax_d2.text(-0.1, 0, 'D2', verticalalignment='center')

for a in range(N_actions):
    # Bind with action index via lambda default arg trick
    radio_buttons['d1'][a].on_clicked(lambda label, a=a: update_d1_mode(label, a))
    radio_buttons['d2'][a].on_clicked(lambda label, a=a: update_d2_mode(label, a))
    

def update_d1_mode(label, action_idx):
    d1_modes[action_idx] = label
    update_d1_weights(action_idx)

def update_d2_mode(label, action_idx):
    d2_modes[action_idx] = label
    update_d2_weights(action_idx)

def update_d1_weights(action_idx):
    start = action_idx * cell_numbers['d1']
    end = start + cell_numbers['d1']
    mode = d1_modes[action_idx]
    for nc in ncs['d1'][start:end]:
        if mode == 'OFF':
            nc.weight[0] = 0
        elif mode == 'ON':
            nc.weight[0] = 2
        elif mode == 'RAND':
            nc.weight[0] = random.uniform(0, 2)

def update_d2_weights(action_idx):
    start = action_idx * cell_numbers['d2']
    end = start + cell_numbers['d2']
    mode = d2_modes[action_idx]

    for nc in ncs['d2'][start:end]:
        if mode == 'OFF':
            nc.weight[0] = 0
        elif mode == 'ON':
            nc.weight[0] = 2
        elif mode == 'RAND':
            nc.weight[0] = random.uniform(0, 2)



# Simulation loop
h.dt = 0.1
h.finitialize()

while True:
    if paused or len(actions) == 0:
        time.sleep(0.1)
        plt.pause(0.1)
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
