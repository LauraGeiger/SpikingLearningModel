from neuron import h#, gui
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, Slider
from matplotlib.collections import EventCollection
import matplotlib.gridspec as gridspec
import time
import random

h.load_file("stdrun.hoc")
h.cvode_active(1)

N_actions = 3  # number of action channels
paused = False
actions = []  # default selected action

def create_cell(name):
    sec = h.Section(name=name)
    sec.insert('hh')
    return sec

# Create neuron populations
d1 = [create_cell(f'D1_{i}') for i in range(N_actions)]
d2 = [create_cell(f'D2_{i}') for i in range(N_actions)]
gpe = [create_cell(f'GPe_{i}') for i in range(N_actions)]
gpi = [create_cell(f'GPi_{i}') for i in range(N_actions)]
thal = [create_cell(f'Thal_{i}') for i in range(N_actions)]
snc = [create_cell(f'SNc_{i}') for i in range(N_actions)]

# Spike detectors and vectors
apc_refs = []
spike_times = {'d1': [], 'd2': [], 'gpe': [], 'gpi': [], 'thal': [], 'snc': []}
for pop, name in zip([d1, d2, gpe, gpi, thal, snc], spike_times.keys()):
    for cell in pop:
        apc = h.APCount(cell(0.5))
        vec = h.Vector()
        apc.record(vec)
        spike_times[name].append(vec)
        apc_refs.append(apc)

def create_stim(cell, start=None, number=1e9, interval=None, weight=None):
    stim = h.NetStim()
    if start: stim.start = start
    if number: stim.number = number
    if interval: stim.interval = interval
    stim.noise = 0
    syn = h.ExpSyn(cell(0.5))
    syn.e = 0
    nc = h.NetCon(stim, syn)
    if weight: nc.weight[0] = weight
    nc.delay = 1
    return stim, syn, nc

# Stimulate tonic activity of GPe
stim_gpe = []
syn_gpe = []
nc_gpe = []
for cell in gpe:
    stim, syn, nc = create_stim(cell, start=0, interval=12.5, weight=2) # 80 Hz
    stim_gpe.append(stim)
    syn_gpe.append(syn)
    nc_gpe.append(nc)

# Stimulate tonic activity of GPi
stim_gpi = []
syn_gpi = []
nc_gpi = []
for cell in gpi:
    stim, syn, nc = create_stim(cell, start=0, interval=10, weight=2) # 100 Hz
    stim_gpi.append(stim)
    syn_gpi.append(syn)
    nc_gpi.append(nc)

# Stimulate tonic activity of Thalamus 
stim_thal = []
syn_thal = []
nc_thal = []
for cell in thal:
    stim, syn, nc = create_stim(cell, start=0, interval=50, weight=2) # 20 Hz
    stim_thal.append(stim)
    syn_thal.append(syn)
    nc_thal.append(nc)


# Create D1 spike train (direct pathway)
stim_d1 = []
syn_d1 = []
nc_d1 = []
for cell in d1:
    stim, syn, nc = create_stim(cell, start=0, number=1e9, interval=20, weight=2)
    stim_d1.append(stim)
    syn_d1.append(syn)
    nc_d1.append(nc)

# Create D2 spike train (indirect pathway)
stim_d2 = []
syn_d2 = []
nc_d2 = []
for cell in d2:
    stim, syn, nc = create_stim(cell, start=0, number=1e9, interval=20, weight=2)
    stim_d2.append(stim)
    syn_d2.append(syn)
    nc_d2.append(nc)


# Stimulate tonic activity of SNc (dopamine)
'''
stim_snc = []
syn_snc = []
nc_snc = []
for cell in snc:
    stim, syn, nc = create_stim(cell, start=0, number=1e9, interval=100, weight=2)
    stim_snc.append(stim)
    syn_snc.append(syn)
    nc_snc.append(nc)
'''

# Connections:

# D1 -> GPi inhibition
d1_gpi_syns = []
d1_gpi_ncs = []
for i in range(N_actions):
    syn = h.ExpSyn(gpi[i](0.5))
    syn.e = -80
    syn.tau = 15
    nc = h.NetCon(d1[i](0.5)._ref_v, syn, sec=d1[i])
    nc.threshold = 0
    nc.weight[0] = 2
    nc.delay = 1
    d1_gpi_syns.append(syn)
    d1_gpi_ncs.append(nc)

# D2 -> GPe inhibition
d2_gpe_syns = []
d2_gpe_ncs = []
for i in range(N_actions):
    syn = h.ExpSyn(gpe[i](0.5))
    syn.e = -80
    syn.tau = 15
    nc = h.NetCon(d2[i](0.5)._ref_v, syn, sec=d2[i])
    nc.threshold = 0
    nc.weight[0] = 1
    nc.delay = 1
    d2_gpe_syns.append(syn)
    d2_gpe_ncs.append(nc)

# GPe -> GPi inhibition 
gpe_gpi_syns = []
gpe_gpi_ncs = []
for i in range(N_actions):
    syn = h.ExpSyn(gpi[i](0.5))
    syn.e = -80
    syn.tau = 15
    nc = h.NetCon(gpe[i](0.5)._ref_v, syn, sec=gpe[i])
    nc.threshold = 0
    nc.weight[0] = 0.3
    nc.delay = 1
    gpe_gpi_syns.append(syn)
    gpe_gpi_ncs.append(nc)

# GPi -> Thal inhibition
gpi_thal_syns = []
gpi_thal_ncs = []
for i in range(N_actions):
    syn = h.ExpSyn(thal[i](0.5))
    syn.e = -80
    syn.tau = 5
    nc = h.NetCon(gpi[i](0.5)._ref_v, syn, sec=gpi[i])
    nc.threshold = 0
    nc.weight[0] = 1
    nc.delay = 1
    gpi_thal_syns.append(syn)
    gpi_thal_ncs.append(nc)

# SNc -> D1 (dopaminergic excitation)
snc_d1_syns = []
snc_d1_ncs = []
for i in range(N_actions):
    syn = h.ExpSyn(d1[i](0.5))
    syn.e = 0      # Excitatory
    syn.tau = 10
    nc = h.NetCon(snc[i](0.5)._ref_v, syn, sec=snc[i])
    nc.threshold = 0
    nc.weight[0] = 0.5  # Tune weight as needed
    nc.delay = 1
    snc_d1_syns.append(syn)
    snc_d1_ncs.append(nc)

# SNc -> D2 (dopaminergic inhibition)
snc_d2_syns = []
snc_d2_ncs = []
for i in range(N_actions):
    syn = h.ExpSyn(d2[i](0.5))
    syn.e = -80   # Inhibitory
    syn.tau = 10
    nc = h.NetCon(snc[i](0.5)._ref_v, syn, sec=snc[i])
    nc.threshold = 0
    nc.weight[0] = 0.5  # Tune weight as needed
    nc.delay = 1
    snc_d2_syns.append(syn)
    snc_d2_ncs.append(nc)


# Recording
t_vec = h.Vector().record(h._ref_t)

v_d1 = [h.Vector().record(cell(0.5)._ref_v) for cell in d1]
v_d2 = [h.Vector().record(cell(0.5)._ref_v) for cell in d2]
v_gpe = [h.Vector().record(cell(0.5)._ref_v) for cell in gpe]
v_gpi = [h.Vector().record(cell(0.5)._ref_v) for cell in gpi]
v_thal = [h.Vector().record(cell(0.5)._ref_v) for cell in thal]
v_snc = [h.Vector().record(cell(0.5)._ref_v) for cell in snc]

# Plotting parameters
plot_interval = 20  # ms
plot_time_range = 200 # ms
last_plot_update = h.t
channels_to_plot = [i for i in range(N_actions)]

plt.ion()
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, len(channels_to_plot)+1, width_ratios=[1]*len(channels_to_plot) + [0.3])
axs = [[fig.add_subplot(gs[i, j]) for j in range(len(channels_to_plot))] for i in range(2)]

# Membrane potential plot
axs[0][0].set_ylabel('Membrane potential (mV)')
lines_d1, lines_d2, lines_gpe, lines_gpi, lines_thal, lines_snc = [], [], [], [], [], []
for i, ch in enumerate(channels_to_plot):
    line_snc, = axs[0][i].plot([], [], label=f'SNc')
    line_d1, = axs[0][i].plot([], [], label=f'D1')
    line_d2, = axs[0][i].plot([], [], label=f'D2')
    line_gpe, = axs[0][i].plot([], [], label=f'GPe')
    line_gpi, = axs[0][i].plot([], [], label=f'GPi')
    line_thal, = axs[0][i].plot([], [], label=f'Thal')
    
    lines_snc.append(line_snc)
    lines_d1.append(line_d1)
    lines_d2.append(line_d2)
    lines_gpe.append(line_gpe)
    lines_gpi.append(line_gpi)
    lines_thal.append(line_thal)

    axs[0][i].set_title(f'Action {channels_to_plot[i]}')
    axs[0][i].legend(loc='upper right')
    axs[0][i].set_xlim(0, plot_time_range)  
    axs[0][i].set_ylim(-85, 65)    

# Spike raster plot
raster_snc = []
raster_d1 = []
raster_d2 = []
raster_gpe = []
raster_gpi = []
raster_thal = []
axs[1][0].set_ylabel('Spike raster')
for i, ch in enumerate(channels_to_plot):
    line_snc, = axs[1][i].plot([], [], 'C0.', markersize=10)
    line_d1, = axs[1][i].plot([], [], 'C1.', markersize=10)
    line_d2, = axs[1][i].plot([], [], 'C2.', markersize=10)
    line_gpe, = axs[1][i].plot([], [], 'C3.', markersize=10)
    line_gpi, = axs[1][i].plot([], [], 'C4.', markersize=10)
    line_thal, = axs[1][i].plot([], [], 'C5.', markersize=10)

    raster_snc.append(line_snc)
    raster_d1.append(line_d1)
    raster_d2.append(line_d2)
    raster_gpe.append(line_gpe)
    raster_gpi.append(line_gpi)
    raster_thal.append(line_thal)

    axs[1][i].set_ylim(0.5, 6.5)
    axs[1][i].set_yticks([1, 2, 3, 4, 5, 6])
    axs[1][i].set_yticklabels(['Thal', 'GPi', 'GPe', 'D2', 'D1', 'SNc'])
    axs[1][i].set_xlim(0, plot_time_range)     

plt.show()

# Pause Button
ax_pause = plt.axes([0.85, 0.75, 0.1, 0.05])
pause_button = Button(ax_pause, 'Pause')

def toggle_pause(event):
    global paused     
    paused = not paused
    pause_button.label.set_text('Continue' if paused else 'Pause')

pause_button.on_clicked(toggle_pause)

# Check Buttons for action selection
ax_action = plt.axes([0.85, 0.5, 0.1, 0.2], frameon=False)
action_button = CheckButtons(ax_action, [f'Action {i}' for i in range(N_actions)], [i in actions for i in range(N_actions)])

# Dopamine level slider (affects SNc firing rate via interval)
ax_dopa = plt.axes([0.85, 0.45, 0.1, 0.03])
dopa_slider = Slider(ax_dopa, 'DA Level', 0.5, 2, valinit=1, valstep=0.1)

DA_level = 1
def update_dopa(val):
    global DA_level
    DA_level = val
    #global weight_d1, weight_d2
    #for i, w in enumerate(weight_d1):
    #    weight_d1[i] = w * val
    #for i, w in enumerate(weight_d2):
    #    weight_d2[i] = w / val
    #print(val, weight_d1, weight_d2)
    

dopa_slider.on_changed(update_dopa)


def toggle_action(label):
    index = int(label.split()[-1])
    if index in actions:
        actions.remove(index)
    else:
        actions.append(index)
    print(f"Active actions: {actions}")

action_button.on_clicked(toggle_action)

# Simulation loop
h.dt = 0.1 #ms
h.finitialize()

weight_d1 = [0.1 * random.randrange(20) for cell in d1] # assign random weight between 0 and 2
weight_d2 = [0.1 * random.randrange(20) for cell in d2] # assign random weight between 0 and 2

while True:
    if paused or len(actions) == 0:
        # Pause simulation
        time.sleep(0.1)
        plt.pause(0.1)
        continue
    
    while (h.t - last_plot_update) < plot_interval: 
        
        # Run simulation
        h.fadvance()

        # Create D1 spike train (direct pathway)
        for i, cell in enumerate(d1): 
            nc_d1[i].weight[0] = weight_d1[i] * DA_level

        # Create D2 spike train (indirect pathway)
        for i, cell in enumerate(d2): 
            nc_d2[i].weight[0] = weight_d2[i] / DA_level
        print(DA_level, [w*DA_level for w in weight_d1], [w/DA_level for w in weight_d2])
        


    # Update plots
    last_plot_update = h.t

    for i, ch in enumerate(channels_to_plot):
        # Update membrane voltage plot
        if t_vec: 
            #lines_snc[i].set_data(t_vec, v_snc[ch])
            #lines_d1[i].set_data(t_vec, v_d1[ch])
            #lines_d2[i].set_data(t_vec, v_d2[ch])
            #lines_gpe[i].set_data(t_vec, v_gpe[ch])
            #lines_gpi[i].set_data(t_vec, v_gpi[ch])
            #lines_thal[i].set_data(t_vec, v_thal[ch])
            #axs[0][i].relim()       # Recalculate limits
            #axs[0][i].autoscale_view()  # Apply new limits
            # Update time axis
            axs[0][i].set_xlim(max(0, last_plot_update - plot_time_range), max(plot_time_range, last_plot_update))


        # Update spike raster plot
        snc_spikes = spike_times['snc'][ch].to_python()
        d1_spikes = spike_times['d1'][ch].to_python()
        d2_spikes = spike_times['d2'][ch].to_python()
        gpe_spikes = spike_times['gpe'][ch].to_python()
        gpi_spikes = spike_times['gpi'][ch].to_python()
        thal_spikes = spike_times['thal'][ch].to_python()
        raster_snc[i].set_data(snc_spikes, [6] * len(snc_spikes))
        raster_d1[i].set_data(d1_spikes, [5] * len(d1_spikes))   
        raster_d2[i].set_data(d2_spikes, [4] * len(d2_spikes))  
        raster_gpe[i].set_data(gpe_spikes, [3] * len(gpe_spikes))   
        raster_gpi[i].set_data(gpi_spikes, [2] * len(gpi_spikes))  
        raster_thal[i].set_data(thal_spikes, [1] * len(thal_spikes)) 
        # Update time axis
        axs[1][i].set_xlim(max(0, last_plot_update - plot_time_range), max(plot_time_range, last_plot_update))

    plt.pause(0.01)


