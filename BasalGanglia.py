from neuron import h#, gui
import matplotlib.pyplot as plt

h.load_file("stdrun.hoc")

N_actions = 12  # number of action channels
actions = [3,5,7]

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

# Spike detectors and vectors
apc_refs = []
spike_times = {'d1': [], 'd2': [], 'gpe': [], 'gpi': [], 'thal': []}
for pop, name in zip([d1, d2, gpe, gpi, thal], spike_times.keys()):
    for cell in pop:
        apc = h.APCount(cell(0.5))
        vec = h.Vector()
        apc.record(vec)
        spike_times[name].append(vec)
        apc_refs.append(apc)

def create_stim(cell, start, number, interval, weight):
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

# Stimulate tonic activity of GPe
stim_gpe = []
syn_gpe = []
nc_gpe = []
for cell in gpe:
    stim, syn, nc = create_stim(cell, start=5, number=1000, interval=20, weight=2)
    stim_gpe.append(stim)
    syn_gpe.append(syn)
    nc_gpe.append(nc)

# Stimulate tonic activity of GPi
stim_gpi = []
syn_gpi = []
nc_gpi = []
for cell in gpi:
    stim, syn, nc = create_stim(cell, start=10, number=1000, interval=20, weight=2)
    stim_gpi.append(stim)
    syn_gpi.append(syn)
    nc_gpi.append(nc)

# Stimulate tonic activity of Thalamus 
stim_thal = []
syn_thal = []
nc_thal = []
for cell in thal:
    stim, syn, nc = create_stim(cell, start=15, number=1000, interval=20, weight=2)
    stim_thal.append(stim)
    syn_thal.append(syn)
    nc_thal.append(nc)

# Create D1 spike train (direct pathway)
stim_d1 = []
syn_d1 = []
nc_d1 = []
for i, cell in enumerate(d1):
    # activate only channel 3 as example
    if i in actions: # facilitate movement
        stim, syn, nc = create_stim(cell, start=0, number=5, interval=20, weight=2)
    else:
        stim, syn, nc = create_stim(cell, start=0, number=0, interval=20, weight=2) 
    stim_d1.append(stim)
    syn_d1.append(syn)
    nc_d1.append(nc)

# Create D2 spike train (indirect pathway)
stim_d2 = []
syn_d2 = []
nc_d2 = []
for i, cell in enumerate(d2):
    # activate channel 7 as example
    if i not in actions: # suppress movement
        stim, syn, nc = create_stim(cell, start=0, number=5, interval=20, weight=2)
    else:
        stim, syn, nc = create_stim(cell, start=0, number=0, interval=20, weight=2)
    stim_d2.append(stim)
    syn_d2.append(syn)
    nc_d2.append(nc)

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
    nc.weight[0] = 1
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
    nc.weight[0] = 1
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

# Recording
t_vec = h.Vector().record(h._ref_t)

v_d1 = [h.Vector().record(cell(0.5)._ref_v) for cell in d1]
v_d2 = [h.Vector().record(cell(0.5)._ref_v) for cell in d2]
v_gpe = [h.Vector().record(cell(0.5)._ref_v) for cell in gpe]
v_gpi = [h.Vector().record(cell(0.5)._ref_v) for cell in gpi]
v_thal = [h.Vector().record(cell(0.5)._ref_v) for cell in thal]

# Run simulation
h.tstop = 100
h.run()

# Plot: membrane potentials of selected channels
channels_to_plot = [0,1,2,3,4,5,6,7,8,9,10,11]  # channels where D1 or D2 activated
fig, axs = plt.subplots(3, len(channels_to_plot), figsize=(14, 10), sharex=True)


for i, ch in enumerate(channels_to_plot):

    axs[0][i].plot(t_vec, v_d1[ch], label=f'D1')
    axs[0][i].plot(t_vec, v_d2[ch], label=f'D2')
    axs[0][i].plot(t_vec, v_gpe[ch], label=f'GPe')
    axs[0][i].plot(t_vec, v_gpi[ch], label=f'GPi')
    axs[0][i].plot(t_vec, v_thal[ch], label=f'Thal')
    #axs[0][i].legend()
    axs[0][i].set_xlabel('Time (ms)')
    if i == 0:
        axs[0][i].set_ylabel('Membrane potential (mV)')
    axs[0][i].set_title(f'Action {channels_to_plot[i]}')


    for t in spike_times['d1'][ch].to_python():
        axs[1][i].plot(t, 3, "C0.", markersize=10)  # D1 
    for t in spike_times['gpi'][ch].to_python():
        axs[1][i].plot(t, 2, "C3.", markersize=10)  # GPi 
    for t in spike_times['thal'][ch].to_python():
        axs[1][i].plot(t, 1, "C4.", markersize=10)  # Thalamus 
    axs[1][i].set_yticks([1, 2, 3], ['Thal', 'GPi', 'D1'])
    axs[1][i].set_xlabel('Time (ms)')
    if i == 0:
        axs[1][i].set_ylabel('Direct Pathway')
    axs[1][i].set_ylim(0.5, 3.5)

    for t in spike_times['d2'][ch].to_python():
        axs[2][i].plot(t, 4, "C1.", markersize=10)  # D2
    for t in spike_times['gpe'][ch].to_python():
        axs[2][i].plot(t, 3, "C2.", markersize=10)  # GPe
    for t in spike_times['gpi'][ch].to_python():
        axs[2][i].plot(t, 2, "C3.", markersize=10)  # GPi 
    for t in spike_times['thal'][ch].to_python():
        axs[2][i].plot(t, 1, "C4.", markersize=10)  # Thalamus 
    axs[2][i].set_yticks([1, 2, 3, 4], ['Thal', 'GPi', 'GPe', 'D2'])
    axs[2][i].set_xlabel('Time (ms)')
    if i == 0:
        axs[2][i].set_ylabel('Indirect Pathway')
    axs[2][i].set_ylim(0.5, 4.5)

plt.tight_layout()
plt.show()
