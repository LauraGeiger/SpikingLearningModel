from neuron import h
import matplotlib.pyplot as plt
import random

h.load_file("stdrun.hoc")

# ----------------------
# Parameters
# ----------------------
N_grasp = 3
N_joints = 6
N_actuators = 6
N_flex = 6
N_pressure = 5

N_PN = N_grasp + N_joints + N_actuators + N_flex + N_pressure
N_GC = 100

N_DCN = 2
N_PC = 2 * N_DCN
N_IO = N_DCN

sim_time = 2000  # ms

# Example input signals
grasp_type = [1, 0, 0]       
joint_mask = [1,1,0,0,0,0]  
actuator_mask = [0,1,0,1,1,0]
flex_sensor = [0.8,0.5,0,0,0,0]  
pressure_sensor = [0,1,0,0,0]    

input_values = grasp_type + joint_mask + actuator_mask + flex_sensor + pressure_sensor

# ----------------------
# Create HH neurons for GCs, PCs, DCNs, IOs
# ----------------------
def create_hh_neuron():
    sec = h.Section()
    sec.L = sec.diam = 10
    sec.insert('hh')
    return {'sec': sec}

# PNs
PNs = [create_hh_neuron() for _ in range(N_PN)]

# GCs
GCs = [create_hh_neuron() for _ in range(N_GC)]

# PCs
PCs = [create_hh_neuron() for _ in range(N_PC)]

# DCNs
DCNs = [create_hh_neuron() for _ in range(N_DCN)]

# IOs
IOs = [create_hh_neuron() for _ in range(N_IO)]

ncs = []
syns = []
# ----------------------
# Connect PNs -> GCs (sparse)
# ----------------------
for gc in GCs:
    pn_subset = random.sample(PNs, 3)
    for pn in pn_subset:
        syn = h.ExpSyn(gc['sec'](0.5))
        syn.tau = 2
        syns.append(syn)
        nc = h.NetCon(pn['sec'](0.5)._ref_v, syn, sec=pn['sec'])
        nc.weight[0] = 0.0005
        nc.threshold = 0  # mV threshold for PN spikes
        ncs.append(nc) 

# GC -> PC
for pc in PCs:
    for gc in GCs:
        syn = h.ExpSyn(pc['sec'](0.5))
        syn.tau = 2
        syns.append(syn)
        nc = h.NetCon(gc['sec'](0.5)._ref_v, syn, sec=gc['sec'])
        nc.weight[0] = 0.1
        nc.threshold = 0  # mV threshold for GC spikes
        ncs.append(nc)

# PC -> DCN (inhibitory)
for dcn_id, dcn in enumerate(DCNs):
    for pc in PCs[dcn_id*N_PC//N_DCN:(dcn_id+1)*N_PC//N_DCN]:
        syn = h.ExpSyn(dcn['sec'](0.5))
        syn.tau = 5
        syn.e = -75  # inhibitory
        syns.append(syn)
        nc = h.NetCon(pc['sec'](0.5)._ref_v, syn, sec=pc['sec'])
        nc.weight[0] = 0.1
        nc.threshold = 0
        ncs.append(nc)

# IO -> PC (climbing fiber)
for io in IOs:
    for pc in PCs:
        syn = h.ExpSyn(pc['sec'](0.5))
        syn.tau = 2
        syns.append(syn)
        nc = h.NetCon(io['sec'](0.5)._ref_v, syn, sec=io['sec'])
        nc.weight[0] = 0.1
        nc.threshold = 0
        ncs.append(nc)

stims = []
for id, (pn, val) in enumerate(zip(PNs, input_values)):
    stim = h.NetStim()
    stim.start = id
    stim.number = 1e9
    stim.interval = 50 / (val+0.01)
    stims.append(stim)
    syn = h.ExpSyn(pn['sec'](0.5))
    syn.e = 0
    syns.append(syn)
    nc = h.NetCon(stim, syn)
    nc.weight[0] = 0.005
    ncs.append(nc)

#PN_stim = []
#for pn, val in zip(PNs, input_values):
#    stim = h.IClamp(pn['sec'](0.5))
#    stim.delay = 0        # start immediately
#    stim.dur = sim_time/2  
#    stim.amp = val * 0.05  # scale current by input value (tune as needed)
#    PN_stim.append(stim)

for dcn in DCNs:
    stim = h.NetStim()
    stim.start = 0
    stim.number = 1e9
    stim.interval = 100
    stims.append(stim)
    syn = h.ExpSyn(dcn['sec'](0.5))
    syn.e = 0
    syns.append(syn)
    nc = h.NetCon(stim, syn)
    nc.weight[0] = 0.01
    ncs.append(nc)

for io_id, io in enumerate(IOs):
    stim = h.NetStim()
    stim.start = 200 * io_id
    stim.number = 1e9
    stim.interval = 500
    stims.append(stim)
    syn = h.ExpSyn(io['sec'](0.5))
    syn.e = 0
    syns.append(syn)
    nc = h.NetCon(stim, syn)
    nc.weight[0] = 0.01
    ncs.append(nc)

# ----------------------
# Record spikes
# ----------------------
def record_spikes_hh(neurons):
    spike_vecs = []
    for n in neurons:
        vec = h.Vector()
        nc = h.NetCon(n['sec'](0.5)._ref_v, None, sec=n['sec'])
        nc.threshold = 0
        nc.record(vec)
        spike_vecs.append(vec)
    return spike_vecs

spikes_PN = record_spikes_hh(PNs)
spikes_GC = record_spikes_hh(GCs)
spikes_PC = record_spikes_hh(PCs)
spikes_DCN = record_spikes_hh(DCNs)
spikes_IO = record_spikes_hh(IOs)

def record_voltage(neurons):
    v_vecs = []
    t_vec = h.Vector()
    t_vec.record(h._ref_t)  # record simulation time
    for n in neurons:
        v = h.Vector()
        v.record(n['sec'](0.5)._ref_v)
        v_vecs.append(v)
    return t_vec, v_vecs

t_vec, v_PN = record_voltage(PNs)
_, v_GC = record_voltage(GCs)
_, v_PC = record_voltage(PCs)
_, v_DCN = record_voltage(DCNs)
_, v_IO = record_voltage(IOs)

# ----------------------
# Run simulation
# ----------------------
h.dt = 0.1
h.finitialize(-65)
h.continuerun(sim_time)

# ----------------------
# Plotting raster
# ----------------------
def plot_raster(spike_vectors, title):
    plt.figure(figsize=(8,4))
    for i, vec in enumerate(spike_vectors):
        plt.vlines(list(vec), i+0.6, i+1.4)
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index')
    plt.title(title)

plot_raster(spikes_PN, 'PNs')
plot_raster(spikes_GC, 'GCs')
plot_raster(spikes_PC, 'PCs')
plot_raster(spikes_DCN, 'DCNs')
plot_raster(spikes_IO, 'IOs')

def plot_voltage(t_vec, v_vecs, title):
    plt.figure(figsize=(10,4))
    for i, v in enumerate(v_vecs):
        plt.plot(list(t_vec), list(v), label=f'{title} {i}')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane potential')
    plt.title(title)
    plt.legend()

plot_voltage(t_vec, v_PN, 'PN membrane potentials')
plot_voltage(t_vec, v_GC, 'GC membrane potentials')
plot_voltage(t_vec, v_PC, 'PC membrane potentials')
plot_voltage(t_vec, v_DCN, 'DCN membrane potentials')
plot_voltage(t_vec, v_IO, 'IO membrane potentials')

plt.show()
