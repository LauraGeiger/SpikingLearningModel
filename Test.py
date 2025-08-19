from neuron import h, gui
import matplotlib.pyplot as plt

h.load_file("stdrun.hoc")

# -----------------------------
# Helper functions
# -----------------------------

def make_hh_neuron():
    sec = h.Section()
    sec.insert('hh')
    return {'sec': sec}

def make_pc():
    return make_hh_neuron()

def make_dcn():
    return make_hh_neuron()

def make_gc():
    sec = h.Section()
    cell = h.IntFire1()
    cell.tau = 10
    cell.refrac = 2
    return {'sec': sec, 'cell': cell}

def make_netstim(start=0, interval=10, number=5):
    ns = h.NetStim()
    ns.start = start
    ns.interval = interval
    ns.number = number
    ns.noise = 0
    return ns

def connect_netstim_to_cell(netstim, target_cell, weight=0.05, delay=1):
    syn = h.ExpSyn(target_cell['sec'](0.5))
    nc = h.NetCon(netstim, syn)
    nc.weight[0] = weight
    nc.delay = delay
    return nc, syn

def connect_gc_to_pc(gc, pc, weight=0.05, delay=1):
    syn = h.ExpSyn(pc['sec'](0.5))
    nc = h.NetCon(gc['cell'], syn)
    nc.weight[0] = weight
    nc.delay = delay
    return nc, syn

def connect_pc_to_dcn(pc, dcn, weight=0.05, delay=1):
    syn = h.ExpSyn(dcn['sec'](0.5))
    nc = h.NetCon(pc['sec'](0.5)._ref_v, syn, sec=pc['sec'])
    nc.weight[0] = weight
    nc.delay = delay
    return nc, syn

def connect_io_to_pc(io, pc, weight=0.05, delay=1):
    syn = h.ExpSyn(pc['sec'](0.5))
    nc = h.NetCon(io, syn)
    nc.weight[0] = weight
    nc.delay = delay
    return nc, syn

# -----------------------------
# Create network
# -----------------------------

# 1 IO
io = make_netstim(start=50, interval=20, number=5)

# 2 DCNs (positive and negative)
DCNs = [make_dcn() for _ in range(2)]

# 2 PCs
PCs = [make_pc() for _ in range(2)]

# 2 GCs
GCs = [make_gc() for _ in range(2)]

# 2 MFs
MFs = [make_netstim(start=10, interval=10, number=5), make_netstim(start=15, interval=12, number=5)]

# -----------------------------
# Connect network
# -----------------------------

# MFs -> GCs
NCs_MF_GC = []
for mf, gc in zip(MFs, GCs):
    nc, syn = connect_netstim_to_cell(mf, gc)
    NCs_MF_GC.append(nc)

# GCs -> PCs
NCs_GC_PC = []
for gc, pc in zip(GCs, PCs):
    nc, syn = connect_gc_to_pc(gc, pc)
    NCs_GC_PC.append(nc)

# PCs -> DCNs
NCs_PC_DCN = []
for pc, dcn in zip(PCs, DCNs):
    nc, syn = connect_pc_to_dcn(pc, dcn)
    NCs_PC_DCN.append(nc)

# IO -> PCs
NCs_IO_PC = []
for pc in PCs:
    nc, syn = connect_io_to_pc(io, pc)
    NCs_IO_PC.append(nc)

# -----------------------------
# Record spikes
# -----------------------------

spikes_pc = [h.Vector() for _ in PCs]
spikes_dcn = [h.Vector() for _ in DCNs]
t_vec = h.Vector()
t_vec.record(h._ref_t)

for pc, vec in zip(PCs, spikes_pc):
    nc = h.NetCon(pc['sec'](0.5)._ref_v, None, sec=pc['sec'])
    nc.threshold = 0
    nc.record(vec)

for dcn, vec in zip(DCNs, spikes_dcn):
    nc = h.NetCon(dcn['sec'](0.5)._ref_v, None, sec=dcn['sec'])
    nc.threshold = 0
    nc.record(vec)

# -----------------------------
# Run simulation
# -----------------------------
h.tstop = 200
h.run()

# -----------------------------
# Plot raster
# -----------------------------
def plot_raster(spikes, title='Spikes'):
    plt.figure(figsize=(8,4))
    for i, vec in enumerate(spikes):
        plt.vlines(list(vec), i+0.5, i+1.5)
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron')
    plt.title(title)
    

plot_raster(spikes_pc, 'PCs')
plot_raster(spikes_dcn, 'DCNs')
plt.show()
