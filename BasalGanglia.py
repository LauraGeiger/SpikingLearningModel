from neuron import h#, gui
import matplotlib.pyplot as plt

h.load_file("stdrun.hoc")

# Cells
def create_cell(name):
    sec = h.Section(name=name)
    sec.insert('hh')
    return sec

d1 = create_cell('D1')
d2 = create_cell('D2')
gpe = create_cell('GPe')
gpi = create_cell('GPi')
thalamus = create_cell('Thalamus')

# Spike detectors
spike_d1 = h.APCount(d1(0.5))
spike_times_d1 = h.Vector()
spike_d1.record(spike_times_d1)

spike_d2 = h.APCount(d2(0.5))
spike_times_d2 = h.Vector()
spike_d2.record(spike_times_d2)

spike_gpe = h.APCount(gpe(0.5))
spike_times_gpe = h.Vector()
spike_gpe.record(spike_times_gpe)

spike_gpi = h.APCount(gpi(0.5))
spike_times_gpi = h.Vector()
spike_gpi.record(spike_times_gpi)

spike_th = h.APCount(thalamus(0.5))
spike_times_th = h.Vector()
spike_th.record(spike_times_th)

# -------- Direct Pathway ----------------- #
#'''
# D1 spike train
stim_d1 = h.NetStim()
stim_d1.start = 100
stim_d1.number = 5
stim_d1.interval = 20
stim_d1.noise = 0

syn_d1 = h.ExpSyn(d1(0.5))
syn_d1.e = 0
nc_d1 = h.NetCon(stim_d1, syn_d1)
nc_d1.weight[0] = 2
nc_d1.delay = 1
#'''

# GPi tonic spikes 
stim_gpi = h.NetStim()
stim_gpi.start = 10
stim_gpi.number = 30
stim_gpi.interval = 20
stim_gpi.noise = 0

syn_gpi = h.ExpSyn(gpi(0.5))
syn_gpi.e = 0
nc_gpi = h.NetCon(stim_gpi, syn_gpi)
nc_gpi.weight[0] = 2
nc_gpi.delay = 1

# Thalamus tonic spikes
stim_th = h.NetStim()
stim_th.start = 15
stim_th.number = 1000
stim_th.interval = 20
stim_th.noise = 0

syn_th = h.ExpSyn(thalamus(0.5))
syn_th.e = 0
nc_th = h.NetCon(stim_th, syn_th)
nc_th.weight[0] = 2
nc_th.delay = 1

# D1 -> GPi inhibition
syn_d1_gpi = h.ExpSyn(gpi(0.5))
syn_d1_gpi.e = -80
syn_d1_gpi.tau = 15
nc_d1_gpi = h.NetCon(d1(0.5)._ref_v, syn_d1_gpi, sec=d1)
nc_d1_gpi.threshold = 0
nc_d1_gpi.weight[0] = 1.5 #1
nc_d1_gpi.delay = 1

# GPi -> Thalamus inhibition
syn_gpi_th = h.ExpSyn(thalamus(0.5))
syn_gpi_th.e = -80
syn_gpi_th.tau = 5
nc_gpi_th = h.NetCon(gpi(0.5)._ref_v, syn_gpi_th, sec=gpi)
nc_gpi_th.threshold = 0
nc_gpi_th.weight[0] = 1
nc_gpi_th.delay = 1


# -------- Indirect Pathway ----------------- #
# D2 spike train (indirect pathway activation)
#'''
stim_d2 = h.NetStim()
stim_d2.start = 250
stim_d2.number = 5
stim_d2.interval = 20
stim_d2.noise = 0

syn_d2 = h.ExpSyn(d2(0.5))
syn_d2.e = 0
nc_d2 = h.NetCon(stim_d2, syn_d2)
nc_d2.weight[0] = 2
nc_d2.delay = 1
#'''

# GPe tonic spikes
stim_gpe = h.NetStim()
stim_gpe.start = 5
stim_gpe.number = 30
stim_gpe.interval = 20
stim_gpe.noise = 0

syn_gpe = h.ExpSyn(gpe(0.5))
syn_gpe.e = 0
nc_gpe = h.NetCon(stim_gpe, syn_gpe)
nc_gpe.weight[0] = 2
nc_gpe.delay = 1

# D2 -> GPe inhibition
syn_d2_gpe = h.ExpSyn(gpe(0.5))
syn_d2_gpe.e = -80
syn_d2_gpe.tau = 15
nc_d2_gpe = h.NetCon(d2(0.5)._ref_v, syn_d2_gpe, sec=d2)
nc_d2_gpe.threshold = 0
nc_d2_gpe.weight[0] = 1
nc_d2_gpe.delay = 1

# GPe -> GPi inhibition
syn_gpe_gpi = h.ExpSyn(gpi(0.5))
syn_gpe_gpi.e = -80
syn_gpe_gpi.tau = 15
nc_gpe_gpi = h.NetCon(gpe(0.5)._ref_v, syn_gpe_gpi, sec=gpe)
nc_gpe_gpi.threshold = 0
nc_gpe_gpi.weight[0] = 0.6#1
nc_gpe_gpi.delay = 1


# Recording
t_vec = h.Vector().record(h._ref_t)
v_d1 = h.Vector().record(d1(0.5)._ref_v)
v_d2 = h.Vector().record(d2(0.5)._ref_v)
v_gpe = h.Vector().record(gpe(0.5)._ref_v)
v_gpi = h.Vector().record(gpi(0.5)._ref_v)
v_th = h.Vector().record(thalamus(0.5)._ref_v)

# Run
h.tstop = 400
h.run()

# Plot
fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

axs[0].plot(t_vec, v_d1, label='D1')
axs[0].plot(t_vec, v_d2, label='D2', linestyle='--')
axs[0].plot(t_vec, v_gpe, label='GPe', linestyle='--')
axs[0].plot(t_vec, v_gpi, label='GPi')
axs[0].plot(t_vec, v_th, label='Thalamus')
axs[0].axvspan(100, 200, color='C0', alpha=0.3, label='direct pathway active')
axs[0].axvspan(250, 350, color='C1', alpha=0.3, label='Indirect pathway active')
axs[0].legend()
axs[0].set_xlabel('Time (ms)')
axs[0].set_ylabel('Membrane potential (mV)')
axs[0].set_title('Basal Ganglia Toy Model')


for t in spike_times_d1:
    axs[1].plot(t, 3, "C0.", markersize=10)  # D1 
for t in spike_times_gpi:
    axs[1].plot(t, 2, "C3.", markersize=10)  # GPi 
for t in spike_times_th:
    axs[1].plot(t, 1, "C4.", markersize=10)  # Thalamus 
axs[1].axvspan(100, 200, color='C0', alpha=0.3, label='direct pathway active')
#axs[1].axvspan(250, 350, color='C1', alpha=0.3, label='indirect pathway active')
axs[1].legend()
axs[1].set_yticks([1, 2, 3], ['Thalamus', 'GPi', 'D1'])
axs[1].set_xlabel('Time (ms)')
axs[1].set_title('Spike Raster Plot')
axs[1].set_ylim(0.5, 3.5)

for t in spike_times_d2:
    axs[2].plot(t, 4, "C1.", markersize=10)  # D2
for t in spike_times_gpe:
    axs[2].plot(t, 3, "C2.", markersize=10)  # GPe
for t in spike_times_gpi:
    axs[2].plot(t, 2, "C3.", markersize=10)  # GPi 
for t in spike_times_th:
    axs[2].plot(t, 1, "C4.", markersize=10)  # Thalamus 
#axs[2].axvspan(100, 200, color='C0', alpha=0.3, label='direct pathway active')
axs[2].axvspan(250, 350, color='C1', alpha=0.3, label='indirect pathway active')
axs[2].legend()
axs[2].set_yticks([1, 2, 3, 4], ['Thalamus', 'GPi', 'GPe', 'D2'])
axs[2].set_xlabel('Time (ms)')
axs[2].set_title('Spike Raster Plot')
axs[2].set_ylim(0.5, 4.5)

plt.tight_layout()
plt.show()

