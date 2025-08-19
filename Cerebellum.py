from neuron import h, gui
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.widgets import Button
from matplotlib.widgets import RadioButtons
from matplotlib.gridspec import GridSpec
import time

# using Python 3.10.16

# --- Granule, Purkinje, Inferior Olive, and Basket Cell Classes ---
class PontineNuclei:
    def __init__(self, gid):
        self.gid = gid
        self.soma = h.Section(name=f'pontine_{gid}')
        self.soma.L = self.soma.diam = 10
        self.soma.insert('hh')

class GranuleCell:
    def __init__(self, gid):
        self.gid = gid
        self.soma = h.Section(name=f'granule_{gid}')
        self.soma.L = self.soma.diam = 10
        self.soma.insert('hh')

class PurkinjeCell:
    def __init__(self, gid):
        self.gid = gid
        self.soma = h.Section(name=f'purkinje_{gid}')
        self.soma.L = self.soma.diam = 50
        self.soma.insert('hh')

class InferiorOliveCell:
    def __init__(self, gid):
        self.gid = gid
        self.soma = h.Section(name=f'inferior_olive_{gid}')
        self.soma.L = self.soma.diam = 20
        self.soma.insert('hh')

class BasketCell:
    def __init__(self, gid):
        self.gid = gid
        self.soma = h.Section(name=f'basket_{gid}')
        self.soma.L = self.soma.diam = 20
        self.soma.insert('hh')

class DeepCerebellarNuclei:
    def __init__(self, gid):
        self.gid = gid
        self.soma = h.Section(name=f'deep_cerebellar{gid}')
        self.soma.L = self.soma.diam = 10
        self.soma.insert('hh')

def init_variables(reset_all=True):
    """Initialize global variables"""
    global fig, gs, ax_network, ax_plots, gs_buttons, animations, purkinje_drawing
    global iter, buttons, state, state_dict, state_grasp_hold_dict, DCN_names
    global colors_purkinje, color_granule, color_inferior_olive, color_basket, color_dcn, color_simple_spike, color_complex_spike, color_error, color_error_hover, color_run, color_run_hover
    global height, width, granule_x, purkinje_x, olive_x, basket_x, dcn_x, granule_y, purkinje_y, olive_y, basket_y, dcn_y
    global t, granule_spikes, purkinje_spikes, inferiorOlive_spikes, basket_spikes, dcn_spikes, v_granule, v_purkinje, v_inferiorOlive, v_basket, v_dcn, t_np, v_granule_np, v_purkinje_np, v_inferiorOlive_np, v_basket_np, v_dcn_np
    global num_granule, num_purkinje, num_inferior_olive, num_basket, num_dcn, pontine_nuclei, granule_cells, purkinje_cells, inferior_olive_cells, basket_cells, deep_cerebellar
    global mf_syns, mf_ncs, pf_syns, pf_ncs, cf_syns, cf_ncs, inh_syns, inh_ncs, inh_dcn_syns, inh_dcn_ncs
    global weights, weights_over_time, pf_initial_weight, cf_initial_weight, basket_initial_weight, dcn_initial_weight, stimuli, frequency, processed_GC_spikes, processed_pairs, errors
    global tau_plus, tau_minus, A_plus, A_minus

    # --- Plotting ---
    try: # Reset figure
        for widget in buttons.values():
            widget.disconnect_events()  # Disconnect event listeners
            del widget  # Delete the widget instance
        for ax in fig.get_axes():
            ax.remove()  # Remove the axis
    except Exception: # Create figure
        plt.ion()  # Turn on interactive mode
        fig = plt.figure(layout="constrained", figsize=[11,7])

    gs, ax_network, ax_plots, gs_buttons = None, None, None, None
    animations = []
    purkinje_drawing = []

    # --- GUI and Control ---
    iter = 0
    buttons = {}
    if reset_all == True: state = 2                           # 1: light obj (GC0), 2: medium obj (GC1), 3: heavy obj (GC2) 
    state_dict = {1: "Light obj.", 2: "Medium obj.", 3: "Heavy obj."}
    state_grasp_hold_dict = {0: "Grasp obj.", 1: "Grasp & hold light obj.", 2: "Grasp & hold medium obj.", 3: "Grasp & hold heavy obj."}
    DCN_names = [
        ["Air Pressure"], 
        ["Timing Thumb Flexion", "Timing Index Finger Flexion"], 
        ["Air Pressure", "Timing Thumb Flexion", "Timing Index Finger Flexion"], 
        ["Timing Thumb\nOpposition & Extension", "Timing Index Finger\nFlexion", "Timing Index Finger\nExtension", "Timing Flexion\nfor Holding"]
    ]
    
    # --- Colors ---
    colors_purkinje = ["steelblue", "darkorange", "mediumseagreen", "crimson", "gold",
    "dodgerblue", "purple", "sienna", "limegreen", "deeppink",
    "teal", "orangered", "indigo", "royalblue", "darkgoldenrod",
    "firebrick", "darkcyan", "tomato", "slateblue", "darkgreen"]
    color_granule = 'darkgoldenrod'
    color_inferior_olive = 'black'
    color_basket = 'darkgray'
    color_dcn = 'darkgray'
    color_simple_spike = 'gold'
    color_complex_spike = 'lightcoral'
    color_error = 'coral'
    color_error_hover = 'lightsalmon'
    color_run = "lightgreen"
    color_run_hover = "palegreen"

    # --- Animation ---
    height, width = None, None
    granule_x, purkinje_x, olive_x, basket_x, dcn_x = None, None, None, None, None
    granule_y, purkinje_y, olive_y, basket_y, dcn_y = None, None, None, None, None

    # --- Spikes and Voltages for Plotting ---
    t = h.Vector()  # First time initialization
    t_np = None
    granule_spikes, purkinje_spikes, inferiorOlive_spikes, basket_spikes, dcn_spikes = None, None, None, None, None
    v_granule, v_purkinje, v_inferiorOlive, v_basket, v_dcn = None, None, None, None, None
    v_granule_np, v_purkinje_np, v_inferiorOlive_np, v_basket_np, v_dcn_np = None, None, None, None, None

    # --- Create Network ---
    N_grasp = 3
    N_joints = 6
    N_actuators = 6
    N_flex = 6
    N_pressure = 5
    if reset_all == True: num_pontine = N_grasp + N_joints + N_actuators + N_flex + N_pressure
    if reset_all == True: num_granule = 20
    if reset_all == True: num_purkinje = 4
    if reset_all == True: num_inferior_olive = 2
    if reset_all == True: num_basket = 2
    if reset_all == True: num_dcn = 2
    pontine_nuclei = [PontineNuclei(i) for i in range(num_pontine)]
    granule_cells = [GranuleCell(i) for i in range(num_granule)]
    purkinje_cells = [PurkinjeCell(i) for i in range(num_purkinje)]
    inferior_olive_cells = [InferiorOliveCell(i) for i in range(num_inferior_olive)]
    basket_cells = [BasketCell(i) for i in range(num_basket)]
    deep_cerebellar = [DeepCerebellarNuclei(i) for i in range(num_dcn)]

    # --- Create Synapses and Connections ---
    mf_syns = [[None for _ in range(num_granule)] for _ in range(num_pontine)] # mossy fiber synapses
    mf_ncs = [[None for _ in range(num_granule)] for _ in range(num_pontine)] # mossy fiber netcons
    pf_syns = [[None for _ in range(num_purkinje)] for _ in range(num_granule)] # parallel fiber synapses
    pf_ncs = [[None for _ in range(num_purkinje)] for _ in range(num_granule)] # parallel fiber netcons
    cf_syns = [[None for _ in range(num_purkinje)] for _ in range(num_inferior_olive)] # climbing fiber synapses
    cf_ncs = [[None for _ in range(num_purkinje)] for _ in range(num_inferior_olive)] # climbing fiber netcons
    inh_syns = [[None for _ in range(num_purkinje)] for _ in range(num_basket)] # inhibitory synapses from basket cell
    inh_ncs = [[None for _ in range(num_purkinje)] for _ in range(num_basket)] # inhibitory netcons from basket cell
    inh_dcn_syns = [[None for _ in range(num_dcn)] for _ in range(num_purkinje)] # inhibitory synapses to DCN
    inh_dcn_ncs = [[None for _ in range(num_dcn)] for _ in range(num_purkinje)] # inhibitory netcons to DCN

    # --- Spikes and Weights ---
    weights = {}
    weights_over_time = { (pre_gid, post_gid): [] for pre_gid in range(num_granule) for post_gid in range(num_purkinje) } # track weights over time
    pf_initial_weight = 0.02 # Parallel fiber initial weight
    cf_initial_weight = 0.3 # Climbing fiber initial weight
    basket_initial_weight = 0.5 # Basket to Purkinje weight
    dcn_initial_weight = 0.5 # Purkinje to DCN weight
    stimuli = []
    frequency = 50 # Hz
    processed_GC_spikes = { (g_gid): set() for g_gid in range(num_granule)} # store the processed granule cell spikes
    processed_pairs = { (pre_id, post_id): set() for pre_id in range(num_granule) for post_id in range(num_purkinje) } # store the processed spike pairs for each (pre_id, post_id)
    errors = [False]*num_dcn

    # --- Learning Parameters ---
    tau_plus = 1 
    tau_minus = 1.5
    A_plus = 0.05  
    A_minus = 0.06


def create_connections():
    """Create synapses between the cells and initialize weights"""
    global weights

    # Granule → Purkinje Connections (excitatory)
    for purkinje in purkinje_cells:
        random_weight = np.random.uniform(0,0.001)
        for granule in granule_cells:
            syn = h.Exp2Syn(purkinje.soma(0.5))
            syn.e = 0 # Excitatory
            syn.tau1 = 1 # Synaptic rise time
            syn.tau2 = 5 # Synaptic decay time
            pf_syns[granule.gid][purkinje.gid] = syn
            nc = h.NetCon(granule.soma(0.5)._ref_v, syn, sec=granule.soma)
            nc.weight[0] = pf_initial_weight + random_weight
            nc.delay = 0
            pf_ncs[granule.gid][purkinje.gid] = nc
            weights[(granule.gid, purkinje.gid)] = nc.weight[0]
    
    # Inferior Olive → Purkinje Connections (excitatory)
    group_size = num_purkinje // num_inferior_olive  # Size of each IO’s Purkinje group
    remainder = num_purkinje % num_inferior_olive  # Handle remainder case

    for inferior_olive in inferior_olive_cells:
        start_idx = inferior_olive.gid * group_size + min(inferior_olive.gid, remainder)  
        end_idx = start_idx + group_size + (1 if inferior_olive.gid < remainder else 0)  

        for purkinje_gid in range(start_idx, end_idx):
            purkinje = purkinje_cells[purkinje_gid]
            syn = h.Exp2Syn(purkinje.soma(0.5))
            syn.e = 0  # Excitatory
            syn.tau1 = 5 # Synaptic rise time
            syn.tau2 = 25 # Synaptic decay time
            cf_syns[inferior_olive.gid][purkinje.gid] = syn
            nc = h.NetCon(inferior_olive.soma(0.5)._ref_v, syn, sec=inferior_olive.soma)
            nc.weight[0] = 0
            nc.delay = 0
            cf_ncs[inferior_olive.gid][purkinje.gid] = nc

    # Basket → Purkinje Connections (inhibitory)
    for basket in basket_cells:
        for purkinje in purkinje_cells:
            syn = h.Exp2Syn(purkinje.soma(0.5))
            syn.e = -70  # Inhibitory
            syn.tau1 = 1 # Synaptic rise time
            syn.tau2 = 5 # Synaptic decay time
            inh_syns[basket.gid][purkinje.gid] = syn
            nc = h.NetCon(basket.soma(0.5)._ref_v, syn, sec=basket.soma)
            nc.weight[0] = basket_initial_weight
            nc.delay = 0
            inh_ncs[basket.gid][purkinje.gid] = nc

    # Purkinje → DCN Connections (inhibitory)
    for purkinje in purkinje_cells:
        for dcn in deep_cerebellar:
            syn = h.Exp2Syn(dcn.soma(0.5))
            syn.e = -70  # Inhibitory
            syn.tau1 = 1 # Synaptic rise time
            syn.tau2 = 5 # Synaptic decay time
            inh_dcn_syns[purkinje.gid][dcn.gid] = syn
            nc = h.NetCon(purkinje.soma(0.5)._ref_v, syn, sec=purkinje.soma)
            nc.weight[0] = dcn_initial_weight
            nc.delay = 0
            inh_dcn_ncs[purkinje.gid][dcn.gid] = nc

def activate_highest_weight_PC(granule_gid):
    """Activate the PC with the highest weight (that is not blocked)"""
    global inh_syns, inh_ncs, stimuli

    # Initialize dictionaries for max_weights and active Purkinje cells
    max_weights = [-np.inf] * num_dcn
    active_purkinje = [None] * num_dcn

    def get_active_purkinje(purkinje, weight, weight_key):
        if weight > max_weights[weight_key]:
                max_weights[weight_key] = weight
                active_purkinje[weight_key] = purkinje

    # Find the Purkinje cell with the highest weight
    for purkinje in purkinje_cells:
        try:
            if v_purkinje_np[purkinje.gid][-1] > -56: # if membrane voltage is above 55 mV
                continue # Skip the blocked Purkinje cell
        except (NameError, IndexError):
            continue

        weight = pf_ncs[granule_gid][purkinje.gid].weight[0]

        for i in range(num_dcn if state > 0 else num_dcn - 1):
            if purkinje.gid >= i * num_purkinje//num_dcn and purkinje.gid < (i+1) * num_purkinje//num_dcn:
                get_active_purkinje(purkinje, weight, i)

    # Set inhibition and climbing fiber weights for all Purkinje cells
    for purkinje in purkinje_cells:
        is_active = purkinje in active_purkinje # Check if the Purkinje cell is active

        # Set inhibition and climbing fiber weights
        new_inh_weight = 0 if is_active else basket_initial_weight
        new_cf_weight = cf_initial_weight if is_active else 0

        for basket in basket_cells:
            if inh_ncs[basket.gid][purkinje.gid] is not None:
                inh_ncs[basket.gid][purkinje.gid].weight[0] = new_inh_weight
        
        for inferior_olive in inferior_olive_cells:
            if cf_ncs[inferior_olive.gid][purkinje.gid] is not None:
                cf_ncs[inferior_olive.gid][purkinje.gid].weight[0] = new_cf_weight


def stimulate_granule_cell():
    """Stimulate Granule Cells Based on State"""
    
    if state > 0:
        g_ids = [state-1] 
    else:
        g_ids = [g_id for g_id in range(num_granule)]
    
    for g_id in g_ids:
        stim = h.IClamp(granule_cells[g_id].soma(0.5))
        stim.delay = 1/frequency*1000 * (iter + 1/3)
        stim.dur = 1
        stim.amp = 0.5
        stimuli.append(stim)

    # Send inhibitory signals to all purkinje cells expect active_purkinje
    for basket in basket_cells:
        basket_stim = h.IClamp(basket.soma(0.5))
        basket_stim.delay = stim.delay # same as granule 
        basket_stim.dur = stim.dur # same as granule 
        basket_stim.amp = stim.amp  # same as granule 
        stimuli.append(basket_stim)

def change_back_error_button_colors():
    """Changes the color of all error buttons back to default color"""
    global buttons
    # Change color of error buttons to default color
    try:
        buttons["error_button"].color = "0.85"
        buttons["error_button"].hovercolor = "0.975"
        buttons["error_button"].ax.figure.canvas.draw_idle()  # Force redraw
    except KeyError: None
    try:
        buttons["error_pressure"].color = "0.85"
        buttons["error_pressure"].hovercolor = "0.975"
        buttons["error_pressure"].ax.figure.canvas.draw_idle()  # Force redraw
    except KeyError: None
    try:
        buttons["error_thumb"].color = "0.85"
        buttons["error_thumb"].hovercolor = "0.975"
        buttons["error_thumb"].ax.figure.canvas.draw_idle()  # Force redraw
    except KeyError: None
    try:
        buttons["error_index"].color = "0.85"
        buttons["error_index"].hovercolor = "0.975"
        buttons["error_index"].ax.figure.canvas.draw_idle()  # Force redraw
    except KeyError: None
    try:
        buttons["error_index_flexion"].color = "0.85"
        buttons["error_index_flexion"].hovercolor = "0.975"
        buttons["error_index_flexion"].ax.figure.canvas.draw_idle()  # Force redraw
    except KeyError: None
    try:
        buttons["error_index_extension"].color = "0.85"
        buttons["error_index_extension"].hovercolor = "0.975"
        buttons["error_index_extension"].ax.figure.canvas.draw_idle()  # Force redraw
    except KeyError: None
    try:
        buttons["error_opposition"].color = "0.85"
        buttons["error_opposition"].hovercolor = "0.975"
        buttons["error_opposition"].ax.figure.canvas.draw_idle()  # Force redraw
    except KeyError: None
    try:
        buttons["error_holding"].color = "0.85"
        buttons["error_holding"].hovercolor = "0.975"
        buttons["error_holding"].ax.figure.canvas.draw_idle()  # Force redraw
    except KeyError: None
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    #plt.pause(1)

def update_granule_stimulation_and_plots(event=None):
    """Stimulates one granule cell and updates the plots """
    global granule_spikes, purkinje_spikes, inferiorOlive_spikes, basket_spikes, dcn_spikes, buttons, iter, ax_network, animations
    global errors

    # Apply errors
    for i, error in enumerate(errors):
        if error:
            update_inferior_olive_stimulation_and_plots(cell_nr=i)

    # Reset errors
    errors = [False]*num_dcn

    try: # change back color of "grasp successul" button
        buttons["success_grasp"].color = "0.85"
        buttons["success_grasp"].hovercolor = "0.975"
        buttons["success_grasp"].ax.figure.canvas.draw_idle()  # Force redraw
    except KeyError: None
    change_back_error_button_colors()
    time.sleep(0.1)
    
    run_simulation(error=True)

    # Activate PC with highest weight
    g_id = (state - 1) % num_granule # choose one GC for calculation and plotting
    activate_highest_weight_PC(g_id)

    # Identify active purkinje cells
    b_id = 0
    p_ids = []
    p_ids.append(next((purkinje.gid for purkinje in purkinje_cells[:num_purkinje//3] if inh_ncs[b_id][purkinje.gid].weight[0] == 0), None))
    p_ids.append(next((purkinje.gid for purkinje in purkinje_cells[num_purkinje//3:2*num_purkinje//3] if inh_ncs[b_id][purkinje.gid].weight[0] == 0), None))
    p_ids.append(next((purkinje.gid for purkinje in purkinje_cells[2*num_purkinje//3:] if inh_ncs[b_id][purkinje.gid].weight[0] == 0), None))
    
    stimulate_granule_cell()
    run_simulation()
    iter += 1
    buttons["run_button"].label.set_text(f"Run iteration {iter}")
    update_spike_and_weight_plot()

    if buttons["network_button"].label.get_text() == "Hide network":
        # Run simple spike animation
        spikes = []
        if state > 0:
            g_ids = [state-1] 
        else:
            g_ids = [g_id for g_id in range(num_granule)]
        

        for g in g_ids:
            for p in p_ids:
                spike, = ax_network.plot([], [], marker='o', color=color_simple_spike, markersize=10)
                spikes.append(spike)
        ani = animation.FuncAnimation(ax_network.figure, update_animation, frames=60, interval = 20, blit=True, repeat=False, fargs=(spikes, 0, p_ids*len(g_ids), g_ids*len(p_ids)))
        animations.append(ani)
        #plt.pause(5)
        #time.sleep(2)
        update_weights_in_network()        

    buttons["run_button"].color = color_run
    buttons["run_button"].hovercolor = color_run_hover
    buttons["run_button"].ax.figure.canvas.draw_idle()  # Force redraw
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    #plt.pause(1)

def stimulate_inferior_olive_cell(i_id=0):
    """Stimulate Inferior Olive"""
    stim = h.IClamp(inferior_olive_cells[i_id].soma(0.5))
    stim.delay = h.t
    stim.dur = 5
    stim.amp = 0.1
    stimuli.append(stim)

def update_inferior_olive_stimulation_and_plots(event=None, cell_nr=0):
    """Stimulates a inferior olive and updates the plots"""
    global buttons, animations
    
    if buttons["network_button"].label.get_text() == "Hide network":
        # Identify active purkinje cell
        b_id = 0
        p_id_first = next((purkinje.gid for purkinje in purkinje_cells[:num_purkinje//3] if inh_ncs[b_id][purkinje.gid].weight[0] == 0), None)
        p_id_second = next((purkinje.gid for purkinje in purkinje_cells[num_purkinje//3:2*num_purkinje//3] if inh_ncs[b_id][purkinje.gid].weight[0] == 0), None)
        p_id_third = next((purkinje.gid for purkinje in purkinje_cells[2*num_purkinje//3:] if inh_ncs[b_id][purkinje.gid].weight[0] == 0), None)
      
        # Run complex spike animation
        spike, = ax_network.plot([], [], marker='o', color=color_complex_spike, markersize=10)
        spikes = [spike]
        i_ids = [cell_nr]
        p_ids = []
        
        if cell_nr == 0: # Trigger complex spike from IO 0
            p_ids.append(p_id_first)
        elif cell_nr == 1: # Trigger complex spike from IO 1
            p_ids.append(p_id_second)
        elif cell_nr == 2: # Trigger complex spike from IO 2
            p_ids.append(p_id_third)
        
        #ani = animation.FuncAnimation(ax_network.figure, update_animation, frames=60, interval = 20, blit=True, repeat=False, fargs=(spikes, 1, p_ids, i_ids))
        #animations.append(ani)
        #plt.pause(5)
        #time.sleep(2)
        
    stimulate_inferior_olive_cell(i_id=cell_nr)
    
    update_spike_and_weight_plot()

def update_state(event=None):
    """Update state variable"""
    global state, buttons
    global pf_ncs, cf_ncs

    state = buttons["state_button"].index_selected + 1
    
    print(f"STATE: {state_dict[state]}")

    if buttons["network_button"].label.get_text() == "Hide network":
        ax_network.cla() # clear network plot
        ax_network.axis("off")
        show_network_graph()
    
    update_spike_and_weight_plot()

def grasp_successfull(event=None):
    """Clears all errors and changes colors of error buttons to default"""
    global errors, buttons

    try: # change color of "grasp successul" button
        buttons["success_grasp"].color = color_run
        buttons["success_grasp"].hovercolor = color_run_hover
        buttons["success_grasp"].ax.figure.canvas.draw_idle()  # Force redraw
    except KeyError: None
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    #plt.pause(1)

    errors = [False for _ in errors] # reset all errors
    change_back_error_button_colors()

def error_detected(event=None, btn_name=None, cell_nr=None):
    """Changes the color of the clicked error button and sets the correct error"""
    global buttons, errors

    if cell_nr != None and errors[cell_nr] == True: # if error is already set, reset error
        errors[cell_nr] = False
        color = "0.85"
        hovercolor = "0.975"
    else: # if error is not set, set error
        if cell_nr != None:
            errors[cell_nr] = True
        color = color_error
        hovercolor = color_error_hover

    try: # change color of clicked error button
        buttons[btn_name].color = color
        buttons[btn_name].hovercolor = hovercolor
        buttons[btn_name].ax.figure.canvas.draw_idle()  # Force redraw
    except KeyError: None
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.draw()
    #plt.pause(1)
    time.sleep(0.1)
   
def toggle_network_graph(event=None):
    """Toggles between showing and hiding the network graph in the GUI"""
    global buttons, ax_network, gs
    if buttons["network_button"].label.get_text() == "Hide network":
        buttons["network_button"].label.set_text("Show network")
        ax_network.cla() # clear network plot
        ax_network.axis("off")
        gs.set_height_ratios([0.1, 1])
        gs_buttons.set_height_ratios([1,2,2,2.5,2.5,1,2,1,1])
    else:
        buttons["network_button"].label.set_text("Hide network")
        gs.set_height_ratios([0.9, 1])
        gs_buttons.set_height_ratios([2,2,2,2.5,2.5,1,2,1,1])
        show_network_graph()
        
    update_spike_and_weight_plot()

def draw_purkinje(ax, x, y, width=0.2, height=1, color='orange', line_width=0.01):
    """Draws a Purkinje neuron with dendrites and a separate soma."""
    purkinje_drawing = []

    # Dendritic tree
    for i in range(num_granule):  # Branching
        top_width =  (line_width[i] if np.isscalar(line_width) is not True else line_width)
        triangle = patches.Polygon([
            (x + (i-1) * (width if i > 0 else width/2) - top_width / 2, y + (i+1) * width),  # Left top
            (x + (i-1) * (width if i > 0 else width/2) + top_width / 2, y + (i+1) * width),  # Right top
            (x, y)  # Bottom center
        ], closed=True, color=color, alpha=0.6)  # Slight transparency

        ax.add_patch(triangle)
        purkinje_drawing.append(triangle)
    
    # Axons
    drawing = ax.plot([x, x], [y, y - height], color=color, lw=2)
    purkinje_drawing.append(drawing[0])

    # Soma (neuron body)
    drawing = ax.scatter(x, y, s=200, color=color, zorder=2)
    purkinje_drawing.append(drawing)

    return purkinje_drawing

def draw_parallel_fiber(ax, x, y, length=5, transparency=1):
    """Draws a parallel fiber extending across Purkinje cells."""
    ax.plot([x - length / 30, x + length], [y , y], color=color_granule, lw=2, alpha=transparency)

def draw_granule_to_parallel(ax, x, y_start, y_end, transparency):
    """Draws a granule cell axon that ascends vertically and forms a parallel fiber."""
    ax.plot([x, x], [y_start, y_end], color=color_granule, lw=2, alpha=transparency)  # Vertical axon
    draw_parallel_fiber(ax, x, y_end, transparency=transparency)  # Horizontal fiber

def draw_climbing_fiber(ax, x, y_start, y_end, width=0.3):
    """Draws a climbing fiber from the Inferior Olive wrapping around a Purkinje cell."""

    ax.plot([x + 0.15, x + 0.15], [y_start, y_end - 0.15], color=color_inferior_olive, lw=2, label="Climbing Fiber")
    ax.plot([x, x + 0.15], [y_end, y_end - 0.15], color=color_inferior_olive, lw=2, label="Climbing Fiber")

    for i in range(num_granule):  # Branching
        branch_x_start = x
        branch_x_end = x + (i-1) * (width if i > 0 else width/2)
        branch_y_start = y_end
        branch_y_end = y_end + (i+1) * width

        dx = branch_x_end - branch_x_start
        dy = branch_y_end - branch_y_start
        length = np.sqrt(dx**2 + dy**2)  
        if length == 0:
            continue  # Avoid division by zero

        t = np.linspace(0, length, 100)
        wave = 0.015 * np.sin(15 * np.pi * t)

        # Compute unit direction
        ux, uy = dx / length, dy / length  # Unit vector along branch
        nx, ny = -uy, ux  # Perpendicular vector for wave oscillation
        x_vals = branch_x_start + ux * t + wave * nx
        y_vals = branch_y_start + uy * t + wave * ny

        ax.plot(x_vals, y_vals, color=color_inferior_olive, lw=1, label="Climbing Fiber")

def calculate_dcn_x_positions(purkinje_x, num_dcn):
    """Calculated the x positions of Deep Cerebellar Nuclei"""
    dcn_x = []

    # Split the purkinje_x into equal segments and compute the middle points
    segment_length = len(purkinje_x) // num_dcn  # Calculate segment size
    
    for i in range(num_dcn):
        # Determine the segment range
        start_idx = i * segment_length
        if i == num_dcn - 1:
            # For the last segment, include all remaining elements
            end_idx = len(purkinje_x)
        else:
            end_idx = (i + 1) * segment_length
        
        # Calculate the average of the current segment
        segment_avg = np.mean(purkinje_x[start_idx:end_idx])
        dcn_x.append(segment_avg)

    return dcn_x

def update_weights_in_network():
    """Updated the weights in the network as increasing or decreasing triangles"""
    global ax_network, purkinje_drawing

    # --- Normalize Triangle Widths ---
    min_w, max_w = min(weights.values()), max(weights.values())
    triangle_widths = np.empty((num_granule, num_purkinje))

    if max_w > min_w:
        for g in range(num_granule):
            for p in range(num_purkinje):
                triangle_widths[g, p] = (weights[(g, p)] - min_w) / (max_w - min_w) * (max_w - min_w)
    else:
        triangle_widths.fill(0.5)

    # Update existing Purkinje triangles
    for i, purkinje_group in enumerate(purkinje_drawing):
        for j, triangle in enumerate(purkinje_group[:-2]):  # Ignore axon & soma
            if isinstance(triangle, patches.Polygon):  
                x, y = triangle.xy[2]  # Fixed bottom point
                top_width = triangle_widths[j, i]
                # Update triangle shape
                new_xy = [
                    (x + (j-1) * (width if j > 0 else width/2) - top_width / 2, y + (j+1) * width),  # Left top
                    (x + (j-1) * (width if j > 0 else width/2) + top_width / 2, y + (j+1) * width),  # Right top
                    (x, y)  # Bottom center
                ]
                triangle.set_xy(new_xy)  # Update vertices
    
    plt.draw()
    #plt.pause(1)

def show_network_graph():
    """Shows the biologically inspired network graph of cerebellar cells and connections"""
    global ax_network, purkinje_drawing
    global height, width, granule_x, purkinje_x, olive_x, basket_x, dcn_x, granule_y, purkinje_y, olive_y, basket_y, dcn_y

    purkinje_drawing = []

    height = 1
    width = 0.2

    granule_x = np.linspace(0.15, 0.5, num_granule)
    purkinje_x = np.linspace(0.9, 4.8, num_purkinje)
    olive_x = purkinje_x[-1] + 0.4
    basket_x = purkinje_x[-1] + 0.4
    dcn_x = calculate_dcn_x_positions(purkinje_x, num_dcn)
    
    granule_y = -height*3/4 
    purkinje_y = 0
    olive_y = np.linspace(-0.9*height, -0.5*height, num_inferior_olive)
    basket_y = purkinje_y 
    dcn_y = -1.3*height

    # Draw Inferior Olive cell
    for inferior_olive in inferior_olive_cells:
        first_purkinje = next((purkinje for purkinje in purkinje_cells if cf_ncs[inferior_olive.gid][purkinje.gid] is not None), 0) # find first connected PC, default PC0
        ax_network.plot([purkinje_x[first_purkinje.gid]+0.15, olive_x], [olive_y[inferior_olive.gid], olive_y[inferior_olive.gid]], color=color_inferior_olive, lw=2, label="Climbing Fiber")
        ax_network.scatter(olive_x, olive_y[inferior_olive.gid], s=100, color=color_inferior_olive, label="Inferior Olive")

    # Draw Basket cell connecting to Purkinje cell somas
    ax_network.plot([purkinje_x[0], basket_x], [basket_y, basket_y], color=color_basket, lw=2)
    ax_network.scatter(basket_x, basket_y, s=100, color=color_basket, label="Basket Cell")
    
    # Draw Granule cells, vertical axons, and parallel fibers
    for granule in granule_cells:
        if state == 0 or granule.gid == state-1:
            transparency = 1
        else:
            transparency = 0.5
        ax_network.scatter(granule_x[granule.gid], granule_y, color=color_granule, s=100, label="Granule Cell", alpha=transparency) 
        draw_granule_to_parallel(ax_network, granule_x[granule.gid], granule_y, purkinje_y + (granule.gid+1) * width, transparency)

    # Draw Purkinje cells
    for purkinje in purkinje_cells:
        for inferior_olive in inferior_olive_cells: 
            if cf_ncs[inferior_olive.gid][purkinje.gid] is not None:  # Check if CF synapse exists
                draw_climbing_fiber(ax_network, purkinje_x[purkinje.gid], olive_y[inferior_olive.gid], purkinje_y, width=width)  # Climbing fibers
        drawing = draw_purkinje(ax_network, purkinje_x[purkinje.gid], purkinje_y, width=width, height=height, color=colors_purkinje[purkinje.gid])
        purkinje_drawing.append(drawing)

    # Draw Deep Cerebellar Nuclei
    for i in range(num_dcn):
        ax_network.scatter(dcn_x[i], dcn_y, color=color_dcn, s=100, label="Deep Cerebellar Nuclei") 
        ax_network.plot([dcn_x[i], dcn_x[i]], [purkinje_y-height, dcn_y], color=color_dcn, lw=2)

        segment_length = num_purkinje // num_dcn  # Length of each segment
    
        for i in range(num_dcn):
            # Determine the segment range
            start_idx = i * segment_length
            if i == num_dcn - 1:
                end_idx = len(purkinje_x)
            else:
                end_idx = (i + 1) * segment_length
            
            # Plot horizontal lines connecting Purkinje cells in each segment
            ax_network.plot([purkinje_x[start_idx], purkinje_x[end_idx-1]], [purkinje_y-height, purkinje_y-height], color=color_dcn, lw=2, label=f'Segment {i+1}')

    # Labels
    ax_network.text(purkinje_x[0] - 0.25, purkinje_y - 0.3, "Purkinje\nCells (PC)", fontsize=10, color=colors_purkinje[0])
    for purkinje in purkinje_cells:
        ax_network.text(purkinje_x[purkinje.gid] + 0.01, purkinje_y - 0.2, f"PC{purkinje.gid+1}", fontsize=10, color=colors_purkinje[purkinje.gid])
    ax_network.text(granule_x[0] - 0.05, granule_y - 0.4, "Granule Cells (GC)", fontsize=10, color=color_granule)
    for granule in granule_cells:
        ax_network.text(granule_x[granule.gid] - 0.05, granule_y - 0.2, f"GC{granule.gid+1}", fontsize=10, color=color_granule)
    ax_network.text(granule_x[1], purkinje_y + (num_granule) * width + 0.1, "Parallel Fibers (PF)", fontsize=10, color=color_granule)
    ax_network.text(olive_x + 0.1, olive_y[0] - 0.2, "Inferior Olives (IO)", fontsize=10, color=color_inferior_olive)
    for inferior_olive in inferior_olive_cells:
        ax_network.text(olive_x + 0.1, olive_y[inferior_olive.gid] - 0.04, f"IO{inferior_olive.gid+1}", fontsize=10, color=color_inferior_olive)
    ax_network.text(purkinje_x[-1] + 0.2, olive_y[-1] + abs(purkinje_y - olive_y[-1]) / 2, "Climbing Fibers (CF)", fontsize=10, color=color_inferior_olive)
    ax_network.text(basket_x + 0.1, basket_y - 0.01, "Basket Cell (BC)", fontsize=10, color=color_basket)
    ax_network.text(dcn_x[0] - 0.4, dcn_y, f"Deep Cerebellar\nNuclei (DCN)", fontsize=10, color=color_dcn)
    for i in range(num_dcn):
        ax_network.text(dcn_x[i] + 0.05, dcn_y, f"DCN{i+1}", fontsize=10, color=color_dcn)

    ax_network.set_xlim([0.0,6.0])

    update_weights_in_network()

    plt.draw()
    #plt.pause(1)

def update_animation(frame, spikes, spike_type=0, p_ids=[], g_or_i_ids=[]):
    """Spike animation for simple spikes and complex spikes"""
    # Animation parameters
    total_steps = 60  # Total frames in animation
    
    for idx, spike in enumerate(spikes):
        # For each spike, get corresponding p_id and g_or_i_id
        p_id = p_ids[idx] if idx < len(p_ids) else 0  # Default to 0 if not enough IDs provided
        g_or_i_id = g_or_i_ids[idx] if idx < len(g_or_i_ids) else 0  # Default to 0 if not enough IDs
        
        # Determine start and end positions based on spike type
        if spike_type == 1:  # Complex Spike from Inferior Olive
            start_x, start_y = olive_x, olive_y[g_or_i_id]
            junction1_x, junction1_y = purkinje_x[p_id] + 0.15, start_y
            junction2_x, junction2_y = junction1_x, purkinje_y - 0.15
        else:  # Simple Spike from Granule Cell
            start_x, start_y = granule_x[g_or_i_id], granule_y
            junction1_x, junction1_y = start_x, purkinje_y + (g_or_i_id + 1) * width
            junction2_x, junction2_y = purkinje_x[p_id] + (g_or_i_id - 1) * (width if g_or_i_id > 0 else width/2), junction1_y
        end_x, end_y = purkinje_x[p_id], purkinje_y

        # Compute segment lengths
        d1 = np.hypot(junction1_x - start_x, junction1_y - start_y)
        d2 = np.hypot(junction2_x - junction1_x, junction2_y - junction1_y)
        d3 = np.hypot(end_x - junction2_x, end_y - junction2_y)
        D_total = d1 + d2 + d3

        # Allocate frames proportionally
        segment_steps1 = round(total_steps * (d1 / D_total))
        segment_steps2 = round(total_steps * (d2 / D_total))
        segment_steps3 = total_steps - (segment_steps1 + segment_steps2)

        # Determine current segment and compute clamped t
        if frame < segment_steps1:  # Move to Junction 1
            t = ((frame + 1) / segment_steps1)
            t = min(max(t, 0), 1)  # Ensure t is in [0,1]
            x_new = start_x + t * (junction1_x - start_x)
            y_new = start_y + t * (junction1_y - start_y)

        elif frame < segment_steps1 + segment_steps2:  # Move to Junction 2
            t = ((frame + 1 - segment_steps1) / segment_steps2)
            t = min(max(t, 0), 1)  # Ensure t is in [0,1]
            x_new = junction1_x + t * (junction2_x - junction1_x)
            y_new = junction1_y + t * (junction2_y - junction1_y)

        else:  # Move to Purkinje Cell
            t = ((frame + 1 - segment_steps1 - segment_steps2) / segment_steps3)
            t = min(max(t, 0), 1)  # Ensure t is in [0,1]
            x_new = junction2_x + t * (end_x - junction2_x)
            y_new = junction2_y + t * (end_y - junction2_y)

        # Update the spike's data
        spike.set_data([x_new], [y_new])

        # If last frame, hide the spike
        if frame == total_steps - 1:
            spike.set_alpha(0)

    return spikes  # Return all updated spikes

def reset(event=None, reset_all=True):
    """Resets the program"""
    global t, iter

    h.finitialize(-65)
    h.frecord_init()
    h.stdinit()
    h.t = 0

    if t is not None:
        t.resize(0)  # Clear old values
    else:
        t= h.Vector()

    init_variables(reset_all)
    create_connections()
    recording()
    
    run_simulation()
    iter += 1
    update_spike_and_weight_plot()

    buttons["run_button"].color = color_run
    buttons["run_button"].hovercolor = color_run_hover
    buttons["run_button"].ax.figure.canvas.draw_idle()  # Force redraw
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    #plt.pause(1)

def update_weights(pre_gid, post_gid, pre_t, post_t):
    """STDP Update Function"""
    delta_t = post_t - pre_t # time between presynaptic spike and postsynaptic spike
    dw = 0
    plasticity = None
    if delta_t > 0: 
        dw = A_plus * np.exp(-delta_t / tau_plus)
        #if dw > 0.001: print(f"[{iter}] LTP: dw= {dw:.3f} GC{pre_gid+1} <-> PC{post_gid+1}")
        if dw > 0.001: 
            plasticity = "LTP"
    elif delta_t < 0:
        dw = -A_minus * np.exp(delta_t / tau_minus)
        #if dw < -0.001: print(f"[{iter}] LTD: dw={dw:.3f} GC{pre_gid+1} <-> PC{post_gid+1}")
        if dw < -0.001: 
            plasticity = "LTD"        

    old_weight = weights[(pre_gid, post_gid)]
    new_weight = old_weight + dw
    
    if plasticity:
        print(f"[{iter}] {plasticity}: weight change {old_weight:.4f}{'+' if dw>0 else ''}{dw:.4f}={new_weight:.4f} at synapses GC{pre_gid+1} <-> PC{post_gid+1}")
    
    # Update weights
    weights[(pre_gid, post_gid)] = new_weight
    pf_ncs[pre_gid][post_gid].weight[0] = new_weight
    
def recording():
    """Records Spiking Activity and Voltages"""
    global t, granule_spikes, purkinje_spikes, inferiorOlive_spikes, basket_spikes, dcn_spikes, v_granule, v_purkinje, v_inferiorOlive, v_basket, v_dcn

    t.record(h._ref_t)  # Reattach to NEURON's time

    granule_spikes =       {i: h.Vector() for i in range(num_granule)}
    purkinje_spikes =      {i: h.Vector() for i in range(num_purkinje)}
    inferiorOlive_spikes = {i: h.Vector() for i in range(num_inferior_olive)}
    basket_spikes =        {i: h.Vector() for i in range(num_basket)}
    dcn_spikes =           {i: h.Vector() for i in range(num_dcn)}

    v_granule =       {i: h.Vector().record(granule_cells[i].soma(0.5)._ref_v) for i in range(num_granule)}
    v_purkinje =      {i: h.Vector().record(purkinje_cells[i].soma(0.5)._ref_v) for i in range(num_purkinje)}
    v_inferiorOlive = {i: h.Vector().record(inferior_olive_cells[i].soma(0.5)._ref_v) for i in range(num_inferior_olive)}
    v_basket =        {i: h.Vector().record(basket_cells[i].soma(0.5)._ref_v) for i in range(num_basket)}
    v_dcn =           {i: h.Vector().record(deep_cerebellar[i].soma(0.5)._ref_v) for i in range(num_dcn)}

    for granule in granule_cells:
        nc = h.NetCon(granule.soma(0.5)._ref_v, None, sec=granule.soma)
        nc.record(granule_spikes[granule.gid])

    for purkinje in purkinje_cells:
        nc = h.NetCon(purkinje.soma(0.5)._ref_v, None, sec=purkinje.soma)
        nc.record(purkinje_spikes[purkinje.gid])

    for inferior_olive in inferior_olive_cells:
        nc = h.NetCon(inferior_olive.soma(0.5)._ref_v, None, sec=inferior_olive.soma)
        nc.record(inferiorOlive_spikes[inferior_olive.gid])

    for basket in basket_cells:
        nc = h.NetCon(basket.soma(0.5)._ref_v, None, sec=basket.soma)
        nc.record(basket_spikes[basket.gid])
    
    for dcn in deep_cerebellar:
        nc = h.NetCon(dcn.soma(0.5)._ref_v, None, sec=dcn.soma)
        nc.record(dcn_spikes[dcn.gid])
    
    h.finitialize(-65) # Set all membrane potentials to -65mV
    
def run_simulation(error=False):
    """Runs the simulation for one iteration and tracks the weights"""
    global granule_spikes, purkinje_spikes, inferiorOlive_spikes, basket_spikes, dcn_spikes
    global iter, spike_times, processed_GC_spikes, processed_pairs, frequency, weights_over_time
    global t_np, v_granule_np, v_purkinje_np, v_inferiorOlive_np, v_basket_np, v_dcn_np

    try:
        buttons["run_button"].color = "0.85"
        buttons["run_button"].hovercolor = "0.975"
        buttons["run_button"].ax.figure.canvas.draw_idle()  # Force redraw
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        #plt.pause(1)
    except Exception: None
    
    if error:
        time_span = 1/8 * 1/frequency*1000
        stop_time = h.t + time_span
    else:
        stop_time = 1/frequency*1000 * (iter + 1) # run 20 ms per iteration

    # Continuously run the simulation and update weights during the simulation
    while h.t < stop_time: 
        h.continuerun(h.t + 1)  # Incrementally run the simulation

        if not error:
            # --- Trigger Purkinje Cell Spike ---
            for g_id in range(num_granule):
                for pre_t in granule_spikes[g_id]:
                    if pre_t > stop_time - 1/frequency*1000: # timespan between last GC stimulation
                        if (pre_t) not in processed_GC_spikes[(g_id)]:
                            processed_GC_spikes[g_id].add((pre_t))

            # --- Apply STDP ---
            for g_id in range(num_granule):
                for p_id in range(num_purkinje):
                    for pre_t in granule_spikes[g_id]:
                        for post_t in purkinje_spikes[p_id]:
                            if (pre_t, post_t) not in processed_pairs[(g_id, p_id)]:
                                #print(f"update weights for GC{g_id+1} <-> PC{p_id+1} pre_t {pre_t:.2f} post_t {post_t:.2f}")
                                update_weights(g_id, p_id, pre_t, post_t)
                                processed_pairs[(g_id, p_id)].add((pre_t, post_t))

                    # Track the weight at the current time step
                    while len(weights_over_time[(g_id, p_id)]) < len(t):
                        weights_over_time[(g_id, p_id)].append(weights[(g_id, p_id)])
        else:
            for g_id in range(num_granule):
                for p_id in range(num_purkinje):
                    # Track the weight at the current time step
                    while len(weights_over_time[(g_id, p_id)]) < len(t):
                        weights_over_time[(g_id, p_id)].append(weights[(g_id, p_id)])
 
    # --- Convert Spike Data ---
    spike_times =      {f"GC{i+1}":  list(granule_spikes[i])       for i in range(num_granule)}
    spike_times.update({f"PC{i+1}":  list(purkinje_spikes[i])      for i in range(num_purkinje)})
    spike_times.update({f"IO{i+1}":  list(inferiorOlive_spikes[i]) for i in range(num_inferior_olive)})
    spike_times.update({f"BC{i+1}":  list(basket_spikes[i])        for i in range(num_basket)})
    spike_times.update({f"DCN{i+1}": list(dcn_spikes[i])           for i in range(num_dcn)})
    
    # --- Convert Voltage Data and Weights ---
    t_np = np.array(t)
    v_granule_np =       np.array([vec.to_python() for vec in v_granule.values()])
    v_purkinje_np =      np.array([vec.to_python() for vec in v_purkinje.values()])
    v_inferiorOlive_np = np.array([vec.to_python() for vec in v_inferiorOlive.values()])
    v_basket_np =        np.array([vec.to_python() for vec in v_basket.values()])
    v_dcn_np =           np.array([vec.to_python() for vec in v_dcn.values()])

def update_spike_and_weight_plot():
    """Updates the plots for the spikes and weights (one weight plot per DCN) and the buttons"""
    global t_np, v_granule_np, v_purkinje_np, v_inferiorOlive_np, v_basket_np, v_dcn_np
    global buttons, fig, gs, ax_network, ax_plots, gs_buttons

    if gs == None or ax_network == None or ax_plots == None:
        try:
            gs = GridSpec(2, 1 + num_dcn + 1, figure=fig, width_ratios=(1+num_purkinje // (num_purkinje//num_dcn))*[1] + [0.35 * num_dcn], height_ratios=[0.1, 1])
        except AttributeError: None
        ax_network = fig.add_subplot(gs[0, :], label="ax_network")
        ax_network.axis("off")
        ax_plots = [None for _ in range(1 + num_dcn)]
        for col in range(1 + num_purkinje // (num_purkinje//num_dcn)):
            ax_plots[col] = fig.add_subplot(gs[1, col], label=f"ax_plots[{col}]")
        gs_buttons = gs[1, -1].subgridspec(9, 1, height_ratios=(1,2,2,2.5,2.5,1,1,1,1))
    else:
        # Clear previous plots
        for col in range(1 + num_dcn):
            ax_plots[col].cla()

    # Share axis
    for col in range(1 + num_dcn): 
        if col > 0:
            ax_plots[col].sharex(ax_plots[0])  # Share x-axis with first column
        if col > 1:
            ax_plots[col].sharey(ax_plots[1])  # Share y-axis with second column

    for granule in granule_cells:

        if granule.gid == (state-1) % num_granule:
            ax1 = ax_plots[0]
            ax1.set_title(f"Spiking Activity")
            #ax1.plot(t_np, v_granule_np[granule.gid], label=f"GC{granule.gid+1 if state > 0 else ''}", color=color_granule, linestyle="dashed")
            ax1.set_xlabel("Time (ms)") 
            ax1.set_ylabel("Membrane Voltage (mV)")

            for i, dcn in enumerate (deep_cerebellar):
                ax1.plot(t_np, v_dcn_np[dcn.gid], label=f"DCN{dcn.gid+1}", color=color_dcn)

                for purkinje in purkinje_cells:
                    # --- Spiking Plot for GC and its connected PCs ---
                    text_blocked = ""
                    try:
                        if v_purkinje_np[purkinje.gid][-1] > -56:
                            text_blocked = " blocked"
                    except IndexError: None
                    ax1.plot(t_np, v_purkinje_np[purkinje.gid], label=f"PC{purkinje.gid+1}{text_blocked}", color=colors_purkinje[purkinje.gid])
                    
                    # --- Weight Plot for GC to all connected PCs ---
                    ax2 = ax_plots[1 + i]
                    ax2.set_title(f"Synaptic Weights")
                    ax2.set_xlabel("Time (ms)")
                    ax2.set_ylabel("Synaptic Weight")
                    if len(weights_over_time[(granule.gid, purkinje.gid)]) > 0:
                        if purkinje.gid >= i * num_purkinje // num_dcn and purkinje.gid < (i + 1) * num_purkinje // num_dcn:
                            ax2.plot(t_np, weights_over_time[(granule.gid, purkinje.gid)], label=f"PC{purkinje.gid+1}{text_blocked}", color=colors_purkinje[purkinje.gid])

            for inferior_olive in inferior_olive_cells:
                ax1.plot(t_np, v_inferiorOlive_np[inferior_olive.gid], label=f"IO{inferior_olive.gid+1 if len(inferior_olive_cells) > 1 else ''}", color=color_inferior_olive, linestyle="dashed")
            #for basket in basket_cells:
            #    ax1.plot(t_np, v_basket_np[basket.gid], label=f"BC{basket.gid+1 if len(basket_cells) > 1 else ''}", color=color_basket, linestyle="dashed")

    # Collect all legend handles and labels for the first column
    handles_first_row = []
    labels_first_row = []
    handles, labels = ax_plots[0].get_legend_handles_labels()
    for l, h in zip(labels, handles):
        if l not in labels_first_row:  # Avoid duplicates
            # Exclude Purkinje cells from the first legend
            #if "PC" not in l:  # Only add non-Purkinje labels
            labels_first_row.append(l)
            handles_first_row.append(h)
    labels_first_row, handles_first_row = zip(*sorted(zip(labels_first_row, handles_first_row), key=lambda x: x[0]))
    
    # Create a single legend for the first column
    spacing = 0.1
    height = 0.5
    ncol_first_legend = 1
    ax_plots[0].legend(handles_first_row, labels_first_row, loc="upper left", bbox_to_anchor=(0, 1), ncol=ncol_first_legend, labelspacing=spacing, handleheight=height)
    legend_height_first_row = 1/20 * (len(labels_first_row) * (height + spacing) - spacing ) / ncol_first_legend
    while legend_height_first_row > ax_plots[0].get_position().height and ncol_first_legend < len(labels_first_row):
        ncol_first_legend += 1  # Increase the number of columns (max is the number of labels)
        ax_plots[0].legend(handles_first_row, labels_first_row, loc="upper left", bbox_to_anchor=(0, 1), ncol=ncol_first_legend, labelspacing=spacing, handleheight=height)
        legend_height_first_row = 1/20 * (len(labels_first_row) * (height + spacing) - spacing ) / ncol_first_legend
    
    # Collect all legend handles and labels for the other columns
    for col in range(1, num_inferior_olive + 1):
        handles_second_row = []
        labels_second_row = []
        handles, labels = ax_plots[col].get_legend_handles_labels()
        for l, h in zip(labels, handles):
            if l not in labels_second_row:
                labels_second_row.append(l)
                handles_second_row.append(h)
        
        # Create a single legend for the other columns
        ncol_second_legend = 1
        ax_plots[col].legend(handles_second_row, labels_second_row, loc="upper left", bbox_to_anchor=(0, 1), ncol=ncol_second_legend, labelspacing=spacing)
        legend_height_second_row = 1/20 * (len(labels_second_row) * (height + spacing) - spacing) / ncol_second_legend
        while legend_height_second_row > ax_plots[col].get_position().height and ncol_second_legend < len(labels_second_row):
            ncol_second_legend += 1  # Increase the number of columns (max is the number of labels)
            ax_plots[col].legend(handles_second_row, labels_second_row, loc="upper left", bbox_to_anchor=(0, 1), ncol=ncol_second_legend, labelspacing=spacing, handleheight=height)
            legend_height_second_row = 1/20 * (len(labels_second_row) * (height + spacing) - spacing) / ncol_second_legend

    # --- Buttons ---

    # Reset Button
    if "reset_button" not in buttons:
        reset_ax = fig.add_subplot(gs_buttons[8], label="reset_button")
        buttons["reset_button"] = Button(reset_ax, "Reset")
        buttons["reset_button"].on_clicked(reset)

    # Network Button
    if "network_button" not in buttons:
        network_ax = fig.add_subplot(gs_buttons[7], label="network_button")
        buttons["network_button"] = Button(network_ax, "Show network")
        buttons["network_button"].on_clicked(toggle_network_graph)

    # Error Button
    if "error_button" not in buttons:
        error_ax = fig.add_subplot(gs_buttons[6], label="error_button")
        buttons["error_button"] = Button(error_ax, "Error")
        buttons["error_button"].on_clicked(lambda event: update_inferior_olive_stimulation_and_plots(event, cell_nr=0))

    # Run Button
    if "run_button" not in buttons:
        run_ax = fig.add_subplot(gs_buttons[5], label="run_button")
        buttons["run_button"] = Button(run_ax, f"Run iteration {iter}")
        buttons["run_button"].on_clicked(update_granule_stimulation_and_plots)

    # State Button
    if "state_button" not in buttons:
        state_ax = fig.add_subplot(gs_buttons[4], label="state_button")
        buttons["state_button"] = RadioButtons(state_ax, list(state_dict.values()), active=state-1)
        buttons["state_button"].on_clicked(update_state)

    plt.draw()
    #plt.pause(1)

def main(reset=True):
    global t, granule_spikes, purkinje_spikes, inferiorOlive_spikes, basket_spikes, dcn_spikes
    global t_np, v_granule_np, v_purkinje_np, v_inferiorOlive_np, v_basket_np, v_dcn_np, iter

    init_variables(reset)
    create_connections()
    recording()

    #h.topology() # prints topology of network
    
    run_simulation()
    iter += 1
    update_spike_and_weight_plot()

    buttons["run_button"].color = color_run
    buttons["run_button"].hovercolor = color_run_hover
    buttons["run_button"].ax.figure.canvas.draw_idle()  # Force redraw
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    #plt.pause(1)

    try:
        while True:
            plt.pause(10)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Simulation stopped by user.")
        plt.close()

    return

main()
    






