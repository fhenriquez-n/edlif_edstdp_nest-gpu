"""
edlif_edstdp_example.py

The simplest simulation that uses the two things this package adds to NEST GPU:
the EdLIF neuron (edlif_psc_exp_nestml) and the ED-STDP synapse (edstdp).

A network of 500 neurons (400 excitatory, 100 inhibitory), every neuron connected to
every other neuron, simulated for one second. The excitatory-to-excitatory synapses
learn with ED-STDP, all the others are static. At the end it prints:
    - the firing rate of each population,
    - the ATP of the neurons (it starts at 100 and goes down with every spike and every
      synaptic event),
    - the weights of the plastic synapses (they all start at 2 pA; if ED-STDP works they
      spread out),
and it draws them in a figure (firing rate over time, ATP, weights). The last lines say whether
each of the three happened.

The values of the neuron and the synapse are those of the lab's default configuration.
The network is simpler than the lab's (NEST GPU's own all_to_all rule also connects each
neuron to itself, and the random numbers are different), so the numbers are similar to
the lab's results but not the same.

Usage, with the NEST GPU of this package installed (see the README):
    source work/install/bin/nestgpu_vars.sh
    python3 examples/edlif_edstdp_example.py
It needs an NVIDIA GPU, and numpy. matplotlib is only for the figure: without it the
example runs and prints everything, but draws nothing.
"""

import os
import sys

import nestgpu as ngpu
import numpy as np

# matplotlib is optional. "Agg" draws into a file: there is no screen on a cluster.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


# ---------------------------------------------------------------------------
# simulation values
# ---------------------------------------------------------------------------
n_exc = 400                   # excitatory neurons: numbers 0 to n_exc - 1
n_inh = 100                   # inhibitory neurons: the numbers after them
n_total = n_exc + n_inh

delay = 1.5                   # ms a spike takes to reach its target, the same for all synapses
weight_exc = 2.0              # pA, strength of an excitatory synapse (also where ED-STDP starts)
weight_inh = 2.0              # pA. Positive: EdLIF has a separate input for inhibition and subtracts it itself

I_e_mean = 166.0              # pA, constant input current of each neuron: normal distribution
I_e_std = 15.0                #     around this mean, so every neuron gets a slightly different one
v0_min = 0.0                  # mV above rest, starting voltage of each neuron: random
v0_max = 13.0                 #     between these two values

t_sim = 1000.0                # ms to simulate (the start, where the network settles, is not left out)
dt = 0.1                      # ms, simulation time step
spike_buffer = 2000           # most spikes NEST GPU keeps for one neuron (a neuron cannot spike more than
                              #     125 times per second: its refractory period is 8 ms)
seed = 0                      # same seed, same starting voltages and currents


# ---------------------------------------------------------------------------
# The parameters of the EdLIF neuron and of the ED-STDP synapse
# ---------------------------------------------------------------------------
neuron_parameters = {
    "E_L": -70.0,             # resting potential (mV)
    "V_th": 15.0,             # spike threshold (mV ABOVE rest, not an absolute voltage)
    "V_reset": 0.0,           # voltage after a spike (mV above rest)
    "C_m": 250.0,             # membrane capacitance (pF)
    "tau_m": 20.0,            # membrane time constant (ms)
    "t_ref": 8.0,             # refractory period (ms): after a spike the neuron cannot fire again
    "tau_syn_ex": 6.0,        # decay of the excitatory synaptic current (ms)
    "tau_syn_in": 6.0,        # decay of the inhibitory synaptic current (ms)
    "gamma": 20.0,            # how strongly a low ATP raises the voltage after a spike
    "E_ap": 2.0,              # ATP cost of one spike (percent of ATP)
    "E_syn_ex": 0.5,          # ATP cost of one excitatory synaptic event (percent), scaled by weight / w_max
    "E_syn_in": 0.5,          # ATP cost of one inhibitory synaptic event (percent), scaled by weight / w_min
    "w_max": 100.0,           # reference excitatory weight for the ATP cost (pA)
    "w_min": 100.0,           # reference inhibitory weight for the ATP cost (pA)
    "tau_ap": 60.0,           # time over which the ATP cost of a spike is paid (ms)
    "tau_syn_atp_ex": 60.0,   # same, for an excitatory synaptic event (ms)
    "tau_syn_atp_in": 60.0,   # same, for an inhibitory synaptic event (ms)
    "K_ATP": 1.0,             # speed of the ATP production towards ATP_h
    "ATP_h": 100.0,           # the ATP level the neuron tries to keep (percent)
    "E_rp": 5.0,              # ATP the resting potential uses (percent per ms)
    "E_hk": 5.0,              # ATP the housekeeping uses (percent per ms)
    "ATP_basal": 10.0,        # E_rp + E_hk
    "fix_atp": 0.0,           # 1 holds the ATP fixed, 0 lets it change
    "ATP": 100.0,             # ATP at the start (percent)
}

synapse_parameters = {
    "lambda": 0.01,           # learning rate
    "alpha": 0.5,             # weakening relative to strengthening
    "eta": 50.0,              # how strongly a low ATP blocks the strengthening
    "Wmax": 100.0,            # weight ceiling (pA). The floor is always 0
    "mu_plus": 0.0,           # weight dependence of the strengthening: 0 = additive
    "mu_minus": 0.0,          # weight dependence of the weakening
    "tau_plus": 20.0,         # time window of the strengthening (ms)
    "tau_minus": 20.0,        # time window of the weakening (ms)
    "ATP_h": 100.0,           # the ATP the synapse compares the postsynaptic ATP with
}


# ---------------------------------------------------------------------------
# NEST GPU settings
# ---------------------------------------------------------------------------
# Plastic synapses need the spike buffer algorithm 0. With NEST GPU's default one they
# are silently ignored. This must come before the connections are made.
ngpu.SetKernelStatus({"spike_buffer_algo": 0})
ngpu.SetTimeResolution(dt)


# ---------------------------------------------------------------------------
# The neurons
# ---------------------------------------------------------------------------
rng = np.random.default_rng(seed)
input_current = rng.normal(I_e_mean, I_e_std, n_total)
initial_voltage = rng.uniform(v0_min, v0_max, n_total)

# {"array": list} gives each neuron its own value, instead of the same for all of them
neuron_parameters["I_e"] = {"array": input_current.tolist()}
neuron_parameters["V_abs"] = {"array": initial_voltage.tolist()}     # V_abs: voltage above rest

# All the neurons are one group. NEST GPU numbers them 0 to n_total - 1.
neurons = ngpu.Create("edlif_psc_exp_nestml", n_total, status_dict=neuron_parameters)
exc = neurons[0:n_exc]                    # the excitatory neurons
inh = neurons[n_exc:n_total]              # the inhibitory neurons

# Store the time of every spike of every neuron
ngpu.ActivateRecSpikeTimes(neurons, spike_buffer)


# ---------------------------------------------------------------------------
# The synapses
# ---------------------------------------------------------------------------
# A synapse group is a learning rule with its parameters.
group = ngpu.CreateSynGroup("edstdp", synapse_parameters)

# "all_to_all": every neuron of the first group is connected to every neuron of the second.
# The EdLIF neuron has two inputs: receptor 0 is the excitatory one, receptor 1 the inhibitory one.
all_to_all = {"rule": "all_to_all"}

# excitatory -> excitatory: the only plastic synapses
ngpu.Connect(exc, exc, all_to_all,
             {"weight": weight_exc, "delay": delay, "receptor": 0, "synapse_group": group})

# excitatory -> inhibitory: static
ngpu.Connect(exc, inh, all_to_all, {"weight": weight_exc, "delay": delay, "receptor": 0})

# inhibitory -> everybody: static, through the inhibitory input
ngpu.Connect(inh, neurons, all_to_all, {"weight": weight_inh, "delay": delay, "receptor": 1})


# ---------------------------------------------------------------------------
# Simulate
# ---------------------------------------------------------------------------
# Calibrate prepares the network on the GPU. It comes after all the neurons and connections
# are made, and before Simulate.
ngpu.Calibrate()
ngpu.Simulate(t_sim)


# ---------------------------------------------------------------------------
# Read the results from the GPU
# ---------------------------------------------------------------------------
# One list of spike times (ms) for each neuron
spike_times = ngpu.GetRecSpikeTimes(neurons)
spikes_per_neuron = np.array([len(neuron_times) for neuron_times in spike_times])

# The same spikes as two arrays, for the figure: the neuron and the time of each spike
senders = np.concatenate([np.full(len(neuron_times), neuron) for neuron, neuron_times in enumerate(spike_times)])
times = np.concatenate([np.asarray(neuron_times, dtype=float) for neuron_times in spike_times])

# The ATP of every neuron at the end of the simulation
atp = np.array(ngpu.GetStatus(neurons, "ATP"), dtype=float).ravel()

# The weights of the plastic synapses: only the connections of the group, from exc to exc
connections = ngpu.GetConnections(exc, exc, syn_group=group.i_syn_group)
weights = np.array(ngpu.GetStatus(connections, "weight"), dtype=float)


# ---------------------------------------------------------------------------
# Print the results
# ---------------------------------------------------------------------------
print()
print(f"Results after {t_sim:.0f} ms")
print(f"  spikes: {spikes_per_neuron.sum()}   most spikes of one neuron: {spikes_per_neuron.max()} "
      f"(buffer {spike_buffer})")
if spikes_per_neuron.max() >= spike_buffer:
    print("  WARNING: the buffer is full, spikes were lost. Raise spike_buffer.")

rate_exc = spikes_per_neuron[:n_exc].sum() / (n_exc * t_sim / 1000.0)
rate_inh = spikes_per_neuron[n_exc:].sum() / (n_inh * t_sim / 1000.0)
atp_exc = atp[:n_exc]
atp_inh = atp[n_exc:]
print(f"  E: rate {rate_exc:.2f} Hz   final ATP mean {atp_exc.mean():.3f} min {atp_exc.min():.3f} max {atp_exc.max():.3f}")
print(f"  I: rate {rate_inh:.2f} Hz   final ATP mean {atp_inh.mean():.3f} min {atp_inh.min():.3f} max {atp_inh.max():.3f}")

if len(weights) > 0:
    print(f"  plastic synapses ({len(weights)}, E -> E): started at {weight_exc:g} pA, now "
          f"mean {weights.mean():.2f}  std {weights.std():.2f}  min {weights.min():.2f}  max {weights.max():.2f}")
else:
    print("  plastic synapses: none found")


# ---------------------------------------------------------------------------
# Did it work?
# ---------------------------------------------------------------------------
# The neurons spiked
spikes_ok = spikes_per_neuron.sum() > 0

# The ATP is a number (a broken build gives NaN) and it went down from 100 with the spikes
atp_ok = bool(np.all(np.isfinite(atp))) and atp.min() < 100.0

# The synapses learned: the weights started all equal, so a spread means ED-STDP changed them
weights_ok = len(weights) > 0 and weights.std() > 0.0

print()
print("Did it work?")
print(f"  the neurons spiked ............... {'yes' if spikes_ok else 'NO'}")
print(f"  the ATP is valid and below 100 ... {'yes' if atp_ok else 'NO'}")
print(f"  ED-STDP changed the weights ...... {'yes' if weights_ok else 'NO'}")


# ---------------------------------------------------------------------------
# The figure
# ---------------------------------------------------------------------------
if plt is None:
    print("\nmatplotlib is not installed: no figure (pip install matplotlib).")
elif not atp_ok:
    print("\nNo figure: the ATP is not valid.")
else:
    # Colors: blue = excitatory neurons, orange = inhibitory neurons. Text and lines are gray.
    surface, ink, ink_soft, grid_color = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
    blue, orange = "#2a78d6", "#eb6834"

    figure, (axis_spikes, axis_atp, axis_weights) = plt.subplots(1, 3, figsize=(15, 4.2), facecolor=surface)

    # Left: the firing rate of each population over time. The spikes are counted in bins of
    # 10 ms, and the count is turned into spikes per neuron per second.
    bin_ms = 10.0
    bin_edges = np.arange(0.0, t_sim + bin_ms, bin_ms)
    bin_centers = bin_edges[:-1] + bin_ms / 2.0
    is_exc = senders < n_exc
    exc_counts, _ = np.histogram(times[is_exc], bins=bin_edges)
    inh_counts, _ = np.histogram(times[~is_exc], bins=bin_edges)
    axis_spikes.plot(bin_centers, exc_counts / (n_exc * bin_ms / 1000.0), color=blue, linewidth=2, label="excitatory")
    axis_spikes.plot(bin_centers, inh_counts / (n_inh * bin_ms / 1000.0), color=orange, linewidth=2, label="inhibitory")
    axis_spikes.set_title("Firing rate (bins of 10 ms)", color=ink, loc="left")
    axis_spikes.set_xlabel("time (ms)")
    axis_spikes.set_ylabel("spikes per neuron per second (Hz)")
    axis_spikes.set_ylim(bottom=0)
    axis_spikes.legend(frameon=False, labelcolor=ink_soft)

    # Middle: the ATP of the neurons at the end (the bars of the two populations are stacked)
    atp_bins = np.linspace(atp.min(), atp.max(), 31)
    axis_atp.hist([atp_exc, atp_inh], bins=atp_bins, stacked=True, color=[blue, orange],
                  edgecolor=surface, linewidth=0.8, label=["excitatory", "inhibitory"])
    axis_atp.set_title("ATP at the end", color=ink, loc="left")
    axis_atp.set_xlabel("ATP (% of the homeostatic level)")
    axis_atp.set_ylabel("neurons")
    axis_atp.legend(frameon=False, labelcolor=ink_soft)

    # Right: the weights of the plastic synapses at the end. They all started at weight_exc.
    axis_weights.hist(weights, bins=50, color=blue, edgecolor=surface, linewidth=0.5, log=True)
    axis_weights.set_title(f"Plastic weights at the end (start: {weight_exc:g} pA)", color=ink, loc="left")
    axis_weights.set_xlabel("weight (pA)")
    axis_weights.set_ylabel("synapses (log scale)")

    # Quiet axes: no frame on top and right, light grid, gray text
    for axis in (axis_spikes, axis_atp, axis_weights):
        axis.set_facecolor(surface)
        axis.grid(color=grid_color, linewidth=0.6)
        axis.set_axisbelow(True)
        axis.tick_params(which="both", colors=ink_soft)       # "both": also the small ticks of the log scale
        axis.xaxis.label.set_color(ink_soft)
        axis.yaxis.label.set_color(ink_soft)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(grid_color)

    figure.tight_layout()
    figure_path = os.path.abspath("edlif_edstdp_example.png")
    figure.savefig(figure_path, dpi=150, facecolor=surface)
    print(f"\nfigure saved in {figure_path}")


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
if spikes_ok and atp_ok and weights_ok:
    print("\nEXAMPLE OK")
else:
    print("\nEXAMPLE FAILED: see the lines with NO above")
    sys.exit(1)
