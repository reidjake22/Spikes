from brian2 import *
import numpy as np
import matplotlib.pyplot as plt

# ——— (Optional) Use the C++ standalone backend with 4 threads ——————
set_device('cpp_standalone')
prefs.devices.cpp_standalone.openmp_threads = 4

# ——— 1) Build & compile the network once —————————————————————————
start_scope()

# Phase durations & rates
phase_duration = 500 * ms
rate1 = 10 * Hz
rate2 = 50 * Hz

# Input neuron group
inp = PoissonGroup(1, rates=rate1)

# Target LIF neuron
target = NeuronGroup(
    1,
    'dv/dt = -v/(10*ms) : 1',
    threshold='v>1',
    reset='v = 0',
    method='exact'
)

# STDP parameters
taupre = 20 * ms
taupost = 20 * ms
wmax = 1.0
apre = 0.01
apost = -apre

# Synapse with STDP “by hand”
syn = Synapses(
    inp, target,
    model='''
        w : 1
        dpre/dt  = -pre/taupre   : 1 (event-driven)
        dpost/dt = -post/taupost : 1 (event-driven)
    ''',
    on_pre='''
        v_post += w
        pre = 1
        w = clip(w + apost*post, 0, wmax)
    ''',
    on_post='''
        post = 1
        w    = clip(w + apre*pre,  0, wmax)
    '''
)
syn.connect()
syn.w = 0.5  # initial weight

# Monitor the synaptic weight
w_mon = StateMonitor(syn, 'w', record=0)

# Compile
# ——— 2) Run first phase ————————————————————————————————————————
run(phase_duration)

# ——— 3) Switch input rate & second phase ————————————————————————
inp.rates = rate2
run(phase_duration)

# ——— 4) Plot results ————————————————————————————————————————
t = w_mon.t / ms
w = w_mon.w[0]

# Weight trace
plt.figure()
plt.plot(t, w, label='w(t)')
plt.axvline(phase_duration/ms, color='k', linestyle='--', label='rate switch')
plt.xlabel('Time (ms)')
plt.ylabel('Synaptic weight')
plt.legend()
plt.title('STDP weight across two runs')

# Input rates trace
rates = np.concatenate([
    np.ones(len(t)//2)*(rate1/Hz),
    np.ones(len(t)//2)*(rate2/Hz),
])
plt.figure()
plt.plot(t, rates)
plt.axvline(phase_duration/ms, color='k', linestyle='--')
plt.xlabel('Time (ms)')
plt.ylabel('Input rate (Hz)')
plt.title('Poisson input rate phases')

plt.show()
