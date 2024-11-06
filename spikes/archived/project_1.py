from brian2 import *
import numpy as n
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class NeuronModel:
    '''A dataclass to hold everything important about a neuron type 

    (may expand on later)
    '''
    def __init__(self, Cm, g_leak, tau_m, V_rest, V_threshold, V_reset, V_reversal, t_refract, model_type='default'):
        self.Cm = Cm
        self.g_leak = g_leak
        self.tau_m = tau_m
        self.V_rest = V_rest
        self.V_threshold = V_threshold
        self.V_reset = V_reset
        self.V_reversal = V_reversal
        self.t_refract = t_refract
        self.model_type = model_type

class SynapticModel:
    """ A dataclass to hold everything important about a synapse type

    Basically, it will define synapses coming from either excitatory or inhibitory neurons.
    e refers to those which synapse onto excitatory cells;
    i refers to those which synapse onto inhibitory cells.
    Therefore, for excitatory neurons -> inhibitory neurons we should use the exc_syn_model.i
    """
    def __init__(self, exc_lambda, exc_tau, inh_lambda, inh_tau, alpha_C = None, alpha_D=None,tau_pre=None,tau_post=None,A_plus=None,A_minus=None):
        self.e = {'lambda': exc_lambda, 'tau': exc_tau}
        self.i = {'lambda': inh_lambda, 'tau': inh_tau}

class NeuralModel:
    """ A model that stores cell types, and synaptic parameters in an easily interpretable format

    it also has a function for generating the correct equations out of these parameters for 
    """
    def __init__(self, exc_neuron_model, exc_syn_model, inh_neuron_model, inh_syn_model, sigma, I_ext=1 * nA):
        self.cell_types = {
            'e': exc_neuron_model,
            'i': inh_neuron_model
        }
        
        self.synaptic_params = {
            'e': exc_syn_model,
            'i': inh_syn_model
        }


        self.stdp_params = {
            'alpha_C': 0.5,
            'alpha_D': 0.5,
            'tau_pre': 3 * ms,
            'tau_post': 5 * ms,
            'A_plus': 0.1,
            'A_minus': 0.1
        }
        self.I_ext = I_ext
        self.sigma = sigma

    def generate_neuron_equation(self, cell_type, stimulus=False):
        equation = '''
        dv/dt = (V_rest-v)/tau_m + (ge * (V_reversal_e-v) + gi * (V_reversal_i-v)) / (tau_m*g_leak) + sigma*xi*(tau_m)**-0.5 : volt
        dge/dt = -ge/tau_e : siemens
        dgi/dt = -gi/tau_i : siemens
        '''
        if stimulus: # place holder for now - return to later
            equation.replace('+ sigma','+ I_ext/(tau_m*g_leak))+ sigma')  # Include the stimulus term if specified

        return equation

class VisNetNetwork(Network):
    ''' This function inherits all the network methods and attributes, but is designed to make it easy to make VisNet style models
    '''
    def __init__(self, Ne, Ni, num_layers, connection, neural_model):
        super().__init__()
        self.num_layers = num_layers
        self.neural_model = neural_model
        self.connection = connection
        self.Ne = Ne
        self.Ni = Ni
        self.construct_network()

    def construct_network(self):
        self.neuron_groups = []
        self.synapses = []
        self.monitors = []
        self.positions = initialize_positions(N=self.Ne)
        ''' This is a function that constructs the VisNetNetwork

       
        '''
        exc_neuron_params = {
            'V_rest': self.neural_model.cell_types['e'].V_rest,
            'V_threshold': self.neural_model.cell_types['e'].V_threshold,
            'V_reset': self.neural_model.cell_types['e'].V_reset,
            'tau_m': self.neural_model.cell_types['e'].tau_m,
            'V_reversal_e': self.neural_model.cell_types['e'].V_reversal,
            'V_reversal_i': self.neural_model.cell_types['i'].V_reversal,
            'g_leak': self.neural_model.cell_types['e'].g_leak,
            'tau_e': self.neural_model.synaptic_params['e'].e['tau'],
            'tau_i': self.neural_model.synaptic_params['i'].e['tau'],
            'lambda_e': self.neural_model.synaptic_params['e'].e['lambda'],
            'lambda_i': self.neural_model.synaptic_params['e'].i['lambda'],
            'sigma': self.neural_model.sigma,
        }

        inh_neuron_params = {
            'V_rest': self.neural_model.cell_types['i'].V_rest,
            'V_threshold': self.neural_model.cell_types['i'].V_threshold,
            'V_reset': self.neural_model.cell_types['i'].V_reset,
            'tau_m': self.neural_model.cell_types['i'].tau_m,
            'V_reversal_e': self.neural_model.cell_types['e'].V_reversal,
            'V_reversal_i': self.neural_model.cell_types['i'].V_reversal,
            'g_leak': self.neural_model.cell_types['i'].g_leak,
            'tau_e': self.neural_model.synaptic_params['e'].i['tau'],
            'tau_i': self.neural_model.synaptic_params['i'].i['tau'],
            'lambda_e': self.neural_model.synaptic_params['i'].e['lambda'],
            'lambda_i': self.neural_model.synaptic_params['i'].i['lambda'],
            'sigma': self.neural_model.sigma,
        }
        S_ei = None
        S_ie = None
        # Now we define the input layer, following the same logic as for the dummy variables. using a poisson group
        # G_e = NeuronGroup(self.Ne,
        #                   self.neural_model.generate_neuron_equation('e',stimulus=True),
        #                   threshold='v>V_threshold',
        #                   reset=f'v=V_reset',
        #                   namespace={**exc_neuron_params,**{'I_ext':1*nA}})
        G_e = PoissonGroup(Ne, rates='10 * Hz')  # Adjust rate as needed
        self.add(self.poisson_input)

        G_i = None
        # Add it to the model

        self.add(G_e)
        self.neuron_groups.append((G_e, G_i))
        '''
        Because we have to synapse onto the next layer, the synapses for layer 0 are actually 
        generated alongside neurons for layer 1. Incidentally, because we use 0 index, the range works as intended
        The first time round, it generates E & I cells for layer 1 and adds them to neuron_group[1].
        Then when we access neuron_groups[layer] we get (G_e<input>, None)
        We then connect that up to the current layer's excitatory cell, by connecting g-0 to g-1

        '''
        for layer in range(self.num_layers):
            print(f"completing neurons of layer {layer+1}")
            G_e = NeuronGroup(self.Ne,
                              self.neural_model.generate_neuron_equation('e'),
                              threshold=f'v > V_threshold',
                              reset=f'v = V_reset',
                              namespace=exc_neuron_params)

            G_i = NeuronGroup(self.Ni,
                              self.neural_model.generate_neuron_equation('i'),
                              threshold=f'v > V_threshold',
                              reset=f'v = V_reset',
                              namespace=inh_neuron_params)
            self.add(G_e, G_i)
            self.neuron_groups.append((G_e, G_i))
            G_e_prev, G_i_prev = self.neuron_groups[layer]
            print(f"building synapses of layer {layer}")
            # Feedforward connections between layers (plastic)
            '''
            the first layer of ff connections aren't plastic
            the next layers are all plastic, but depending on set up will be fully connected 
            with previous layer or connected via radius or some sort
            '''
            if (layer == 0):
                S_ff = Synapses(G_e_prev,
                                G_e,
                                )
                print(f"connecting synapses of layer {layer}")
                S_ff.connect(j='i')
            else:
                # flagging the tau_pre - what's going on here? Oh it's a STDP parameter
                S_ff = Synapses(G_e_prev,
                                G_e,
                                model='''
                    w : 1
                    dapre/dt = -apre/tau_pre : 1 (clock-driven)
                    dapost/dt = -apost/tau_post : 1 (clock-driven)
                    ''',
                    on_pre='''
                    ge_post += lambda_e * w
                    apre += alpha_C * (1-apre)
                    w += -w * apost * A_minus
                    ''',
                    on_post=f'''
                    apost += alpha_D * (1-apost)
                    w += (1-w) * apre * A_plus
                    ''',
                    namespace={**exc_neuron_params,**neural_model.stdp_params}
                )
                print(f"connecting synapses of layer {layer}")
                if self.connection == 'fc':
                    S_ff.connect()

                elif self.connection == 'rf':
                    S_ff.connect(condition='')

                else:
                    S_ff.connect(condition=self.connection)
            self.add(S_ff)     
            '''
            finally w append the synapses related to this the previous layer.
            This includes all afferent connections.
            So having just made layer 1 neurons we have layer 0 synapses on the books
            '''
            self.synapses.append((S_ff, S_ei, S_ie))
            # Now we can generate EI and IE connections for the current neuronal layer before the next loop (not plastic)
    
            S_ei = Synapses(G_e,
                            G_i,
                            model='w : 1',
                            on_pre='ge_post += lambda_i',
                            namespace=exc_neuron_params
                            )
            S_ie = Synapses(G_i,
                            G_e,
                            model='w : 1',
                            on_pre='gi_post += lambda_e',
                            namespace=inh_neuron_params
                            )
        # Connect and add them to the network
        print(f"connecting synapses of final layer")
        S_ei.connect()
        S_ie.connect()

        self.add(S_ei)
        self.add(S_ie)
        # at the end of the loop we just have to add layer N's neurons, we do this here
        self.synapses.append((None, S_ei, S_ie))
        # Now we have all them connected, and completed according
    def add_monitors(self, layers, monitor_type='voltage', record_exc=True, record_inh=True, dt=None):
        """Add monitors to specified layers and store them in a dictionary.
        
        Args:
            layers (list): List of layer indices to add monitors.
            monitor_type (str): 'voltage' or 'spike' to specify type of monitor.
            record_exc (bool): If True, monitor excitatory neurons.
            record_inh (bool): If True, monitor inhibitory neurons.
            dt (Quantity): The time step for the monitor recording.
        
        Returns:
            dict: A dictionary of monitors with keys as names.
            name = layer_0_exc_spike_monitor
        """
        self.monitors = {}  # This could also be initialized in the __init__ if you want to access it elsewhere
        for layer_index in layers:
            exc_neurons, inh_neurons = self.neuron_groups[layer_index]
            if record_exc and exc_neurons:
                name = f"layer_{layer_index}_exc_{monitor_type}_monitor"
                if monitor_type == 'voltage':
                    monitor = StateMonitor(exc_neurons, 'v', record=True, dt=dt)
                elif monitor_type == 'spike':
                    monitor = SpikeMonitor(exc_neurons)
                self.add(monitor)
                self.monitors[name] = monitor

            if record_inh and inh_neurons:
                name = f"layer_{layer_index}_inh_{monitor_type}_monitor"
                if monitor_type == 'voltage':
                    monitor = StateMonitor(inh_neurons, 'v', record=True, dt=dt)
                elif monitor_type == 'spike':
                    monitor = SpikeMonitor(inh_neurons)
                self.add(monitor)
                self.monitors[name] = monitor

        return self.monitors
    def initialize_positions(N, grid_type='square', width=None, height=None):
        if grid_type == 'square':
            side = int(np.sqrt(N))  # Determine the side length of a square grid
            positions = [(x, y) for x in range(side) for y in range(side)]
        elif grid_type == 'rectangular' and width and height:
            positions = [(x, y) for x in range(width) for y in range(height)]
        else:
            pass
# Create instances of NeuronModel for excitatory and inhibitory neurons
e_neuron_model = NeuronModel(
    Cm=500 * pF,
    g_leak=25 * nS,
    tau_m=20 * ms,
    V_rest=-74 * mV,
    V_threshold=-53*mV,
    V_reset=-57*mV,
    V_reversal=0 * mV,
    t_refract=2 * ms
    )

i_neuron_model = NeuronModel(
    Cm=214 * pF,
    g_leak=18 * nS,
    tau_m=12 * ms,
    V_rest=-82 * mV,
    V_threshold=-53*mV,
    V_reset=-58*mV,
    V_reversal=-70 * mV,
    t_refract=2 * ms
    )
exc_syn_model = SynapticModel(
    exc_lambda = 4 * nS,
    exc_tau = 2 * ms,
    inh_lambda = 5.0 * nS,
    inh_tau = 2 * ms
    )
inh_syn_model = SynapticModel(
    exc_lambda = 0.5 * nS,
    exc_tau = 5 * ms,
    inh_lambda = 5.0 * nS,
    inh_tau = 5 * ms
    )
Ne = 10
Ni = 3

# Create an instance of NeuralModel
neural_model = NeuralModel(e_neuron_model, exc_syn_model, i_neuron_model, inh_syn_model, 0.015 * mV)

# Example of constructing a network with 3 layers using the NeuralModel instance
network = VisNetNetwork(Ne, Ni, num_layers=3, neural_model=neural_model,connection='fc')




# Run the network
voltage_monitor_input = StateMonitor(network.neuron_groups[0][0], 'v', record=True, dt=10*ms)
voltage_monitor_first = StateMonitor(network.neuron_groups[1][0], 'v', record=True, dt=10*ms)
voltage_monitor_second = StateMonitor(network.neuron_groups[2][0], 'v', record=True, dt=1*ms)
network.add(voltage_monitor_input, voltage_monitor_first, voltage_monitor_second)

# Run the network
print("Beginning run")
network.run(20 * ms, report='text', report_period=2 * ms)
print("Run completed")

# Visualization setup
fig, ax = plt.subplots()

def update(frame):
    ax.clear()
    ax.set_title("Voltage Animation")
    
    for layer_index, voltage_monitor in enumerate(voltage_monitors):
        voltages = voltage_monitor.v[:, frame]
        
        # Clip the voltages for color mapping
        clipped_voltages = np.clip(voltages / mV, -80, 50)
        colors = plt.cm.RdYlGn((clipped_voltages + 80) / 130)  # Scale between 0 and 1
        
        for i, voltage in enumerate(voltages):
            ax.add_patch(plt.Rectangle((layer_index, i), 1, 1, color=colors[i]))

    ax.set_xlim(0, len(voltage_monitors))
    ax.set_ylim(0, max(Ne, Ni))
    ax.set_xticks([])
    ax.set_yticks([])

# Create the color bar
norm = plt.Normalize(vmin=-80, vmax=50)
sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=norm)
sm.set_array([])

cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Voltage (mV)')

ani = animation.FuncAnimation(fig, update, frames=len(voltage_monitors[0].t), interval=20, repeat=False)
plt.show()
