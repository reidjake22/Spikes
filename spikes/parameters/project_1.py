from brian2 import *
import numpy as np
"""THIS IS PRETTY CRAP RIGHT NOW HONESTLY In an ideal world,
 it is flexible enough that there is an add class method to add heterogeneous types of neurons to excitatory/ inhib 
 and a method to return the neuron and synaptic parameters in some ordered form in order to input into the VisNetNetwork class, in order to form the right neuron groups.
 """
"""
OKAY SO TO DO: SORT THE INPUT SO THAT THERE IS THE CORRECT EQUATION, 
SORT THE SYNAPTIC CONNECTIONS SO THAT THEY MAKE SENSE FOR FC AND RF
CHECK CHATGPT DIDNT FUCK UP THE PARAMETERS ANYWHERE
WRITE SOME TESTS
THE GOOD NEWS IS WE CAN HOPEFULLY TEST IF THE EQUATIONS ARE IDENTICAL USING .is
RUN THE CODE
GET A NICE NETWORK RUNNING AND START PLAYING AROUND WITH THE OTHER SIDE OF THINGS
THE ACTUAL RUNNING OF THE NETWORK YGM
THERE IS SO MUCH FURTHER TO GO BUT HOPEFULLY THIS SPEEDS ME ON MY WAY SOMEWHAT
Regarding inputs:
    Can do the poison group inputs
    Can do SpikeGenerator inputs
    Can do Step Current inputs:
        do I do that to first or second layer?
        I think first layer...
        
"""

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

    (may expand on later)
    """
    def __init__(self, exc_lambda, exc_tau, inh_lambda, inh_tau, alpha_C = None, alpha_D=None,tau_pre=None,tau_post=None,A_plus=None,A_minus=None):
        self.e = {'lambda': exc_lambda, 'tau': exc_tau},
        self.i = {'lambda': inh_lambda, 'tau': inh_tau},
        # self.alpha_C = alpha_C
        # self.alpha_D = alpha_D
        # self.tau_pre = tau_pre
        # self.tau_post = tau_post
        # self.A_plus = A_plus
        # self.A_minus = A_minus

class NeuralModel:
    """ A model that stores cell types, and synaptic parameters in an easily interpretable format

    it also has a function for generating the correct equations out of these parameters for 
    """
    def __init__(self, exc_neuron_model, exc_syn_model, inh_neuron_model, inh_syn_model, sigma):
        self.cell_types = {
            'e': {
                'Cm': exc_neuron_model.Cm,
                'g_leak': exc_neuron_model.g_leak,
                'tau_m': exc_neuron_model.tau_m,
                'V_rest': exc_neuron_model.V_rest,
                'V_threshold': exc_neuron_model.V_threshold,
                'V_reset': exc_neuron_model.V_reset,
                'V_reversal': exc_neuron_model.V_reversal,
                't_refract': exc_neuron_model.t_refract
            },
            'i': {
                'Cm': inh_neuron_model.Cm,
                'g_leak': inh_neuron_model.g_leak,
                'tau_m': inh_neuron_model.tau_m,
                'V_rest': inh_neuron_model.V_rest,
                'V_threshold': inh_neuron_model.V_threshold,
                'V_reset': inh_neuron_model.V_reset,
                'V_reversal': inh_neuron_model.V_reversal,
                't_refract': inh_neuron_model.t_refract
            }
        }
        
        self.synaptic_params = {
            'e' : {
                'e': {'lambda': exc_syn_model.e['lambda'], 'tau': exc_syn_model.e['tau']},
                'i': {'lambda': exc_syn_model.i['lambda'], 'tau': exc_syn_model.i['tau']}
            },
            'i' : {
                'e': {'lambda': inh_syn_model.e['lambda'], 'tau': inh_syn_model.e['tau']},
                'i': {'lambda': inh_syn_model.i['lambda'], 'tau': inh_syn_model.i['tau']}
            }
        }


        self.stdp_params = {
            'alpha_C': 0.5,
            'alpha_D': 0.5,
            'tau_pre': 3 * ms,
            'tau_post': 5 * ms,
            'A_plus': 0.1,
            'A_minus': 0.1
        }
        
        self.sigma = sigma

    def generate_neuron_equation(self, cell_type, stimulus=False):
        params = self.cell_types[cell_type]
        V_rest = params['V_rest']
        tau_m = params['tau_m']
        V_reversal_e = self.cell_types['e']['V_reversal']
        V_reversal_i = self.cell_types['i']['V_reversal']
        g_leak = params['g_leak']
        tau_e = self.synaptic_params['e'][cell_type]
        tau_i = self.synaptic_params['i'][cell_type]
        sigma=self.sigma
        equation = f'''
        dv/dt = ({V_rest}-v)/{tau_m} + (ge * ({V_reversal_e}-v) + gi * ({V_reversal_i}-v)) / ({tau_m}*{g_leak}) + {sigma}*xi*({tau_m})**-0.5 : volt
        dge/dt = -ge/{tau_e} : siemens
        dgi/dt = -gi/{tau_i} : siemens
        '''
        if stimulus: # place holder for now - return to later
            equation = equation.replace('/(tau_m*g_leak)', '+ stimulus(t,i)/(tau_m*g_leak)')
        return equation

class VisNetNetwork(Network):
    ''' This function inherits all the network methods and attributes, but is designed to make it easy to make VisNet style models


    '''
    def __init__(self, num_layers, connection, neural_model):
        super().__init__()
        self.num_layers = num_layers
        self.neural_model = neural_model
        self.connection = connection
        self.construct_network()

    def construct_network(self):
        ''' This is a function that constructs the VisNetNetwork

       
        '''
        neuron_groups = []
        synapses = []
        '''
        What's happening here? 
        Okay, slightly hacky, basically, we first create S_ei, S_ie as the layer's synapses for interneurons
        we start at layer 0, so obviously they dont exist.
        '''

        S_ei = None
        S_ie = None
        # Now we define the input layer, following the same logic as for the dummy variables
        G_e = NeuronGroup(self.neural_model.cell_types['e']['Cm'],
                                  self.neural_model.generate_neuron_equation('e'),
                                  threshold=f'v > {self.neural_model.cell_types["e"]["V_threshold"]}',
                                  reset=f'v = {self.neural_model.cell_types["e"]["V_reset"]}')
        G_i = None
        # Add it to the model
        self.add(G_e)

        # Add it to the index along with 
        neuron_groups.append((G_e, G_i))

        '''
        Because we have to synapse onto the next layer, the synapses for layer 0 are actually 
        generated alongside neurons for layer 1. Incidentally, because we use 0 index, the range works as intended
        The first time round, it generates E & I cells for layer 1 and adds them to neuron_group[1].
        Then when we access neuron_groups[layer] we get (G_e<input>, None)
        We then connect that up to the current layer's excitatory cell, by connecting g-0 to g-1

        '''
        for layer in range(self.num_layers):
            G_e = NeuronGroup(self.neural_model.cell_types['e']['Cm'],
                              self.neural_model.generate_neuron_equation('e'),
                              threshold=f'v > {self.neural_model.cell_types["e"]["V_threshold"]}',
                              reset=f'v = {self.neural_model.cell_types["e"]["V_reset"]}')

            G_i = NeuronGroup(self.neural_model.cell_types['i']['Cm'],
                              self.neural_model.generate_neuron_equation('i'),
                              threshold=f'v > {self.neural_model.cell_types["i"]["V_threshold"]}',
                              reset=f'v = {self.neural_model.cell_types["i"]["V_reset"]}')
            self.add(G_e, G_i)
            neuron_groups.append((G_e, G_i))
            G_e_prev, G_i_prev = neuron_groups[layer]

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
                S_ff.connect(j='i')
            else:
                S_ff = Synapses(G_e_prev, G_e, model='''
                    w : 1
                    dapre/dt = -apre/tau_pre : 1 (clock-driven)
                    dapost/dt = -apost/tau_post : 1 (clock-driven)
                    ''',
                    on_pre=f'''
                    ge_post += {self.neural_model.synaptic_params['ee']['lambda']} * w
                    apre += {self.neural_model.stdp_params['alpha_C']} * (1-apre)
                    w += -w * apost * {self.neural_model.stdp_params['A_minus']}
                    ''',
                    on_post=f'''
                    apost += {self.neural_model.stdp_params['alpha_D']} * (1-apost)
                    w += (1-w) * apre * {self.neural_model.stdp_params['A_plus']}
                    '''
                )
                if self.connection == 'fc':
                    S_ff.connect(condition='')

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
            synapses.append((S_ff, S_ei, S_ie))
            # Now we can generate EI and IE connections for the current neuronal layer before the next loop (not plastic)
            S_ei = Synapses(G_e,
                            G_i,
                            model='w : 1',
                            on_pre='ge_post += 0.1*nS',
                            )
            S_ie = Synapses(G_i,
                            G_e,
                            model='w : 1',
                            on_pre='gi_post += 0.1*nS',
                            )
            # Connect and add them to the network
            S_ei.connect()
            S_ie.connect()

            self.add(S_ei)
            self.add(S_ie)
        # at the end of the loop we just have to add layer N's neurons, we do this here
        synapses.append((None, S_ei, S_ie))
        # Now we have all them connected, and completed according
        print(f"the dimensions of the neuron_groups array is {np.ndim(neuron_groups)} and the dimensions of the synapses array is {np.ndim(synapses)}")

# Create instances of NeuronModel for excitatory and inhibitory neurons
e_neuron_model = NeuronModel(
    Cm=500 * pF,
    g_leak=25 * nS,
    tau_m=20 * ms,
    V_rest=-74 * mV,
    V_threshold=-53 * mV,
    V_reset=-57 * mV,
    V_reversal=0 * mV,
    t_refract=2 * ms
)

i_neuron_model = NeuronModel(
    Cm=214 * pF,
    g_leak=18 * nS,
    tau_m=12 * ms,
    V_rest=-82 * mV,
    V_threshold=-53 * mV,
    V_reset=-58 * mV,
    V_reversal=-70 * mV,
    t_refract=2 * ms
)

# Create an instance of NeuralModel
neural_model = NeuralModel(e_neuron_model, i_neuron_model)

# Example of constructing a network with 3 layers using the NeuralModel instance
network = VisNetNetwork(num_layers=3, neural_model=neural_model)
