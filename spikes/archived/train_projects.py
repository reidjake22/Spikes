from brian2 import *
def project_1():

    """A project to remind myself of the simulations, and familiarise myself with the Brian2 neurons

    Involves building a simple fully connected 3 layer network of 10 neurons, then running the network with some noisy input
    """
    from spikes.archived import project_1 as p1

    # Parameters:
    excit_neuron = '''
    dv/dt = (V_rest_e-v)/tau_m_e + (ge*(V_reversal_e-v) + gi*(V_reversal_i-v))/(tau_m_e*g_leak_e) + sigma*xi*(tau_m_e)**-0.5 : volt
    dge/dt = -ge/tau_ee : siemens
    dgi/dt = -gi/tau_ie : siemens
    '''
    #including inputs
    input_neuron = '''
    dv/dt = (V_rest_e-v)/tau_m_e + (ge*(V_reversal_e-v) + gi*(V_reversal_i-v) + stimulus(t,i))/(tau_m_e*g_leak_e) + sigma*xi*(tau_m_e)**-0.5 : volt
    dge/dt = -ge/tau_ee : siemens
    dgi/dt = -gi/tau_ie : siemens
    '''
    # Inhibitory
    inhib_neuron = '''
    dv/dt = (V_rest_i-v)/tau_m_i + (ge*(V_reversal_e-v) + gi*(V_reversal_i-v))/(tau_m_i * g_leak_i) + sigma*xi*(tau_m_i)**-0.5 : volt
    dge/dt = -ge/tau_ei : siemens
    dgi/dt = -gi/tau_ii : siemens
    '''

    # Setting up Network: IT SEEMS TO ME THIS BIT COULD DEFINITELY BE OPTIMISED THERE MUST BE SOME WAY OF PASSING THE SAME PARAMETERS - INDEED I THINK YOU COULD PASS THEM AS A TUPLE TO BE UNPACKED OR DICTIONARY
    class VisNet_Network(Network):
        #This guy will make all the network and synapse objects
        def __init__(self, n_layers, parameters, equations)


    excit_setup = {"N": p1.Ne,
            
                     "model": p1.cell_E1,
                     "threshold": 'v > p1.V_threshold_e',
                     "reset": 'v=p1.V_reset_e',
                     "refractory": p1.t_refract_e,
                     "method": 'euler'}
    inhib_setup = {"N": p1.Ni,
                   "model": p1.cell_I,
                   "threshold": 'v > p1.V_threshold_i',
                   "reset":'v=p1.V_reset_i',
                   "refractory": p1.t_refract_i,
                   "method": 'euler'}
    
    l0 = NeuronGroup(**excit_setup)
    l1_e = NeuronGroup(**excit_setup)
    l2_e = NeuronGroup(**excit_setup)
    l3_e = NeuronGroup(**excit_setup)

    l1_i = NeuronGroup(**inhib_setup)
    l2_i = NeuronGroup(**inhib_setup)
    l3_i = NeuronGroup(**inhib_setup)
    
    # Setting up Synapses:
    ei_setup = {"model": p1.syn_EI,
                "on_pre": p1.syn_EI_on_pre,
                "delay": None,
                "method": 'euler',
    }
    ie_setup = {"model": p1.syn_IE,
                "on_pre": p1.syn_IE_on_pre,
                "delay": None,
                "method": 'euler',
    }
    ee_setup = {"model": p1.syn_EE,
                "on_pre": p1.syn_EE_on_pre,
                "on_post": p1.syn_EE_on_post,
                "delay": None,
                "method": 'euler',
    }
    
    l0_ee = Synapses(l0,l1_e,**ee_setup)
    l1_ee = Synapses(l1_e,l2_e,**ee_setup)
    l2_ee = Synapses(l2_e,l3_e,**ee_setup)

    l1_ei = Synapses(l1_e,l1_i, **ei_setup)
    l2_ei = Synapses(l2_e, l2_i, **ei_setup)
    l3_ei = Synapses(l3_e,l3_i, **ei_setup)

    l1_ie = Synapses(l1_i,l1_e, **ie_setup)
    l2_ie = Synapses(l2_i,l2_e, **ie_setup)
    l3_ie = Synapses(l3_i,l3_e, **ie_setup)
    print("connecting excitatory")
    l0_ee.connect(i= "j")
    l1_ee.connect()
    l2_ee.connect()
    print("connecting excitatory to inhib")
    l1_ei.connect()
    l2_ei.connect()
    l3_ei.connect()
    print("connecting inhibitory to excitatory")
    l1_ie.connect()
    l2_ie.connect()
    l3_ie.connect()
    '''
    So basically, I've made a few improvements here
    - turns out I was simulating plasticity without ever using it in the interneurons
    In addition, instead of setting w to 1, I'm setting w to 
    '''
    
    l0_ee
    l1_ee
    l2_ee
    l3_ee

    l1_ei
    l2_ei
    l3_ei

    l1_ie
    l2_ie
    l3_ie

    L1_EI.w = 1
    L1_IE.w = 1
    L1_EE.w = 'rand()'
    L2_EI.w = 1
    L2_IE.w = 1
    L1_EE.plastic = True

