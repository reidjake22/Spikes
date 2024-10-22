from brian2 import *


class SynapseParameters:
    def __init__(
        self,
        lambda_e=None,
        A_minus=None,
        A_plus=None,
        alpha_C=None,
        alpha_D=None,
        tau_pre=None,
        tau_post=None,
        synapse_type=None,
    ):
        self.lambda_e = lambda_e
        self.A_minus = A_minus
        self.A_plus = A_plus
        self.alpha_C = alpha_C
        self.alpha_D = alpha_D
        self.tau_pre = tau_pre
        self.tau_post = tau_post

        self.check_valid_parameters(
            lambda_e=lambda_e,
            A_minus=A_minus,
            A_plus=A_plus,
            alpha_C=alpha_C,
            alpha_D=alpha_D,
            tau_pre=tau_pre,
            tau_post=tau_post,
            synapse_type=synapse_type,
        )

    def check_valid_parameters(self, **params):
        """
        Check if the parameters provided are valid
        """
        synapse_type = params.pop("synapse_type", None)
        if synapse_type == "stdp":
            for key, value in params.items():
                if value is None:
                    raise ValueError(f"Parameter {key} is not provided")
        elif synapse_type == "non_stdp":
            for key, value in params.items():
                if key == "lambda_e" and value is None:
                    raise ValueError(
                        f"lambda_e is not provided, this is needed for non_stdp synapses"
                    )


class StdpSynapseSpecs:
    def __init__(self, lambda_e, A_minus, A_plus, alpha_C, alpha_D, tau_pre, tau_post):
        self.parameters = SynapseParameters(
            lambda_e=lambda_e,
            A_minus=A_minus,
            A_plus=A_plus,
            alpha_C=alpha_C,
            alpha_D=alpha_D,
            tau_pre=tau_pre,
            tau_post=tau_post,
            synapse_type="stdp",
        )

    def create_synapses(self, afferent_group, efferent_group):
        synapses = Synapses(
            afferent_group,
            efferent_group,
            model="""
                    lambda_e: 1
                    A_minus: 1
                    A_plus: 1
                    alpha_C: 1
                    alpha_D: 1
                    apre: 1
                    apost: 1
                    w: 1
                    lastupdate_pre: second
                    lastupdate_post: second
                    """,
            on_pre="""
                    ge_post += lambda_e * w
                    apre = apre * exp((lastupdate_post - t)/tau_pre)
                    lastupdate_pre = t
                    apre += alpha_C
                    apre = clip(apre,0,1)
                    w += - apost * A_minus 
                    w = clip(w,0,1)
                    """,
            on_post="""
                    apost = apost * exp((lastupdate_pre - t)/tau_post)
                    lastupdate_post = t
                    apost += alpha_D
                    w += apre * A_plus
                    w = clip(w,0,1)
                    """,
        )
        return synapses

    def define_synapse_parameters(self, synapses):
        synapses.lambda_e = self.parameters.lambda_e
        synapses.A_minus = self.parameters.A_minus
        synapses.A_plus = self.parameters.A_plus
        synapses.alpha_C = self.parameters.alpha_C
        synapses.alpha_D = self.parameters.alpha_D
        synapses.tau_pre = self.parameters.tau_pre
        synapses.tau_post = self.parameters.tau_post
        synapses.apre = 0
        synapses.apost = 0
        synapses.w = 0.1
        synapses.lastupdate_pre = 0 * ms
        synapses.lastupdate_post = 0 * ms

    def connect_synapses(self, synapses):
        self.define_synapse_parameters(synapses)
        synapses.connect(j="i")


class NonStdpSynapseSpecs:
    def __init__(self, lambda_e):
        self.parameters = SynapseParameters(
            lambda_e=lambda_e, synapse_type="non_stdp"
        )  # Will this mean that we could end up using None 0 values without realising?

    def create_synapses(self, afferent_group, efferent_group):
        synapses = Synapses(
            afferent_group,
            efferent_group,
            model="""
                        w: 1
                        lambda_e: 1
                        """,
            on_pre="""
                        ge_post += lambda_e * w # This is to make the weights marry up! Maybe look at this
                        """,
        )
        return synapses

    def define_synapse_parameters(self, synapses):
        synapses.lambda_e = self.parameters.lambda_e
        synapses.w = 0.1

    def connect_synapses(self, synapses):
        self.define_synapse_parameters(synapses)
        synapses.connect(j="i")
