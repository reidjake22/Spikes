===== Network Components =====

==================================NeuronGroups==================================
Name                Neuron Count   Parameters               Model               
================================================================================
e_2                 4096           _spikespace=123, ga=2.4e-07, ge=1.49e-06, gi=0, v=-0.022, Cm=5e-10, V_reset=-0.057, V_rest=-0.074, V_reversal_a=-0.09, V_reversal_e=0, V_reversal_i=-0.07, V_threshold=-0.053, column=31.5, g_leak=2.5e-08, lastspike=50, not_refractory=0, row=31.5, sigma=1.5e-05, t_refract=0, tau_a=0.08, tau_ee=0.002, tau_ie=0.005, tau_m=0.02dga/dt = -ga/tau_a : S
dge/dt = -ge/tau_ee : S
dgi/dt = -gi/tau_ie : S
dv/dt = (V_rest-v)/tau_m + (ge*(V_reversal_e-v) + gi*(V_reversal_i-v) + ga*(V_reversal_a-v))/(tau_m*g_leak) : V
Cm : F
V_reset : V
V_rest : V
V_reversal_a : V
V_reversal_e : V
V_reversal_i : V
V_threshold : V
column : 1 (constant)
g_leak : S
lastspike : s
not_refractory : 1
row : 1 (constant)
sigma : V
t_refract : s
tau_a : s
tau_ee : s
tau_ie : s
tau_m : s
i_2                 1024           _spikespace=0, ge=0, gi=6.49e-07, v=-0.0703, Cm=2.14e-10, V_reset=-0.058, V_rest=-0.082, V_reversal_e=0, V_reversal_i=-0.07, V_threshold=-0.053, column=15.5, g_leak=1.8e-08, lastspike=-1e+04, not_refractory=1, row=15.5, sigma=1.5e-05, t_refract=0, tau_ei=0.002, tau_ii=0.005, tau_m=0.012dge/dt = -ge/tau_ei : S
dgi/dt = -gi/tau_ii : S
dv/dt = (V_rest-v)/tau_m + (ge*(V_reversal_e-v) + gi*(V_reversal_i-v))/(tau_m * g_leak) : V
Cm : F
V_reset : V
V_rest : V
V_reversal_e : V
V_reversal_i : V
V_threshold : V
column : 1 (constant)
g_leak : S
lastspike : s
not_refractory : 1
row : 1 (constant)
sigma : V
t_refract : s
tau_ei : s
tau_ii : s
tau_m : s
e_3                 4096           _spikespace=149, ga=2.4e-07, ge=1.87e-06, gi=0, v=-0.0182, Cm=5e-10, V_reset=-0.057, V_rest=-0.074, V_reversal_a=-0.09, V_reversal_e=0, V_reversal_i=-0.07, V_threshold=-0.053, column=31.5, g_leak=2.5e-08, lastspike=50, not_refractory=0, row=31.5, sigma=1.5e-05, t_refract=0, tau_a=0.08, tau_ee=0.002, tau_ie=0.005, tau_m=0.02dga/dt = -ga/tau_a : S
dge/dt = -ge/tau_ee : S
dgi/dt = -gi/tau_ie : S
dv/dt = (V_rest-v)/tau_m + (ge*(V_reversal_e-v) + gi*(V_reversal_i-v) + ga*(V_reversal_a-v))/(tau_m*g_leak) : V
Cm : F
V_reset : V
V_rest : V
V_reversal_a : V
V_reversal_e : V
V_reversal_i : V
V_threshold : V
column : 1 (constant)
g_leak : S
lastspike : s
not_refractory : 1
row : 1 (constant)
sigma : V
t_refract : s
tau_a : s
tau_ee : s
tau_ie : s
tau_m : s
e_4                 4096           _spikespace=127, ga=2.4e-07, ge=1.79e-06, gi=0, v=-0.0196, Cm=5e-10, V_reset=-0.057, V_rest=-0.074, V_reversal_a=-0.09, V_reversal_e=0, V_reversal_i=-0.07, V_threshold=-0.053, column=31.5, g_leak=2.5e-08, lastspike=50, not_refractory=0, row=31.5, sigma=1.5e-05, t_refract=0, tau_a=0.08, tau_ee=0.002, tau_ie=0.005, tau_m=0.02dga/dt = -ga/tau_a : S
dge/dt = -ge/tau_ee : S
dgi/dt = -gi/tau_ie : S
dv/dt = (V_rest-v)/tau_m + (ge*(V_reversal_e-v) + gi*(V_reversal_i-v) + ga*(V_reversal_a-v))/(tau_m*g_leak) : V
Cm : F
V_reset : V
V_rest : V
V_reversal_a : V
V_reversal_e : V
V_reversal_i : V
V_threshold : V
column : 1 (constant)
g_leak : S
lastspike : s
not_refractory : 1
row : 1 (constant)
sigma : V
t_refract : s
tau_a : s
tau_ee : s
tau_ie : s
tau_m : s
i_3                 1024           _spikespace=0, ge=0, gi=6.57e-07, v=-0.0703, Cm=2.14e-10, V_reset=-0.058, V_rest=-0.082, V_reversal_e=0, V_reversal_i=-0.07, V_threshold=-0.053, column=15.5, g_leak=1.8e-08, lastspike=-1e+04, not_refractory=1, row=15.5, sigma=1.5e-05, t_refract=0, tau_ei=0.002, tau_ii=0.005, tau_m=0.012dge/dt = -ge/tau_ei : S
dgi/dt = -gi/tau_ii : S
dv/dt = (V_rest-v)/tau_m + (ge*(V_reversal_e-v) + gi*(V_reversal_i-v))/(tau_m * g_leak) : V
Cm : F
V_reset : V
V_rest : V
V_reversal_e : V
V_reversal_i : V
V_threshold : V
column : 1 (constant)
g_leak : S
lastspike : s
not_refractory : 1
row : 1 (constant)
sigma : V
t_refract : s
tau_ei : s
tau_ii : s
tau_m : s
i_4                 1024           _spikespace=0, ge=0, gi=6.48e-07, v=-0.0704, Cm=2.14e-10, V_reset=-0.058, V_rest=-0.082, V_reversal_e=0, V_reversal_i=-0.07, V_threshold=-0.053, column=15.5, g_leak=1.8e-08, lastspike=-1e+04, not_refractory=1, row=15.5, sigma=1.5e-05, t_refract=0, tau_ei=0.002, tau_ii=0.005, tau_m=0.012dge/dt = -ge/tau_ei : S
dgi/dt = -gi/tau_ii : S
dv/dt = (V_rest-v)/tau_m + (ge*(V_reversal_e-v) + gi*(V_reversal_i-v))/(tau_m * g_leak) : V
Cm : F
V_reset : V
V_rest : V
V_reversal_e : V
V_reversal_i : V
V_threshold : V
column : 1 (constant)
g_leak : S
lastspike : s
not_refractory : 1
row : 1 (constant)
sigma : V
t_refract : s
tau_ei : s
tau_ii : s
tau_m : s
e_1                 4096           _spikespace=142, ga=3.51e-07, ge=2.26e-07, gi=0, v=-0.0554, Cm=5e-10, V_reset=-0.057, V_rest=-0.074, V_reversal_a=-0.09, V_reversal_e=0, V_reversal_i=-0.07, V_threshold=-0.053, column=31.5, g_leak=2.5e-08, lastspike=45.7, not_refractory=0.258, row=31.5, sigma=1.5e-05, t_refract=0, tau_a=0.08, tau_ee=0.002, tau_ie=0.005, tau_m=0.02dga/dt = -ga/tau_a : S
dge/dt = -ge/tau_ee : S
dgi/dt = -gi/tau_ie : S
dv/dt = (V_rest-v)/tau_m + (ge*(V_reversal_e-v) + gi*(V_reversal_i-v) + ga*(V_reversal_a-v))/(tau_m*g_leak) : V
Cm : F
V_reset : V
V_rest : V
V_reversal_a : V
V_reversal_e : V
V_reversal_i : V
V_threshold : V
column : 1 (constant)
g_leak : S
lastspike : s
not_refractory : 1
row : 1 (constant)
sigma : V
t_refract : s
tau_a : s
tau_ee : s
tau_ie : s
tau_m : s
i_1                 1024           _spikespace=0, ge=0, gi=5.06e-07, v=-0.0712, Cm=2.14e-10, V_reset=-0.058, V_rest=-0.082, V_reversal_e=0, V_reversal_i=-0.07, V_threshold=-0.053, column=15.5, g_leak=1.8e-08, lastspike=-1e+04, not_refractory=1, row=15.5, sigma=1.5e-05, t_refract=0, tau_ei=0.002, tau_ii=0.005, tau_m=0.012dge/dt = -ge/tau_ei : S
dgi/dt = -gi/tau_ii : S
dv/dt = (V_rest-v)/tau_m + (ge*(V_reversal_e-v) + gi*(V_reversal_i-v))/(tau_m * g_leak) : V
Cm : F
V_reset : V
V_rest : V
V_reversal_e : V
V_reversal_i : V
V_threshold : V
column : 1 (constant)
g_leak : S
lastspike : s
not_refractory : 1
row : 1 (constant)
sigma : V
t_refract : s
tau_ei : s
tau_ii : s
tau_m : s


====================================Synapses====================================
Name                Source -> Target    Synapse Count  Parameters               Model               
================================================================================
efe_1               e_1 -> e_2          409380         apost=1.44, apre=1.11, alpha_C=0.5, alpha_D=0.5, lambda_e=3e-08, learning_rate=0.04, plasticity=0, tau_c=0.005, tau_d=0.005, w=0.521, lastupdate=50, ga_pre=3.71e-07, ge_pre=2.36e-07, gi_pre=0, v_pre=-0.0552, Cm_pre=5e-10, V_reset_pre=-0.057, V_rest_pre=-0.074, V_reversal_a_pre=-0.09, V_reversal_e_pre=0, V_reversal_i_pre=-0.07, V_threshold_pre=-0.053, column_pre=31.2, g_leak_pre=2.5e-08, lastspike_pre=48.6, not_refractory_pre=0.22, row_pre=31.2, sigma_pre=1.5e-05, t_refract_pre=0, tau_a_pre=0.08, tau_ee_pre=0.002, tau_ie_pre=0.005, tau_m_pre=0.02, ga_post=2.4e-07, ga=2.4e-07, ge_post=1.55e-06, ge=1.55e-06, gi_post=0, gi=0, v_post=-0.0213, v=-0.0213, Cm_post=5e-10, Cm=5e-10, V_reset_post=-0.057, V_reset=-0.057, V_rest_post=-0.074, V_rest=-0.074, V_reversal_a_post=-0.09, V_reversal_a=-0.09, V_reversal_e_post=0, V_reversal_e=0, V_reversal_i_post=-0.07, V_reversal_i=-0.07, V_threshold_post=-0.053, V_threshold=-0.053, column_post=31.2, column=31.2, g_leak_post=2.5e-08, g_leak=2.5e-08, lastspike_post=50, lastspike=50, not_refractory_post=0, not_refractory=0, row_post=31.2, row=31.2, sigma_post=1.5e-05, sigma=1.5e-05, t_refract_post=0, t_refract=0, tau_a_post=0.08, tau_a=0.08, tau_ee_post=0.002, tau_ee=0.002, tau_ie_post=0.005, tau_ie=0.005, tau_m_post=0.02, tau_m=0.02alpha_C : 1
alpha_D : 1
lambda_e : S
learning_rate : 1
plasticity : 1
tau_c : s
tau_d : s
w : 1
ele_1               e_1 -> e_1          35344          apost=1.19, apre=1.26, alpha_C=0.5, alpha_D=0.5, lambda_e=2e-08, learning_rate=0.04, plasticity=0, tau_c=0.005, tau_d=0.005, w=0.547, lastupdate=49, ga_pre=3.59e-07, ge_pre=2.31e-07, gi_pre=0, v_pre=-0.0552, Cm_pre=5e-10, V_reset_pre=-0.057, V_rest_pre=-0.074, V_reversal_a_pre=-0.09, V_reversal_e_pre=0, V_reversal_i_pre=-0.07, V_threshold_pre=-0.053, column_pre=31.2, g_leak_pre=2.5e-08, lastspike_pre=48.4, not_refractory_pre=0.231, row_pre=31.2, sigma_pre=1.5e-05, t_refract_pre=0, tau_a_pre=0.08, tau_ee_pre=0.002, tau_ie_pre=0.005, tau_m_pre=0.02, ga_post=3.59e-07, ga=3.59e-07, ge_post=2.31e-07, ge=2.31e-07, gi_post=0, gi=0, v_post=-0.0552, v=-0.0552, Cm_post=5e-10, Cm=5e-10, V_reset_post=-0.057, V_reset=-0.057, V_rest_post=-0.074, V_rest=-0.074, V_reversal_a_post=-0.09, V_reversal_a=-0.09, V_reversal_e_post=0, V_reversal_e=0, V_reversal_i_post=-0.07, V_reversal_i=-0.07, V_threshold_post=-0.053, V_threshold=-0.053, column_post=31.2, column=31.2, g_leak_post=2.5e-08, g_leak=2.5e-08, lastspike_post=48.1, lastspike=48.1, not_refractory_post=0.235, not_refractory=0.235, row_post=31.2, row=31.2, sigma_post=1.5e-05, sigma=1.5e-05, t_refract_post=0, t_refract=0, tau_a_post=0.08, tau_a=0.08, tau_ee_post=0.002, tau_ee=0.002, tau_ie_post=0.005, tau_ie=0.005, tau_m_post=0.02, tau_m=0.02alpha_C : 1
alpha_D : 1
lambda_e : S
learning_rate : 1
plasticity : 1
tau_c : s
tau_d : s
w : 1
efe_2               e_2 -> e_3          409045         apost=1.43, apre=1.42, alpha_C=0.5, alpha_D=0.5, lambda_e=3e-08, learning_rate=0.04, plasticity=0, tau_c=0.005, tau_d=0.005, w=0.524, lastupdate=50, ga_pre=2.4e-07, ge_pre=1.56e-06, gi_pre=0, v_pre=-0.0212, Cm_pre=5e-10, V_reset_pre=-0.057, V_rest_pre=-0.074, V_reversal_a_pre=-0.09, V_reversal_e_pre=0, V_reversal_i_pre=-0.07, V_threshold_pre=-0.053, column_pre=31.2, g_leak_pre=2.5e-08, lastspike_pre=50, not_refractory_pre=0, row_pre=31.2, sigma_pre=1.5e-05, t_refract_pre=0, tau_a_pre=0.08, tau_ee_pre=0.002, tau_ie_pre=0.005, tau_m_pre=0.02, ga_post=2.4e-07, ga=2.4e-07, ge_post=1.98e-06, ge=1.98e-06, gi_post=0, gi=0, v_post=-0.0173, v=-0.0173, Cm_post=5e-10, Cm=5e-10, V_reset_post=-0.057, V_reset=-0.057, V_rest_post=-0.074, V_rest=-0.074, V_reversal_a_post=-0.09, V_reversal_a=-0.09, V_reversal_e_post=0, V_reversal_e=0, V_reversal_i_post=-0.07, V_reversal_i=-0.07, V_threshold_post=-0.053, V_threshold=-0.053, column_post=31.3, column=31.3, g_leak_post=2.5e-08, g_leak=2.5e-08, lastspike_post=50, lastspike=50, not_refractory_post=0, not_refractory=0, row_post=31.3, row=31.3, sigma_post=1.5e-05, sigma=1.5e-05, t_refract_post=0, t_refract=0, tau_a_post=0.08, tau_a=0.08, tau_ee_post=0.002, tau_ee=0.002, tau_ie_post=0.005, tau_ie=0.005, tau_m_post=0.02, tau_m=0.02alpha_C : 1
alpha_D : 1
lambda_e : S
learning_rate : 1
plasticity : 1
tau_c : s
tau_d : s
w : 1
efe_3               e_3 -> e_4          409413         apost=1.42, apre=1.43, alpha_C=0.5, alpha_D=0.5, lambda_e=3e-08, learning_rate=0.04, plasticity=0, tau_c=0.005, tau_d=0.005, w=0.523, lastupdate=50, ga_pre=2.4e-07, ge_pre=1.99e-06, gi_pre=0, v_pre=-0.0172, Cm_pre=5e-10, V_reset_pre=-0.057, V_rest_pre=-0.074, V_reversal_a_pre=-0.09, V_reversal_e_pre=0, V_reversal_i_pre=-0.07, V_threshold_pre=-0.053, column_pre=31.2, g_leak_pre=2.5e-08, lastspike_pre=50, not_refractory_pre=0, row_pre=31.2, sigma_pre=1.5e-05, t_refract_pre=0, tau_a_pre=0.08, tau_ee_pre=0.002, tau_ie_pre=0.005, tau_m_pre=0.02, ga_post=2.4e-07, ga=2.4e-07, ge_post=1.92e-06, ge=1.92e-06, gi_post=0, gi=0, v_post=-0.0184, v=-0.0184, Cm_post=5e-10, Cm=5e-10, V_reset_post=-0.057, V_reset=-0.057, V_rest_post=-0.074, V_rest=-0.074, V_reversal_a_post=-0.09, V_reversal_a=-0.09, V_reversal_e_post=0, V_reversal_e=0, V_reversal_i_post=-0.07, V_reversal_i=-0.07, V_threshold_post=-0.053, V_threshold=-0.053, column_post=31.3, column=31.3, g_leak_post=2.5e-08, g_leak=2.5e-08, lastspike_post=50, lastspike=50, not_refractory_post=0, not_refractory=0, row_post=31.3, row=31.3, sigma_post=1.5e-05, sigma=1.5e-05, t_refract_post=0, t_refract=0, tau_a_post=0.08, tau_a=0.08, tau_ee_post=0.002, tau_ee=0.002, tau_ie_post=0.005, tau_ie=0.005, tau_m_post=0.02, tau_m=0.02alpha_C : 1
alpha_D : 1
lambda_e : S
learning_rate : 1
plasticity : 1
tau_c : s
tau_d : s
w : 1
pfe_post_0          p_0 -> e_1          204250         ga_post=3.54e-07, ga=3.54e-07, ge_post=2.28e-07, ge=2.28e-07, gi_post=0, gi=0, v_post=-0.0553, v=-0.0553, Cm_post=5e-10, Cm=5e-10, V_reset_post=-0.057, V_reset=-0.057, V_rest_post=-0.074, V_rest=-0.074, V_reversal_a_post=-0.09, V_reversal_a=-0.09, V_reversal_e_post=0, V_reversal_e=0, V_reversal_i_post=-0.07, V_reversal_i=-0.07, V_threshold_post=-0.053, V_threshold=-0.053, column_post=31.6, column=31.6, g_leak_post=2.5e-08, g_leak=2.5e-08, lastspike_post=46.2, lastspike=46.2, not_refractory_post=0.249, not_refractory=0.249, row_post=31.7, row=31.7, sigma_post=1.5e-05, sigma=1.5e-05, t_refract_post=0, t_refract=0, tau_a_post=0.08, tau_a=0.08, tau_ee_post=0.002, tau_ee=0.002, tau_ie_post=0.005, tau_ie=0.005, tau_m_post=0.02, tau_m=0.02                    
ile_1               i_1 -> e_1          123017         lambda_e=2e-08, w=1, ge_pre=0, gi_pre=5.38e-07, v_pre=-0.0712, Cm_pre=2.14e-10, V_reset_pre=-0.058, V_rest_pre=-0.082, V_reversal_e_pre=0, V_reversal_i_pre=-0.07, V_threshold_pre=-0.053, column_pre=15.2, g_leak_pre=1.8e-08, lastspike_pre=-1e+04, not_refractory_pre=1, row_pre=15.2, sigma_pre=1.5e-05, t_refract_pre=0, tau_ei_pre=0.002, tau_ii_pre=0.005, tau_m_pre=0.012, ga_post=3.72e-07, ga=3.72e-07, ge_post=2.35e-07, ge=2.35e-07, gi_post=0, gi=0, v_post=-0.0553, v=-0.0553, Cm_post=5e-10, Cm=5e-10, V_reset_post=-0.057, V_reset=-0.057, V_rest_post=-0.074, V_rest=-0.074, V_reversal_a_post=-0.09, V_reversal_a=-0.09, V_reversal_e_post=0, V_reversal_e=0, V_reversal_i_post=-0.07, V_reversal_i=-0.07, V_threshold_post=-0.053, V_threshold=-0.053, column_post=31, column=31, g_leak_post=2.5e-08, g_leak=2.5e-08, lastspike_post=47.8, lastspike=47.8, not_refractory_post=0.232, not_refractory=0.232, row_post=31, row=31, sigma_post=1.5e-05, sigma=1.5e-05, t_refract_post=0, t_refract=0, tau_a_post=0.08, tau_a=0.08, tau_ee_post=0.002, tau_ee=0.002, tau_ie_post=0.005, tau_ie=0.005, tau_m_post=0.02, tau_m=0.02lambda_e : S
w : 1  
eli_1               e_1 -> i_1          8836           lambda_i=3e-08, w=1, ga_pre=3.58e-07, ge_pre=2.31e-07, gi_pre=0, v_pre=-0.0551, Cm_pre=5e-10, V_reset_pre=-0.057, V_rest_pre=-0.074, V_reversal_a_pre=-0.09, V_reversal_e_pre=0, V_reversal_i_pre=-0.07, V_threshold_pre=-0.053, column_pre=31.3, g_leak_pre=2.5e-08, lastspike_pre=48.6, not_refractory_pre=0.227, row_pre=31.3, sigma_pre=1.5e-05, t_refract_pre=0, tau_a_pre=0.08, tau_ee_pre=0.002, tau_ie_pre=0.005, tau_m_pre=0.02, ge_post=0, ge=0, gi_post=5.22e-07, gi=5.22e-07, v_post=-0.0712, v=-0.0712, Cm_post=2.14e-10, Cm=2.14e-10, V_reset_post=-0.058, V_reset=-0.058, V_rest_post=-0.082, V_rest=-0.082, V_reversal_e_post=0, V_reversal_e=0, V_reversal_i_post=-0.07, V_reversal_i=-0.07, V_threshold_post=-0.053, V_threshold=-0.053, column_post=15.2, column=15.2, g_leak_post=1.8e-08, g_leak=1.8e-08, lastspike_post=-1e+04, lastspike=-1e+04, not_refractory_post=1, not_refractory=1, row_post=15.2, row=15.2, sigma_post=1.5e-05, sigma=1.5e-05, t_refract_post=0, t_refract=0, tau_ei_post=0.002, tau_ei=0.002, tau_ii_post=0.005, tau_ii=0.005, tau_m_post=0.012, tau_m=0.012lambda_i : S
w : 1  
ile_2               i_2 -> e_2          122954         lambda_e=2e-08, w=1, ge_pre=0, gi_pre=6.77e-07, v_pre=-0.0703, Cm_pre=2.14e-10, V_reset_pre=-0.058, V_rest_pre=-0.082, V_reversal_e_pre=0, V_reversal_i_pre=-0.07, V_threshold_pre=-0.053, column_pre=15.2, g_leak_pre=1.8e-08, lastspike_pre=-1e+04, not_refractory_pre=1, row_pre=15.2, sigma_pre=1.5e-05, t_refract_pre=0, tau_ei_pre=0.002, tau_ii_pre=0.005, tau_m_pre=0.012, ga_post=2.4e-07, ga=2.4e-07, ge_post=1.55e-06, ge=1.55e-06, gi_post=0, gi=0, v_post=-0.0213, v=-0.0213, Cm_post=5e-10, Cm=5e-10, V_reset_post=-0.057, V_reset=-0.057, V_rest_post=-0.074, V_rest=-0.074, V_reversal_a_post=-0.09, V_reversal_a=-0.09, V_reversal_e_post=0, V_reversal_e=0, V_reversal_i_post=-0.07, V_reversal_i=-0.07, V_threshold_post=-0.053, V_threshold=-0.053, column_post=30.9, column=30.9, g_leak_post=2.5e-08, g_leak=2.5e-08, lastspike_post=50, lastspike=50, not_refractory_post=0, not_refractory=0, row_post=31, row=31, sigma_post=1.5e-05, sigma=1.5e-05, t_refract_post=0, t_refract=0, tau_a_post=0.08, tau_a=0.08, tau_ee_post=0.002, tau_ee=0.002, tau_ie_post=0.005, tau_ie=0.005, tau_m_post=0.02, tau_m=0.02lambda_e : S
w : 1  
ebe_3               e_3 -> e_2          41271          apost=1.42, apre=1.43, alpha_C=0.5, alpha_D=0.5, lambda_e=2e-08, learning_rate=0.04, plasticity=0, tau_c=0.005, tau_d=0.005, w=0.527, lastupdate=50, ga_pre=2.4e-07, ge_pre=1.96e-06, gi_pre=0, v_pre=-0.0174, Cm_pre=5e-10, V_reset_pre=-0.057, V_rest_pre=-0.074, V_reversal_a_pre=-0.09, V_reversal_e_pre=0, V_reversal_i_pre=-0.07, V_threshold_pre=-0.053, column_pre=31.2, g_leak_pre=2.5e-08, lastspike_pre=50, not_refractory_pre=0, row_pre=31.3, sigma_pre=1.5e-05, t_refract_pre=0, tau_a_pre=0.08, tau_ee_pre=0.002, tau_ie_pre=0.005, tau_m_pre=0.02, ga_post=2.4e-07, ga=2.4e-07, ge_post=1.56e-06, ge=1.56e-06, gi_post=0, gi=0, v_post=-0.0212, v=-0.0212, Cm_post=5e-10, Cm=5e-10, V_reset_post=-0.057, V_reset=-0.057, V_rest_post=-0.074, V_rest=-0.074, V_reversal_a_post=-0.09, V_reversal_a=-0.09, V_reversal_e_post=0, V_reversal_e=0, V_reversal_i_post=-0.07, V_reversal_i=-0.07, V_threshold_post=-0.053, V_threshold=-0.053, column_post=31.3, column=31.3, g_leak_post=2.5e-08, g_leak=2.5e-08, lastspike_post=50, lastspike=50, not_refractory_post=0, not_refractory=0, row_post=31.3, row=31.3, sigma_post=1.5e-05, sigma=1.5e-05, t_refract_post=0, t_refract=0, tau_a_post=0.08, tau_a=0.08, tau_ee_post=0.002, tau_ee=0.002, tau_ie_post=0.005, tau_ie=0.005, tau_m_post=0.02, tau_m=0.02alpha_C : 1
alpha_D : 1
lambda_e : S
learning_rate : 1
plasticity : 1
tau_c : s
tau_d : s
w : 1
ele_2               e_2 -> e_2          35344          apost=1.42, apre=1.43, alpha_C=0.5, alpha_D=0.5, lambda_e=2e-08, learning_rate=0.04, plasticity=0, tau_c=0.005, tau_d=0.005, w=0.527, lastupdate=50, ga_pre=2.4e-07, ge_pre=1.52e-06, gi_pre=0, v_pre=-0.0216, Cm_pre=5e-10, V_reset_pre=-0.057, V_rest_pre=-0.074, V_reversal_a_pre=-0.09, V_reversal_e_pre=0, V_reversal_i_pre=-0.07, V_threshold_pre=-0.053, column_pre=31.2, g_leak_pre=2.5e-08, lastspike_pre=50, not_refractory_pre=0, row_pre=31.2, sigma_pre=1.5e-05, t_refract_pre=0, tau_a_pre=0.08, tau_ee_pre=0.002, tau_ie_pre=0.005, tau_m_pre=0.02, ga_post=2.4e-07, ga=2.4e-07, ge_post=1.52e-06, ge=1.52e-06, gi_post=0, gi=0, v_post=-0.0216, v=-0.0216, Cm_post=5e-10, Cm=5e-10, V_reset_post=-0.057, V_reset=-0.057, V_rest_post=-0.074, V_rest=-0.074, V_reversal_a_post=-0.09, V_reversal_a=-0.09, V_reversal_e_post=0, V_reversal_e=0, V_reversal_i_post=-0.07, V_reversal_i=-0.07, V_threshold_post=-0.053, V_threshold=-0.053, column_post=31.2, column=31.2, g_leak_post=2.5e-08, g_leak=2.5e-08, lastspike_post=50, lastspike=50, not_refractory_post=0, not_refractory=0, row_post=31.2, row=31.2, sigma_post=1.5e-05, sigma=1.5e-05, t_refract_post=0, t_refract=0, tau_a_post=0.08, tau_a=0.08, tau_ee_post=0.002, tau_ee=0.002, tau_ie_post=0.005, tau_ie=0.005, tau_m_post=0.02, tau_m=0.02alpha_C : 1
alpha_D : 1
lambda_e : S
learning_rate : 1
plasticity : 1
tau_c : s
tau_d : s
w : 1
ele_3               e_3 -> e_3          35344          apost=1.43, apre=1.42, alpha_C=0.5, alpha_D=0.5, lambda_e=2e-08, learning_rate=0.04, plasticity=0, tau_c=0.005, tau_d=0.005, w=0.534, lastupdate=50, ga_pre=2.4e-07, ge_pre=1.91e-06, gi_pre=0, v_pre=-0.0179, Cm_pre=5e-10, V_reset_pre=-0.057, V_rest_pre=-0.074, V_reversal_a_pre=-0.09, V_reversal_e_pre=0, V_reversal_i_pre=-0.07, V_threshold_pre=-0.053, column_pre=31.2, g_leak_pre=2.5e-08, lastspike_pre=50, not_refractory_pre=0, row_pre=31.2, sigma_pre=1.5e-05, t_refract_pre=0, tau_a_pre=0.08, tau_ee_pre=0.002, tau_ie_pre=0.005, tau_m_pre=0.02, ga_post=2.4e-07, ga=2.4e-07, ge_post=1.91e-06, ge=1.91e-06, gi_post=0, gi=0, v_post=-0.0179, v=-0.0179, Cm_post=5e-10, Cm=5e-10, V_reset_post=-0.057, V_reset=-0.057, V_rest_post=-0.074, V_rest=-0.074, V_reversal_a_post=-0.09, V_reversal_a=-0.09, V_reversal_e_post=0, V_reversal_e=0, V_reversal_i_post=-0.07, V_reversal_i=-0.07, V_threshold_post=-0.053, V_threshold=-0.053, column_post=31.2, column=31.2, g_leak_post=2.5e-08, g_leak=2.5e-08, lastspike_post=50, lastspike=50, not_refractory_post=0, not_refractory=0, row_post=31.2, row=31.2, sigma_post=1.5e-05, sigma=1.5e-05, t_refract_post=0, t_refract=0, tau_a_post=0.08, tau_a=0.08, tau_ee_post=0.002, tau_ee=0.002, tau_ie_post=0.005, tau_ie=0.005, tau_m_post=0.02, tau_m=0.02alpha_C : 1
alpha_D : 1
lambda_e : S
learning_rate : 1
plasticity : 1
tau_c : s
tau_d : s
w : 1
ile_4               i_4 -> e_4          122841         lambda_e=2e-08, w=1, ge_pre=0, gi_pre=6.75e-07, v_pre=-0.0703, Cm_pre=2.14e-10, V_reset_pre=-0.058, V_rest_pre=-0.082, V_reversal_e_pre=0, V_reversal_i_pre=-0.07, V_threshold_pre=-0.053, column_pre=15.2, g_leak_pre=1.8e-08, lastspike_pre=-1e+04, not_refractory_pre=1, row_pre=15.2, sigma_pre=1.5e-05, t_refract_pre=0, tau_ei_pre=0.002, tau_ii_pre=0.005, tau_m_pre=0.012, ga_post=2.4e-07, ga=2.4e-07, ge_post=1.88e-06, ge=1.88e-06, gi_post=0, gi=0, v_post=-0.0187, v=-0.0187, Cm_post=5e-10, Cm=5e-10, V_reset_post=-0.057, V_reset=-0.057, V_rest_post=-0.074, V_rest=-0.074, V_reversal_a_post=-0.09, V_reversal_a=-0.09, V_reversal_e_post=0, V_reversal_e=0, V_reversal_i_post=-0.07, V_reversal_i=-0.07, V_threshold_post=-0.053, V_threshold=-0.053, column_post=30.9, column=30.9, g_leak_post=2.5e-08, g_leak=2.5e-08, lastspike_post=50, lastspike=50, not_refractory_post=0, not_refractory=0, row_post=31, row=31, sigma_post=1.5e-05, sigma=1.5e-05, t_refract_post=0, t_refract=0, tau_a_post=0.08, tau_a=0.08, tau_ee_post=0.002, tau_ee=0.002, tau_ie_post=0.005, tau_ie=0.005, tau_m_post=0.02, tau_m=0.02lambda_e : S
w : 1  
ebe_4               e_4 -> e_3          40826          apost=1.43, apre=1.42, alpha_C=0.5, alpha_D=0.5, lambda_e=2e-08, learning_rate=0.04, plasticity=0, tau_c=0.005, tau_d=0.005, w=0.52, lastupdate=50, ga_pre=2.4e-07, ge_pre=1.88e-06, gi_pre=0, v_pre=-0.0188, Cm_pre=5e-10, V_reset_pre=-0.057, V_rest_pre=-0.074, V_reversal_a_pre=-0.09, V_reversal_e_pre=0, V_reversal_i_pre=-0.07, V_threshold_pre=-0.053, column_pre=31.1, g_leak_pre=2.5e-08, lastspike_pre=50, not_refractory_pre=0, row_pre=31.3, sigma_pre=1.5e-05, t_refract_pre=0, tau_a_pre=0.08, tau_ee_pre=0.002, tau_ie_pre=0.005, tau_m_pre=0.02, ga_post=2.4e-07, ga=2.4e-07, ge_post=1.96e-06, ge=1.96e-06, gi_post=0, gi=0, v_post=-0.0174, v=-0.0174, Cm_post=5e-10, Cm=5e-10, V_reset_post=-0.057, V_reset=-0.057, V_rest_post=-0.074, V_rest=-0.074, V_reversal_a_post=-0.09, V_reversal_a=-0.09, V_reversal_e_post=0, V_reversal_e=0, V_reversal_i_post=-0.07, V_reversal_i=-0.07, V_threshold_post=-0.053, V_threshold=-0.053, column_post=31.1, column=31.1, g_leak_post=2.5e-08, g_leak=2.5e-08, lastspike_post=50, lastspike=50, not_refractory_post=0, not_refractory=0, row_post=31.4, row=31.4, sigma_post=1.5e-05, sigma=1.5e-05, t_refract_post=0, t_refract=0, tau_a_post=0.08, tau_a=0.08, tau_ee_post=0.002, tau_ee=0.002, tau_ie_post=0.005, tau_ie=0.005, tau_m_post=0.02, tau_m=0.02alpha_C : 1
alpha_D : 1
lambda_e : S
learning_rate : 1
plasticity : 1
tau_c : s
tau_d : s
w : 1
ele_4               e_4 -> e_4          35344          apost=1.42, apre=1.43, alpha_C=0.5, alpha_D=0.5, lambda_e=2e-08, learning_rate=0.04, plasticity=0, tau_c=0.005, tau_d=0.005, w=0.534, lastupdate=50, ga_pre=2.4e-07, ge_pre=1.83e-06, gi_pre=0, v_pre=-0.0192, Cm_pre=5e-10, V_reset_pre=-0.057, V_rest_pre=-0.074, V_reversal_a_pre=-0.09, V_reversal_e_pre=0, V_reversal_i_pre=-0.07, V_threshold_pre=-0.053, column_pre=31.2, g_leak_pre=2.5e-08, lastspike_pre=50, not_refractory_pre=0, row_pre=31.2, sigma_pre=1.5e-05, t_refract_pre=0, tau_a_pre=0.08, tau_ee_pre=0.002, tau_ie_pre=0.005, tau_m_pre=0.02, ga_post=2.4e-07, ga=2.4e-07, ge_post=1.82e-06, ge=1.82e-06, gi_post=0, gi=0, v_post=-0.0192, v=-0.0192, Cm_post=5e-10, Cm=5e-10, V_reset_post=-0.057, V_reset=-0.057, V_rest_post=-0.074, V_rest=-0.074, V_reversal_a_post=-0.09, V_reversal_a=-0.09, V_reversal_e_post=0, V_reversal_e=0, V_reversal_i_post=-0.07, V_reversal_i=-0.07, V_threshold_post=-0.053, V_threshold=-0.053, column_post=31.2, column=31.2, g_leak_post=2.5e-08, g_leak=2.5e-08, lastspike_post=50, lastspike=50, not_refractory_post=0, not_refractory=0, row_post=31.2, row=31.2, sigma_post=1.5e-05, sigma=1.5e-05, t_refract_post=0, t_refract=0, tau_a_post=0.08, tau_a=0.08, tau_ee_post=0.002, tau_ee=0.002, tau_ie_post=0.005, tau_ie=0.005, tau_m_post=0.02, tau_m=0.02alpha_C : 1
alpha_D : 1
lambda_e : S
learning_rate : 1
plasticity : 1
tau_c : s
tau_d : s
w : 1
eli_3               e_3 -> i_3          8836           
lambda_i=3e-08, 
w=1,
 ga_pre=2.4e-07,
  ge_pre=1.91e-06,
   gi_pre=0,
    v_pre=-0.0178,
     Cm_pre=5e-10,
      V_reset_pre=-0.057,
       V_rest_pre=-0.074,
        V_reversal_a_pre=-0.09,
         V_reversal_e_pre=0,
          V_reversal_i_pre=-0.07,
           V_threshold_pre=-0.053,
            column_pre=31.3,
             g_leak_pre=2.5e-08,
              lastspike_pre=50,
               not_refractory_pre=0,
                row_pre=31.3,
                 sigma_pre=1.5e-05,
                  t_refract_pre=0,
                   tau_a_pre=0.08,
                    tau_ee_pre=0.002,
                     tau_ie_pre=0.005,
                      tau_m_pre=0.02,
                       ge_post=0,
                        ge=0,
                         gi_post=6.76e-07,
                          gi=6.76e-07,
                           v_post=-0.0703,
                            v=-0.0703,
                             Cm_post=2.14e-10,
                              Cm=2.14e-10,
                               V_reset_post=-0.058,
                                V_reset=-0.058,
                                 V_rest_post=-0.082,
                                  V_rest=-0.082,
                                   V_reversal_e_post=0,
                                    V_reversal_e=0,
                                     V_reversal_i_post=-0.07,
                                      V_reversal_i=-0.07,
                                       V_threshold_post=-0.053,
                                        V_threshold=-0.053,
                                         column_post=15.2,
                                          column=15.2,
                                           g_leak_post=1.8e-08,
                                            g_leak=1.8e-08,
                                             lastspike_post=-1e+04,
                                              lastspike=-1e+04,
                                               not_refractory_post=1,
                                                not_refractory=1,
                                                 row_post=15.2,
                                                  row=15.2,
                                                   sigma_post=1.5e-05,
                                                    sigma=1.5e-05,
                                                     t_refract_post=0,
                                                      t_refract=0,
                                                       tau_ei_post=0.002,
                                                        tau_ei=0.002,
                                                         tau_ii_post=0.005,
                                                          tau_ii=0.005,
                                                           tau_m_post=0.012,
                                                            tau_m=0.012
                                                            lambda_i : S
w : 1  
eli_2               e_2 -> i_2          8836           lambda_i=3e-08, w=1, ga_pre=2.4e-07, ge_pre=1.52e-06, gi_pre=0, v_pre=-0.0215, Cm_pre=5e-10, V_reset_pre=-0.057, V_rest_pre=-0.074, V_reversal_a_pre=-0.09, V_reversal_e_pre=0, V_reversal_i_pre=-0.07, V_threshold_pre=-0.053, column_pre=31.3, g_leak_pre=2.5e-08, lastspike_pre=50, not_refractory_pre=0, row_pre=31.3, sigma_pre=1.5e-05, t_refract_pre=0, tau_a_pre=0.08, tau_ee_pre=0.002, tau_ie_pre=0.005, tau_m_pre=0.02, ge_post=0, ge=0, gi_post=6.67e-07, gi=6.67e-07, v_post=-0.0703, v=-0.0703, Cm_post=2.14e-10, Cm=2.14e-10, V_reset_post=-0.058, V_reset=-0.058, V_rest_post=-0.082, V_rest=-0.082, V_reversal_e_post=0, V_reversal_e=0, V_reversal_i_post=-0.07, V_reversal_i=-0.07, V_threshold_post=-0.053, V_threshold=-0.053, column_post=15.2, column=15.2, g_leak_post=1.8e-08, g_leak=1.8e-08, lastspike_post=-1e+04, lastspike=-1e+04, not_refractory_post=1, not_refractory=1, row_post=15.2, row=15.2, sigma_post=1.5e-05, sigma=1.5e-05, t_refract_post=0, t_refract=0, tau_ei_post=0.002, tau_ei=0.002, tau_ii_post=0.005, tau_ii=0.005, tau_m_post=0.012, tau_m=0.012lambda_i : S
w : 1  
ile_3               i_3 -> e_3          122564         lambda_e=2e-08, w=1, ge_pre=0, gi_pre=6.86e-07, v_pre=-0.0703, Cm_pre=2.14e-10, V_reset_pre=-0.058, V_rest_pre=-0.082, V_reversal_e_pre=0, V_reversal_i_pre=-0.07, V_threshold_pre=-0.053, column_pre=15.2, g_leak_pre=1.8e-08, lastspike_pre=-1e+04, not_refractory_pre=1, row_pre=15.2, sigma_pre=1.5e-05, t_refract_pre=0, tau_ei_pre=0.002, tau_ii_pre=0.005, tau_m_pre=0.012, ga_post=2.4e-07, ga=2.4e-07, ge_post=1.96e-06, ge=1.96e-06, gi_post=0, gi=0, v_post=-0.0174, v=-0.0174, Cm_post=5e-10, Cm=5e-10, V_reset_post=-0.057, V_reset=-0.057, V_rest_post=-0.074, V_rest=-0.074, V_reversal_a_post=-0.09, V_reversal_a=-0.09, V_reversal_e_post=0, V_reversal_e=0, V_reversal_i_post=-0.07, V_reversal_i=-0.07, V_threshold_post=-0.053, V_threshold=-0.053, column_post=31, column=31, g_leak_post=2.5e-08, g_leak=2.5e-08, lastspike_post=50, lastspike=50, not_refractory_post=0, not_refractory=0, row_post=30.9, row=30.9, sigma_post=1.5e-05, sigma=1.5e-05, t_refract_post=0, t_refract=0, tau_a_post=0.08, tau_a=0.08, tau_ee_post=0.002, tau_ee=0.002, tau_ie_post=0.005, tau_ie=0.005, tau_m_post=0.02, tau_m=0.02lambda_e : S

w : 1  
eli_4               e_4 -> i_4          8836           lambda_i=3e-08, w=1, ga_pre=2.4e-07, ge_pre=1.83e-06, gi_pre=0, v_pre=-0.0192, Cm_pre=5e-10, V_reset_pre=-0.057, V_rest_pre=-0.074, V_reversal_a_pre=-0.09, V_reversal_e_pre=0, V_reversal_i_pre=-0.07, V_threshold_pre=-0.053, column_pre=31.3, g_leak_pre=2.5e-08, lastspike_pre=50, not_refractory_pre=0, row_pre=31.3, sigma_pre=1.5e-05, t_refract_pre=0, tau_a_pre=0.08, tau_ee_pre=0.002, tau_ie_pre=0.005, tau_m_pre=0.02, ge_post=0, ge=0, gi_post=6.66e-07, gi=6.66e-07, v_post=-0.0703, v=-0.0703, Cm_post=2.14e-10, Cm=2.14e-10, V_reset_post=-0.058, V_reset=-0.058, V_rest_post=-0.082, V_rest=-0.082, V_reversal_e_post=0, V_reversal_e=0, V_reversal_i_post=-0.07, V_reversal_i=-0.07, V_threshold_post=-0.053, V_threshold=-0.053, column_post=15.2, column=15.2, g_leak_post=1.8e-08, g_leak=1.8e-08, lastspike_post=-1e+04, lastspike=-1e+04, not_refractory_post=1, not_refractory=1, row_post=15.2, row=15.2, sigma_post=1.5e-05, sigma=1.5e-05, t_refract_post=0, t_refract=0, tau_ei_post=0.002, tau_ei=0.002, tau_ii_post=0.005, tau_ii=0.005, tau_m_post=0.012, tau_m=0.012lambda_i : S

w : 1  
ebe_2               e_2 -> e_1          41119          apost=1.11, apre=1.44, alpha_C=0.5, alpha_D=0.5, lambda_e=2e-08, learning_rate=0.04, plasticity=0, tau_c=0.005, tau_d=0.005, w=0.582, lastupdate=50, ga_pre=2.4e-07, ge_pre=1.56e-06, gi_pre=0, v_pre=-0.0211, Cm_pre=5e-10, V_reset_pre=-0.057, V_rest_pre=-0.074, V_reversal_a_pre=-0.09, V_reversal_e_pre=0, V_reversal_i_pre=-0.07, V_threshold_pre=-0.053, column_pre=31.3, g_leak_pre=2.5e-08, lastspike_pre=50, not_refractory_pre=0, row_pre=31.1, sigma_pre=1.5e-05, t_refract_pre=0, tau_a_pre=0.08, tau_ee_pre=0.002, tau_ie_pre=0.005, tau_m_pre=0.02, ga_post=3.72e-07, ga=3.72e-07, ge_post=2.44e-07, ge=2.44e-07, gi_post=0, gi=0, v_post=-0.0548, v=-0.0548, Cm_post=5e-10, Cm=5e-10, V_reset_post=-0.057, V_reset=-0.057, V_rest_post=-0.074, V_rest=-0.074, V_reversal_a_post=-0.09, V_reversal_a=-0.09, V_reversal_e_post=0, V_reversal_e=0, V_reversal_i_post=-0.07, V_reversal_i=-0.07, V_threshold_post=-0.053, V_threshold=-0.053, column_post=31.3, column=31.3, g_leak_post=2.5e-08, g_leak=2.5e-08, lastspike_post=48.6, lastspike=48.6, not_refractory_post=0.218, not_refractory=0.218, row_post=31.2, row=31.2, sigma_post=1.5e-05, sigma=1.5e-05, t_refract_post=0, t_refract=0, tau_a_post=0.08, tau_a=0.08, tau_ee_post=0.002, tau_ee=0.002, tau_ie_post=0.005, tau_ie=0.005, tau_m_post=0.02, tau_m=0.02alpha_C : 1
alpha_D : 1
lambda_e : S
learning_rate : 1
plasticity : 1
tau_c : s
tau_d : s
w : 1


