NEURON {
    POINT_PROCESS STDP
    RANGE tau_pre, tau_post, A_plus, A_minus, w, max_weight, min_weight
    NONSPECIFIC_CURRENT i
    NET_RECEIVE (weight, t_pre, t_post)
}

PARAMETER {
    tau_pre = 20 (ms)
    tau_post = 20 (ms)
    A_plus = 0.01
    A_minus = 0.012
    w = 0.5
    max_weight = 1
    min_weight = 0
}

ASSIGNED {
    i (nA)
}

INITIAL {
    i = 0
}

NET_RECEIVE(weight, t_pre, t_post) {
    if (flag == 0) {  : Pre-synaptic spike
        t_pre = t
        if (t_post > 0) {
            w = w + A_plus * exp(-(t - t_post)/tau_pre)
            if (w > max_weight) {
                w = max_weight
            }
        }
        i = -w
    } else if (flag == 1) {  : Post-synaptic spike
        t_post = t
        if (t_pre > 0) {
            w = w - A_minus * exp(-(t - t_pre)/tau_post)
            if (w < min_weight) {
                w = min_weight
            }
        }
    }
}
