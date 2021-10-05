import numpy as np
import matplotlib.pyplot as plt

font = {'weight': 'normal', 'size': 14}
plt.rc('font', **font)

# Initialize data, parameters and variables

x = np.array([[-1, -1, -1], [1, -1, 1],
              [-1, 1, 1], [1, 1, -1],
              [1, -1, -1], [-1, 1, -1],
              [-1, -1, 1], [1, 1, 1]])

P_D = np.array([0.25, 0.25, 0.25, 0.25, 0, 0, 0, 0])
n_visible = 3
n_hidden_array = np.array([1, 2, 4, 8])
n_inner = 1000
n_outer = 1000
CD_K = 100
eta = 0.1
mb_size = np.array([2, 3, 20, 20])
n_trials = 2000
n_averages = 10  # This sets the number of averages of D_KL. Set to 1 to run just one time.
th_D_KL = np.zeros(len(n_hidden_array))
num_D_KL = np.zeros(len(n_hidden_array))
P_B_mat = np.zeros((n_averages, 8))
num_D_KL_ave = np.zeros((n_averages, len(n_hidden_array)))

# Functions


def init_weights(_n_visible_neurons, _n_hidden_neurons):
    w_init_output = np.random.normal(size=(_n_visible_neurons, _n_hidden_neurons))
    theta_v_init_output = np.zeros(_n_visible_neurons)
    theta_h_init_output = np.zeros(_n_hidden_neurons)
    return w_init_output, theta_h_init_output, theta_v_init_output


def init_delta_weights(_n_visible_neurons, _n_hidden_neurons):
    delta_w_init = np.zeros((_n_visible_neurons, _n_hidden_neurons))
    delta_theta_v_init = np.zeros(_n_visible_neurons)
    delta_theta_h_init = np.zeros(_n_hidden_neurons)
    return delta_w_init, delta_theta_h_init, delta_theta_v_init


def choose_pattern(_patterns, _rows):
    chosen_pattern_row = np.random.randint(_rows)
    chosen_pattern = _patterns[chosen_pattern_row, :]
    return chosen_pattern


def stochastic_update(_local_field):
    _r = np.random.rand()
    _p = 1 / (1 + np.exp(-2 * _local_field))
    if _r < _p:
        return 1
    else:
        return -1


def local_field_h(_v, _w, _theta_h):
    _local_field_h = np.dot(_v, _w) - _theta_h
    if _local_field_h.any() > 10:
        print("local_field_h > 10")
    return _local_field_h


def local_field_v(_h, _w, _theta_v):
    _local_field_v = np.dot(_w, _h) - _theta_v
    if _local_field_v.any() > 10:
        print("local_field_v > 10")
    return _local_field_v


def update_hidden(_v, _w, _theta_h):
    _local_field_h = local_field_h(_v, _w, _theta_h)
    _len_h = len(_theta_h)
    _h = np.zeros(_len_h)
    for i in range(_len_h):
        _h[i] = stochastic_update(_local_field_h[i])
    return _h


def update_visible(_h, _w, _theta_v):
    _local_field_v = local_field_v(_h, _w, _theta_v)
    _len_v = len(_theta_v)
    _v = np.zeros(_len_v)
    for i in range(_len_v):
        _v[i] = stochastic_update(_local_field_v[i])
    return _v


def update_delta_w(_eta, _v0, _b_h0, _v, _b_h):
    _delta_w_update = _eta * (np.outer(_v0, np.tanh(_b_h0)) - np.outer(_v, np.tanh(_b_h)))
    return _delta_w_update


def update_delta_theta_v(_eta, _v0, _v):
    _delta_theta_v_update = - _eta * (_v0 - _v)
    return _delta_theta_v_update


def update_delta_theta_h(_eta, _b_h0, _b_h):
    _delta_theta_h_update = - _eta * (np.tanh(_b_h0) - np.tanh(_b_h))
    return _delta_theta_h_update


def compare_patterns(_v, _x, _n_outer, _n_inner):
    _P_B = np.zeros(8)
    for l in range(8):
        if np.array_equal(_v, _x[l, :]):
            _P_B[l] = 1 / (_n_outer * _n_inner)
    return _P_B


def theoretical_D_KL(_n_hidden, _n_visible):
    if _n_hidden < 2 ** (_n_visible - 1) - 1:
        _th_D_KL = np.log(2) * (_n_visible - (np.log2(_n_hidden + 1))
                                - (_n_hidden + 1) / 2 ** (np.log2(_n_hidden + 1)))
    else:
        _th_D_KL = 0
    return _th_D_KL


def numerical_D_KL(_P_B, _P_D):
    _D_KL = np.sum(P_D[:4] * np.log(P_D[:4] / P_B[:4]))
    if np.isnan(_D_KL) or np.isinf(_D_KL):
        print(f"Warning! D_KL contains nan or inf!")
    return _D_KL


def run_trials(_n_visible, _n_hidden, _n_trials, _mb_size, _x, _K, _eta):
    _w, _theta_h, _theta_v = init_weights(_n_visible, _n_hidden)
    # Trials loop
    for i in range(_n_trials):
        _delta_w, _delta_theta_h, _delta_theta_v = init_delta_weights(_n_visible, _n_hidden)
        # Minibatch loop
        for j in range(_mb_size):
            _v = choose_pattern(_x, 4)
            _v0 = np.copy(_v)
            _h = update_hidden(_v, _w, _theta_h)
            for k in range(_K):
                _v = update_visible(_h, _w, _theta_v)
                _h = update_hidden(_v, _w, _theta_h)
            _b_h0 = local_field_h(_v0, _w, _theta_h)
            _b_h = local_field_h(_v, _w, _theta_h)

            # Update delta_w, delta_theta_h, delta_theta_v
            _delta_w += update_delta_w(_eta, _v0, _b_h0, _v, _b_h)
            _delta_theta_v += update_delta_theta_v(_eta, _v0, _v)
            _delta_theta_h += update_delta_theta_h(_eta, _b_h0, _b_h)
        _w += _delta_w
        _theta_v += _delta_theta_v
        _theta_h += _delta_theta_h

    return _w, _theta_h, _theta_v


def sample_distribution(_n_outer, _n_inner, _x, _w, _theta_h, _theta_v):
    _P_B = np.zeros(8)
    for i in range(_n_outer):
        _mu = np.random.randint(8)
        _v = choose_pattern(_x, 8)
        _h = update_hidden(_v, _w, _theta_h)
        if i % 100 == 0:
            print(f'Outer loop: {i}.')
        for j in range(_n_inner):
            _v = update_visible(_h, _w, _theta_v)
            _h = update_hidden(_v, _w, _theta_h)
            _P_B += compare_patterns(_v, _x, _n_outer, _n_inner)
    return _P_B


# Pipeline
for index, n_hidden in enumerate(n_hidden_array):
    w, theta_h, theta_v = run_trials(n_visible, n_hidden, n_trials, mb_size[index], x, CD_K, eta)
    for average in range(n_averages):
        P_B = sample_distribution(n_outer, n_inner, x, w, theta_h, theta_v)
        P_B_mat[average, :] = P_B[:]
        num_D_KL_ave[average, index] = numerical_D_KL(P_B, P_D)
    print(f'Hidden variables: {n_hidden}.')
    print(f'Weight matrix:')
    print(w)
    print(f'theta_h:')
    print(theta_h)
    print(f'theta_v:')
    print(theta_v)
    print(f'P_B:')
    print(P_B)
    print(f'-------------------------')
    th_D_KL[index] = theoretical_D_KL(n_hidden, n_visible)
    P_B_ave = np.average(P_B_mat, axis=0)
print(num_D_KL_ave)

# Plot all but last without label for clean legend
for average in range(n_averages - 1):
    plt.plot(n_hidden_array, num_D_KL_ave[average, :], 'b-', alpha=0.5)
# Plot last with legend
plt.plot(n_hidden_array, num_D_KL_ave[-1, :], 'b-', alpha=0.5, label='Simulated $D_{KL}$')
plt.plot(n_hidden_array, np.mean(num_D_KL_ave, axis=0), 'ko--', linewidth=2,
         label='Average $D_{KL}$ over ' + str(n_averages) + ' MCMC-runs.')
plt.plot(n_hidden_array, th_D_KL, 'r-', label='Theoretical $D_{KL}$')

plt.xlabel("$M$")
plt.ylabel("$D_{KL}$")
plt.title("$D_{KL}$ for " + str(n_trials) + " trials.")
plt.grid()
plt.legend()
plt.show()
