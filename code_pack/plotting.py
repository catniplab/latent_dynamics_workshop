import matplotlib.pyplot as plt
import numpy as np

def plot_two_d_vector_field_from_data(dynamics_func, axs, axs_range, P=None):
    x = np.linspace(min(axs_range['x_min'], -2), max(axs_range['x_max'], 2), 100)
    y = np.linspace(min(axs_range['y_min'], -2), max(axs_range['y_max'], 2), 100)

    X, Y = np.meshgrid(x, y)
    u, v = np.zeros(X.shape), np.zeros(Y.shape)
    std_output = np.zeros_like(X)
    NI, NJ = Y.shape

    for i in range(NI):
        for j in range(NJ):
            x = X[i, j]
            y = Y[i, j]

            vec_in = np.array([x, y])
            # ode always needs 0th time point, so we take the first mapping which is not 0
            vec_out = dynamics_func(vec_in)[1]

            if P is None:
                s = (vec_out - vec_in)
            else:
                s = (vec_out - vec_in) @ np.transpose(P)

            u[i, j] = s[0]
            v[i, j] = s[1]

    # axs.contourf(X, Y, std_output, cmap='seismic', alpha=0.2)
    axs.streamplot(X, Y, u, v, color='blue', linewidth=0.5, arrowsize=0.5)


