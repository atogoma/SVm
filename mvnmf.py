import numpy as np
from sklearn.preprocessing import normalize

from .initialization import initialize_nmf, nnls, beta_divergence, normalize_WH


EPSILON = np.finfo(np.float32).eps
EPSILON2 = np.finfo(np.float16).eps


def _volume_logdet(W, delta):
    K = W.shape[1]
    volume = np.log10(np.linalg.det(W.T @ W + delta * np.eye(K)))
    return volume


def _loss_mvnmf(X, W, H, Lambda, delta):
    reconstruction_error = beta_divergence(X, W @ H, beta=1, square_root=False)
    volume = _volume_logdet(W, delta)
    loss = reconstruction_error + Lambda * volume
    return loss, reconstruction_error, volume


def _solve_mvnmf(X, W, H, lambda_tilde=1e-5, delta=1.0, gamma=1.0,
                 max_iter=200, min_iter=100, tol=1e-4,
                 conv_test_freq=10, conv_test_baseline=None, verbose=0):
    """Mvnmf solver
    X : array-like of shape (n_features, n_samples)
        Constant input matrix.

    W : array-like of shape (n_features, n_components)
        Initial guess.

    H : array-like of shape (n_components, n_samples)
        Initial guess.

    lambda_tilde : float
        Hyperparameter.

    delta : float
        Hyperparameter.

    gamma : float
        Initial step size for backtracking line search. Should be between 0 and 1. If -1,
        then backtracking line search is skipped.

    max_iter : int, default=200
        Maximum number of iterations.

    min_iter : int, default=100
        Minimum number of iterations.

    tol : float, default=1e-4
        Tolerance of the stopping condition.

    conv_test_freq : int, default=10
        Convergence test frequency. Convergence test is performed every conv_test_freq iterations.

    conv_test_baseline : float, default=None
        Baseline for convergence test. If None, the initial loss is taken as the baseline.

    verbose : int, default=0
        Verbosity level.

    """
    if (type(X) != np.ndarray) or (not np.issubdtype(X.dtype, np.floating)):
        X = np.array(X).astype(float)
    if (type(W) != np.ndarray) or (not np.issubdtype(W.dtype, np.floating)):
        W = np.array(W).astype(float)
    if (type(H) != np.ndarray) or (not np.issubdtype(H.dtype, np.floating)):
        H = np.array(H).astype(float)
    n_features, n_samples = X.shape
    n_components = W.shape[1]
    M = n_features
    T = n_samples
    K = n_components
    ##############################################
    #### Algorithm
    ##############################################
    # First normalize W
    W, H = normalize_WH(W, H)
    # Clip small values: important.
    W = W.clip(EPSILON)
    H = H.clip(EPSILON)
    # Calculate Lambda from labmda_tilde
    reconstruction_error = beta_divergence(X, W @ H, beta=1, square_root=False)
    volume = _volume_logdet(W, delta)
    Lambda = lambda_tilde * reconstruction_error / abs(volume)
    loss = reconstruction_error + Lambda * volume
    # Useful constants
    ones = np.ones((M, T))
    # Baseline of convergence test
    if conv_test_baseline is None:
        conv_test_baseline = loss
    elif type(conv_test_baseline) is str and conv_test_baseline == 'min-iter':
        pass
    else:
        conv_test_baseline = float(conv_test_baseline)
    # Loop
    losses = [loss]
    reconstruction_errors = [reconstruction_error]
    volumes = [volume]
    line_search_steps = []
    gammas = [gamma]
    loss_previous = loss # Loss in the last iteration
    loss_previous_conv_test = loss # Loss in the last convergence test
    converged = False
    for n_iter in range(1, max_iter + 1):
        # Update H
        H = H * ( ( W.T @ (X/(W @ H)) ) / (W.T @ ones) )
        H = H.clip(EPSILON)
        # Update W
        Y = np.linalg.inv(W.T @ W + delta * np.eye(K))
        Y_plus = np.maximum(Y, 0)
        Y_minus = np.maximum(-Y, 0)
        JHT = ones @ H.T
        LWYm = Lambda * (W @ Y_minus)
        LWY = Lambda * (W @ (Y_plus + Y_minus))
        numerator = ( (JHT - 4 * LWYm)**2 + 8 * LWY * ((X/(W @ H)) @ H.T) )**0.5 - JHT + 4 * LWYm
        denominator = 4 * LWY
        Wup = W * (numerator / denominator)
        Wup = Wup.clip(EPSILON)
        # Backtracking line search for W
        if gamma != -1:
            W_new = (1 - gamma) * W + gamma * Wup
            W_new, H_new = normalize_WH(W_new, H)
            W_new = W_new.clip(EPSILON)
            H_new = H_new.clip(EPSILON)
            loss, reconstruction_error, volume = _loss_mvnmf(X, W_new, H_new, Lambda, delta)
            line_search_step = 0
            while (loss > loss_previous) and (gamma > 1e-16):
                gamma = gamma * 0.8
                W_new = (1 - gamma) * W + gamma * Wup
                W_new, H_new = normalize_WH(W_new, H)
                W_new = W_new.clip(EPSILON)
                H_new = H_new.clip(EPSILON)
                loss, reconstruction_error, volume = _loss_mvnmf(X, W_new, H_new, Lambda, delta)
                line_search_step += 1
            W = W_new
            H = H_new
        else:
            line_search_step = 0
            W = Wup
            W, H = normalize_WH(W, H)
            W = W.clip(EPSILON)
            H = H.clip(EPSILON)
        line_search_steps.append(line_search_step)
        # Update gamma
        if gamma != -1:
            gamma = min(gamma*2.0, 1.0)
        gammas.append(gamma)
        # Losses
        loss, reconstruction_error, volume = _loss_mvnmf(X, W, H, Lambda, delta)
        losses.append(loss)
        reconstruction_errors.append(reconstruction_error)
        volumes.append(volume)
        loss_previous = loss
        # Convergence test
        if n_iter == min_iter and conv_test_baseline == 'min-iter':
            conv_test_baseline = loss
        if n_iter >= min_iter and tol > 0 and n_iter % conv_test_freq == 0:
            relative_loss_change = (loss_previous_conv_test - loss) / conv_test_baseline
            if (loss <= loss_previous_conv_test) and (relative_loss_change <= tol):
                converged = True
            else:
                converged = False
            if verbose:
                print('Epoch %02d reached. Loss: %.3g. Loss in the previous convergence test: %.3g. '
                      'Baseline: %.3g. Relative loss change: %.3g' %
                      (n_iter, loss, loss_previous_conv_test, conv_test_baseline, relative_loss_change))
            loss_previous_conv_test = loss
        # If converged, stop
        if converged and n_iter >= min_iter:
            break

    losses = np.array(losses)
    reconstruction_errors = np.array(reconstruction_errors)
    volumes = np.array(volumes)
    line_search_steps = np.array(line_search_steps)
    gammas = np.array(gammas)

    return W, H, n_iter, converged, Lambda, losses, reconstruction_errors, volumes, line_search_steps, gammas


class MVNMF:
    def __init__(self,
                 X,
                 n_components,
                 init='cluster',
                 init_W_custom=None,
                 init_H_custom=None,
                 lambda_tilde=1e-5,
                 delta=1.0,
                 gamma=1.0,
                 max_iter=200,
                 min_iter=100,
                 tol=1e-4,
                 conv_test_freq=10,
                 conv_test_baseline=None,
                 verbose=0
                 ):
        if (type(X) != np.ndarray) or (not np.issubdtype(X.dtype, np.floating)):
            X = np.array(X).astype(float)
        self.X = X
        self.n_components = n_components
        self.init = init
        if init_W_custom is not None:
            if (type(init_W_custom) != np.ndarray) or (not np.issubdtype(init_W_custom.dtype, np.floating)):
                init_W_custom = np.array(init_W_custom).astype(float)
        if init_H_custom is not None:
            if (type(init_H_custom) != np.ndarray) or (not np.issubdtype(init_H_custom.dtype, np.floating)):
                init_H_custom = np.array(init_H_custom).astype(float)
        self.init_W_custom = init_W_custom
        self.init_H_custom = init_H_custom
        self.lambda_tilde = lambda_tilde
        self.delta = delta
        self.gamma = gamma
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.tol = tol
        self.conv_test_freq = conv_test_freq
        self.conv_test_baseline = conv_test_baseline
        self.verbose = verbose

    def fit(self):
        W_init, H_init = initialize_nmf(self.X, self.n_components,
                                        init=self.init,
                                        init_W_custom=self.init_W_custom,
                                        init_H_custom=self.init_H_custom)
        self.W_init = W_init
        self.H_init = H_init

        (_W, _H, n_iter, converged, Lambda, losses, reconstruction_errors,
            volumes, line_search_steps, gammas) = _solve_mvnmf(
            X=self.X, W=self.W_init, H=self.H_init, lambda_tilde=self.lambda_tilde,
            delta=self.delta, gamma=self.gamma, max_iter=self.max_iter,
            min_iter=self.min_iter, tol=self.tol,
            conv_test_freq=self.conv_test_freq,
            conv_test_baseline=self.conv_test_baseline,
            verbose=self.verbose)
        self.n_iter = n_iter
        self.converged = converged
        self.Lambda = Lambda
        # Normalize W and perform NNLS to recalculate H
        W = normalize(_W, norm='l1', axis=0)
        H = nnls(self.X, W)
        #
        self._W = _W
        self._H = _H
        self._loss = losses[-1]
        self._reconstruction_error = reconstruction_errors[-1]
        self._volume = volumes[-1]
        #
        self.W = W
        self.H = H
        loss, reconstruction_error, volume = _loss_mvnmf(self.X, self.W, self.H, self.Lambda, self.delta)
        self.loss = loss
        self.reconstruction_error = reconstruction_error
        self.volume = volume
        
        #self.loss_track = losses
        #self.reconstruction_error_track = reconstruction_errors
        #self.volume_track = volumes
        #self.line_search_step_track = line_search_steps
        #self.gamma_track = gammas

        return self