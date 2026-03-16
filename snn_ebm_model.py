"""
Continuous SNN + Energy-Based Model Hybrid Architecture in Brian2

This module implements a hybrid continuous Spiking Neural Network (SNN) with 
per-neuron Energy-Based Models (EBM) as specified in architecture.md.

Key features:
- Continuous spike events with strength ∈ [0, 1]
- Per-neuron Hopfield-style energy function
- STDP + EBM gradient learning
- Distance-dependent sparse connectivity
- Energy transfer mechanism
"""

import numpy as np
from brian2 import *

# Store Brian2's seed function before it gets shadowed by parameter names
brian2_seed = seed


# =============================================================================
# Energy-Based Model Functions (Per-Neuron)
# =============================================================================

class PerNeuronEBM:
    """
    Per-neuron Energy-Based Model with Modern Hopfield dynamics.
    
    Each neuron maintains its own energy function and memory patterns.
    Energy function: E_i(q) = -(1/β) * log(Σ_k exp(β * x_k^T * q)) + (λ/2) * ||q||²
    """
    
    def __init__(self, n_neurons, state_dim, n_patterns=5, beta=5.0, lambda_reg=0.01):
        """
        Initialize per-neuron EBM.
        
        Parameters
        ----------
        n_neurons : int
            Number of neurons in the layer
        state_dim : int
            Dimension of neuron state vector q (e.g., [V, a_pre, a_post, ...])
        n_patterns : int
            Number of stored memory patterns per neuron
        beta : float
            Inverse temperature (controls sharpness of energy landscape)
        lambda_reg : float
            Regularization coefficient for ||q||² term
        """
        self.n_neurons = n_neurons
        self.state_dim = state_dim
        self.n_patterns = n_patterns
        self.beta = beta
        self.lambda_reg = lambda_reg
        
        # Initialize memory patterns for each neuron (random orthonormal patterns)
        self.memory_patterns = self._initialize_memory_patterns()
        
        # Precompute pattern matrix for efficiency
        # X_i shape: (state_dim, n_patterns) for each neuron
        self.X = self.memory_patterns  # Shape: (n_neurons, state_dim, n_patterns)
    
    def _initialize_memory_patterns(self):
        """Initialize random orthonormal memory patterns for each neuron."""
        patterns = np.zeros((self.n_neurons, self.state_dim, self.n_patterns))
        for i in range(self.n_neurons):
            # Random patterns - use fewer patterns than state_dim for QR
            n_actual_patterns = min(self.n_patterns, self.state_dim)
            raw = np.random.randn(self.state_dim, n_actual_patterns)
            patterns[i, :, :n_actual_patterns], _ = np.linalg.qr(raw)
        return patterns
    
    def compute_energy(self, q, neuron_idx=None):
        """
        Compute energy E_i(q) for given state vector(s).

        Parameters
        ----------
        q : ndarray
            State vector(s), shape (state_dim,) or (n_neurons, state_dim)
        neuron_idx : int or None
            Specific neuron index, or None for all neurons

        Returns
        -------
        E : ndarray
            Energy value(s)
        """
        if q.ndim == 1:
            q = q.reshape(1, -1)

        n = q.shape[0] if neuron_idx is None else 1
        E = np.zeros(n)

        if neuron_idx is not None:
            X_i = self.X[neuron_idx]  # (state_dim, n_patterns)
            # LogSumExp energy: E = -(1/β) * log(Σ_k exp(β * x_k^T * q))
            logits = X_i.T @ q.T  # (n_patterns, n)
            max_logit = np.max(logits, axis=0, keepdims=True)
            logsumexp = max_logit + np.log(np.sum(np.exp(logits - max_logit), axis=0))
            E = -(1.0 / self.beta) * logsumexp

            # Regularization term
            E += (self.lambda_reg / 2) * np.sum(q ** 2, axis=1)
        else:
            # Batch computation for all neurons
            for i in range(n):
                X_i = self.X[i]
                logits = X_i.T @ q[i].T
                max_logit = np.max(logits)
                logsumexp = max_logit + np.log(np.sum(np.exp(logits - max_logit)))
                E[i] = -(1.0 / self.beta) * logsumexp
                E[i] += (self.lambda_reg / 2) * np.sum(q[i] ** 2)

        return E
    
    def compute_energy_gradient(self, q, neuron_idx=None):
        """
        Compute energy gradient ∂E/∂q for given state vector(s).

        ∂E/∂q = -softmax(β * X^T * q) @ X + λ * q

        Parameters
        ----------
        q : ndarray
            State vector(s), shape (state_dim,) or (n_neurons, state_dim)
        neuron_idx : int or None
            Specific neuron index, or None for all neurons

        Returns
        -------
        dE_dq : ndarray
            Gradient vector(s), same shape as q
        """
        if q.ndim == 1:
            q = q.reshape(1, -1)

        n = q.shape[0]
        dE_dq = np.zeros_like(q)

        if neuron_idx is not None:
            X_i = self.X[neuron_idx]  # (state_dim, n_patterns)
            logits = X_i.T @ q[0]  # (n_patterns,)

            # Softmax attention weights
            exp_logits = np.exp(self.beta * (logits - np.max(logits)))
            softmax_weights = exp_logits / (np.sum(exp_logits) + 1e-10)

            # Gradient: -X @ softmax + λ * q
            dE_dq[0] = -X_i @ softmax_weights + self.lambda_reg * q[0]
        else:
            for i in range(n):
                X_i = self.X[i]
                logits = X_i.T @ q[i]

                exp_logits = np.exp(self.beta * (logits - np.max(logits)))
                softmax_weights = exp_logits / (np.sum(exp_logits) + 1e-10)

                dE_dq[i] = -X_i @ softmax_weights + self.lambda_reg * q[i]

        return dE_dq
    
    def compute_dE_dW(self, q, spikes_pre, neuron_idx):
        """
        Compute energy gradient w.r.t. synaptic weights: ∂E/∂W_ij
        
        ∂E/∂W_ij = -spike_j * softmax(β * X_i^T * q_i) + λ * W_ij
        
        Parameters
        ----------
        q : ndarray
            State vector for neuron i, shape (state_dim,)
        spikes_pre : ndarray
            Spike strengths from presynaptic neurons
        neuron_idx : int
            Postsynaptic neuron index
            
        Returns
        -------
        dE_dW : ndarray
            Gradient w.r.t. incoming weights, shape (n_pre,)
        """
        X_i = self.X[neuron_idx]
        logits = X_i.T @ q
        
        exp_logits = np.exp(self.beta * (logits - np.max(logits)))
        softmax_weights = exp_logits / (np.sum(exp_logits) + 1e-10)
        
        # Gradient term from energy
        dE_dW = -spikes_pre * softmax_weights[:len(spikes_pre)] if len(spikes_pre) <= self.n_patterns else \
                -spikes_pre * np.mean(softmax_weights)
        
        return dE_dW
    
    def retrieve_pattern(self, q_init, neuron_idx=0, n_steps=10, dt=0.1):
        """
        Perform energy-based retrieval (Hopfield dynamics).
        
        dq/dt = -∂E/∂q
        
        Parameters
        ----------
        q_init : ndarray
            Initial state vector
        neuron_idx : int
            Which neuron's energy landscape to use for retrieval
        n_steps : int
            Number of retrieval steps
        dt : float
            Step size
            
        Returns
        -------
        q_final : ndarray
            Retrieved state after energy minimization
        """
        q = q_init.copy()
        for _ in range(n_steps):
            grad = self.compute_energy_gradient(q.reshape(1, -1), neuron_idx=neuron_idx)[0]
            q = q - dt * grad
        return q


# =============================================================================
# Distance-Dependent Connectivity
# =============================================================================

def create_distance_matrix(n_neurons, positions=None, space_dim=2, grid_size=10.0):
    """
    Create distance matrix for neurons in metric space.
    
    Parameters
    ----------
    n_neurons : int
        Number of neurons
    positions : ndarray, optional
        Predefined positions, shape (n_neurons, space_dim)
    space_dim : int
        Dimension of space (1, 2, or 3)
    grid_size : float
        Size of the space
        
    Returns
    -------
    positions : ndarray
        Neuron positions
    distance_matrix : ndarray
        Pairwise distance matrix, shape (n_neurons, n_neurons)
    """
    if positions is None:
        positions = np.random.rand(n_neurons, space_dim) * grid_size
    
    # Compute pairwise Euclidean distance using numpy broadcasting
    # ||a - b||² = ||a||² + ||b||² - 2*a·b
    sq_norms = np.sum(positions ** 2, axis=1)
    distance_matrix = np.sqrt(
        np.maximum(
            sq_norms.reshape(-1, 1) + sq_norms.reshape(1, -1) - 2 * positions @ positions.T,
            0  # Avoid negative due to floating point errors
        )
    )
    return positions, distance_matrix


def create_sparse_connectivity(distance_matrix, p0=0.3, d0=5.0, max_range=None):
    """
    Create sparse connectivity based on distance-dependent probability.
    
    P_connect(i, j) = p0 * exp(-distance(i, j) / d0)
    
    Parameters
    ----------
    distance_matrix : ndarray
        Pairwise distance matrix
    p0 : float
        Base connection probability at zero distance
    d0 : float
        Distance decay constant
    max_range : float, optional
        Maximum connection range (None for no cutoff)
        
    Returns
    -------
    conn_matrix : ndarray
        Binary connectivity matrix (True = connected)
    conn_weights : ndarray
        Connection weights (distance-dependent)
    """
    n = distance_matrix.shape[0]
    
    # Connection probability matrix
    prob_matrix = p0 * np.exp(-distance_matrix / d0)
    np.fill_diagonal(prob_matrix, 0)  # No self-connections
    
    # Random sampling for connectivity
    rand_matrix = np.random.rand(n, n)
    conn_matrix = rand_matrix < prob_matrix
    
    if max_range is not None:
        conn_matrix[distance_matrix > max_range] = False
    
    # Base weights decay with distance
    conn_weights = np.exp(-distance_matrix / d0) * conn_matrix
    
    return conn_matrix, conn_weights


# =============================================================================
# Brian2 Network Model
# =============================================================================

class ContinuousLIFEBM:
    """
    Continuous LIF + Per-Neuron EBM Hybrid Network in Brian2.
    
    Features:
    - Standard LIF membrane dynamics
    - Continuous spike output (graded strength based on suprathreshold voltage)
    - Per-neuron EBM for memory storage
    - STDP + EBM gradient learning
    - Support for excitatory and inhibitory connections
    """
    
    def __init__(self, n_neurons=100, n_layers=1, neurons_per_layer=None,
                 tau=15*ms, tau_tr1=20*ms, tau_tr2=20*ms,
                 eta_stdp=0.005, mu_ebm=0.5, lambda_decay=0.0005,
                 beta=2.0, alpha=0.05, gamma=0.1,
                 space_dim=2, grid_size=10.0, p0=0.3, d0=5.0,
                 n_patterns=5, seed=None,
                 v_threshold=0.5, v_reset=0.0, v_scale=2.0, tau_refrac=5*ms,
                 homeostasis=True, target_rate=5*Hz,
                 reward_modulation=True, eta_reward=0.1,
                 error_driven=True, eta_error=0.05):
        """
        Initialize the continuous LIF-EBM network.

        Parameters
        ----------
        n_neurons : int
            Total neurons (if single layer) or neurons per layer
        n_layers : int
            Number of layers
        neurons_per_layer : list, optional
            Override n_neurons with specific layer sizes
        tau : Quantity
            Membrane time constant
        tau_tr1 : Quantity
            STDP trace 1 decay
        tau_tr2 : Quantity
            STDP trace 2 decay
        eta_stdp : float
            STDP learning rate
        mu_ebm : float
            EBM gradient modulation strength
        lambda_decay : float
            Weight decay coefficient
        beta : float
            EBM inverse temperature
        alpha : float
            Energy transfer coefficient
        gamma : float
            Energy gradient coupling to membrane potential
        space_dim : int
            Dimension of neuron embedding space
        grid_size : float
            Size of the embedding space
        p0 : float
            Base connection probability
        d0 : float
            Distance decay for connectivity
        n_patterns : int
            Number of memory patterns per neuron
        seed : int, optional
            Random seed for reproducibility
        v_threshold : float
            Spike threshold
        v_reset : float
            Reset potential after spike
        v_scale : float
            Scaling for continuous spike output
        tau_refrac : Quantity
            Refractory period
        """
        if seed is not None:
            np.random.seed(seed)
            brian2_seed(seed)
        
        self.n_layers = n_layers
        self.neurons_per_layer = neurons_per_layer if neurons_per_layer else [n_neurons] * n_layers
        self.n_neurons_total = sum(self.neurons_per_layer)
        
        # LIF parameters
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.v_scale = v_scale
        self.tau_refrac = tau_refrac
        
        # Time constants
        self.tau = tau
        self.tau_tr1 = tau_tr1
        self.tau_tr2 = tau_tr2
        
        # Learning parameters
        self.eta_stdp = eta_stdp
        self.mu_ebm = mu_ebm
        self.lambda_decay = lambda_decay
        
        # EBM parameters
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        
        # Connectivity parameters
        self.space_dim = space_dim
        self.grid_size = grid_size
        self.p0 = p0
        self.d0 = d0

        # Homeostasis parameters
        self.homeostasis = homeostasis
        self.target_rate = target_rate

        # Reward-modulated plasticity parameters
        self.reward_modulation = reward_modulation
        self.eta_reward = eta_reward
        self.eligibility_trace = None  # Will be initialized in _build_network

        # Error-driven learning parameters
        self.error_driven = error_driven
        self.eta_error = eta_error
        self.prediction_error = None  # Will be computed during learning

        # State dimension for EBM: [V, a_pre, a_post]
        self.state_dim = 3

        # Initialize per-neuron EBM
        self.ebm = PerNeuronEBM(
            n_neurons=self.n_neurons_total,
            state_dim=self.state_dim,
            n_patterns=n_patterns,
            beta=beta,
            lambda_reg=lambda_decay
        )
        
        # Create neuron positions and distance matrix
        self.positions, self.distance_matrix = create_distance_matrix(
            self.n_neurons_total, space_dim=space_dim, grid_size=grid_size
        )
        
        # Create sparse connectivity
        self.conn_matrix, self.conn_weights = create_sparse_connectivity(
            self.distance_matrix, p0=p0, d0=d0
        )
        
        # Build Brian2 network
        self._build_network()
    
    def _build_network(self):
        """Build the Brian2 NeuronGroup and Synapses."""
        
        # Continuous LIF equations with homeostasis
        # - Standard LIF membrane dynamics
        # - Continuous spike output based on suprathreshold voltage
        # - Homeostatic inhibition based on firing rate
        neuron_eqs = '''
        dV/dt = (-V + I_syn + I_EBM + I_ext + I_inh_homeo) / tau : 1 (unless refractory)
        dtrace1/dt = -trace1 / tau_tr1 : 1
        dtrace2/dt = -trace2 / tau_tr2 : 1
        drate/dt = (spike_output - rate) / (100*ms) : 1  # Running average firing rate
        I_syn : 1
        I_EBM : 1
        I_ext : 1
        I_inh_homeo : 1  # Homeostatic inhibition current
        influence : 1  # Energy-based influence factor (high energy = low influence)
        # Continuous spike strength: how much V exceeds threshold
        spike_output = clip((V - v_thresh) / v_scale, 0, 1) : 1
        v_thresh : 1
        v_scale : 1
        v_reset : 1
        tau : second
        tau_tr1 : second
        tau_tr2 : second
        '''
        
        # Create neuron group with continuous LIF
        self.neurons = NeuronGroup(
            self.n_neurons_total,
            neuron_eqs,
            threshold='V > v_thresh',
            reset='V = v_reset; trace1 += 1',
            refractory=self.tau_refrac,
            method='euler',
            name='continuous_lif'
        )

        # Initialize LIF parameters
        self.neurons.v_thresh = self.v_threshold
        self.neurons.v_scale = self.v_scale
        self.neurons.v_reset = self.v_reset

        # Initialize parameters
        self.neurons.tau = self.tau
        self.neurons.tau_tr1 = self.tau_tr1
        self.neurons.tau_tr2 = self.tau_tr2
        self.neurons.V = 'rand() * 0.1'  # Small random initial voltage
        self.neurons.trace1 = 0
        self.neurons.trace2 = 0
        self.neurons.rate = 0
        self.neurons.I_syn = 0
        self.neurons.I_EBM = 0
        self.neurons.I_ext = 0
        self.neurons.I_inh_homeo = 0
        self.neurons.influence = 1.0  # Energy-based influence factor (updated externally)

        # Synapse equations with reward-modulated plasticity and eligibility traces
        synapse_eqs = '''
        # Weight update with STDP, EBM, decay, and reward modulation.
        # Decay pulls W toward W_target (not toward 0). Without this, excitatory
        # weights erode to 0 over epochs and recall with partial cues fails.
        dW/dt = (eta_stdp * (trace1 * spike_output_post - trace2 * spike_output_pre)
                + mu_ebm * dE_dW - lambda_decay * (W - W_target)) * 1000*Hz : 1 (clock-driven)
        # Eligibility trace: stores recent STDP activity for reward modulation
        deligibility/dt = -eligibility / (50*ms) + (trace1 * spike_output_post - trace2 * spike_output_pre) * 1000*Hz : 1
        dE_dW : 1  # EBM gradient (updated externally)
        eta_stdp : 1
        mu_ebm : 1
        lambda_decay : 1
        W_min : 1
        W_max : 1
        W_target : 1  # Target weight for normalization
        '''

        # Synaptic transmission modulated by presynaptic influence factor
        # High-energy neurons have reduced impact
        on_pre_eqs = '''
        I_syn_post += W * spike_output_pre * influence_pre
        '''
        
        # Create synapses based on connectivity matrix
        pre_indices, post_indices = np.where(self.conn_matrix)

        self.synapses = Synapses(
            self.neurons, self.neurons,
            synapse_eqs,
            on_pre=on_pre_eqs,
            name='stdp_ebm'
        )

        self.synapses.connect(i=pre_indices, j=post_indices)

        # Initialize weights with balanced E/I for stability
        # Start with small random weights centered around zero
        initial_weights = np.random.randn(len(pre_indices)) * 0.2
        
        # Ensure roughly 60% excitatory, 40% inhibitory for balance
        inhibitory_mask = np.random.rand(len(initial_weights)) < 0.4
        initial_weights[inhibitory_mask] = -np.abs(initial_weights[inhibitory_mask]) * 0.5  # Weaker inhibition
        initial_weights[~inhibitory_mask] = np.abs(initial_weights[~inhibitory_mask]) * 0.5  # Moderate excitation

        self.synapses.W = initial_weights
        self.synapses.dE_dW = 0
        # Tighter bounds to prevent runaway excitation
        self.synapses.W_min = np.full(len(self.synapses), -0.3)
        self.synapses.W_max = np.full(len(self.synapses), 0.5)
        # W_target is sign-dependent: excitatory synapses pulled toward +0.15,
        # inhibitory toward -0.10. A uniform value of 0.25 would drag inhibitory
        # weights toward positive territory, destroying E/I balance.
        w_targets = np.where(initial_weights >= 0, 0.15, -0.10)
        self.synapses.W_target = w_targets

        # Learning parameters
        self.synapses.eta_stdp = self.eta_stdp
        self.synapses.mu_ebm = self.mu_ebm
        self.synapses.lambda_decay = self.lambda_decay

        # Initialize eligibility traces to zero
        self.synapses.eligibility = 0

        # Store eligibility trace for reward modulation
        self.eligibility_trace = np.zeros(len(self.synapses))

        # Spike monitors
        self.spike_monitor = SpikeMonitor(self.neurons, record=True)
        self.state_monitor = StateMonitor(
            self.neurons, ['V', 'spike_output', 'I_syn', 'I_EBM'],
            record=[0]  # Record only first neuron for efficiency
        )
        
        # Create explicit network
        self.net = Network(self.neurons, self.synapses, self.spike_monitor, self.state_monitor)
    
    def update_ebm_gradients(self):
        """
        Update EBM gradients for all neurons and homeostatic inhibition.
        Also enforce weight bounds to prevent instability.
        
        High-energy neurons have reduced influence on stable neurons.
        """
        # Get current neuron states
        V = self.neurons.V[:]
        trace1 = self.neurons.trace1[:]
        trace2 = self.neurons.trace2[:]
        rate = self.neurons.rate[:]

        # Build state vectors q_i = [V_i, trace1_i, trace2_i]
        Q = np.column_stack([V, trace1, trace2])  # (n_neurons, state_dim)

        # Compute energy for each neuron
        energies = np.array([self.ebm.compute_energy(Q[i], neuron_idx=i)[0]
                            for i in range(self.n_neurons_total)])
        
        # Compute energy-based influence modulation
        # High energy = less influence (exponential decay)
        # Normalize energies to [0, 1] range for modulation
        E_min = np.min(energies)
        E_max = np.max(energies)
        E_normalized = (energies - E_min) / (E_max - E_min + 1e-10)
        
        # Influence factor: exp(-k * E_normalized) where k controls strength
        # High energy neurons have ~0.3x influence, low energy have ~1.0x
        influence_factor = np.exp(-2.0 * E_normalized).flatten()
        
        # Store influence factor for use in synaptic transmission
        self.influence_factor = influence_factor
        
        # Update neuron influence variable
        self.neurons.influence = influence_factor

        # Compute energy gradients for each neuron
        dE_dq = self.ebm.compute_energy_gradient(Q)

        # Update I_EBM = -γ * ∂E/∂V (gradient affects membrane potential)
        # Reduced coupling for stability
        self.neurons.I_EBM = -self.gamma * dE_dq[:, 0]

        # Homeostatic inhibition: neurons firing above target rate get inhibitory current
        if self.homeostasis:
            target_rate = float(self.target_rate) / 1000  # Convert Hz to rate
            # Proportional control: more firing = more inhibition
            self.neurons.I_inh_homeo = -0.3 * np.maximum(rate - target_rate, 0)

        # Compute dE/dW for each synapse with bounded gradient
        # Modulate by presynaptic influence factor
        spike_outputs = 1.0 / (1.0 + np.exp(-np.clip(V, -10, 10)))

        for syn_idx in range(len(self.synapses)):
            pre_idx = self.synapses.i[syn_idx]
            post_idx = self.synapses.j[syn_idx]

            # Get presynaptic spike strength modulated by influence
            spike_pre = spike_outputs[pre_idx] * influence_factor[pre_idx]

            # Compute dE/dW for postsynaptic neuron - bounded to prevent explosion
            dE_dW = self.ebm.compute_dE_dW(Q[post_idx], np.array([spike_pre]), post_idx)

            # Clamp the EBM gradient to prevent destabilization
            dE_dW = np.clip(dE_dW, -0.1, 0.1)

            self.synapses.dE_dW[syn_idx] = dE_dW if np.isscalar(dE_dW) else dE_dW[0]

        # Enforce hard weight bounds after gradient update
        W = self.synapses.W[:]
        W = np.clip(W, self.synapses.W_min, self.synapses.W_max)
        self.synapses.W = W

    def normalize_weights(self, target_mean=0.25):
        """
        Normalize weights to maintain E/I balance and prevent runaway dynamics.
        
        This scales all weights to maintain a target mean while preserving
        the sign (excitatory vs inhibitory) of each synapse.
        """
        W = self.synapses.W[:]
        
        # Get bounds as scalars (they're uniform arrays)
        w_min = self.synapses.W_min[0]
        w_max = self.synapses.W_max[0]
        
        # Separate excitatory and inhibitory
        exc_mask = W > 0
        inh_mask = W < 0
        
        # Normalize excitatory weights
        if np.any(exc_mask):
            exc_W = W[exc_mask]
            current_mean = np.mean(exc_W)
            if current_mean > 0:
                # Scale to target
                exc_W = exc_W * (target_mean / current_mean)
                # Clip to bounds
                exc_W = np.clip(exc_W, 0, w_max)
                W[exc_mask] = exc_W
        
        # Normalize inhibitory weights (keep them weaker)
        if np.any(inh_mask):
            inh_W = W[inh_mask]
            current_mean = np.mean(np.abs(inh_W))
            target_inh = target_mean * 0.5  # Inhibitory weights are weaker
            if current_mean > 0:
                inh_W = inh_W * (target_inh / current_mean)
                inh_W = np.clip(inh_W, w_min, 0)
                W[inh_mask] = inh_W
        
        self.synapses.W = W

    def apply_reward(self, reward):
        """
        Apply reward-modulated plasticity to update weights.
        
        Reward-modulated STDP: ΔW = η_reward * reward * eligibility_trace
        
        Parameters
        ----------
        reward : float
            Reward signal in range [-1, 1]
            - Positive reward: strengthen synapses with positive eligibility
            - Negative reward: weaken synapses with positive eligibility
        """
        if not self.reward_modulation:
            return
        
        # Get current eligibility traces from synapses
        eligibility = self.synapses.eligibility[:]
        
        # Reward-modulated update: ΔW = η * r * e
        dW_reward = self.eta_reward * reward * eligibility
        
        # Apply to weights
        W = self.synapses.W[:]
        W = W + dW_reward
        
        # Clip to bounds
        W = np.clip(W, self.synapses.W_min, self.synapses.W_max)
        self.synapses.W = W
        
        # Reset eligibility traces after reward is applied
        self.synapses.eligibility = 0

    def compute_prediction_error(self, target_activity, actual_activity):
        """
        Compute prediction error for error-driven learning.
        
        Parameters
        ----------
        target_activity : ndarray
            Target activity pattern (what should fire)
        actual_activity : ndarray
            Actual neuron activity (what did fire)
        
        Returns
        -------
        error : ndarray
            Prediction error (target - actual)
        """
        if not self.error_driven:
            return None
        
        self.prediction_error = target_activity - actual_activity
        return self.prediction_error

    def apply_error_driven_learning(self, error):
        """
        Apply error-driven learning to adjust weights.
        
        Error-driven update: ΔW_ij = η_error * error_i * spike_output_j
        
        This strengthens connections from active neurons to neurons that
        should have fired but didn't (positive error), and weakens
        connections to neurons that fired but shouldn't have (negative error).
        
        Parameters
        ----------
        error : ndarray
            Prediction error for each neuron
        """
        if not self.error_driven:
            return
        
        # Get spike outputs
        V = self.neurons.V[:]
        spike_outputs = 1.0 / (1.0 + np.exp(-np.clip(V, -10, 10)))
        
        # Compute error-driven weight updates
        W = self.synapses.W[:]
        pre_indices = self.synapses.i[:]
        post_indices = self.synapses.j[:]
        
        for syn_idx in range(len(self.synapses)):
            pre_idx = pre_indices[syn_idx]
            post_idx = post_indices[syn_idx]
            
            # Error-driven update: ΔW = η * error_post * spike_pre
            dW_error = self.eta_error * error[post_idx] * spike_outputs[pre_idx]
            W[syn_idx] += dW_error
        
        # Clip to bounds
        W = np.clip(W, self.synapses.W_min, self.synapses.W_max)
        self.synapses.W = W

    def set_input(self, input_current, neuron_indices=None):
        """
        Set external input current to specified neurons.
        
        Parameters
        ----------
        input_current : ndarray or float
            Input current values
        neuron_indices : list or None
            Indices of neurons to receive input (None = all)
        """
        if neuron_indices is None:
            neuron_indices = np.arange(self.n_neurons_total)
        
        if np.isscalar(input_current):
            input_current = np.ones(len(neuron_indices)) * input_current
        
        self.neurons.I_ext[neuron_indices] = input_current
    
    def get_output(self, layer_idx=None):
        """
        Get continuous spike output from neurons.
        
        Parameters
        ----------
        layer_idx : int or None
            Layer index (None = all neurons)
            
        Returns
        -------
        spike_outputs : ndarray
            Continuous spike strengths ∈ [0, 1]
        """
        V = self.neurons.V[:]
        spike_outputs = 1.0 / (1.0 + np.exp(-V))
        
        if layer_idx is not None:
            start = sum(self.neurons_per_layer[:layer_idx])
            end = start + self.neurons_per_layer[layer_idx]
            return spike_outputs[start:end]
        
        return spike_outputs
    
    def run(self, duration, dt=0.5*ms, update_ebm_every=5):
        """
        Run the simulation.
        
        Parameters
        ----------
        duration : Quantity
            Simulation duration
        dt : Quantity
            Simulation timestep
        update_ebm_every : int
            Update EBM gradients every N steps
            
        Returns
        -------
        results : dict
            Simulation results (spike times, states, etc.)
        """
        # Set the simulation timestep
        defaultclock.dt = dt
        
        # Run simulation using explicit network
        self.net.run(duration, report='text')
        
        # Update EBM gradients at the end (for demonstration)
        self.update_ebm_gradients()
        
        # Compile results
        results = {
            'spike_times': self.spike_monitor.spike_trains(),
            'spike_count': self.spike_monitor.count,
            'V': self.state_monitor.V,
            'spike_output': self.state_monitor.spike_output,
            'I_syn': self.state_monitor.I_syn,
            'I_EBM': self.state_monitor.I_EBM,
            't': self.state_monitor.t,
            'weights': self.synapses.W[:],
            'positions': self.positions
        }
        
        return results
    
    def get_energy_stats(self):
        """
        Compute energy statistics for the network.
        
        Returns
        -------
        stats : dict
            Energy statistics
        """
        V = self.neurons.V[:]
        trace1 = self.neurons.trace1[:]
        trace2 = self.neurons.trace2[:]
        Q = np.column_stack([V, trace1, trace2])
        
        energies = np.array([self.ebm.compute_energy(Q[i], neuron_idx=i)[0] 
                            for i in range(self.n_neurons_total)])
        
        return {
            'mean_energy': np.mean(energies),
            'std_energy': np.std(energies),
            'min_energy': np.min(energies),
            'max_energy': np.max(energies),
            'total_energy': np.sum(energies)
        }
    
    def plot_network(self, ax=None, show_weights=True):
        """
        Visualize the network structure.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on
        show_weights : bool
            Show synaptic weights as connection lines
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.collections import LineCollection
        except ImportError:
            print("Matplotlib not available")
            return
        
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        pos = self.positions
        
        # Draw connections
        if show_weights and len(self.synapses) > 0:
            weights = self.synapses.W[:]
            lines = []
            colors = []
            
            for i in range(len(self.synapses)):
                pre_idx = self.synapses.i[i]
                post_idx = self.synapses.j[i]
                w = weights[i]
                
                lines.append([(pos[pre_idx, 0], pos[pre_idx, 1]),
                             (pos[post_idx, 0], pos[post_idx, 1])])
                
                # Red for excitatory, blue for inhibitory
                if w > 0:
                    colors.append([1, 0, 0, min(abs(w) / 5, 0.5)])  # Red, alpha scaled
                else:
                    colors.append([0, 0, 1, min(abs(w) / 5, 0.5)])  # Blue, alpha scaled
            
            lc = LineCollection(lines, colors=colors, linewidths=0.5)
            ax.add_collection(lc)
        
        # Draw neurons
        scatter = ax.scatter(pos[:, 0], pos[:, 1], c='green', s=50, 
                            alpha=0.7, edgecolors='white', linewidth=0.5)
        
        ax.set_xlim(-1, self.grid_size + 1)
        ax.set_ylim(-1, self.grid_size + 1)
        ax.set_aspect('equal')
        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
        ax.set_title(f'Network Structure ({self.n_neurons_total} neurons, {len(self.synapses)} synapses)')
    
    def plot_activity(self, results, ax_v=None, ax_spike=None):
        """
        Plot membrane potential and spike activity.
        
        Parameters
        ----------
        results : dict
            Simulation results from run()
        ax_v : matplotlib.axes.Axes, optional
            Axes for voltage trace
        ax_spike : matplotlib.axes.Axes, optional
            Axes for spike raster
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available")
            return
        
        if ax_v is None or ax_spike is None:
            fig, (ax_v, ax_spike) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        
        # Voltage trace for sample neurons
        t = results['t'] / ms
        sample_neurons = [0, 5, 10]
        
        for neuron_id in sample_neurons:
            if neuron_id < len(results['V']):
                ax_v.plot(t, results['V'][neuron_id], label=f'Neuron {neuron_id}')
        
        ax_v.axhline(y=self.v_threshold, color='r', linestyle='--', label='Threshold')
        ax_v.set_ylabel('Membrane Potential V')
        ax_v.set_title('Membrane Potential Dynamics')
        ax_v.legend(loc='upper right')
        ax_v.grid(True, alpha=0.3)
        
        # Spike raster
        spike_trains = results['spike_times']
        for neuron_id, times in spike_trains.items():
            if len(times) > 0:
                ax_spike.vlines(times / ms, neuron_id - 0.5, neuron_id + 0.5, 
                               color='blue', alpha=0.5, linewidth=0.5)
        
        ax_spike.set_xlabel('Time (ms)')
        ax_spike.set_ylabel('Neuron ID')
        ax_spike.set_title('Spike Raster Plot')
        ax_spike.set_ylim(-0.5, self.n_neurons_total - 0.5)
    
    def plot_weight_distribution(self, ax=None):
        """Plot synaptic weight distribution."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available")
            return
        
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        
        weights = self.synapses.W[:]
        
        # Separate excitatory and inhibitory
        exc_weights = weights[weights > 0]
        inh_weights = weights[weights < 0]
        
        ax.hist(exc_weights, bins=30, alpha=0.7, color='red', label='Excitatory')
        ax.hist(inh_weights, bins=30, alpha=0.7, color='blue', label='Inhibitory')
        
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('Synaptic Weight')
        ax.set_ylabel('Count')
        ax.set_title(f'Weight Distribution (mean={np.mean(weights):.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)


# =============================================================================
# Example Usage and Testing
# =============================================================================

def run_example():
    """Run a simple example simulation."""
    print("Initializing Continuous SNN + EBM Network...")
    
    # Create network
    net = ContinuousSNNEBM(
        n_neurons=50,
        n_layers=1,
        tau=15*ms,
        eta_stdp=0.005,
        mu_ebm=0.5,
        beta=5.0,
        alpha=0.05,
        p0=0.3,
        seed=42
    )
    
    print(f"Created {net.n_neurons_total} neurons")
    print(f"Created {len(net.synapses)} synapses")
    print(f"Connectivity: {len(net.synapses) / (net.n_neurons_total**2) * 100:.2f}%")
    
    # Apply input to subset of neurons
    input_neurons = np.arange(10)
    net.set_input(2.0, input_neurons)

    print("\nRunning simulation for 100ms...")

    # Run simulation
    results = net.run(100*ms, dt=0.5*ms, update_ebm_every=5)
    
    # Print results
    print(f"\nSimulation complete!")
    print(f"Total spikes: {sum(results['spike_count'])}")
    print(f"Mean firing rate: {sum(results['spike_count']) / net.n_neurons_total / 0.1:.2f} Hz")
    
    energy_stats = net.get_energy_stats()
    print(f"\nEnergy Statistics:")
    print(f"  Mean: {energy_stats['mean_energy']:.4f}")
    print(f"  Std:  {energy_stats['std_energy']:.4f}")
    print(f"  Range: [{energy_stats['min_energy']:.4f}, {energy_stats['max_energy']:.4f}]")
    
    return net, results


def run_memory_task():
    """Run a simple memory pattern task."""
    print("\n" + "="*60)
    print("Memory Pattern Task")
    print("="*60)
    
    # Smaller network for clearer patterns
    net = ContinuousSNNEBM(
        n_neurons=30,
        n_layers=1,
        n_patterns=3,
        beta=10.0,
        mu_ebm=1.0,
        p0=0.5,
        seed=123
    )
    
    # Store a specific pattern
    pattern = np.random.choice([-0.5, 0.5], size=net.state_dim)
    net.ebm.memory_patterns[0, :, 0] = pattern  # Store in first neuron
    
    # Present partial cue
    cue = pattern * 0.3 + np.random.randn(net.state_dim) * 0.1
    net.neurons.V[:5] = cue[0]  # Partial input
    
    print("Running retrieval dynamics...")
    results = net.run(200*ms, dt=0.1*ms, update_ebm_every=5)
    
    # Check final state
    final_V = net.neurons.V[:]
    print(f"Initial cue: V = {cue[0]:.4f}")
    print(f"Final state: V = {final_V[0]:.4f}")
    print(f"Target pattern: V = {pattern[0]:.4f}")
    
    return net, results


if __name__ == "__main__":
    # Run example simulation
    net, results = run_example()
    
    # Optional: run memory task
    # net_mem, results_mem = run_memory_task()
    
    # Plotting (requires matplotlib)
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Spike raster
        ax = axes[0, 0]
        spike_trains = results['spike_times']
        for neuron_id, times in spike_trains.items():
            if len(times) > 0:
                ax.vlines(times, neuron_id - 0.5, neuron_id + 0.5, color='blue', alpha=0.5)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Neuron ID')
        ax.set_title('Spike Raster Plot')
        
        # Membrane potential trace (sample neuron)
        ax = axes[0, 1]
        sample_neuron = 0
        ax.plot(results['t'] / ms, results['V'][sample_neuron], 'b-', label='V')
        ax.plot(results['t'] / ms, results['spike_output'][sample_neuron], 'r--', label='spike output')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Value')
        ax.set_title(f'Neuron {sample_neuron} Dynamics')
        ax.legend()
        
        # Weight distribution
        ax = axes[1, 0]
        ax.hist(results['weights'], bins=30, color='green', alpha=0.7)
        ax.set_xlabel('Weight')
        ax.set_ylabel('Count')
        ax.set_title('Synaptic Weight Distribution')
        
        # Neuron positions
        ax = axes[1, 1]
        pos = results['positions']
        ax.scatter(pos[:, 0], pos[:, 1], c='blue', s=50, alpha=0.6)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Neuron Positions in 2D Space')
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig('snn_ebm_results.png', dpi=150)
        print("\nResults saved to snn_ebm_results.png")
        
    except ImportError:
        print("\nMatplotlib not available - skipping plots")
