# Continuous SNN + Energy-Based Model Architecture

## Overview

This project implements a **hybrid continuous Spiking Neural Network (SNN) with per-neuron Energy-Based Models (EBM)** in **Brian2**. Unlike traditional SNNs that fire binary spikes, this architecture fires **continuous spike events with strength ∈ [0, 1]**, while each neuron minimizes its own Hopfield-style energy function.

---

## Core Concepts

### Traditional SNN vs Continuous SNN vs This Hybrid

| Traditional SNN | Continuous SNN (Rate) | **This Hybrid** |
|----------------|----------------------|-----------------|
| Binary spikes (0 or 1) | Continuous output [0, 1] | **Continuous spike events** ∈ [0, 1] |
| All-or-nothing firing | Graded activation | **Graded spike events** with timing |
| No energy dynamics | No energy dynamics | **Per-neuron EBM** modulates dynamics |
| STDP or backprop | Backprop only | **STDP + EBM gradient** learning |

---

## Architecture Components

### 1. Neuron Model (Continuous SNN)

Each neuron maintains:

**Membrane Potential ODE:**
```
dV_i/dt = -V_i/τ + I_syn_i + I_EBM_i + I_ext_i
```

**Continuous Spike Output:**
```
spike_i = σ(V_i)  where σ(x) = 1 / (1 + exp(-x))  ∈ [0, 1]
```

**Soft Reset (after spike event):**
```
V_i ← V_i - reset_value  (not hard zero — preserves state)
```

**STDP Traces:**
```
da_pre_i/dt = -a_pre_i / τ_pre + spike_i
da_post_i/dt = -a_post_i / τ_post + spike_i
```

**Variables:**
| Symbol | Meaning |
|--------|---------|
| `V_i` | Membrane potential of neuron i |
| `τ` | Membrane time constant (~10-20 ms) |
| `I_syn_i` | Synaptic input current |
| `I_EBM_i` | Energy-based modulation current |
| `I_ext_i` | External input current |
| `spike_i` | Continuous spike strength [0, 1] |
| `a_pre_i`, `a_post_i` | STDP timing traces |

---

### 2. Per-Neuron Energy Function (EBM)

**Each neuron has its own energy function** — no symmetry required between connections.

**Energy Formula (Modern Hopfield / LogSumExp):**
```
E_i(q) = -(1/β) * log(Σ_k exp(β * x_k^T * q)) + (λ/2) * ||q||²
```

**Where:**
| Symbol | Meaning |
|--------|---------|
| `q` | Neuron's internal state vector |
| `x_k` | k-th stored memory pattern for neuron i |
| `β` | Inverse temperature (controls sharpness, ~1-10) |
| `λ` | Regularization coefficient |

**Internal State:**
```
q_i = [V_i, a_pre_i, a_post_i, synaptic_state_i]
```

**Energy Retrieval Dynamics (Continuous Hopfield):**
```
dq_i/dt = -∂E_i/∂q_i + Σ_j W_ij * spike_j
```

**Energy-Modulated Current:**
```
I_EBM_i = -∂E_i/∂V_i  (energy gradient affects membrane potential)
```

**Characteristics:**
- ✧ Exponential memory capacity (~exp(c·d))
- ✧ Single-step rapid convergence
- ✧ Equivalent to transformer attention mechanism
- ✧ **Per-neuron** — no global energy, no symmetry constraint
- ✗ Requires β tuning to avoid metastable states

---

### 3. Network Structure (Brian2)

```
┌─────────────────────────────────────────────────────┐
│              Input Layer                            │
│  (Continuous spike encoding ∈ [0, 1])               │
│  I_ext drives initial spike events                  │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│           Hidden Layer(s)                           │
│  dV/dt = -V/τ + I_syn + I_EBM + I_ext               │
│  spike = sigmoid(V) ∈ [0, 1]                        │
│  Per-neuron E_i(q) energy minimization              │
│  STDP traces: a_pre, a_post                         │
│  Sparse, distance-dependent connectivity            │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│             Output Layer                            │
│  (Continuous spike decoding ∈ [0, 1])               │
│  Readout: weighted sum of spike outputs             │
└─────────────────────────────────────────────────────┘
```

---

### 4. Synapse Model

**Synaptic Current:**
```
I_syn_i = Σ_j W_ij * spike_j
```

**Distance-Dependent Connectivity:**
```
W_ij = W_base * exp(-distance(i, j) / σ)  if connected
W_ij = 0                                   otherwise
```

**Sparse Connectivity:**
- Neurons do **NOT** have all-to-all connections
- Connection probability: `P_connect = p_0 * exp(-distance(i, j) / d_0)`
- Reduces computational cost, mimics biological realism

**Synaptic Dynamics (optional alpha function):**
```
dI_syn/dt = -I_syn / τ_syn
```

---

### 5. Distance-Based Energy Cost

Neurons exist in a **metric space** (1D line, 2D grid, 3D volume, or abstract embedding).

**Energy Cost for Signal Transmission:**
```
E_cost_ij = α * spike_j * distance(i, j)
```

**When neuron j fires onto neuron i:**
```
E_i ← E_i + E_cost_ij  (energy penalty added)
```

**Effect:**
- Farther connections cost more energy
- Long-range connections naturally suppressed unless strongly useful
- Encourages local clustering with selective long-range links

**Distance Formula (example for 2D grid):**
```
distance(i, j) = sqrt((x_i - x_j)² + (y_i - y_j)²)
```

---

### 6. Energy Transfer Mechanism

When neuron `j` fires a spike onto neuron `i`:

```
1. Synaptic current injected:
   I_syn_i += W_ij * spike_j

2. Energy transferred (cost):
   ΔE_i = α * spike_j * distance(i, j)
   E_i ← E_i + ΔE_i

3. Energy gradient modulates membrane potential:
   V_i ← V_i - γ * ∂E_i/∂V_i
```

**Physical Interpretation:**
- Firing a spike **costs energy** proportional to distance
- Receiving neuron's energy landscape is **perturbed**
- Energy gradient **pushes back** on membrane potential
- Creates homeostatic regulation of activity

```
Neuron j ──────→ Neuron i
   │                    │
   │  (spike + energy)  │
   │───────────────────→│
   │←───────────────────│
   │  (energy penalty)  │
```

---

### 7. Learning Rule: STDP + EBM Gradient

**Full Synaptic Plasticity:**
```
dW_ij/dt = η_STDP * (a_pre_i * spike_j - a_post_j * spike_i)
           + μ * ∂E_i/∂W_ij
           - λ_decay * W_ij
```

**Term Breakdown:**

| Term | Formula | Effect |
|------|---------|--------|
| **LTP** | `a_pre_i * spike_j` | Pre fires before post → potentiation |
| **LTD** | `a_post_j * spike_i` | Post fires before pre → depression |
| **EBM Gradient** | `∂E_i/∂W_ij` | Minimize neuron i's energy |
| **Weight Decay** | `λ_decay * W_ij` | Prevent unbounded growth |

**EBM Gradient Term:**
```
∂E_i/∂W_ij = -spike_j * softmax(β * X_i^T * q_i) + λ * W_ij
```

**Graded STDP Characteristics:**
- Stronger spikes → larger STDP updates
- Weaker spikes → smaller updates
- No spike → no update (trace decays naturally)
- **Continuous, analog plasticity** — not binary

---

### 8. Key Parameters

| Parameter | Symbol | Typical Range | Role |
|-----------|--------|---------------|------|
| Membrane time constant | `τ` | 10-20 ms | Voltage decay rate |
| STDP pre trace decay | `τ_pre` | 10-50 ms | Presynaptic trace memory |
| STDP post trace decay | `τ_post` | 10-50 ms | Postsynaptic trace memory |
| STDP learning rate | `η_STDP` | 0.001-0.01 | Plasticity strength |
| EBM modulation strength | `μ` | 0.1-1.0 | EBM gradient influence |
| Inverse temperature | `β` | 1-10 | Energy landscape sharpness |
| Weight decay | `λ_decay` | 0.0001-0.001 | Regularization |
| Energy transfer coefficient | `α` | 0.01-0.1 | Distance cost scaling |
| Energy gradient coupling | `γ` | 0.1-1.0 | EBM → voltage coupling |
| Connection probability | `p_0` | 0.1-0.5 | Base connectivity |
| Distance decay | `σ`, `d_0` | 1-10 (units) | Spatial connection range |

---

## Implementation Notes (Brian2)

### Neuron Group Definition
```python
neurons = NeuronGroup(N,
    '''
    dV/dt = -V/τ + I_syn + I_EBM + I_ext : 1
    da_pre/dt = -a_pre/τ_pre : 1
    da_post/dt = -a_post/τ_post : 1
    I_syn : 1
    I_EBM : 1
    I_ext : 1
    spike = 1 / (1 + exp(-V)) : 1  # continuous spike output
    ''',
    threshold='rand() < spike',  # stochastic spiking proportional to strength
    reset='V -= reset_value; a_pre += 1',
    method='euler'
)
```

### Synapse Definition
```python
synapses = Synapses(neurons, neurons,
    '''
    dW/dt = η * (a_pre * spike_post - a_post * spike_pre) 
            + μ * dE_dW - λ * W : 1 (clock-driven)
    dE_dW : 1  # precomputed from EBM
    ''',
    on_pre='''
    I_syn_post += W * spike_pre
    a_pre += 1
    ''',
    on_post='a_post += 1'
)
```

### Energy Dynamics (Custom Update)
```python
# Run at each timestep or every few timesteps
def update_energy_and_gradients():
    for i in range(N):
        q_i = get_neuron_state(i)  # [V, a_pre, a_post, ...]
        E_i = compute_energy(q_i, X_i, β, λ)
        dE_dq = compute_energy_gradient(q_i, X_i, β, λ)
        neurons.I_EBM[i] = -dE_dq[V_component]
        
        for j in connected_pre(i):
            dE_dW = -spikes[j] * softmax(β * X_i.T @ q_i) + λ * W[j,i]
            synapses.dE_dW[j, i] = dE_dW
```

### Distance Matrix (Precomputed)
```python
# For N neurons in 2D grid
positions = np.random.rand(N, 2) * grid_size
distance_matrix = cdist(positions, positions, 'euclidean')
W_base = np.exp(-distance_matrix / σ)
W_base[distance_matrix > max_range] = 0  # sparse
```

---

## Input/Output Encoding

### Input Encoding
```
Raw input x ∈ ℝ^d → I_ext = W_encode @ x
```
- External current drives initial spike events
- Continuous encoding preserves signal intensity

### Output Decoding
```
Output y = W_decode @ spike_outputs
```
- Weighted sum of continuous spike strengths
- Can add readout layer (linear or nonlinear)

---

## Future Considerations

- [ ] Implement Brian2 neuron and synapse equations
- [ ] Design network topology (1D/2D/3D or abstract)
- [ ] Precompute distance matrix for efficiency
- [ ] Implement per-neuron EBM storage and gradient computation
- [ ] Add input/output encoding/decoding layers
- [ ] Define memory patterns X_i for each neuron
- [ ] Testing: memory capacity, pattern completion, energy dynamics
- [ ] Optimization: batched EBM updates, GPU acceleration

---

## Summary: What Makes This Hybrid Unique

| Feature | Traditional SNN | Hopfield EBM | **This Hybrid** |
|---------|----------------|--------------|-----------------|
| Spike type | Binary events | No spikes | **Continuous events** |
| Energy | None | Global symmetric | **Per-neuron, asymmetric** |
| Learning | STDP | Energy minimization | **STDP + EBM gradient** |
| Dynamics | Membrane ODE | State retrieval | **Coupled V + q ODEs** |
| Connectivity | Sparse or dense | All-to-all | **Sparse, distance-dependent** |
| Biological realism | Medium | Low | **High** |

---

*Architecture specification for continuous SNN + EBM hybrid in Brian2*
