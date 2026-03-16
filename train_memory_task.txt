"""
Train the Continuous LIF + EBM on Pattern Completion Tasks

This task demonstrates the model's key capabilities:
- Per-neuron EBM stores memory patterns
- STDP + EBM gradient learning
- Pattern completion from partial cues
- Continuous spike transmission (graded strength)
"""

import numpy as np
from brian2 import *
from snn_ebm_model import ContinuousLIFEBM, PerNeuronEBM
import time


# =============================================================================
# Pattern Completion Task
# =============================================================================

class PatternCompletionTask:
    """
    Train the network to store and recall spatial patterns.
    
    Each neuron stores memory patterns in its EBM.
    During recall, partial cues trigger pattern completion.
    """
    
    def __init__(self, n_neurons=100, n_patterns=5, pattern_size=10,
                 v_threshold=0.5, v_reset=0.0, v_scale=2.0):
        """
        Parameters
        ----------
        n_neurons : int
            Number of neurons in the network
        n_patterns : int
            Number of patterns to store
        pattern_size : int
            Number of active neurons per pattern
        v_threshold : float
            LIF spike threshold
        v_reset : float
            LIF reset potential
        v_scale : float
            Scaling for continuous spike output
        """
        self.n_neurons = n_neurons
        self.n_patterns = n_patterns
        self.pattern_size = pattern_size

        # Create network with continuous LIF neurons FIRST (need positions for patterns)
        # Fixed parameters for stability with reward-modulated and error-driven learning
        self.net = ContinuousLIFEBM(
            n_neurons=n_neurons,
            n_layers=1,
            tau=15*ms,
            tau_tr1=20*ms,
            tau_tr2=20*ms,
            eta_stdp=0.002,  # Very low STDP (reward will modulate)
            mu_ebm=0.0,  # Disable EBM on synapses
            lambda_decay=0.00005,  # Minimal weight decay
            beta=2.0,  # Lower beta = smoother energy landscape
            alpha=0.05,
            gamma=0.05,  # Very reduced EBM coupling for stability
            space_dim=2,
            grid_size=10.0,
            p0=0.3,
            d0=5.0,
            n_patterns=n_patterns,
            seed=42,
            v_threshold=v_threshold,
            v_reset=v_reset,
            v_scale=v_scale,
            tau_refrac=5*ms,
            homeostasis=True,
            target_rate=20*Hz,  # Higher target rate
            reward_modulation=True,
            eta_reward=0.05,  # was 0.02 — too small to move weights within ~100 epochs
            error_driven=True,
            eta_error=0.05    # was 0.01 — too small; error signal was correct but invisible
        )

        # Generate diverse patterns AFTER network is created (need positions)
        self.patterns = self._generate_patterns()

        # Store patterns in each neuron's EBM
        self._store_patterns_in_ebm()

        # Weight bounds already set in model, but reinforce here
        self.net.synapses.W_min = np.full(len(self.net.synapses), -0.3)
        self.net.synapses.W_max = np.full(len(self.net.synapses), 0.5)

    def _generate_patterns(self):
        """Generate diverse binary patterns with different structures."""
        patterns = []
        
        # Type 1: Localized clusters (spatially contiguous groups)
        for _ in range(3):
            pattern = np.zeros(self.n_neurons)
            # Pick a random center neuron
            center = np.random.randint(0, self.n_neurons)
            # Activate nearby neurons based on distance
            for i in range(self.n_neurons):
                dist = np.sqrt(np.sum((self.net.positions[i] - self.net.positions[center])**2))
                if dist < 2.5:  # Local cluster radius
                    pattern[i] = 1.0
            patterns.append(pattern)
        
        # Type 2: Striped patterns (based on x-position)
        for i in range(2):
            pattern = np.zeros(self.n_neurons)
            stripe_start = i * 3
            for n in range(self.n_neurons):
                x = self.net.positions[n, 0]
                if stripe_start <= (x % 6) < stripe_start + 3:
                    pattern[n] = 1.0
            patterns.append(pattern)
        
        # Type 3: Checkerboard pattern
        pattern = np.zeros(self.n_neurons)
        for n in range(self.n_neurons):
            x, y = self.net.positions[n]
            if (int(x) + int(y)) % 2 == 0:
                pattern[n] = 1.0
        patterns.append(pattern)
        
        # Type 4: Random sparse patterns (few active neurons)
        for _ in range(3):
            pattern = np.zeros(self.n_neurons)
            active_indices = np.random.choice(
                self.n_neurons, size=max(5, self.pattern_size // 3), replace=False
            )
            pattern[active_indices] = 1.0
            patterns.append(pattern)
        
        # Type 5: Random dense patterns (many active neurons)
        for _ in range(2):
            pattern = np.zeros(self.n_neurons)
            active_indices = np.random.choice(
                self.n_neurons, size=min(self.n_neurons - 5, self.pattern_size * 2), replace=False
            )
            pattern[active_indices] = 1.0
            patterns.append(pattern)
        
        # Type 6: Edge detectors (left vs right, top vs bottom)
        # Left half
        pattern = np.zeros(self.n_neurons)
        for n in range(self.n_neurons):
            if self.net.positions[n, 0] < self.net.grid_size / 2:
                pattern[n] = 1.0
        patterns.append(pattern)
        
        # Right half
        pattern = np.zeros(self.n_neurons)
        for n in range(self.n_neurons):
            if self.net.positions[n, 0] >= self.net.grid_size / 2:
                pattern[n] = 1.0
        patterns.append(pattern)
        
        # Top half
        pattern = np.zeros(self.n_neurons)
        for n in range(self.n_neurons):
            if self.net.positions[n, 1] >= self.net.grid_size / 2:
                pattern[n] = 1.0
        patterns.append(pattern)
        
        # Bottom half
        pattern = np.zeros(self.n_neurons)
        for n in range(self.n_neurons):
            if self.net.positions[n, 1] < self.net.grid_size / 2:
                pattern[n] = 1.0
        patterns.append(pattern)
        
        # Type 7: Diagonal patterns
        pattern = np.zeros(self.n_neurons)
        for n in range(self.n_neurons):
            x, y = self.net.positions[n]
            if x > y:
                pattern[n] = 1.0
        patterns.append(pattern)
        
        pattern = np.zeros(self.n_neurons)
        for n in range(self.n_neurons):
            x, y = self.net.positions[n]
            if x + y > self.net.grid_size:
                pattern[n] = 1.0
        patterns.append(pattern)
        
        # Type 8: Ring pattern (annulus)
        pattern = np.zeros(self.n_neurons)
        center = np.array([self.net.grid_size / 2, self.net.grid_size / 2])
        for n in range(self.n_neurons):
            dist = np.sqrt(np.sum((self.net.positions[n] - center)**2))
            if 2.5 < dist < 4.5:
                pattern[n] = 1.0
        patterns.append(pattern)
        
        # Type 9: Cross pattern
        pattern = np.zeros(self.n_neurons)
        for n in range(self.n_neurons):
            x, y = self.net.positions[n]
            if abs(x - self.net.grid_size/2) < 2 or abs(y - self.net.grid_size/2) < 2:
                pattern[n] = 1.0
        patterns.append(pattern)
        
        # Type 10: Corner patterns
        for corner in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            pattern = np.zeros(self.n_neurons)
            cx, cy = corner[0] * self.net.grid_size, corner[1] * self.net.grid_size
            for n in range(self.n_neurons):
                dist = np.sqrt((self.net.positions[n, 0] - cx)**2 + 
                              (self.net.positions[n, 1] - cy)**2)
                if dist < 3:
                    pattern[n] = 1.0
            patterns.append(pattern)
        
        # Fill remaining patterns with random ones if needed
        while len(patterns) < self.n_patterns:
            pattern = np.zeros(self.n_neurons)
            active_indices = np.random.choice(
                self.n_neurons, size=self.pattern_size, replace=False
            )
            pattern[active_indices] = 1.0
            patterns.append(pattern)
        
        return np.array(patterns[:self.n_patterns])
    
    def _store_patterns_in_ebm(self):
        """Store patterns in each neuron's EBM memory.
        
        Store patterns with realistic trace values to match operating conditions.
        When a neuron is active (V=1), it typically has small positive traces from spiking.
        """
        for i in range(self.n_neurons):
            for p in range(self.n_patterns):
                state = np.zeros(3)
                state[0] = self.patterns[p, i]  # V component (0 or 1)
                # Active neurons typically have small positive traces from recent spiking
                state[1] = 0.05 * self.patterns[p, i]  # trace1
                state[2] = 0.05 * self.patterns[p, i]  # trace2
                # Normalize to unit vector so the softmax attention in the EBM
                # operates on a well-conditioned, non-flat energy landscape.
                # Without this, patterns stored as raw {0,1} vectors are not
                # orthonormal, causing uniform softmax weights and a flat energy.
                norm = np.linalg.norm(state) + 1e-10
                self.net.ebm.memory_patterns[i, :, p] = state / norm
        
        # Update the pattern matrix in EBM
        self.net.ebm.X = self.net.ebm.memory_patterns
    
    def present_pattern(self, pattern_idx, cue_strength=0.7, noise=0.1):
        """
        Present a partial cue of a stored pattern.
        """
        pattern = self.patterns[pattern_idx].copy()
        
        active_neurons = np.where(pattern > 0)[0]
        n_cue_neurons = max(1, int(len(active_neurons) * cue_strength))
        cue_neurons = np.random.choice(active_neurons, size=n_cue_neurons, replace=False)
        
        self.net.neurons.I_ext[:] = 0
        self.net.neurons.I_ext[cue_neurons] = 2.0
        
        noise_neurons = np.setdiff1d(np.arange(self.n_neurons), cue_neurons)
        if len(noise_neurons) > 0:
            self.net.neurons.I_ext[noise_neurons] = np.random.randn(len(noise_neurons)) * noise
        
        return cue_neurons
    
    def train(self, n_epochs=10, pattern_duration=50*ms, recall_duration=100*ms):
        """Train the network on pattern completion."""
        print(f"\nTraining on {self.n_patterns} patterns for {n_epochs} epochs...")
        print(f"Network: {self.n_neurons} neurons, {len(self.net.synapses)} synapses")

        history = {
            'epoch': [],
            'mean_weight': [],
            'mean_energy': [],
            'recall_accuracy': [],
            'excitatory_weight': [],
            'inhibitory_weight': [],
            'mean_reward': [],   # initialised here so the print line never hits a KeyError
            'mean_error': []     # even if reward/error computation raises an exception
        }

        for epoch in range(n_epochs):
            epoch_start = time.time()
            
            epoch_rewards = []
            epoch_errors = []

            # Training phase: present each pattern with reward/error feedback
            for p in range(self.n_patterns):
                # Snapshot cumulative spike counts BEFORE this presentation.
                # self.net.spike_monitor.count is a running total since t=0,
                # so diffing before/after gives spikes fired in this window only.
                count_before = self.net.spike_monitor.count[:].copy()

                self.present_pattern(p, cue_strength=1.0, noise=0.0)
                self.net.run(pattern_duration, dt=0.5*ms, update_ebm_every=5)

                # Spikes fired during this presentation window
                count_after = self.net.spike_monitor.count[:].copy()
                spikes_this_window = count_after - count_before  # shape: (n_neurons,)

                # Binary activity: did the neuron fire at all during the window?
                # This correctly captures neurons that fired-and-reset, unlike
                # reading instantaneous V which sees them at v_reset=0 post-spike.
                actual_activity = (spikes_this_window > 0).astype(float)
                target_activity = self.patterns[p]
                
                # Compute and apply error-driven learning
                error = self.net.compute_prediction_error(target_activity, actual_activity)
                if error is not None:
                    self.net.apply_error_driven_learning(error)
                    epoch_errors.append(np.mean(np.abs(error)))
                
                # Update EBM gradients
                self.net.update_ebm_gradients()
                
                # Compute reward based on how well the pattern was recalled
                # Reward = correlation between target and actual activity
                reward = np.corrcoef(target_activity.flatten(), actual_activity.flatten())[0, 1]
                if np.isnan(reward):
                    reward = 0.0
                
                # Apply reward-modulated plasticity
                self.net.apply_reward(reward)
                epoch_rewards.append(reward)
                
                # Reset for next pattern
                self.net.neurons.I_ext[:] = 0
                self.net.neurons.V[:] = np.random.rand(self.n_neurons) * 0.1
                self.net.neurons.trace1[:] = 0.02
                self.net.neurons.trace2[:] = 0.02

            # Recall phase
            accuracy = self._test_recall(recall_duration)

            # NOTE: normalize_weights removed. It classified by current sign, so after
            # STDP + lambda_decay drifted weights toward 0, it saw almost no positives
            # and rescaled only inhibitory ones, leaving W_mean negative even after
            # "normalizing". Worse, it wiped pattern-specific structure STDP had learned.
            # Hard bounds (W_min/W_max) in the synapse equations prevent runaway.

            # Record metrics
            history['epoch'].append(epoch)
            weights = self.net.synapses.W[:]
            history['mean_weight'].append(np.mean(weights))
            history['excitatory_weight'].append(np.mean(weights[weights > 0]))
            history['inhibitory_weight'].append(np.mean(weights[weights < 0]))

            energy_stats = self.net.get_energy_stats()
            history['mean_energy'].append(energy_stats['mean_energy'])
            history['recall_accuracy'].append(accuracy)
            
            history['mean_reward'].append(float(np.mean(epoch_rewards)) if epoch_rewards else float('nan'))
            history['mean_error'].append(float(np.mean(epoch_errors)) if epoch_errors else float('nan'))

            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1}/{n_epochs}: "
                  f"Accuracy={accuracy*100:.1f}%, "
                  f"Reward={history['mean_reward'][-1]:.3f}, "
                  f"Error={history['mean_error'][-1]:.4f}, "
                  f"Energy={energy_stats['mean_energy']:.4f}, "
                  f"W_mean={history['mean_weight'][-1]:.4f}, "
                  f"Time={epoch_time:.1f}s")
        
        return history
    
    def _test_recall(self, duration=100*ms):
        """Test pattern completion ability."""
        accuracies = []

        for p in range(self.n_patterns):
            cue_neurons = self.present_pattern(p, cue_strength=0.5, noise=0.1)

            # Record voltage during recall
            from brian2 import StateMonitor
            monitor = StateMonitor(self.net.neurons, 'V', record=True)
            self.net.net.add(monitor)

            self.net.run(duration, dt=0.5*ms, update_ebm_every=5)

            # Get peak voltage during recall
            peak_V = np.max(monitor.V, axis=1)

            # Remove monitor
            self.net.net.remove(monitor)

            pattern_neurons = np.where(self.patterns[p] > 0)[0]
            non_pattern_neurons = np.where(self.patterns[p] == 0)[0]

            pattern_activity = np.mean(peak_V[pattern_neurons])
            non_pattern_activity = np.mean(peak_V[non_pattern_neurons])

            # More robust accuracy: pattern neurons should fire more than non-pattern
            # Use a margin for better discrimination
            accuracy = (pattern_activity > non_pattern_activity + 0.1)
            accuracies.append(float(accuracy))

            self.net.neurons.I_ext[:] = 0
            self.net.neurons.V[:] = np.random.rand(self.n_neurons) * 0.1
            # Reset traces to operating point
            self.net.neurons.trace1[:] = 0.02
            self.net.neurons.trace2[:] = 0.02

        return np.mean(accuracies)
    
    def visualize_results(self, history, results=None):
        """Create comprehensive visualization of training results."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available")
            return

        fig = plt.figure(figsize=(18, 14))

        # 1. Training accuracy
        ax1 = fig.add_subplot(3, 3, 1)
        ax1.plot(history['epoch'], np.array(history['recall_accuracy']) * 100, 'b-o', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Pattern Completion Accuracy')
        ax1.set_ylim([0, 110])
        ax1.grid(True, alpha=0.3)

        # 2. Reward signal over epochs
        ax2 = fig.add_subplot(3, 3, 2)
        if 'mean_reward' in history:
            ax2.plot(history['epoch'], history['mean_reward'], 'g-o', linewidth=2, label='Reward')
            ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Reward (correlation)')
            ax2.set_title('Reward Signal (R-STDP)')
            ax2.set_ylim([-1.1, 1.1])
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No reward data', ha='center', va='center')

        # 3. Prediction error
        ax3 = fig.add_subplot(3, 3, 3)
        if 'mean_error' in history:
            ax3.plot(history['epoch'], history['mean_error'], 'r-o', linewidth=2)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Mean |Error|')
            ax3.set_title('Prediction Error (Error-Driven Learning)')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No error data', ha='center', va='center')

        # 4. Network structure
        ax4 = fig.add_subplot(3, 3, 4)
        self.net.plot_network(ax=ax4)

        # 5. Weight evolution with E/I balance
        ax5 = fig.add_subplot(3, 3, 5)
        ax5.plot(history['epoch'], history['mean_weight'], 'g-o', label='Mean', linewidth=2)
        ax5.plot(history['epoch'], history['excitatory_weight'], 'r--', label='Excitatory')
        ax5.plot(history['epoch'], history['inhibitory_weight'], 'b--', label='Inhibitory')
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Weight')
        ax5.set_title('Synaptic Weight Evolution')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Pattern visualization - show stored patterns
        ax6 = fig.add_subplot(3, 3, 6)
        self.plot_patterns(ax6)

        # 7. Weight distribution
        ax7 = fig.add_subplot(3, 3, 7)
        self.net.plot_weight_distribution(ax=ax7)

        # 8. Activity (if results available)
        ax8 = fig.add_subplot(3, 3, 8)
        if results is not None and 'spike_times' in results:
            spike_trains = results['spike_times']
            for neuron_id, times in spike_trains.items():
                if len(times) > 0 and neuron_id < 20:  # Show first 20 neurons
                    ax8.vlines(times / ms, neuron_id - 0.5, neuron_id + 0.5,
                              color='blue', alpha=0.5, linewidth=0.5)
            ax8.set_xlabel('Time (ms)')
            ax8.set_ylabel('Neuron ID')
            ax8.set_title('Spike Raster (first 20 neurons)')
            ax8.set_ylim(-0.5, 19.5)
        else:
            ax8.text(0.5, 0.5, 'Run simulation\nfor activity plot',
                    ha='center', va='center', transform=ax8.transAxes)
            ax8.set_xlim(0, 1)
            ax8.set_ylim(0, 1)

        # 9. E/I ratio over time
        ax9 = fig.add_subplot(3, 3, 9)
        exc = np.array(history['excitatory_weight'])
        inh = np.abs(np.array(history['inhibitory_weight']))
        ratio = exc / (inh + 1e-6)
        ax9.plot(history['epoch'], ratio, 'orange', linewidth=2)
        ax9.set_xlabel('Epoch')
        ax9.set_ylabel('E/I Weight Ratio')
        ax9.set_title('Excitatory/Inhibitory Balance')
        ax9.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('lif_ebm_results.png', dpi=150, bbox_inches='tight')
        print("\nResults saved to lif_ebm_results.png")
    
    def plot_patterns(self, ax=None):
        """Visualize the stored patterns as spatial maps."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import ListedColormap
        except ImportError:
            print("Matplotlib not available")
            return
        
        if ax is None:
            fig, ax = plt.subplots()
        
        # Show first 9 patterns in a 3x3 grid
        n_show = min(9, self.n_patterns)
        
        # Create subplots within the axis
        for p in range(n_show):
            # Calculate position in 3x3 grid
            row = p // 3
            col = p % 3
            
            # Create inset axes
            left = 0.1 + col * 0.27
            bottom = 0.1 + (2 - row) * 0.27
            width = 0.25
            height = 0.25
            
            ax_pattern = ax.inset_axes([left, bottom, width, height])
            
            # Plot pattern
            pattern = self.patterns[p]
            positions = self.net.positions
            
            scatter = ax_pattern.scatter(positions[:, 0], positions[:, 1], 
                                        c=pattern, cmap='viridis', 
                                        s=30, alpha=0.8,
                                        vmin=0, vmax=1)
            ax_pattern.set_xlim(0, self.net.grid_size)
            ax_pattern.set_ylim(0, self.net.grid_size)
            ax_pattern.set_aspect('equal')
            ax_pattern.axis('off')
            ax_pattern.set_title(f'P{p}', fontsize=8)
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.axis('off')
        ax.set_title('Stored Patterns (spatial view)', fontsize=10)


# =============================================================================
# Demo: Single Pattern Completion
# =============================================================================

def run_demo():
    """Run a single pattern completion demo with full visualization."""
    print("=" * 60)
    print("CONTINUOUS LIF + EBM: PATTERN COMPLETION DEMO")
    print("=" * 60)
    
    # Create task with diverse pattern types
    task = PatternCompletionTask(
        n_neurons=50,
        n_patterns=6,   # Hopfield capacity ≈ 0.14 * N ≈ 7; 20 patterns caused
                        # destructive interference and guaranteed failure to learn
        pattern_size=15,
        v_threshold=0.5,
        v_reset=0.0,
        v_scale=2.0
    )
    
    print(f"\nNetwork created:")
    print(f"  - {task.n_neurons} continuous LIF neurons")
    print(f"  - {len(task.net.synapses)} synapses (E/I mixed)")
    print(f"  - {task.n_patterns} stored patterns")
    
    # Show initial state
    print("\nInitial network state:")
    print(f"  - Mean weight: {np.mean(task.net.synapses.W[:]):.4f}")
    print(f"  - Excitatory: {np.sum(task.net.synapses.W[:] > 0)} synapses")
    print(f"  - Inhibitory: {np.sum(task.net.synapses.W[:] < 0)} synapses")
    
    # Train with more epochs to show energy minimization trend
    history = task.train(
        n_epochs=100,
        pattern_duration=50*ms,   # 20ms was only ~1.3x tau=15ms; neurons need
                                  # ≥3 tau to charge past threshold reliably
        recall_duration=30*ms
    )
    
    # Run final test with recording
    print("\nRunning final recall test...")
    task.present_pattern(0, cue_strength=0.5)
    results = task.net.run(50*ms, dt=0.5*ms, update_ebm_every=5)
    
    # Visualize
    task.visualize_results(history, results)
    
    # Print final stats
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Final accuracy: {history['recall_accuracy'][-1]*100:.1f}%")
    print(f"Final mean weight: {history['mean_weight'][-1]:.4f}")
    print(f"Final energy: {history['mean_energy'][-1]:.4f}")
    print(f"Total spikes: {sum(results['spike_count'])}")
    
    # Show spike strength distribution
    spike_outputs = np.clip((task.net.neurons.V[:] - task.net.v_threshold) / task.net.v_scale, 0, 1)
    print(f"\nContinuous spike output stats:")
    print(f"  - Mean: {np.mean(spike_outputs):.4f}")
    print(f"  - Max: {np.max(spike_outputs):.4f}")
    print(f"  - Active (output > 0.1): {np.sum(spike_outputs > 0.1)} neurons")
    
    return task, history, results


if __name__ == "__main__":
    task, history, results = run_demo()
