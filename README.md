## Beam-GD: A Beam Search-Inspired Variant of Gradient Descent

Most gradient-based optimization methods are inherently greedy: they descend the steepest slope at each step using a single model trajectory. While this works well in many smooth landscapes, it can lead to suboptimal outcomes in high-dimensional or noisy problems where local minima and saddle points abound.

Inspired by beam search from discrete optimization, we propose a simple yet effective variant: **beam-style gradient descent**.

### Key Idea

Maintain multiple candidate models in parallel. At each iteration:

1. **Update** each model independently using SGD (optionally with injected noise to encourage exploration).
2. **Evaluate** their performance on a held-out validation set.
3. **Select** the top-performing $k$ models and discard the rest.
4. **Repeat** the process for the next epoch.

This approach allows the algorithm to:

* Explore multiple promising regions of the loss landscape.
* Escape poor local optima via stochastic updates.
* Maintain only the most promising optimization paths based on generalization (validation loss), not just training progress.

### Benefits

* **Better Exploration:** Noise and multiple starts prevent premature convergence.
* **Validation-Guided Search:** Regular selection ensures generalization guides optimization.
* **Anytime Behavior:** Returns the best-so-far model at each step.

### Empirical Result

In a small-scale binary classification task, beam-style gradient descent (with $k=3$) consistently achieved lower validation loss than a standard single-trajectory baseline, highlighting its potential even in simple settings.
