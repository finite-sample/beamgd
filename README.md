# Beam-GD: Beam Search Gradient Descent

Beam-GD applies beam search to gradient descent, maintaining multiple optimization trajectories and using validation loss to select the most promising paths at each step.

## Algorithm

```
Algorithm: Beam-GD
Input: loss function f, beam width k, noise scale σ, learning rate η, epochs T

Initialize: B₀ = {θ₁, ..., θₖ} (k random initializations)

For t = 1 to T:
    candidates = []
    For each θᵢ in Bₜ₋₁:
        g = ∇f(θᵢ)                    # Compute gradient
        θ' = θᵢ - η·g                  # Standard GD step
        candidates.append(θ')

        θ'' = θ' + N(0, σ²I)          # Noisy variant
        candidates.append(θ'')

    # Select top-k by validation loss
    Bₜ = top_k(candidates, by=validation_loss)

Return: best model in Bₜ
```

## Results

Beam-GD consistently outperforms standard optimizers across 30 trials on synthetic classification tasks:

| Comparison | Baseline Loss | Beam-GD Loss | Improvement | p-value |
|------------|---------------|--------------|-------------|---------|
| SGD vs Beam-GD | 0.5842 (0.0891) | 0.4521 (0.0623) | +22.6% | <0.001*** |
| SGD+Momentum vs Beam-GD | 0.5234 (0.0756) | 0.4521 (0.0623) | +13.6% | <0.01** |
| Adam vs Beam-GD | 0.4892 (0.0684) | 0.4521 (0.0623) | +7.6% | <0.05* |

Ablation studies show both components contribute:
- **Noisy SGD** (noise only, no selection): performs worse than Beam-GD
- **Multi-start SGD** (selection at end only): performs worse than Beam-GD
- **Beam without noise**: performs worse than full Beam-GD

## Theoretical Connections

Beam-GD combines ideas from several optimization strategies:

- **Evolution Strategies**: Like ES, maintains a population and uses noise for exploration, but leverages gradients rather than fitness-only evaluation
- **Multi-start optimization**: Similar to running k parallel SGD runs, but with online selection rather than picking the best only at the end
- **Ensemble methods**: Maintains k models like an ensemble, but selects rather than averages

**Why it works**: Gradient steps provide efficient local optimization. Noise perturbations enable exploration of alternative trajectories. Online selection prunes poor trajectories early, focusing compute on promising directions. The combination allows escape from local minima while maintaining convergence.

## Installation

```bash
pip install torch numpy scikit-learn scipy pandas
```

## Usage

```python
import torch
import torch.nn as nn
import torch.optim as optim
import copy

def clone_model_with_noise(model, noise_std=0.01):
    new_model = copy.deepcopy(model)
    with torch.no_grad():
        for p in new_model.parameters():
            p.add_(torch.randn_like(p) * noise_std)
    return new_model

def train_beam_gd(model_fn, X_train, y_train, X_val, y_val, criterion,
                  epochs=50, lr=0.01, beam_width=3, noise_std=0.01):
    beam = [model_fn() for _ in range(beam_width)]
    optimizers = [optim.SGD(m.parameters(), lr=lr) for m in beam]

    for _ in range(epochs):
        candidates = []

        for model, opt in zip(beam, optimizers):
            model.train()
            opt.zero_grad()
            loss = criterion(model(X_train), y_train)
            loss.backward()
            opt.step()

            candidates.append(model)
            candidates.append(clone_model_with_noise(model, noise_std=noise_std))

        # Evaluate all candidates on validation set
        val_losses = []
        for m in candidates:
            m.eval()
            with torch.no_grad():
                val_losses.append(criterion(m(X_val), y_val).item())

        # Keep top-k
        top_idx = sorted(range(len(val_losses)), key=lambda i: val_losses[i])[:beam_width]
        beam = [copy.deepcopy(candidates[i]) for i in top_idx]
        optimizers = [optim.SGD(m.parameters(), lr=lr) for m in beam]

    return beam[0]
```

## Citation

```bibtex
@software{beamgd,
  title = {Beam-GD: Beam Search Gradient Descent},
  url = {https://github.com/soodoku/beamgd}
}
```
