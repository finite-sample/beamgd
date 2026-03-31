"""
Comprehensive evaluation framework for Beam-GD optimizer.
Includes statistical testing, modern optimizer baselines, ablation studies, and MNIST evaluation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import stats
import pandas as pd
import warnings
import copy
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Callable

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
def set_seeds(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


# =============================================================================
# Model Definitions
# =============================================================================

class SimpleNet(nn.Module):
    """Simple neural network for binary classification."""
    def __init__(self, input_dim: int = 20, hidden_dim: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class LeNet(nn.Module):
    """LeNet-style CNN for MNIST classification."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class LogisticRegression(nn.Module):
    """Logistic regression model."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


# =============================================================================
# Optimizer Variants
# =============================================================================

def clone_model_with_noise(model: nn.Module, noise_std: float = 0.01) -> nn.Module:
    """Clone a model and add Gaussian noise to parameters."""
    new_model = copy.deepcopy(model)
    with torch.no_grad():
        for p in new_model.parameters():
            p.add_(torch.randn_like(p) * noise_std)
    return new_model


def evaluate_model(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
                   criterion: nn.Module) -> float:
    """Evaluate model loss on given data."""
    model.eval()
    with torch.no_grad():
        preds = model(X)
        return criterion(preds, y).item()


def train_baseline_sgd(model_fn: Callable, X_train: torch.Tensor, y_train: torch.Tensor,
                       X_val: torch.Tensor, y_val: torch.Tensor, criterion: nn.Module,
                       epochs: int = 50, lr: float = 0.01) -> Tuple[nn.Module, List[float], List[float]]:
    """Train with vanilla SGD."""
    model = model_fn()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    train_losses, val_losses = [], []

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        val_losses.append(evaluate_model(model, X_val, y_val, criterion))

    return model, train_losses, val_losses


def train_sgd_momentum(model_fn: Callable, X_train: torch.Tensor, y_train: torch.Tensor,
                       X_val: torch.Tensor, y_val: torch.Tensor, criterion: nn.Module,
                       epochs: int = 50, lr: float = 0.01, momentum: float = 0.9) -> Tuple[nn.Module, List[float], List[float]]:
    """Train with SGD + momentum."""
    model = model_fn()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    train_losses, val_losses = [], []

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        val_losses.append(evaluate_model(model, X_val, y_val, criterion))

    return model, train_losses, val_losses


def train_adam(model_fn: Callable, X_train: torch.Tensor, y_train: torch.Tensor,
               X_val: torch.Tensor, y_val: torch.Tensor, criterion: nn.Module,
               epochs: int = 50, lr: float = 0.001) -> Tuple[nn.Module, List[float], List[float]]:
    """Train with Adam optimizer."""
    model = model_fn()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses, val_losses = [], []

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        val_losses.append(evaluate_model(model, X_val, y_val, criterion))

    return model, train_losses, val_losses


def train_sgd_lr_decay(model_fn: Callable, X_train: torch.Tensor, y_train: torch.Tensor,
                       X_val: torch.Tensor, y_val: torch.Tensor, criterion: nn.Module,
                       epochs: int = 50, lr: float = 0.01, decay_rate: float = 0.95) -> Tuple[nn.Module, List[float], List[float]]:
    """Train with SGD + learning rate decay."""
    model = model_fn()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
    train_losses, val_losses = [], []

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_losses.append(loss.item())
        val_losses.append(evaluate_model(model, X_val, y_val, criterion))

    return model, train_losses, val_losses


def train_beam_gd(model_fn: Callable, X_train: torch.Tensor, y_train: torch.Tensor,
                  X_val: torch.Tensor, y_val: torch.Tensor, criterion: nn.Module,
                  epochs: int = 50, lr: float = 0.01, beam_width: int = 3,
                  noise_std: float = 0.01) -> Tuple[nn.Module, List[float], List[float]]:
    """Train with Beam Search Gradient Descent."""
    beam = [model_fn() for _ in range(beam_width)]
    optimizers = [optim.SGD(m.parameters(), lr=lr) for m in beam]
    train_losses, val_losses = [], []

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

        all_val_losses = [evaluate_model(m, X_val, y_val, criterion) for m in candidates]
        top_idx = np.argsort(all_val_losses)[:beam_width]
        beam = [copy.deepcopy(candidates[i]) for i in top_idx]
        optimizers = [optim.SGD(m.parameters(), lr=lr) for m in beam]

        best_train_loss = criterion(beam[0](X_train), y_train).item()
        train_losses.append(best_train_loss)
        val_losses.append(all_val_losses[top_idx[0]])

    return beam[0], train_losses, val_losses


def train_beam_gd_momentum(model_fn: Callable, X_train: torch.Tensor, y_train: torch.Tensor,
                            X_val: torch.Tensor, y_val: torch.Tensor, criterion: nn.Module,
                            epochs: int = 50, lr: float = 0.01, momentum: float = 0.9,
                            beam_width: int = 3, noise_std: float = 0.01) -> Tuple[nn.Module, List[float], List[float]]:
    """Train with Beam Search GD using momentum optimizer internally."""
    beam = [model_fn() for _ in range(beam_width)]
    optimizers = [optim.SGD(m.parameters(), lr=lr, momentum=momentum) for m in beam]
    train_losses, val_losses = [], []

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

        all_val_losses = [evaluate_model(m, X_val, y_val, criterion) for m in candidates]
        top_idx = np.argsort(all_val_losses)[:beam_width]
        beam = [copy.deepcopy(candidates[i]) for i in top_idx]
        optimizers = [optim.SGD(m.parameters(), lr=lr, momentum=momentum) for m in beam]

        best_train_loss = criterion(beam[0](X_train), y_train).item()
        train_losses.append(best_train_loss)
        val_losses.append(all_val_losses[top_idx[0]])

    return beam[0], train_losses, val_losses


# =============================================================================
# Ablation Study Variants
# =============================================================================

def train_noisy_sgd(model_fn: Callable, X_train: torch.Tensor, y_train: torch.Tensor,
                    X_val: torch.Tensor, y_val: torch.Tensor, criterion: nn.Module,
                    epochs: int = 50, lr: float = 0.01, noise_std: float = 0.01) -> Tuple[nn.Module, List[float], List[float]]:
    """SGD with noise injection but NO beam selection (isolates noise contribution)."""
    model = model_fn()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    train_losses, val_losses = [], []

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * noise_std)

        train_losses.append(loss.item())
        val_losses.append(evaluate_model(model, X_val, y_val, criterion))

    return model, train_losses, val_losses


def train_multistart_sgd(model_fn: Callable, X_train: torch.Tensor, y_train: torch.Tensor,
                         X_val: torch.Tensor, y_val: torch.Tensor, criterion: nn.Module,
                         epochs: int = 50, lr: float = 0.01, n_starts: int = 3) -> Tuple[nn.Module, List[float], List[float]]:
    """Multi-start SGD: run n_starts parallel SGD, pick best at END (isolates selection)."""
    models = [model_fn() for _ in range(n_starts)]
    optimizers = [optim.SGD(m.parameters(), lr=lr) for m in models]
    all_train_losses = [[] for _ in range(n_starts)]
    all_val_losses = [[] for _ in range(n_starts)]

    for epoch in range(epochs):
        for i, (model, opt) in enumerate(zip(models, optimizers)):
            model.train()
            opt.zero_grad()
            loss = criterion(model(X_train), y_train)
            loss.backward()
            opt.step()

            all_train_losses[i].append(loss.item())
            all_val_losses[i].append(evaluate_model(model, X_val, y_val, criterion))

    final_val_losses = [losses[-1] for losses in all_val_losses]
    best_idx = np.argmin(final_val_losses)

    return models[best_idx], all_train_losses[best_idx], all_val_losses[best_idx]


def train_beam_no_noise(model_fn: Callable, X_train: torch.Tensor, y_train: torch.Tensor,
                        X_val: torch.Tensor, y_val: torch.Tensor, criterion: nn.Module,
                        epochs: int = 50, lr: float = 0.01, beam_width: int = 3) -> Tuple[nn.Module, List[float], List[float]]:
    """Beam search WITHOUT noise (isolates beam structure contribution)."""
    beam = [model_fn() for _ in range(beam_width)]
    optimizers = [optim.SGD(m.parameters(), lr=lr) for m in beam]
    train_losses, val_losses = [], []

    for _ in range(epochs):
        for model, opt in zip(beam, optimizers):
            model.train()
            opt.zero_grad()
            loss = criterion(model(X_train), y_train)
            loss.backward()
            opt.step()

        all_val_losses = [evaluate_model(m, X_val, y_val, criterion) for m in beam]
        top_idx = np.argsort(all_val_losses)[:beam_width]
        beam = [copy.deepcopy(beam[i]) for i in top_idx]
        optimizers = [optim.SGD(m.parameters(), lr=lr) for m in beam]

        best_train_loss = criterion(beam[0](X_train), y_train).item()
        train_losses.append(best_train_loss)
        val_losses.append(all_val_losses[top_idx[0]])

    return beam[0], train_losses, val_losses


# =============================================================================
# Statistical Analysis
# =============================================================================

@dataclass
class StatisticalResult:
    """Container for statistical analysis results."""
    method1_name: str
    method2_name: str
    method1_mean: float
    method1_std: float
    method2_mean: float
    method2_std: float
    t_statistic: float
    p_value_ttest: float
    wilcoxon_statistic: Optional[float]
    p_value_wilcoxon: Optional[float]
    cohens_d: float
    ci_lower: float
    ci_upper: float
    n_trials: int
    improvement_pct: float


def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0.0


def compute_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """Compute confidence interval for the mean."""
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean - h, mean + h


def statistical_comparison(losses1: np.ndarray, losses2: np.ndarray,
                           name1: str, name2: str) -> StatisticalResult:
    """Perform comprehensive statistical comparison between two methods."""
    mean1, std1 = float(np.mean(losses1)), float(np.std(losses1, ddof=1))
    mean2, std2 = float(np.mean(losses2)), float(np.std(losses2, ddof=1))

    t_stat, p_ttest = stats.ttest_rel(losses1, losses2)

    try:
        wilcoxon_result = stats.wilcoxon(losses1, losses2)
        w_stat = float(wilcoxon_result[0])
        p_wilcoxon = float(wilcoxon_result[1])
    except (ValueError, IndexError):
        w_stat, p_wilcoxon = None, None

    cohens_d = compute_cohens_d(losses1, losses2)

    diff = losses1 - losses2
    ci_lower, ci_upper = compute_confidence_interval(diff)

    improvement_pct = float((mean1 - mean2) / mean1 * 100) if mean1 != 0 else 0.0

    return StatisticalResult(
        method1_name=name1,
        method2_name=name2,
        method1_mean=mean1,
        method1_std=std1,
        method2_mean=mean2,
        method2_std=std2,
        t_statistic=float(t_stat),
        p_value_ttest=float(p_ttest),
        wilcoxon_statistic=w_stat,
        p_value_wilcoxon=p_wilcoxon,
        cohens_d=cohens_d,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_trials=len(losses1),
        improvement_pct=improvement_pct
    )


def format_statistical_results(results: List[StatisticalResult]) -> pd.DataFrame:
    """Format statistical results into a DataFrame."""
    rows = []
    for r in results:
        sig_level = ""
        if r.p_value_ttest < 0.001:
            sig_level = "***"
        elif r.p_value_ttest < 0.01:
            sig_level = "**"
        elif r.p_value_ttest < 0.05:
            sig_level = "*"

        effect_size = ""
        d = abs(r.cohens_d)
        if d >= 0.8:
            effect_size = "large"
        elif d >= 0.5:
            effect_size = "medium"
        elif d >= 0.2:
            effect_size = "small"
        else:
            effect_size = "negligible"

        rows.append({
            "Comparison": f"{r.method1_name} vs {r.method2_name}",
            f"{r.method1_name} Loss": f"{r.method1_mean:.4f} ({r.method1_std:.4f})",
            f"{r.method2_name} Loss": f"{r.method2_mean:.4f} ({r.method2_std:.4f})",
            "Improvement (%)": f"{r.improvement_pct:+.2f}%",
            "p-value (t-test)": f"{r.p_value_ttest:.4f}{sig_level}",
            "p-value (Wilcoxon)": f"{r.p_value_wilcoxon:.4f}" if r.p_value_wilcoxon else "N/A",
            "Cohen's d": f"{r.cohens_d:.3f} ({effect_size})",
            "95% CI": f"[{r.ci_lower:.4f}, {r.ci_upper:.4f}]",
            "n": r.n_trials
        })

    return pd.DataFrame(rows)


# =============================================================================
# Data Loading
# =============================================================================

def load_synthetic_classification(n_samples: int = 500, n_features: int = 20,
                                   test_size: float = 0.2, val_size: float = 0.25,
                                   seed: int = 42) -> Dict:
    """Load synthetic classification dataset."""
    X, y = make_classification(n_samples=n_samples, n_features=n_features,
                                n_informative=n_features // 2, n_redundant=n_features // 4,
                                n_classes=2, random_state=seed)
    X = StandardScaler().fit_transform(X)
    y = y.astype(np.float32)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_size, random_state=seed)

    return {
        "X_train": torch.tensor(X_train, dtype=torch.float32),
        "y_train": torch.tensor(y_train, dtype=torch.float32).view(-1, 1),
        "X_val": torch.tensor(X_val, dtype=torch.float32),
        "y_val": torch.tensor(y_val, dtype=torch.float32).view(-1, 1),
        "X_test": torch.tensor(X_test, dtype=torch.float32),
        "y_test": torch.tensor(y_test, dtype=torch.float32).view(-1, 1),
        "input_dim": n_features,
    }


def load_mnist(n_train: int = 10000, n_test: int = 2000, seed: int = 42) -> Dict:
    """Load MNIST dataset."""
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = np.asarray(mnist.data)
    y = np.asarray(mnist.target).astype(int)

    X = X / 255.0

    np.random.seed(seed)
    train_idx = np.random.choice(60000, n_train, replace=False)
    test_idx = 60000 + np.random.choice(10000, n_test, replace=False)

    X_train_full = X[train_idx]
    y_train_full = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=seed)

    X_train = torch.tensor(X_train, dtype=torch.float32).reshape(-1, 1, 28, 28)
    X_val = torch.tensor(X_val, dtype=torch.float32).reshape(-1, 1, 28, 28)
    X_test = torch.tensor(X_test, dtype=torch.float32).reshape(-1, 1, 28, 28)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


# =============================================================================
# MNIST Training Functions
# =============================================================================

def evaluate_mnist(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
    """Evaluate MNIST model, return (loss, accuracy)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        outputs = model(X)
        loss = criterion(outputs, y).item()
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y).float().mean().item()
    return loss, accuracy


def train_mnist_sgd(X_train: torch.Tensor, y_train: torch.Tensor,
                    X_val: torch.Tensor, y_val: torch.Tensor,
                    epochs: int = 20, lr: float = 0.01, batch_size: int = 64) -> Tuple[nn.Module, List[float], List[float]]:
    """Train LeNet on MNIST with SGD."""
    model = LeNet()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_losses, val_losses = [], []

    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(loader))
        val_loss, _ = evaluate_mnist(model, X_val, y_val)
        val_losses.append(val_loss)

    return model, train_losses, val_losses


def train_mnist_sgd_momentum(X_train: torch.Tensor, y_train: torch.Tensor,
                              X_val: torch.Tensor, y_val: torch.Tensor,
                              epochs: int = 20, lr: float = 0.01, momentum: float = 0.9,
                              batch_size: int = 64) -> Tuple[nn.Module, List[float], List[float]]:
    """Train LeNet on MNIST with SGD + momentum."""
    model = LeNet()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_losses, val_losses = [], []

    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(loader))
        val_loss, _ = evaluate_mnist(model, X_val, y_val)
        val_losses.append(val_loss)

    return model, train_losses, val_losses


def train_mnist_adam(X_train: torch.Tensor, y_train: torch.Tensor,
                     X_val: torch.Tensor, y_val: torch.Tensor,
                     epochs: int = 20, lr: float = 0.001,
                     batch_size: int = 64) -> Tuple[nn.Module, List[float], List[float]]:
    """Train LeNet on MNIST with Adam."""
    model = LeNet()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_losses, val_losses = [], []

    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(loader))
        val_loss, _ = evaluate_mnist(model, X_val, y_val)
        val_losses.append(val_loss)

    return model, train_losses, val_losses


def train_mnist_beam_gd(X_train: torch.Tensor, y_train: torch.Tensor,
                        X_val: torch.Tensor, y_val: torch.Tensor,
                        epochs: int = 20, lr: float = 0.01, beam_width: int = 3,
                        noise_std: float = 0.005, batch_size: int = 64) -> Tuple[nn.Module, List[float], List[float]]:
    """Train LeNet on MNIST with Beam Search GD."""
    beam = [LeNet() for _ in range(beam_width)]
    optimizers = [optim.SGD(m.parameters(), lr=lr) for m in beam]
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_losses, val_losses = [], []

    for _ in range(epochs):
        for X_batch, y_batch in loader:
            candidates = []

            for model, opt in zip(beam, optimizers):
                model.train()
                opt.zero_grad()
                loss = criterion(model(X_batch), y_batch)
                loss.backward()
                opt.step()

                candidates.append(model)
                candidates.append(clone_model_with_noise(model, noise_std=noise_std))

            all_val_losses = [evaluate_mnist(m, X_val, y_val)[0] for m in candidates]
            top_idx = np.argsort(all_val_losses)[:beam_width]
            beam = [copy.deepcopy(candidates[i]) for i in top_idx]
            optimizers = [optim.SGD(m.parameters(), lr=lr) for m in beam]

        best_train_loss = criterion(beam[0](X_train[:1000]), y_train[:1000]).item()
        train_losses.append(best_train_loss)
        val_loss, _ = evaluate_mnist(beam[0], X_val, y_val)
        val_losses.append(val_loss)

    return beam[0], train_losses, val_losses


def train_mnist_beam_gd_momentum(X_train: torch.Tensor, y_train: torch.Tensor,
                                  X_val: torch.Tensor, y_val: torch.Tensor,
                                  epochs: int = 20, lr: float = 0.01, momentum: float = 0.9,
                                  beam_width: int = 3, noise_std: float = 0.005,
                                  batch_size: int = 64) -> Tuple[nn.Module, List[float], List[float]]:
    """Train LeNet on MNIST with Beam Search GD + momentum."""
    beam = [LeNet() for _ in range(beam_width)]
    optimizers = [optim.SGD(m.parameters(), lr=lr, momentum=momentum) for m in beam]
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_losses, val_losses = [], []

    for _ in range(epochs):
        for X_batch, y_batch in loader:
            candidates = []

            for model, opt in zip(beam, optimizers):
                model.train()
                opt.zero_grad()
                loss = criterion(model(X_batch), y_batch)
                loss.backward()
                opt.step()

                candidates.append(model)
                candidates.append(clone_model_with_noise(model, noise_std=noise_std))

            all_val_losses = [evaluate_mnist(m, X_val, y_val)[0] for m in candidates]
            top_idx = np.argsort(all_val_losses)[:beam_width]
            beam = [copy.deepcopy(candidates[i]) for i in top_idx]
            optimizers = [optim.SGD(m.parameters(), lr=lr, momentum=momentum) for m in beam]

        best_train_loss = criterion(beam[0](X_train[:1000]), y_train[:1000]).item()
        train_losses.append(best_train_loss)
        val_loss, _ = evaluate_mnist(beam[0], X_val, y_val)
        val_losses.append(val_loss)

    return beam[0], train_losses, val_losses


# =============================================================================
# Experiment Runners
# =============================================================================

def run_statistical_validation(n_trials: int = 30, epochs: int = 50, seed_base: int = 42) -> pd.DataFrame:
    """
    Experiment 1: Statistical validation on synthetic classification.
    Run n_trials comparisons between Beam-GD and baselines.
    """
    print(f"\n{'='*60}")
    print("EXPERIMENT 1: Statistical Validation")
    print(f"{'='*60}")
    print(f"Running {n_trials} trials...")

    results = {
        "SGD": [],
        "SGD+Momentum": [],
        "Adam": [],
        "SGD+LRDecay": [],
        "Beam-GD": [],
    }

    for trial in range(n_trials):
        seed = seed_base + trial
        set_seeds(seed)

        data = load_synthetic_classification(seed=seed)
        model_fn = lambda: SimpleNet(input_dim=data["input_dim"])
        criterion = nn.BCELoss()

        _, _, sgd_val = train_baseline_sgd(
            model_fn, data["X_train"], data["y_train"],
            data["X_val"], data["y_val"], criterion, epochs=epochs)
        results["SGD"].append(sgd_val[-1])

        _, _, sgd_mom_val = train_sgd_momentum(
            model_fn, data["X_train"], data["y_train"],
            data["X_val"], data["y_val"], criterion, epochs=epochs)
        results["SGD+Momentum"].append(sgd_mom_val[-1])

        _, _, adam_val = train_adam(
            model_fn, data["X_train"], data["y_train"],
            data["X_val"], data["y_val"], criterion, epochs=epochs)
        results["Adam"].append(adam_val[-1])

        _, _, sgd_decay_val = train_sgd_lr_decay(
            model_fn, data["X_train"], data["y_train"],
            data["X_val"], data["y_val"], criterion, epochs=epochs)
        results["SGD+LRDecay"].append(sgd_decay_val[-1])

        _, _, beam_val = train_beam_gd(
            model_fn, data["X_train"], data["y_train"],
            data["X_val"], data["y_val"], criterion, epochs=epochs)
        results["Beam-GD"].append(beam_val[-1])

        if (trial + 1) % 10 == 0:
            print(f"  Completed {trial + 1}/{n_trials} trials")

    statistical_results = []
    for baseline in ["SGD", "SGD+Momentum", "Adam", "SGD+LRDecay"]:
        result = statistical_comparison(
            np.array(results[baseline]),
            np.array(results["Beam-GD"]),
            baseline, "Beam-GD"
        )
        statistical_results.append(result)

    return format_statistical_results(statistical_results)


def run_ablation_study(n_trials: int = 30, epochs: int = 50, seed_base: int = 42) -> pd.DataFrame:
    """
    Experiment 3: Ablation study to understand component contributions.
    """
    print(f"\n{'='*60}")
    print("EXPERIMENT 3: Ablation Study")
    print(f"{'='*60}")
    print(f"Running {n_trials} trials...")

    results = {
        "Full Beam-GD": [],
        "Noisy SGD": [],
        "Multi-start SGD": [],
        "Beam (no noise)": [],
        "Vanilla SGD": [],
    }

    for trial in range(n_trials):
        seed = seed_base + trial
        set_seeds(seed)

        data = load_synthetic_classification(seed=seed)
        model_fn = lambda: SimpleNet(input_dim=data["input_dim"])
        criterion = nn.BCELoss()

        _, _, sgd_val = train_baseline_sgd(
            model_fn, data["X_train"], data["y_train"],
            data["X_val"], data["y_val"], criterion, epochs=epochs)
        results["Vanilla SGD"].append(sgd_val[-1])

        _, _, beam_val = train_beam_gd(
            model_fn, data["X_train"], data["y_train"],
            data["X_val"], data["y_val"], criterion, epochs=epochs)
        results["Full Beam-GD"].append(beam_val[-1])

        _, _, noisy_val = train_noisy_sgd(
            model_fn, data["X_train"], data["y_train"],
            data["X_val"], data["y_val"], criterion, epochs=epochs)
        results["Noisy SGD"].append(noisy_val[-1])

        _, _, multistart_val = train_multistart_sgd(
            model_fn, data["X_train"], data["y_train"],
            data["X_val"], data["y_val"], criterion, epochs=epochs)
        results["Multi-start SGD"].append(multistart_val[-1])

        _, _, beam_nonoise_val = train_beam_no_noise(
            model_fn, data["X_train"], data["y_train"],
            data["X_val"], data["y_val"], criterion, epochs=epochs)
        results["Beam (no noise)"].append(beam_nonoise_val[-1])

        if (trial + 1) % 10 == 0:
            print(f"  Completed {trial + 1}/{n_trials} trials")

    statistical_results = []
    for variant in ["Vanilla SGD", "Noisy SGD", "Multi-start SGD", "Beam (no noise)"]:
        result = statistical_comparison(
            np.array(results[variant]),
            np.array(results["Full Beam-GD"]),
            variant, "Full Beam-GD"
        )
        statistical_results.append(result)

    return format_statistical_results(statistical_results)


def run_mnist_evaluation(n_trials: int = 10, epochs: int = 20, seed_base: int = 42) -> pd.DataFrame:
    """
    Experiment 2: MNIST evaluation with proper backpropagation.
    """
    print(f"\n{'='*60}")
    print("EXPERIMENT 2: MNIST Evaluation")
    print(f"{'='*60}")
    print(f"Running {n_trials} trials...")

    data = load_mnist()

    results = {
        "SGD": {"loss": [], "accuracy": []},
        "SGD+Momentum": {"loss": [], "accuracy": []},
        "Adam": {"loss": [], "accuracy": []},
        "Beam-GD": {"loss": [], "accuracy": []},
    }

    for trial in range(n_trials):
        seed = seed_base + trial
        set_seeds(seed)
        print(f"  Trial {trial + 1}/{n_trials}...")

        model, _, _ = train_mnist_sgd(
            data["X_train"], data["y_train"],
            data["X_val"], data["y_val"], epochs=epochs)
        loss, acc = evaluate_mnist(model, data["X_test"], data["y_test"])
        results["SGD"]["loss"].append(loss)
        results["SGD"]["accuracy"].append(acc)

        model, _, _ = train_mnist_sgd_momentum(
            data["X_train"], data["y_train"],
            data["X_val"], data["y_val"], epochs=epochs)
        loss, acc = evaluate_mnist(model, data["X_test"], data["y_test"])
        results["SGD+Momentum"]["loss"].append(loss)
        results["SGD+Momentum"]["accuracy"].append(acc)

        model, _, _ = train_mnist_adam(
            data["X_train"], data["y_train"],
            data["X_val"], data["y_val"], epochs=epochs)
        loss, acc = evaluate_mnist(model, data["X_test"], data["y_test"])
        results["Adam"]["loss"].append(loss)
        results["Adam"]["accuracy"].append(acc)

        model, _, _ = train_mnist_beam_gd(
            data["X_train"], data["y_train"],
            data["X_val"], data["y_val"], epochs=epochs)
        loss, acc = evaluate_mnist(model, data["X_test"], data["y_test"])
        results["Beam-GD"]["loss"].append(loss)
        results["Beam-GD"]["accuracy"].append(acc)

    statistical_results = []
    for baseline in ["SGD", "SGD+Momentum", "Adam"]:
        result = statistical_comparison(
            np.array(results[baseline]["loss"]),
            np.array(results["Beam-GD"]["loss"]),
            baseline, "Beam-GD"
        )
        statistical_results.append(result)

    summary_df = format_statistical_results(statistical_results)

    print("\n--- MNIST Accuracy Summary ---")
    for method, data_dict in results.items():
        acc_mean = np.mean(data_dict["accuracy"]) * 100
        acc_std = np.std(data_dict["accuracy"]) * 100
        print(f"{method}: {acc_mean:.2f}% (+/- {acc_std:.2f}%)")

    return summary_df


def run_all_experiments():
    """Run all experiments and compile results."""
    print("\n" + "=" * 80)
    print("BEAM-GD COMPREHENSIVE EVALUATION")
    print("=" * 80)

    print("\nRunning Experiment 1: Statistical Validation...")
    stat_results = run_statistical_validation(n_trials=30)
    print("\n--- Statistical Validation Results ---")
    print(stat_results.to_string())

    print("\nRunning Experiment 2: MNIST Evaluation...")
    mnist_results = run_mnist_evaluation(n_trials=10)
    print("\n--- MNIST Evaluation Results ---")
    print(mnist_results.to_string())

    print("\nRunning Experiment 3: Ablation Study...")
    ablation_results = run_ablation_study(n_trials=30)
    print("\n--- Ablation Study Results ---")
    print(ablation_results.to_string())

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 80)

    return {
        "statistical_validation": stat_results,
        "mnist_evaluation": mnist_results,
        "ablation_study": ablation_results,
    }


if __name__ == "__main__":
    results = run_all_experiments()
