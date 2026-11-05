"""Confidence calibration (temperature scaling) and selective prediction ("I'm not sure")."""
import numpy as np
import torch


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Find one number T so that softmax(logits / T) gives honest probabilities.

    T > 1 softens over-confident predictions; T < 1 sharpens under-confident ones.
    The predicted class never changes (dividing by T keeps the order of the logits),
    so accuracy is untouched: only the confidence becomes truthful.
    T is chosen by minimising negative log-likelihood on held-out validation data.
    """
    z = torch.tensor(logits, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    log_t = torch.zeros(1, requires_grad=True)  # optimise log(T) so T stays positive
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp())


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def risk_coverage(probs: np.ndarray, labels: np.ndarray):
    """Answer only the most confident images and abstain on the rest.

    Returns (coverage, accuracy on answered images, threshold) for every possible cut-off.
    coverage = fraction of photos the app answers; the rest get "not sure, ask an expert".
    """
    conf = probs.max(1)
    correct = probs.argmax(1) == labels
    order = np.argsort(-conf)
    n = np.arange(1, len(conf) + 1)
    return n / len(conf), np.cumsum(correct[order]) / n, conf[order]
