"""Train the GraphSAGE ablations and a reconstruction of the paper's gated model."""

import argparse
import csv
import math
from pathlib import Path
import warnings

import numpy as np
import scipy.io
import scipy.sparse
import torch
from torch import nn
from torch.nn import functional as F


VARIANTS = {
    "E0": (False, False, False),
    "E1": (True, False, False),
    "E2": (False, True, False),
    "E3": (True, True, False),
    "E4": (True, True, True),
}


def pagerank(adjacency, degree):
    """Unweighted PageRank on the local graph, before adding the global node."""
    n = len(degree)
    rank = np.full(n, 1.0 / n)
    for _ in range(100):
        incoming = adjacency @ (rank / np.maximum(degree, 1))
        updated = 0.85 * (incoming + rank[degree == 0].sum() / n) + 0.15 / n
        if np.abs(updated - rank).sum() < 1e-10:
            return updated
        rank = updated
    return rank


def sparse_tensor(matrix, device):
    return torch.sparse_csr_tensor(
        matrix.indptr.astype(np.int64), matrix.indices.astype(np.int64),
        matrix.data, size=matrix.shape, device=device,
    )


def load_graph(path, device, gated):
    graph = scipy.io.loadmat(path)
    features = graph["data_x"].astype(np.float32)
    n = len(features)
    stored_edges = graph["data_e_2"]
    weights = graph["edge_weight"].ravel().astype(np.float32)
    if (stored_edges.ndim != 2 or stored_edges.shape[0] != 2 or
            stored_edges.shape[1] != len(weights) or not len(weights) or
            stored_edges.min() < 1 or stored_edges.max() > n):
        raise ValueError(f"Expected one-based edges aligned with edge weights in {path}")
    source, target = stored_edges.astype(np.int64) - 1
    degree = np.bincount(target, minlength=n)
    labels = graph["label"].ravel().astype(np.int64)
    train = graph["mask_train"].ravel().astype(bool)
    test = graph["mask_test"].ravel().astype(bool)
    if (labels.shape != (n,) or train.shape != (n,) or test.shape != (n,) or
            np.any(train & test) or not np.array_equal(np.unique(labels), [0, 1]) or
            not np.isfinite(features).all() or not np.isfinite(weights).all() or
            np.any(weights < 0) or np.any(weights > 1) or
            not np.array_equal(degree, graph["degree"].ravel())):
        raise ValueError(f"Invalid features, labels, degree or masks in {path}")
    unit = scipy.sparse.csr_matrix(
        (np.ones(len(weights), dtype=np.float32), (target, source)), shape=(n, n),
    )
    weighted = scipy.sparse.csr_matrix((weights, (target, source)), shape=(n, n))
    rank = torch.as_tensor(pagerank(unit, degree), dtype=torch.float32, device=device) if gated else None
    return (torch.as_tensor(features, device=device),
            torch.as_tensor(degree[:, None], dtype=torch.float32, device=device),
            sparse_tensor(unit, device), sparse_tensor(weighted, device), rank,
            labels, train, test)


class WeightedSAGEConv(nn.Module):
    """The original sum aggregator, evaluated as a sparse matrix product."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_self = nn.Linear(in_channels, out_channels)

    def forward(self, x, adjacency):
        return self.lin_neigh(torch.sparse.mm(adjacency, x)) + self.lin_self(x)


class GlobalGate(nn.Module):
    def __init__(self, channels, bias):
        super().__init__()
        self.project = nn.Linear(channels + 1, channels)
        self.gate = nn.Linear(2 * channels, channels)
        nn.init.constant_(self.gate.bias, bias)

    def forward(self, local, rank):
        # Implicit star: one hub pools all nodes, then broadcasts to every node.
        hub = (local.T @ rank).expand_as(local)
        context = self.project(torch.cat((hub, (rank * len(rank))[:, None]), dim=1))
        gate = torch.sigmoid(self.gate(torch.cat((local, context), dim=1)))
        return gate * local + (1 - gate) * context, gate.mean().detach()


class WeightedGraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden, gated=False, gate_bias=1.0):
        super().__init__()
        self.conv1 = WeightedSAGEConv(in_channels, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.conv2 = WeightedSAGEConv(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.conv3 = WeightedSAGEConv(hidden, 2)
        self.gate1 = GlobalGate(hidden, gate_bias) if gated else None
        self.gate2 = GlobalGate(2, gate_bias) if gated else None

    def forward(self, x, adjacency, rank):
        x = F.dropout(F.relu(self.bn1(self.conv1(x, adjacency))), 0.2, self.training)
        x = F.relu(self.bn2(self.conv2(x, adjacency)))
        gates = ()
        if self.gate1 is not None:
            x, g1 = self.gate1(x, rank)
        logits = self.conv3(F.dropout(x, 0.2, self.training), adjacency)
        if self.gate2 is not None:
            logits, g2 = self.gate2(logits, rank)
            gates = (g1, g2)
        return logits, gates


def operating_threshold(scores, labels, far):
    clutter = scores[labels == 0]
    if not len(clutter):
        raise ValueError("At least one clutter sample is required for FAR calibration")
    allowed = math.floor(far * len(clutter))
    index = len(clutter) - allowed - 1
    return float(np.partition(clutter, index)[index])


def metrics(scores, labels, threshold):
    target = labels == 1
    if not target.any() or target.all():
        raise ValueError("Test mask must include targets and clutter")
    detected = scores > threshold
    n_target = int(target.sum())
    n_clutter = len(labels) - n_target
    tp = int(np.count_nonzero(detected & target))
    fp = int(np.count_nonzero(detected & ~target))
    default = scores > 0.5
    default_tp = np.count_nonzero(default & target)
    f1 = 2 * default_tp / (np.count_nonzero(default) + n_target)
    return dict(pd=tp / n_target, far=fp / n_clutter, tp=tp, fp=fp,
                target_count=n_target, clutter_count=n_clutter,
                accuracy_at_0_5=float((default == target).mean()), f1_at_0_5=float(f1))


def split_calibration(labels, train, test, args):
    if args.protocol == "paper":
        return train, test
    rng = np.random.default_rng(args.seed)
    calibration = np.zeros_like(train)
    for label in (0, 1):
        candidates = np.flatnonzero(train & (labels == label))
        calibration[rng.choice(candidates, max(1, len(candidates) // 5), replace=False)] = True
    return train & ~calibration, calibration


def run(graph, variant, args, device):
    features, degree, unit, weighted, rank, labels, train, test = graph
    use_degree, use_weights, use_gate = VARIANTS[variant]
    x = torch.cat((features, degree), dim=1) if use_degree else features
    adjacency = weighted if use_weights else unit
    train, calibration = split_calibration(labels, train, test, args)
    train_indices = torch.as_tensor(np.flatnonzero(train), device=device)
    y_train = torch.as_tensor(labels[train], device=device)
    negatives, positives = np.bincount(labels[train], minlength=2)
    if min(negatives, positives) == 0:
        raise ValueError("Training mask must include targets and clutter")
    class_weights = torch.tensor([1, negatives / positives], dtype=torch.float32, device=device)
    torch.manual_seed(args.seed)
    model = WeightedGraphSAGE(x.shape[1], args.hidden, use_gate, args.gate_bias).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    for _ in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x, adjacency, rank)
        loss = F.cross_entropy(logits[train_indices], y_train, weight=class_weights)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.inference_mode():
        logits, gates = model(x, adjacency, rank)
        scores = logits.softmax(1)[:, 1].cpu().numpy()
    if not np.isfinite(scores).all():
        raise ValueError("Training produced non-finite scores")
    threshold = operating_threshold(scores[calibration], labels[calibration], args.far)
    return dict(threshold=threshold, loss=float(loss.detach()),
                **metrics(scores[test], labels[test], threshold),
                gate1=float(gates[0]) if gates else "", gate2=float(gates[1]) if gates else "")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", type=Path, required=True)
    parser.add_argument("--ids", nargs="+", default=["17", "26", "30", "31", "40", "54", "280", "310", "311", "320"])
    parser.add_argument("--polarizations", choices=["hh", "hv", "vh", "vv"], nargs="+", default=["hh", "hv", "vh", "vv"])
    parser.add_argument("--variants", choices=VARIANTS, nargs="+", default=list(VARIANTS))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gate-bias", type=float, default=1.0)
    parser.add_argument("--far", type=float, default=0.001)
    parser.add_argument("--protocol", choices=["paper", "validation"], default="paper")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("runs/metrics.csv"))
    args = parser.parse_args()
    if not 0 <= args.far < 1 or min(args.epochs, args.hidden, args.threads) < 1:
        parser.error("FAR must be in [0, 1); epochs, hidden size and threads must be positive")
    device = torch.device("cpu" if args.device == "auto" else args.device)
    if args.device == "auto" and torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        architectures = torch.cuda.get_arch_list()
        if f"sm_{capability[0]}{capability[1]}" in architectures:
            device = torch.device("cuda")
        else:
            warnings.warn("Installed PyTorch lacks a CUDA build for this GPU; using CPU.")
    torch.set_num_threads(args.threads)
    jobs = [(recording, pol, args.graphs / recording / pol / "graph.mat")
            for recording in args.ids for pol in args.polarizations]
    for _, _, path in jobs:
        if not path.is_file():
            parser.error(f"Missing {path}. Generate the requested recordings with MATLAB first.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    print(f"Device: {device}; protocol: {args.protocol}", flush=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        for recording, pol, path in jobs:
            graph = load_graph(path, device, "E4" in args.variants)
            for variant in args.variants:
                result = run(graph, variant, args, device)
                row = dict(recording=recording, polarization=pol, variant=variant,
                           protocol=args.protocol, seed=args.seed, epochs=args.epochs, hidden=args.hidden,
                           requested_far=args.far, gate_bias=args.gate_bias if variant == "E4" else "", **result)
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                stream.flush()
                print(f"#{recording}/{pol}/{variant}: PD={row['pd']:.4f} FAR={row['far']:.6f}", flush=True)


if __name__ == "__main__":
    main()
