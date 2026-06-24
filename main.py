import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from rich.console import Console
from rich.table import Table

from data.synthetic import make_dataloaders, majority_baseline, VOCAB_SIZE, SEQ_LEN
from model import DualTransformerClassifier
from training.loop import run_training
from training.diagnostics import (
    experiment_a_gradient_norms,
    experiment_b_ablation,
    experiment_c_perturbation,
    experiment_10_comparison,
)

console = Console()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[bold]Device:[/bold] {device}")

    train_loader, test_loader = make_dataloaders(n_train=4000, n_test=1000, batch_size=64)
    baseline = majority_baseline(train_loader.dataset)
    console.print(f"[bold]Majority baseline:[/bold] {baseline:.1%}")

    model = DualTransformerClassifier(vocab_size=VOCAB_SIZE, d_model=32, n_heads=2)
    model.to(device)

    console.rule("[bold blue]Training")
    run_training(model, train_loader, test_loader, n_epochs=20, device=device)

    console.rule("[bold blue]Diagnostics")
    console.print("\n[bold]Experiment A — Gradient norms[/bold]")
    experiment_a_gradient_norms(model, train_loader, device)

    console.print("\n[bold]Experiment B — Ablation (b=0)[/bold]")
    ablation = experiment_b_ablation(train_loader, test_loader, device)
    console.print(ablation)

    console.print("\n[bold]Experiment C — Perturbation[/bold]")
    perturb = experiment_c_perturbation(model, test_loader, device)
    console.print(perturb)

    console.rule("[bold blue]Phase 10 — b-Component Contribution Audit")
    results = experiment_10_comparison(
        train_loader, test_loader, device,
        d_model=32, n_heads=2, n_epochs=20,
        vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN,
    )

    # ── summary table ─────────────────────────────────────────────────────────
    tbl = Table(title="Four-way comparison", show_lines=True)
    tbl.add_column("Variant")
    tbl.add_column("d_model", justify="right")
    tbl.add_column("Params", justify="right")
    tbl.add_column("Test acc", justify="right")
    tbl.add_column("b-mag epoch 1", justify="right")
    tbl.add_column("b-mag epoch 20", justify="right")

    label_map = {
        "dual":       "Dual (b dead)",
        "real_small": "Real-small",
        "real_big":   "Real-big (matched)",
        "fixed_dual": "Fixed-dual (b->logit)",
    }
    for key, label in label_map.items():
        r = results[key]
        bm = r["b_mag_per_epoch"]
        b1  = f"{bm[0]:.6f}"  if bm else "n/a"
        b20 = f"{bm[-1]:.6f}" if bm else "n/a"
        tbl.add_row(label, str(r["d_model"]), str(r["n_params"]),
                    f"{r['test_acc']:.4f}", b1, b20)
    console.print(tbl)

    # ── per-epoch accuracy for every variant ─────────────────────────────────
    console.print("\n[bold]Per-epoch test accuracy[/bold]")
    acc_tbl = Table(show_lines=False)
    acc_tbl.add_column("Epoch", justify="right")
    for label in label_map.values():
        acc_tbl.add_column(label, justify="right")

    n_epochs = len(next(iter(results.values()))["history"])
    for i in range(n_epochs):
        row = [str(i + 1)]
        for key in label_map:
            row.append(f"{results[key]['history'][i]['test_acc']:.4f}")
        acc_tbl.add_row(*row)
    console.print(acc_tbl)

    # ── b-magnitude evolution (dual variants) ────────────────────────────────
    console.print("\n[bold]b-magnitude per epoch (mean |b_repr| after attention)[/bold]")
    b_tbl = Table(show_lines=False)
    b_tbl.add_column("Epoch", justify="right")
    b_tbl.add_column("Dual (b dead)", justify="right")
    b_tbl.add_column("Fixed-dual (b->logit)", justify="right")
    for i in range(n_epochs):
        bd  = results["dual"]["b_mag_per_epoch"][i]
        bfd = results["fixed_dual"]["b_mag_per_epoch"][i]
        b_tbl.add_row(str(i + 1), f"{bd:.8f}", f"{bfd:.8f}")
    console.print(b_tbl)

    console.print("\n[bold green]Phase 10 complete.[/bold green]")


if __name__ == "__main__":
    main()
