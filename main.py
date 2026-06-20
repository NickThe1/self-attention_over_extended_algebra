import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from rich.console import Console

from data.synthetic import make_dataloaders, majority_baseline, VOCAB_SIZE
from model import DualTransformerClassifier
from training.loop import run_training
from training.diagnostics import experiment_a_gradient_norms, experiment_b_ablation, experiment_c_perturbation

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


if __name__ == "__main__":
    main()
