import copy
import gc
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from config import Config
from model import Regressor

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

sns.set_theme()
sns.set_context("paper")


def evaluate(model, data_loader, config):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0
    correct = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(config.device)
            targets = targets.to(config.device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            predicted = outputs.argmax(dim=1)
            correct += (predicted == targets).sum().item()

    average_loss = total_loss / max(total_samples, 1)
    accuracy = correct / max(total_samples, 1)
    model.train()
    return average_loss, accuracy


def run_swe_training(
    model,
    epochs_per_round,
    rounds,
    neurons_per_round,
    experiment_id,
    train_loader,
    val_loader,
    test_loader,
    config,
    train_dataset,
):
    round_metrics = []
    model.set_lr(0.001)
    print(f"{'=' * 50}\nStarting Training with swe Method (Experiment {experiment_id})\n{'=' * 50}")

    for round_idx, round_number in enumerate(tqdm(range(1, rounds + 1), desc="Rounds")):
        model.train()
        print(f"\n[Round {round_number}/{rounds}] {'-' * 30}")

        best_val_loss = float("inf")
        best_state = None
        patience = 20 + (10 if round_number == rounds else 0)
        patience_counter = 0

        max_epochs = epochs_per_round[round_idx]
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for epoch in range(1, max_epochs + 1):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            for inputs, targets in train_loader:
                inputs = inputs.to(config.device)
                targets = targets.to(config.device)

                batch_loss = model.update(inputs, targets)
                batch_size = inputs.size(0)
                epoch_loss += batch_loss * batch_size

                with torch.no_grad():
                    predictions = model(inputs).argmax(dim=1)
                epoch_total += batch_size
                epoch_correct += (predictions == targets).sum().item()

            epoch_loss /= max(len(train_loader.dataset), 1)

            model.eval()
            val_loss_total = 0.0
            val_samples = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(config.device)
                    targets = targets.to(config.device)
                    logits = model(inputs)
                    loss = model.criterion(logits, targets)
                    batch_size = inputs.size(0)
                    val_loss_total += loss.item() * batch_size
                    val_samples += batch_size
            val_loss = val_loss_total / max(val_samples, 1)

            print(f"  [Epoch {epoch}/{max_epochs}] TrainLoss: {epoch_loss:.4f} | ValLoss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

            model.train()

        if best_state is not None:
            model.load_state_dict(best_state)

        model.train()
        train_accuracy = epoch_correct / max(epoch_total, 1)
        test_loss, test_accuracy = evaluate(model, test_loader, config)

        round_metrics.append(
            {
                "round": round_number,
                "train_loss": epoch_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
            }
        )

        print(f"  Training Loss: {epoch_loss:.4f}, Training Accuracy: {train_accuracy * 100:.2f}%")
        print(f"  Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy * 100:.2f}%")

        if round_number != rounds:
            full_train_loader = DataLoader(train_dataset, batch_size=100, shuffle=False)
            _ = model.split(
                "swe",
                full_train_loader,
                neurons_per_round[round_idx],
                draw_details=["swe", round_number, experiment_id],
            )

        model.create_optimizer()
        model.set_lr(0.001)

    print(f"{'=' * 50}\nTraining Completed with swe Method\n{'=' * 50}")
    return round_metrics


def run_swe_experiments(
    epochs_per_round,
    rounds,
    neurons_per_round,
    num_experiments,
    input_size,
    hidden_size,
    output_size,
    train_loader,
    val_loader,
    test_loader,
    train_dataset,
):
    config = Config()

    template = Regressor(config, input_size, hidden_size, output_size)
    base_state = copy.deepcopy(template.state_dict())
    del template

    metrics_keys = ("train_loss", "train_accuracy", "test_loss", "test_accuracy")
    aggregated_metrics = {key: [] for key in metrics_keys}

    for experiment in range(1, num_experiments + 1):
        model = Regressor(config, input_size, hidden_size, output_size)
        model.load_state_dict(copy.deepcopy(base_state))
        model = model.to(config.device)

        round_metrics = run_swe_training(
            model=model,
            epochs_per_round=epochs_per_round,
            rounds=rounds,
            neurons_per_round=neurons_per_round,
            experiment_id=experiment,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=config,
            train_dataset=train_dataset,
        )

        for key in metrics_keys:
            aggregated_metrics[key].append([metrics[key] for metrics in round_metrics])

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mean_metrics = {key: np.mean(aggregated_metrics[key], axis=0) for key in metrics_keys}
    std_metrics = {key: np.std(aggregated_metrics[key], axis=0) for key in metrics_keys}

    return mean_metrics, std_metrics


def plot_swe_metrics(
    mean_metrics,
    std_metrics,
    rounds,
    final_train_loss,
    final_test_loss,
    filename_train="/gpu-data2/ncha/projects/extension/output/classification_train_loss_plot.pdf",
    filename_test="/gpu-data2/ncha/projects/extension/output/classification_test_loss_plot.pdf",
):
    rounds_axis = list(range(rounds))

    plt.figure(figsize=(8, 6))
    plt.plot(rounds_axis, mean_metrics["train_loss"], label="swe", marker="o")
    plt.fill_between(
        rounds_axis,
        mean_metrics["train_loss"] - std_metrics["train_loss"],
        mean_metrics["train_loss"] + std_metrics["train_loss"],
        alpha=0.2,
        label="Std Dev",
    )
    if final_train_loss:
        plt.axhline(y=final_train_loss, linestyle="--", label="Final Training Loss")
    plt.xlabel("Rounds")
    plt.ylabel("Mean Training Loss")
    plt.title("Mean Training Loss: swe")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename_train, format="pdf", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(rounds_axis, mean_metrics["test_loss"], label="swe", marker="o")
    plt.fill_between(
        rounds_axis,
        mean_metrics["test_loss"] - std_metrics["test_loss"],
        mean_metrics["test_loss"] + std_metrics["test_loss"],
        alpha=0.2,
        label="Std Dev",
    )
    if final_test_loss:
        plt.axhline(y=final_test_loss, linestyle="--", label=f"Final Test Loss: {final_test_loss:.4f}")
    plt.xlabel("Rounds")
    plt.ylabel("Mean Test Loss")
    plt.title("Mean Test Loss: swe")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename_test, format="pdf", dpi=300)
    plt.close()


def running_algorithm(
    input_size,
    hidden_size,
    output_size,
    train_loader,
    val_loader,
    test_loader,
    final_train_loss,
    final_test_loss,
    train_dataset,
):
    num_experiments = 3
    rounds = 8

    epochs_per_round = [250, 250, 250, 250, 250, 250, 250, 250]
    neurons_per_round = [36, 47, 60, 80, 102, 134, 174]

    mean_metrics, std_metrics = run_swe_experiments(
        epochs_per_round=epochs_per_round,
        rounds=rounds,
        neurons_per_round=neurons_per_round,
        num_experiments=num_experiments,
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_dataset=train_dataset,
    )

    print(mean_metrics)
    print(std_metrics)
    plot_swe_metrics(mean_metrics, std_metrics, rounds, final_train_loss, final_test_loss)


if __name__ == "__main__":
    original_stdout = sys.stdout
    log_path = "/gpu-data2/ncha/projects/extension/output/ablation_class3hlbackbone_cifar100.txt"

    with open(log_path, "w") as file:
        sys.stdout = file

        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )

        eval_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )

        root = "/gpu-data2/ncha/projects/extension/data"
        cifar_train_aug = datasets.CIFAR100(root=root, train=True, download=True, transform=train_transform)
        cifar_train_eval = datasets.CIFAR100(root=root, train=True, download=False, transform=eval_transform)
        cifar_test = datasets.CIFAR100(root=root, train=False, download=True, transform=eval_transform)

        targets = np.array(cifar_train_aug.targets)
        num_classes = 100
        train_ratio = 0.8

        train_idx, val_idx = [], []
        rng = np.random.default_rng(42)
        for cls in range(num_classes):
            cls_idx = np.where(targets == cls)[0]
            rng.shuffle(cls_idx)
            n_train = int(len(cls_idx) * train_ratio)
            train_idx.extend(cls_idx[:n_train])
            val_idx.extend(cls_idx[n_train:])

        train_idx = np.array(train_idx)
        val_idx = np.array(val_idx)
        rng.shuffle(train_idx)
        rng.shuffle(val_idx)

        train_dataset = Subset(cifar_train_aug, train_idx.tolist())
        val_dataset = Subset(cifar_train_eval, val_idx.tolist())

        train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(cifar_test, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)

        input_size = 784
        hidden_size = 188
        output_size = 100

        final_train_loss = 0.0
        final_test_loss = 0.0

        running_algorithm(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            final_train_loss=final_train_loss,
            final_test_loss=final_test_loss,
            train_dataset=train_dataset,
        )

    sys.stdout = original_stdout
