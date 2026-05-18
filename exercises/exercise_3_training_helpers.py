import copy
import random

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from process_census_data import prepare_census_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 99


def set_seed(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()

train_x, val_x, test_x, train_y, val_y, test_y = prepare_census_data(
    random_state=SEED,
    split_validation=True,
    rebalance=False,
)

split_overview = pd.DataFrame(
    {
        "split": ["train", "validation", "test"],
        "rows": [len(train_x), len(val_x), len(test_x)],
        "positive_rate": [train_y.mean(), val_y.mean(), test_y.mean()],
    }
)

train_x_tensor = torch.tensor(train_x.to_numpy(), dtype=torch.float32)
val_x_tensor = torch.tensor(val_x.to_numpy(), dtype=torch.float32)
test_x_tensor = torch.tensor(test_x.to_numpy(), dtype=torch.float32)
train_y_tensor = torch.tensor(train_y.to_numpy(), dtype=torch.float32).unsqueeze(1)
val_y_tensor = torch.tensor(val_y.to_numpy(), dtype=torch.float32).unsqueeze(1)
test_y_tensor = torch.tensor(test_y.to_numpy(), dtype=torch.float32).unsqueeze(1)


def standardize_splits(train_features, *other_feature_splits):
    mean = train_features.mean(dim=0, keepdim=True)
    std = train_features.std(dim=0, keepdim=True)
    std = torch.where(std == 0, torch.ones_like(std), std)

    normalized_splits = [(train_features - mean) / std]
    for features in other_feature_splits:
        normalized_splits.append((features - mean) / std)

    return normalized_splits, mean, std


scaled_features, train_mean, train_std = standardize_splits(
    train_x_tensor, val_x_tensor, test_x_tensor
)
train_x_scaled, val_x_scaled, test_x_scaled = scaled_features


def make_weighted_sampler(targets):
    class_counts = torch.bincount(targets.squeeze(1).long())
    class_weights = 1.0 / class_counts.float().clamp(min=1)
    sample_weights = class_weights[targets.squeeze(1).long()]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def make_loaders(
    train_features,
    val_features,
    test_features,
    train_targets,
    val_targets,
    test_targets,
    *,
    batch_size=512,
    balance_strategy="none",
):
    train_dataset = TensorDataset(train_features, train_targets)
    val_dataset = TensorDataset(val_features, val_targets)
    test_dataset = TensorDataset(test_features, test_targets)

    sampler = None
    shuffle = True
    pos_weight = None

    if balance_strategy == "weighted_sampler":
        sampler = make_weighted_sampler(train_targets)
        shuffle = False
    elif balance_strategy == "pos_weight":
        positives = train_targets.sum()
        negatives = len(train_targets) - positives
        pos_weight = torch.tensor(
            [negatives / positives.clamp(min=1.0)],
            dtype=torch.float32,
            device=device,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 2)

    return train_loader, val_loader, test_loader, pos_weight


def get_activation(name):
    if name == "relu":
        return nn.ReLU()
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()

    raise ValueError(f"Unsupported activation: {name}")


class ConfigurableNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_dims,
        activation="relu",
        dropout=0.0,
        use_batch_norm=False,
        num_classes=1,
        output_activation=None,
    ):
        super().__init__()

        layers = []
        in_features = input_size

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = hidden_dim

        layers.append(nn.Linear(in_features, num_classes))
        self.network = nn.Sequential(*layers)
        self.output_activation = output_activation

    def forward(self, x):
        logits = self.network(x)
        if self.output_activation is not None:
            logits = get_activation(self.output_activation)(logits)

        return logits


def initialize_model(model, strategy="xavier_uniform"):
    for module in model.modules():
        if not isinstance(module, nn.Linear):
            continue

        if strategy == "xavier_uniform":
            nn.init.xavier_uniform_(module.weight)
        elif strategy == "xavier_normal":
            nn.init.xavier_normal_(module.weight)
        elif strategy == "kaiming_uniform":
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        elif strategy == "large_normal":
            nn.init.normal_(module.weight, mean=0.0, std=2.0)
        elif strategy == "zeros":
            nn.init.zeros_(module.weight)
        else:
            raise ValueError(f"Unsupported initialization: {strategy}")

        nn.init.zeros_(module.bias)

    return model


def evaluate_model(model, data_loader, device, loss_fn=None, threshold=0.5):
    model.eval()
    all_targets = []
    all_probabilities = []
    total_loss = 0.0

    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)

            logits = model(features)
            probabilities = torch.sigmoid(logits)

            all_probabilities.append(probabilities.cpu())
            all_targets.append(targets.cpu())

            if loss_fn is not None:
                total_loss += loss_fn(logits, targets).item() * features.size(0)

    y_true = torch.cat(all_targets).squeeze(1).numpy()
    y_score = torch.cat(all_probabilities).squeeze(1).numpy()
    y_pred = (y_score >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "positive_prediction_rate": y_pred.mean(),
    }

    if loss_fn is not None:
        metrics["loss"] = total_loss / len(data_loader.dataset)

    return metrics


def compute_gradient_norm(model):
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        total += parameter.grad.detach().norm(2).item() ** 2

    return total**0.5


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    *,
    epochs=10,
    learning_rate=1e-3,
    optimizer_name="adam",
    weight_decay=0.0,
    gradient_clip_norm=None,
    pos_weight=None,
    seed=SEED,
):
    if seed is not None:
        set_seed(seed)

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    history = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        gradient_norms = []

        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = loss_fn(logits, targets)
            loss.backward()

            gradient_norms.append(compute_gradient_norm(model))

            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)

            optimizer.step()
            running_loss += loss.item() * features.size(0)

        validation_metrics = evaluate_model(model, val_loader, device, loss_fn)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": running_loss / len(train_loader.dataset),
                "mean_grad_norm": sum(gradient_norms) / max(len(gradient_norms), 1),
                **validation_metrics,
            }
        )

    return model, pd.DataFrame(history)


BASELINE_CONFIG = {
    "input_size": train_x_tensor.shape[1],
    "hidden_dims": [64, 32],
    "activation": "relu",
    "dropout": 0.1,
    "use_batch_norm": False,
    "output_activation": None,
    "init": "xavier_uniform",
    "learning_rate": 1e-3,
    "optimizer_name": "adam",
    "weight_decay": 1e-4,
    "gradient_clip_norm": None,
    "epochs": 10,
    "batch_size": 512,
    "balance_strategy": "none",
    "use_scaled_features": True,
    "subset_fraction": 1.0,
    "positive_class_keep_fraction": 1.0,
    "label_noise": 0.0,
    "train_feature_noise": 0.0,
    "rare_feature_fraction": 0.0,
    "rare_feature_scale": 25.0,
    "distribution_shift_strength": 0.0,
    "leak_validation_into_train": False,
    "seed": SEED,
}


EXERCISE_SCENARIOS = {
    "Vanishing gradients": {
        "category": "optimization",
        "symptom": "Gradient norms collapse and validation loss barely improves.",
        "first_fixes_to_try": "ReLU or LeakyReLU, better initialization, batch norm, fewer layers.",
        "overrides": {
            "hidden_dims": [256, 256, 256, 256, 256, 256, 256, 256],
            "activation": "sigmoid",
            "learning_rate": 1e-4,
        },
    },
    "Exploding gradients": {
        "category": "optimization",
        "symptom": "Loss spikes, gradients blow up, and training becomes unstable.",
        "first_fixes_to_try": "Lower learning rate, clip gradients, safer initialization.",
        "overrides": {
            "hidden_dims": [256, 512, 512, 512],
            "activation": "relu",
            "init": "large_normal",
            "optimizer_name": "sgd",
            "learning_rate": 1e-1,
        },
    },
    "Slow convergence": {
        "category": "optimization",
        "symptom": "Loss moves in the right direction but very slowly.",
        "first_fixes_to_try": "Increase learning rate, switch optimizer, batch norm, feature scaling.",
        "overrides": {
            "optimizer_name": "sgd",
            "learning_rate": 1e-5,
            "batch_size": 4096,
            "weight_decay": 1e-2,
        },
    },
    "Getting stuck in local minima or saddle points": {
        "category": "optimization",
        "symptom": "Training plateaus early and then barely escapes.",
        "first_fixes_to_try": "Change optimizer, adjust learning rate, add noise via smaller batches.",
        "overrides": {
            "hidden_dims": [128, 128, 128, 128, 128, 128],
            "activation": "tanh",
            "optimizer_name": "sgd",
            "learning_rate": 1e-4,
            "batch_size": 4096,
        },
    },
    "Poor learning rate (too high or too low)": {
        "category": "optimization",
        "symptom": "Either oscillation or nearly no progress.",
        "first_fixes_to_try": "Try orders of magnitude up or down and inspect the curves.",
        "overrides": {
            "optimizer_name": "sgd",
            "learning_rate": 2e-1,
        },
    },
    "Insufficient training data": {
        "category": "data",
        "symptom": "Training looks good but generalization is weak.",
        "first_fixes_to_try": "Use more data, simplify the model, add regularization.",
        "overrides": {
            "subset_fraction": 0.003,
            "epochs": 50,
        },
    },
    # "Noisy or mislabeled data": {
    #     "category": "data",
    #     "symptom": "Training and validation both become noisy or inconsistent.",
    #     "first_fixes_to_try": "Reduce label noise, clean data, use robust monitoring.",
    #     "overrides": {
    #         "label_noise": 0.4,
    #         "train_feature_noise": 0.2,
    #     },
    # },
    "Class imbalance": {
        "category": "data",
        "symptom": "Accuracy looks acceptable but recall and F1 stay weak.",
        "first_fixes_to_try": "Use pos_weight or a weighted sampler, then compare metrics.",
        "overrides": {
            "balance_strategy": "none",
            "positive_class_keep_fraction": 0.15,
        },
    },
    "Poor feature scaling": {
        "category": "data",
        "symptom": "Optimization becomes erratic because input scales differ.",
        "first_fixes_to_try": "Enable standardization and inspect convergence again.",
        "overrides": {
            "use_scaled_features": False,
            "rare_feature_fraction": 0.6,
            "rare_feature_scale": 100.0,
        },
    },
    "Sparse or rare features": {
        "category": "data",
        "symptom": "A few weak features carry signal and are hard to learn from.",
        "first_fixes_to_try": "Rescale rare features, add capacity, inspect feature engineering.",
        "overrides": {
            "rare_feature_fraction": 0.6,
            "rare_feature_scale": 100.0,
        },
    },
    "Overfitting": {
        "category": "generalization",
        "symptom": "Training improves while validation degrades or stalls.",
        "first_fixes_to_try": "Add dropout or weight decay, reduce capacity, stop earlier.",
        "overrides": {
            "hidden_dims": [512, 512, 256, 128],
            "dropout": 0.0,
            "weight_decay": 0.0,
            "epochs": 10,
            "subset_fraction": 0.12,
        },
    },
    "Underfitting": {
        "category": "generalization",
        "symptom": "Both training and validation stay poor.",
        "first_fixes_to_try": "Increase capacity, train longer, reduce regularization.",
        "overrides": {
            "hidden_dims": [4],
            "epochs": 10,
            "learning_rate": 1e-4,
            "weight_decay": 1e-2,
        },
    },
    "High variance / high bias": {
        "category": "generalization",
        "symptom": "Large train-validation gap or weak performance everywhere.",
        "first_fixes_to_try": "Diagnose whether you need more regularization or more capacity.",
        "overrides": {
            "hidden_dims": [512, 512, 256],
            "dropout": 0.0,
            "weight_decay": 0.0,
            "subset_fraction": 0.1,
            "epochs": 10,
        },
    },
    "Bad weight initialization": {
        "category": "architecture",
        "symptom": "Network fails to break symmetry or starts in a bad scale.",
        "first_fixes_to_try": "Use Xavier or Kaiming initialization matched to the activation.",
        "overrides": {
            "init": "zeros",
        },
    },
    "Poor model architecture choice": {
        "category": "architecture",
        "symptom": "Model capacity or structure does not match the problem.",
        "first_fixes_to_try": "Change depth, width, normalization, or activation family.",
        "overrides": {
            "hidden_dims": [],
        },
    },
    "Activation function issues (e.g., dying ReLUs, saturation)": {
        "category": "architecture",
        "symptom": "Units saturate or stop updating.",
        "first_fixes_to_try": "Use LeakyReLU, GELU, lower learning rate, or add normalization.",
        "overrides": {
            "hidden_dims": [256, 256, 256, 256],
            "activation": "relu",
            "learning_rate": 1e-1,
        },
    },
    "Too little regularization": {
        "category": "regularization",
        "symptom": "Model memorizes the training set too easily.",
        "first_fixes_to_try": "Add dropout, weight decay, or early stopping.",
        "overrides": {
            "hidden_dims": [512, 512, 256],
            "dropout": 0.0,
            "weight_decay": 0.0,
            "epochs": 10,
            "subset_fraction": 0.15,
        },
    },
    "Too much regularization": {
        "category": "regularization",
        "symptom": "Model struggles to fit even the training data.",
        "first_fixes_to_try": "Reduce dropout and weight decay, increase epochs or width.",
        "overrides": {
            "dropout": 0.75,
            "weight_decay": 5e-1,
        },
    },
    "Numerical instability (e.g., division by zero, log(0))": {
        "category": "stability",
        "symptom": "Loss becomes NaN or infinite, or jumps unpredictably.",
        "first_fixes_to_try": "Lower learning rate, inspect activations, clip gradients, guard edge cases.",
        "overrides": {
            "init": "large_normal",
            "learning_rate": 2e-1,
            "optimizer_name": "sgd",
            "use_scaled_features": False,
        },
    },
    "Implementation bugs (e.g., incorrect gradients)": {
        "category": "stability",
        "symptom": "Metrics do not make sense even when the code seems to run.",
        "first_fixes_to_try": "Check tensor shapes, loss-input expectations, and gradient flow.",
        "overrides": {
            "output_activation": "sigmoid",
        },
    },
    "Batch size issues (too small or too large)": {
        "category": "optimization",
        "symptom": "Training is either noisy or too smooth and slow to adapt.",
        "first_fixes_to_try": "Compare small and large batches while watching speed and F1.",
        "overrides": {
            "batch_size": 8192,
            "optimizer_name": "sgd",
        },
    },
    "Data leakage": {
        "category": "evaluation",
        "symptom": "Validation performance looks unrealistically strong.",
        "first_fixes_to_try": "Remove leaked rows or features and rebuild the split.",
        "overrides": {
            "epochs": 15,
            "leak_validation_into_train": True,
        },
    },
    "Wrong evaluation metric": {
        "category": "evaluation",
        "symptom": "The chosen metric hides the failure mode you care about.",
        "first_fixes_to_try": "Compare accuracy, precision, recall, F1, and prediction rate.",
        "overrides": {},
    },
    "Train-test distribution mismatch": {
        "category": "evaluation",
        "symptom": "Validation and test behavior diverge because the data shifts.",
        "first_fixes_to_try": "Measure the shift, retrain on better data, or adapt preprocessing.",
        "overrides": {
            "distribution_shift_strength": 6.0,
        },
    },
    "Compute limitations (training too slow)": {
        "category": "systems",
        "symptom": "The model is too expensive for interactive iteration.",
        "first_fixes_to_try": "Use a smaller model, larger batches, or fewer epochs while prototyping.",
        "overrides": {
            "hidden_dims": [768, 768, 512, 256],
            "batch_size": 32,
            "epochs": 10,
        },
    },
    "Memory constraints": {
        "category": "systems",
        "symptom": "The chosen batch or model size exceeds available memory.",
        "first_fixes_to_try": "Reduce batch size, shrink the model, or move less data at once.",
        "overrides": {
            "hidden_dims": [768, 768, 512],
            "batch_size": 4096,
        },
    },
}

scenario_table = pd.DataFrame(
    [
        {
            "issue": issue,
            "category": spec["category"],
            "symptom": spec["symptom"],
            "first_fixes_to_try": spec["first_fixes_to_try"],
        }
        for issue, spec in EXERCISE_SCENARIOS.items()
    ]
).sort_values(["category", "issue"]).reset_index(drop=True)


def prepare_experiment_data(config):
    if config["use_scaled_features"]:
        train_features = train_x_scaled.clone()
        val_features = val_x_scaled.clone()
        test_features = test_x_scaled.clone()
    else:
        train_features = train_x_tensor.clone()
        val_features = val_x_tensor.clone()
        test_features = test_x_tensor.clone()

    train_targets = train_y_tensor.clone()
    val_targets = val_y_tensor.clone()
    test_targets = test_y_tensor.clone()

    positive_class_keep_fraction = config.get("positive_class_keep_fraction", 1.0)
    if positive_class_keep_fraction < 1.0:
        positive_indices = torch.where(train_targets.squeeze(1) == 1)[0]
        negative_indices = torch.where(train_targets.squeeze(1) == 0)[0]
        keep_count = max(1, int(len(positive_indices) * positive_class_keep_fraction))
        kept_positive_indices = positive_indices[torch.randperm(len(positive_indices))[:keep_count]]
        selected_indices = torch.cat([negative_indices, kept_positive_indices])
        selected_indices = selected_indices[torch.randperm(len(selected_indices))]
        train_features = train_features[selected_indices]
        train_targets = train_targets[selected_indices]

    rare_feature_fraction = config.get("rare_feature_fraction", 0.0)
    if rare_feature_fraction > 0:
        num_features = train_features.shape[1]
        num_rare_features = max(1, int(num_features * rare_feature_fraction))
        rare_indices = torch.arange(num_features - num_rare_features, num_features)
        scale = config.get("rare_feature_scale", 25.0)

        for features in (train_features, val_features, test_features):
            features[:, rare_indices] = features[:, rare_indices] / scale

    train_feature_noise = config.get("train_feature_noise", 0.0)
    if train_feature_noise > 0:
        train_features = train_features + torch.randn_like(train_features) * train_feature_noise

    label_noise = config.get("label_noise", 0.0)
    if label_noise > 0:
        noise_mask = torch.rand(len(train_targets)) < label_noise
        train_targets[noise_mask] = 1.0 - train_targets[noise_mask]

    subset_fraction = config.get("subset_fraction", 1.0)
    if subset_fraction < 1.0:
        subset_size = max(32, int(len(train_features) * subset_fraction))
        subset_indices = torch.randperm(len(train_features))[:subset_size]
        train_features = train_features[subset_indices]
        train_targets = train_targets[subset_indices]

    if config.get("leak_validation_into_train", False):
        train_features = torch.cat([train_features, val_features], dim=0)
        train_targets = torch.cat([train_targets, val_targets], dim=0)

    distribution_shift_strength = config.get("distribution_shift_strength", 0.0)
    if distribution_shift_strength > 0:
        shift_columns = min(10, train_features.shape[1])
        val_features[:, :shift_columns] = (
            val_features[:, :shift_columns] + distribution_shift_strength
        )
        test_features[:, :shift_columns] = (
            test_features[:, :shift_columns] + distribution_shift_strength
        )

    return train_features, val_features, test_features, train_targets, val_targets, test_targets


def build_experiment_config(issue=None, custom_overrides=None):
    if isinstance(issue, int):
        issue = scenario_table.loc[issue, "issue"]
    config = copy.deepcopy(BASELINE_CONFIG)
    if issue is not None:
        config.update(EXERCISE_SCENARIOS[issue]["overrides"])
    if custom_overrides is not None:
        config.update(custom_overrides)

    return config


def get_symptoms_for_issue(issue):
    if isinstance(issue, int):
        issue = scenario_table.loc[issue, "issue"]
    return scenario_table.loc[scenario_table["issue"] == issue, "symptom"].values[0]

def get_fixes_for_issue(issue):
    if isinstance(issue, int):
        issue = scenario_table.loc[issue, "issue"]
    return scenario_table.loc[scenario_table["issue"] == issue, "first_fixes_to_try"].values[0]


def run_experiment(issue=None, custom_overrides=None):
    config = build_experiment_config(issue, custom_overrides)
    data_splits = prepare_experiment_data(config)
    train_loader, val_loader, test_loader, pos_weight = make_loaders(
        *data_splits,
        batch_size=config["batch_size"],
        balance_strategy=config["balance_strategy"],
    )

    model = ConfigurableNN(
        input_size=config["input_size"],
        hidden_dims=config["hidden_dims"],
        activation=config["activation"],
        dropout=config["dropout"],
        use_batch_norm=config["use_batch_norm"],
        output_activation=config["output_activation"],
    ).to(device)
    model = initialize_model(model, config["init"])

    trained_model, training_history = train_model(
        model,
        train_loader,
        val_loader,
        device,
        epochs=config["epochs"],
        learning_rate=config["learning_rate"],
        optimizer_name=config["optimizer_name"],
        weight_decay=config["weight_decay"],
        gradient_clip_norm=config["gradient_clip_norm"],
        pos_weight=pos_weight,
        seed=config["seed"],
    )

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    test_metrics = pd.Series(evaluate_model(trained_model, test_loader, device, loss_fn))

    return config, trained_model, training_history, test_metrics


__all__ = [
    "SEED",
    "scenario_table",
    "split_overview",
    "run_experiment",
]