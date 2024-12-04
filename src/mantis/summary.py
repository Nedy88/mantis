"""Summarize stats for the model and training."""

from rich import print
from rich.table import Table
from torch import nn


def format_large_number(number: float) -> str:
    """Pretty formatting for large numbers."""
    if number < 1e-3:  # noqa: PLR2004
        return "-"
    for unit in ["", "K", "M", "B", "T"]:
        if abs(number) < 1000:  # noqa: PLR2004
            return f"{number:.1f}{unit}"
        number /= 1000
    return f"{number}"


def get_param_count_detailed(model: nn.Module) -> dict[str, tuple[int, int]]:
    """Get the total number of params grouped by first level modules."""
    param_count: dict[str, tuple[int, int]] = {}
    for name, param in model.named_parameters():
        module_name = name.split(".")[0]
        if module_name not in param_count:
            param_count[module_name] = (0, 0)
        trainable, non_trainable = param_count[module_name]
        if param.requires_grad:
            trainable += param.numel()
        else:
            non_trainable += param.numel()
        param_count[module_name] = (trainable, non_trainable)
    return param_count


def print_table_with_param_counts(model: nn.Module, title: str) -> None:
    """Print a summary with number of model parameters grouped by top-level module."""
    table = Table(title=title, style="bold dark_sea_green2", title_style="bold orange1")
    table.add_column("Component", justify="right")
    table.add_column("Trainable params", justify="center", style="bold orchid")
    table.add_column("Non-trainable params", justify="center", style="orchid")
    param_counts = get_param_count_detailed(model)
    total_trainable, total_non_trainable = 0, 0
    for module_name, count in param_counts.items():
        table.add_row(module_name, format_large_number(count[0]), format_large_number(count[1]))
        total_trainable += count[0]
        total_non_trainable += count[1]
    table.add_section()
    table.add_row(
        "Total",
        format_large_number(total_trainable),
        format_large_number(total_non_trainable),
        style="bold",
    )
    print(table)
