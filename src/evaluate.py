"""Evaluate trained pipelines and produce a comparison table."""

from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline


def evaluate_pipeline(pipeline: Pipeline, X_train, y_train, X_test, y_test) -> dict:
    """Compute train accuracy, test accuracy, and macro F1 for one pipeline."""
    train_acc = accuracy_score(y_train, pipeline.predict(X_train))
    test_acc = accuracy_score(y_test, pipeline.predict(X_test))
    test_f1 = f1_score(y_test, pipeline.predict(X_test), average="macro")
    return {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
    }


def print_comparison(results: dict[str, dict]) -> None:
    """Print a comparison table across models. results = {name: metrics_dict}."""
    print(f"\n{'Model':<22} {'Train Acc':>10} {'Test Acc':>10} {'Test F1':>10}")
    print("-" * 54)
    for name, metrics in results.items():
        print(
            f"{name:<22} "
            f"{metrics['train_accuracy']:>10.4f} "
            f"{metrics['test_accuracy']:>10.4f} "
            f"{metrics['test_macro_f1']:>10.4f}"
        )


def print_classification_report(pipeline: Pipeline, X_test, y_test, target_names) -> None:
    """Detailed per-class metrics for one pipeline."""
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=target_names))


def select_best(results: dict[str, dict], metric: str = "test_accuracy") -> str:
    """Return the name of the pipeline with the highest score on the given metric."""
    return max(results, key=lambda name: results[name][metric])
