from __future__ import annotations


def confusion_matrix(y_true: list[int], y_pred: list[int], num_classes: int) -> list[list[int]]:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for truth, pred in zip(y_true, y_pred):
        matrix[truth][pred] += 1
    return matrix


def classification_report(y_true: list[int], y_pred: list[int], class_names: list[str]) -> dict:
    matrix = confusion_matrix(y_true, y_pred, len(class_names))
    total = max(len(y_true), 1)
    correct = sum(matrix[i][i] for i in range(len(class_names)))
    per_class = {}
    f1_values = []

    for idx, name in enumerate(class_names):
        tp = matrix[idx][idx]
        fp = sum(matrix[row][idx] for row in range(len(class_names))) - tp
        fn = sum(matrix[idx][col] for col in range(len(class_names))) - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        f1_values.append(f1)
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(matrix[idx]),
        }

    return {
        "accuracy": correct / total,
        "macro_f1": sum(f1_values) / max(len(f1_values), 1),
        "per_class": per_class,
    }

