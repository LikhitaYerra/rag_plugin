import csv
import random
import math
import json
from typing import List, Tuple


def generate_rows(n: int = 100) -> List[Tuple[str, int, str, str, int]]:
    rows = []
    for i in range(n):
        age = random.randint(18, 90)
        gender = random.choice(["Male", "Female"])
        ethnicity = random.choice(["A", "B", "C"])
        _id = f"ID{i:05d}"
        # simple rule: higher age -> higher risk
        prob = 1 / (1 + math.exp(-(age - 50) / 10))
        label = 1 if random.random() < prob else 0
        rows.append((_id, age, gender, ethnicity, label))
    return rows


def train_simple_model(rows: List[Tuple[str, int, str, str, int]], epochs: int = 200, lr: float = 0.001):
    weights = {"age": 0.0, "gender": 0.0, "ethnicity": 0.0, "bias": 0.0}
    for _ in range(epochs):
        for _, age, gender, eth, label in rows:
            g = 1.0 if gender == "Male" else 0.0
            e = {"A": 0.0, "B": 1.0, "C": 2.0}[eth]
            z = weights["age"] * age + weights["gender"] * g + weights["ethnicity"] * e + weights["bias"]
            pred = 1.0 / (1.0 + math.exp(-z))
            error = label - pred
            weights["age"] += lr * error * age
            weights["gender"] += lr * error * g
            weights["ethnicity"] += lr * error * e
            weights["bias"] += lr * error
    return weights


def main():
    rows = generate_rows()
    with open("data/sample_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Age", "Gender", "Ethnicity", "Disease"])
        writer.writerows(rows)

    weights = train_simple_model(rows)
    with open("data/model.json", "w") as f:
        json.dump(weights, f)

    print("Generated data/sample_data.csv and data/model.json")


if __name__ == "__main__":
    main()
