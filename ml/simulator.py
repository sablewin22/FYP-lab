import copy
import pandas as pd
from ml.predict import predict


def simulate(base_input: dict, scenarios: list[dict]) -> list[dict]:
    base_class, base_probs, _ = predict(base_input)
    results = []

    for scenario in scenarios:
        modified = copy.deepcopy(base_input)
        modified.update(scenario["changes"])
        scenario_class, scenario_probs, _ = predict(modified)

        deltas = {}
        for cls in base_probs:
            deltas[cls] = round(scenario_probs[cls] - base_probs[cls], 4)

        results.append({
            "name": scenario["name"],
            "class": scenario_class,
            "probabilities": scenario_probs,
            "deltas": deltas,
        })

    return results, base_class, base_probs
