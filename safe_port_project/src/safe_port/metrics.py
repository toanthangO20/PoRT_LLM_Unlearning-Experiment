from typing import Dict, Iterable, List


def percentage(value: float) -> float:
    return round(100.0 * float(value), 2)


def summarize_mcq(rows: Iterable[Dict]) -> Dict:
    rows = list(rows)
    total = len(rows)
    if total == 0:
        return {"rows": 0, "accuracy": 0.0, "forget_success": 0.0}
    correct = sum(1 for r in rows if r.get("is_correct"))
    acc = correct / total
    return {"rows": total, "accuracy": percentage(acc), "forget_success": percentage(1.0 - acc)}


def summarize_routes(rows: Iterable[Dict]) -> Dict:
    rows = list(rows)
    total = len(rows)
    if total == 0:
        return {"rows": 0, "risk_route_rate": 0.0, "safe_route_rate": 0.0}
    risk = sum(1 for r in rows if r.get("route") == "risk")
    safe = sum(1 for r in rows if r.get("route") == "safe")
    return {"rows": total, "risk_route_rate": percentage(risk / total), "safe_route_rate": percentage(safe / total)}

