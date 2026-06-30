from dataclasses import dataclass
from typing import Dict


RISK_KEYWORDS = {
    "bio": ["pathogen", "culture", "toxin", "viral", "bacteria"],
    "cyber": ["exploit", "payload", "privilege escalation", "credential", "malware"],
    "chem": ["synthesis", "precursor", "toxic", "reaction yield", "purification"],
}


SAFE_INTENT_WORDS = ["defensive", "safety", "education", "policy", "high-level", "awareness", "prevention"]


@dataclass
class RouteDecision:
    route: str
    risk_score: float
    confidence: float
    reason: str


def score_text_risk(text: str) -> float:
    lower = text.lower()
    hits = 0
    for words in RISK_KEYWORDS.values():
        hits += sum(1 for word in words if word in lower)
    safe_hits = sum(1 for word in SAFE_INTENT_WORDS if word in lower)
    score = min(1.0, hits / 4.0)
    if safe_hits:
        score = max(0.0, score - min(0.35, 0.12 * safe_hits))
    return float(score)


def post_judge(prompt: str, answer: str, router_cfg: Dict, entropy: float = 0.0, disagreement: float = 0.0) -> RouteDecision:
    prompt_risk = score_text_risk(prompt)
    answer_risk = score_text_risk(answer)
    risk_score = min(1.0, 0.45 * prompt_risk + 0.45 * answer_risk + 0.05 * entropy + 0.05 * disagreement)
    confidence = max(0.0, min(1.0, 1.0 - abs(risk_score - 0.5)))
    risk_threshold = float(router_cfg.get("risk_threshold", 0.55))
    confidence_threshold = float(router_cfg.get("confidence_threshold", 0.60))
    if risk_score < risk_threshold and confidence >= confidence_threshold:
        return RouteDecision("safe", risk_score, confidence, "risk below threshold and confidence acceptable")
    return RouteDecision("risk", risk_score, confidence, "risk high or confidence low")


def safe_rethink_response() -> str:
    return (
        "I cannot provide operational or harmful details. I can help with high-level safety, "
        "defensive, educational, or policy-oriented information instead."
    )

