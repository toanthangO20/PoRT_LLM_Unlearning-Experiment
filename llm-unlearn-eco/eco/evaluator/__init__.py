from .answer_prob import AnswerProb
from .choice_by_top_logit import ChoiceByTopLogit
from .choice_by_top_prob import ChoiceByTopProb
from .normalized_answer_prob import NormalizedAnswerProb
from .rouge_recall import ROUGERecall
from .truth_ratio import TruthRatio

try:
    from .rouge import ROUGE
except ModuleNotFoundError:
    ROUGE = None

_classes = [
        AnswerProb,
        ChoiceByTopLogit,
        ChoiceByTopProb,
        NormalizedAnswerProb,
        ROUGERecall,
        TruthRatio,
    ]
if ROUGE is not None:
    _classes.append(ROUGE)

evaluator_classes = {c.name: c for c in _classes}
