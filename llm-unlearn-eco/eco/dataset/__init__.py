from .mmlu import MMLU
from .tofu import TOFU, TOFUPerturbed
from .truthfulqa import TruthfulQA
from .wmdp import WMDP, WMDPBio, WMDPChem, WMDPCyber


dataset_classes = {
    c.name: c
    for c in [
        MMLU,
        TOFU,
        TOFUPerturbed,
        TruthfulQA,
        WMDP,
        WMDPBio,
        WMDPChem,
        WMDPCyber,
    ]
}
