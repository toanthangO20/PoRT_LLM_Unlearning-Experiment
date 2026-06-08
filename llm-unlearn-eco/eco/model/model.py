import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)

from eco.utils import load_yaml


def _resolve_torch_dtype(dtype_name):
    if dtype_name is None:
        return torch.float16 if torch.cuda.is_available() else torch.float32
    if dtype_name == "auto":
        return "auto"
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"Unsupported torch_dtype in model config: {dtype_name}")
    return dtype_map[dtype_name]


class HFModel:
    def __init__(
        self,
        model_name,
        model_path=None,
        config_path="./config",
        generation_config=None,
    ):
        self.model_name = model_name
        self.model_config = load_yaml(f"{config_path}/{model_name}.yaml")
        load_in_4bit = self.model_config.get("load_in_4bit", False)
        load_in_8bit = self.model_config.get("load_in_8bit", False)
        quantization_config = (
            BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
            )
            if load_in_4bit or load_in_8bit
            else None
        )
        model_args = {
            "torch_dtype": _resolve_torch_dtype(self.model_config.get("torch_dtype")),
            "device_map": "auto",
            "quantization_config": quantization_config,
            "trust_remote_code": self.model_config.get(
                "trust_remote_code",
                False
                if "c4ai-command-r-v01" in model_name.lower()
                or "falcon" in model_name.lower()
                or "phi-1_5" in model_name.lower()
                else True
            ),
        }
        if self.model_config.get("attn_implementation"):
            model_args["attn_implementation"] = self.model_config["attn_implementation"]

        model_source = model_path if model_path else self.model_config["hf_name"]
        hf_config = AutoConfig.from_pretrained(
            model_source,
            trust_remote_code=model_args["trust_remote_code"],
        )
        if not hasattr(hf_config, "pad_token_id"):
            hf_config.pad_token_id = getattr(hf_config, "eos_token_id", None)
        model_args["config"] = hf_config
        self.model = AutoModelForCausalLM.from_pretrained(model_source, **model_args)

        num_parameters = sum(p.numel() for p in self.model.parameters())
        print(f"Number of parameters: {num_parameters}")

        tokenizer_args = {
            "trust_remote_code": self.model_config.get(
                "trust_remote_code",
                False
                if "c4ai-command-r-v01" in model_name.lower()
                or "falcon" in model_name.lower()
                or "phi-1_5" in model_name.lower()
                else True
            )
        }

        self.tokenizer = AutoTokenizer.from_pretrained(
            (
                model_source
                if "openelm" not in model_name.lower()
                else "meta-llama/Llama-2-7b-hf"
            ),
            **tokenizer_args,
        )

        self.model.generation_config = (
            GenerationConfig(do_sample=False, use_cache=True)
            if generation_config is None
            else generation_config
        )
        self.device = (
            self.model.device
            if hasattr(self.model, "device")
            else next(self.model.parameters()).device
        )
        self.generation_config = self.model.generation_config
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Prevent error caused by padding_side for qwen model
        if "qwen" in model_name.lower() or "starcoder2" in model_name.lower():
            self.tokenizer.padding_side = "left"

    def __call__(self, *args, **kwargs):
        # Remove the "prompts" key from the kwargs if it exists
        for key in ["prompts", "answers"]:
            if key in kwargs:
                kwargs.pop(key, None)
        # Prevent error caused by token_type_ids for OLMo-7B-Instruct model
        if (
            "olmo" in self.model_name.lower()
            or "qwen" in self.model_name.lower()
            or self.model_name == "falcon-180B-chat"
        ):
            kwargs.pop("token_type_ids", None)
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        # Remove the "prompts" key from the kwargs if it exists
        for key in ["prompts"]:
            if key in kwargs:
                kwargs.pop(key, None)
        # Prevent error caused by token_type_ids for OLMo-7B-Instruct model
        if "olmo" in self.model_name.lower() or self.model_name == "falcon-180B-chat":
            kwargs.pop("token_type_ids", None)
        return self.model.generate(*args, **kwargs)
