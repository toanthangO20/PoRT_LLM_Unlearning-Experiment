import torch


def _get_context_window(model, tokenizer):
    candidates = []
    model_obj = getattr(model, "model", model)
    config = getattr(model_obj, "config", None)
    for attr in ["max_position_embeddings", "n_positions", "max_sequence_length"]:
        value = getattr(config, attr, None)
        if isinstance(value, int) and 0 < value < 1_000_000:
            candidates.append(value)

    tokenizer_max_length = getattr(tokenizer, "model_max_length", None)
    if (
        isinstance(tokenizer_max_length, int)
        and 0 < tokenizer_max_length < 1_000_000
    ):
        candidates.append(tokenizer_max_length)

    return min(candidates) if candidates else None


class ChoiceByTopLogit:
    name = "choice_by_top_logit"

    def __init__(self, save_logits=False, truncation_side="left"):
        super().__init__()
        self.save_logits = save_logits
        self.logits = []
        self.truncation_side = truncation_side

    def evaluate(self, prompts, answers, model, tokenizer):
        padding_side = tokenizer.padding_side

        inputs = prompts
        context_window = _get_context_window(model, tokenizer)
        tokenizer_kwargs = {
            "padding": "longest",
            "return_tensors": "pt",
        }
        if context_window is not None:
            tokenizer_kwargs.update({"truncation": True, "max_length": context_window})

        original_truncation_side = getattr(tokenizer, "truncation_side", None)
        if original_truncation_side is not None:
            tokenizer.truncation_side = self.truncation_side
        try:
            prompt_encoding = tokenizer(inputs, **tokenizer_kwargs).to(model.device)
        finally:
            if original_truncation_side is not None:
                tokenizer.truncation_side = original_truncation_side

        choice_encoding = (
            tokenizer(answers[0], return_tensors="pt", add_special_tokens=False)
            .input_ids.to(model.device)
            .squeeze(1)
        )

        with torch.no_grad():
            logits = model(**prompt_encoding, prompts=prompts, answers=None).logits

        # Get the top logit for each choice
        top_logit_choices = []
        for i, (_, attn_mask) in enumerate(
            zip(prompt_encoding["input_ids"], prompt_encoding["attention_mask"])
        ):
            # Find the last token of the prompt and get the logit
            prompt_end = sum(attn_mask) - 1 if padding_side == "right" else -1
            choice = torch.argmax(logits[i, prompt_end, choice_encoding], dim=-1).item()
            top_logit_choices.append(choice)
            if self.save_logits:
                self.logits.append(logits[i, prompt_end, :].view(1, -1).cpu())

        return top_logit_choices
