import os, sys, re
from typing import Dict, List, Any, Union
import torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model.model import Transformer
from model.vocab.tokenizer import Tokenizer
import config


class EndpointHandler:
    def __init__(self, path: str = ""):
        self.base_dir = path or REPO_ROOT

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt_path = os.path.join(self.base_dir, "epoch_10.pt")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Missing checkpoint at: {ckpt_path}")

        self.model = Transformer().to(self.device)

        ckpt = torch.load(ckpt_path, map_location=self.device)

        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt

        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

        self.tokenizer = Tokenizer()

    def _last_token_logits(self, model_out: torch.Tensor) -> torch.Tensor:
        if model_out.dim() == 3:      
            return model_out[0, -1, :]
        if model_out.dim() == 2:      
            return model_out[-1, :]
        raise ValueError(f"Unexpected model output shape: {tuple(model_out.shape)}")

    @torch.inference_mode()
    def _generate_one(self, prompt: str) -> str:
        encoded = torch.as_tensor(
            self.tokenizer.encode(prompt),
            dtype=torch.long,
            device=self.device,
        )

        currtoken = ""
        outputstring = ""
        countcheck = 0

        while currtoken != "<END>" and countcheck < config.max_tokens:
            logits = self._last_token_logits(self.model(encoded))

            if config.argmax:
                next_id = int(torch.argmax(logits).item())
            else:
                probs = torch.softmax(logits / config.temperature, dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1).item())

            currtoken = self.tokenizer.decode([next_id]).strip()

            if re.match(r"^[.,!?;:]", currtoken):
                if outputstring.endswith(" "):
                    outputstring = outputstring[:-1]
                outputstring += currtoken + " "
            else:
                outputstring += currtoken + " "

            encoded = torch.cat(
                [encoded, torch.tensor([next_id], dtype=torch.long, device=self.device)],
                dim=0,
            )
            if encoded.numel() > config.max_seq_length:
                encoded = encoded[-config.max_seq_length :]

            countcheck += 1

        text = re.sub("<BEGIN>", "\n\n", outputstring)
        text = re.sub("<END>", "\n\n", text)
        return "AURELIUS: " + text

    def __call__(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        inputs = data.get("inputs", data)

        if isinstance(inputs, dict):
            inputs = inputs.get("text", "")

        if isinstance(inputs, list):
            return [{"generated_text": self._generate_one(str(x))} for x in inputs]

        return [{"generated_text": self._generate_one(str(inputs))}]
