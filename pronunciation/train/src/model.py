"""Wav2vec2 with a learnable-layer-weighted stress prediction head."""

import torch
import torch.nn as nn
from transformers import (
    Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)


class Wav2Vec2StressModel(nn.Module):
    """Freeze wav2vec2 backbone, train a learnable layer-weighted stress head.

    Mallela et al. (2025) found that stress info is strongest in the middle
    transformer layers, not the final one. So instead of using just the last
    hidden state, we take all layers and let the head learn a weighted sum.
    This is the SUPERB-style probing approach.
    """

    def __init__(self, model_name: str = "anchpop/lexide-pronunciation-phoneme-xls-r-2b", num_labels: int = 3):
        super().__init__()
        self.backbone = Wav2Vec2ForCTC.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size  # 1920 on xls-r-2b
        num_layers = self.backbone.config.num_hidden_layers + 1  # +1 for embedding (49 on xls-r-2b)

        # Freeze the entire backbone (including the original CTC head)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        # Learnable weights across layers (softmax-normalized at use time)
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))

        # Stress prediction head
        self.stress_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels),
        )

    def forward(self, input_values, attention_mask=None):
        with torch.no_grad():
            outputs = self.backbone(
                input_values=input_values,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # Stack all layers: (num_layers, batch, frames, hidden_size)
        stacked = torch.stack(outputs.hidden_states, dim=0)

        # Weighted sum across layers (softmax weights for stable training)
        weights = torch.softmax(self.layer_weights, dim=0)
        weighted = (stacked * weights.view(-1, 1, 1, 1)).sum(dim=0)  # (batch, frames, hidden)

        stress_logits = self.stress_head(weighted)

        return {
            "stress_logits": stress_logits,
            "ctc_logits": outputs.logits,  # from frozen original head
            "layer_weights": weights,
        }

    def get_trainable_params(self):
        yield self.layer_weights
        yield from self.stress_head.parameters()


def load_processor(model_name: str = "anchpop/lexide-pronunciation-phoneme-xls-r-2b"):
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_name)
    return Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
