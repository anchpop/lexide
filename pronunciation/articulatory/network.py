"""Phone-conditioned controls. Speaker identity has no articulation path."""
import torch
from torch import nn

import unicodedata

from maeda import NAMES


class NasalLoss(nn.Module):
    """Only the nasal velum hinge remains: PB1 has no MRI velum surface."""
    def __init__(self, vocabulary):
        super().__init__()
        vowels = set('aeiouyæɑɒɔəɚɛɜɞɤɨɪɯɵœøʊʌɐʉʏ')
        normalized = [unicodedata.normalize('NFD', p) for p in vocabulary]
        self.register_buffer('nasal', torch.tensor([
            p[0] in 'mnŋɲɴ' or (p[0] in vowels and '̃' in p)
            for p in normalized], dtype=torch.bool))

    def forward(self, art, scale, phone, relative):
        coupling = .001 + 1.8 * art[:, 7].sigmoid()**3
        mask = self.nasal[phone] & (relative >= .3) & (relative <= .7)
        return ((((.3-coupling)/.3).relu().square())*mask).sum() / mask.sum().clamp_min(1)



class ControlNet(nn.Module):
    def __init__(self, phones, speakers, languages=1):
        super().__init__()
        self.phone = nn.Embedding(phones, 32)
        self.language = nn.Embedding(languages, 4)
        self.speaker = nn.Embedding(speakers, 8)
        self.context = nn.Sequential(
            nn.Conv1d(37, 96, 5, padding=2), nn.SiLU(),
            nn.Conv1d(96, 96, 5, dilation=3, padding=6), nn.SiLU(),
            nn.Conv1d(96, 96, 5, dilation=9, padding=18), nn.SiLU())
        self.articulation = nn.Linear(96, len(NAMES))
        self.source = nn.Linear(98, 4)
        self.character = nn.Linear(8, 6)
        nn.init.zeros_(self.articulation.weight)
        nn.init.zeros_(self.articulation.bias)
        self.articulation.bias.data[-1] = -2.
        nn.init.zeros_(self.source.weight)
        self.source.bias.data.copy_(torch.tensor([0., -7., -7., 0.]))
        nn.init.zeros_(self.character.weight)
        nn.init.zeros_(self.character.bias)

    def forward(self, phone, relative, language, speaker, f0, voiced):
        language = self.language(language).expand(len(phone), -1)
        inputs = torch.cat((self.phone(phone), language, relative[:, None]), -1)
        context = self.context(inputs.T[None])[0].T
        articulation = self.articulation(context)
        # One vector for the whole recording, never a frame-dependent speaker input.
        character = self.character(self.speaker(speaker))
        source_inputs = torch.cat((context, (f0/150).log()[:, None], voiced[:, None]), -1)
        source = self.source(source_inputs) + character[2:]
        scale = .85 + .3*character[:2].sigmoid()
        return articulation, source, scale
