"""Phone-conditioned controls. Speaker identity has no articulation path."""
import math

import torch
from torch import nn

import unicodedata

from maeda import NAMES, geometry


class GestureLoss(nn.Module):
    """Weak phone-class inequalities, only in the middle 40% of a phone.

    Native neutral PB1 dorsal vertices 12..20 map to sections 17..28;
    front vertices 21..26 map to 30..37. Exclude lips from tongue targets.
    These broad regions are heuristics, not gold articulator trajectories.
    """
    names = ('bilabial', 'nasal', 'oral', 'alveolar', 'velar', 'vowel_pharynx')

    def __init__(self, vocabulary):
        super().__init__()
        vowels = set('aeiouyæɑɒɔəɚɛɜɞɤɨɪɯɵœøʊʌɐʉʏ')
        consonants = set('bcdfghjklmnpqrstvwxyzðŋɕɖɟɡɣɥɦɫɲɴɸɹɻɽɾʁʂʃʈʋʎʑʒʔʕʝβθχçħ')
        rows = []
        for phone in vocabulary:
            normalized = unicodedata.normalize('NFD', phone)
            base = normalized[0]
            nasal = base in 'mnŋɲɴ' or (base in vowels and '̃' in normalized)
            rows.append([phone in {'m', 'b', 'p'}, nasal,
                         base in consonants and not nasal,
                         phone in {'t', 'd', 'n', 's', 'z', 'l', 'ɾ'},
                         phone in {'k', 'g', 'ɡ', 'ŋ'},
                         base in vowels and base not in 'ɑɔɒ'])
        self.register_buffer('classes', torch.tensor(rows, dtype=torch.bool))

    def forward(self, art, scale, phone, relative):
        areas, _, coupling, _ = geometry(art, scale, True)
        masks = self.classes[phone] & ((relative >= .3) & (relative <= .7))[:, None]
        # Normalize by each tolerance before squaring: cm² units must not make
        # the nasal target weaker than the competing rest-posture prior.
        violations = torch.stack((((areas[:, -1]-.3)/.3).relu(),
            ((.3-coupling)/.3).relu(), ((coupling-.1)/.1).relu(),
            ((areas[:, 29:38].amin(-1)-.5)/.5).relu(),
            ((areas[:, 17:29].amin(-1)-.5)/.5).relu(),
            # Low-pharynx opening is a heuristic, not MRI supervision: Annot16's
            # tongue/wall surfaces do not cover these cuts. Back open vowels
            # /ɑ ɔ ɒ/ may genuinely constrict here; preserve that freedom.
            ((.5-areas[:, 4:11].amin(-1))/.5).relu()), -1).square()
        # The competing oral constriction includes the front tongue AND lips,
        # but not the pharynx. A 1.25 area ratio demands a modest 20% dorsal
        # advantage, independent of scale; it is a heuristic, not MRI-derived.
        dorsal = areas[:, 17:29].amin(-1)
        front_lips = areas[:, 29:40].amin(-1)
        dominance = (dorsal.log()-front_lips.log()+math.log(1.25)).relu().square()
        violations = violations + torch.nn.functional.pad(dominance[:, None], (4, 1))
        # Per-class means keep rare gestures visible without a dense phone target.
        return (violations*masks).sum(0) / masks.sum(0).clamp_min(1)



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
        self.source = nn.Linear(98, 2)
        self.character = nn.Linear(8, 4)
        nn.init.zeros_(self.articulation.weight)
        nn.init.zeros_(self.articulation.bias)
        self.articulation.bias.data[-1] = -2.
        nn.init.zeros_(self.source.weight)
        self.source.bias.data.copy_(torch.tensor([0., -2.]))
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
