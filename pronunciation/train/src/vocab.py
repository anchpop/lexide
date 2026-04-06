"""IPA vocabulary for CTC model — maps IPA codepoints to token IDs."""

import json


BLANK_TOKEN = "<blank>"
UNK_TOKEN = "<unk>"
SPACE_TOKEN = " "

# Reserved token IDs
BLANK_ID = 0
UNK_ID = 1
SPACE_ID = 2


class IPAVocab:
    def __init__(self, tokens: list[str]):
        """Initialize from ordered list of tokens (index 0 must be <blank>)."""
        assert tokens[BLANK_ID] == BLANK_TOKEN
        assert tokens[UNK_ID] == UNK_TOKEN
        assert tokens[SPACE_ID] == SPACE_TOKEN
        self.tokens = tokens
        self.token_to_id = {t: i for i, t in enumerate(tokens)}

    @classmethod
    def from_file(cls, path: str) -> "IPAVocab":
        with open(path) as f:
            data = json.load(f)
        return cls(data["tokens"])

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"tokens": self.tokens}, f, ensure_ascii=False, indent=2)

    def encode(self, ipa_string: str) -> list[int]:
        """Convert IPA string to list of token IDs (character-level)."""
        return [self.token_to_id.get(ch, UNK_ID) for ch in ipa_string]

    def decode(self, ids: list[int]) -> str:
        """Convert token IDs back to IPA string, skipping blank and unk."""
        return "".join(
            self.tokens[i] for i in ids if i not in (BLANK_ID, UNK_ID)
        )

    @property
    def blank_id(self) -> int:
        return BLANK_ID

    @property
    def size(self) -> int:
        return len(self.tokens)
