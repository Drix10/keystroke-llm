"""Character-to-key translation and character-level tokenization.

CHAR_TO_KEY maps every typeable US QWERTY character to the corresponding key
name used in the Kreo Hive 75 JSON profile.  Lower-case letters map directly
(a->a), upper-case letters map to their lower-case physical key (A->a), digits
map directly, and shifted-symbol pairs share the same physical key (= / +,
[ / {, etc.).
"""

import string

# Mapping from input character to Kreo Hive profile physical key name.
# Physical keyboard switches are unshifted; typing '!' or 'A' physically depresses
# the '1' or 'a' key switch with Shift held. Therefore, all shifted symbols and uppercase
# letters must map to their corresponding physical switch name to illuminate the correct key.
CHAR_TO_KEY = {
    **{c: c for c in string.ascii_lowercase},
    **{c.upper(): c for c in string.ascii_lowercase},
    **{d: d for d in string.digits},
    " ": "space", "\t": "tab", "\n": "enter", "\r": "enter", "\b": "backspace",
    "-": "minus", "=": "equal", "[": "lbracket", "]": "rbracket", "\\": "backslash",
    ";": "semicolon", "'": "quote", ",": "comma", ".": "period", "/": "slash", "`": "grave",
    # Shifted punctuation — map directly to the base physical switch
    "!": "1", "@": "2", "#": "3", "$": "4", "%": "5", "^": "6", "&": "7", "*": "8",
    "(": "9", ")": "0", "_": "minus", "+": "equal", "{": "lbracket", "}": "rbracket",
    "|": "backslash", ":": "semicolon", '"': "quote", "<": "comma", ">": "period",
    "?": "slash", "~": "grave",
}

SHIFTED_CHARS = set('~!@#$%^&*()_+{}|:"<>?' + string.ascii_uppercase)

# Valid typeable characters for next-key prediction (alphabets, digits, space, basic punctuation)
# Excludes control codes, unmapped symbols, and non-typeable tokens
VALID_PREDICTIVE_CHARS = set(
    string.ascii_lowercase
    + string.ascii_uppercase
    + string.digits
    + " "
    + ".,?!'\"-;:()/_=+[]"
)


def char_to_key_name(ch: str) -> str | None:
    return CHAR_TO_KEY.get(ch, None)


def is_shifted(ch: str) -> bool:
    return ch in SHIFTED_CHARS


def format_prediction_label(ch: str) -> str:
    """Formats a predicted character for clean user-facing terminal display.
    
    Clarifies both the intended character and the physical target keycap.
    """
    if ch == " ":
        return "[Space]"
    if ch == "\n":
        return "[Enter]"
    if ch == "\t":
        return "[Tab]"
    if ch == "\b":
        return "[Backspace]"
    key = char_to_key_name(ch)
    if not key:
        return repr(ch)
    # For alphabetic letters, display as the clean physical key (e.g. 'w', 'n', 'e')
    if len(key) == 1 and key.isalpha():
        return f"'{key}'"
    # For shifted punctuation, clarify the physical base key
    if is_shifted(ch) and ch != key:
        return f"'{ch}' [Key {key.upper()}]"
    return f"'{ch}'"


def get_default_vocab() -> list[str]:
    vocab = ["<unk>", " ", "\n", "\t"]
    vocab.extend(list(string.digits))
    vocab.extend(list(string.ascii_lowercase))
    vocab.extend(list(string.ascii_uppercase))
    for s in string.punctuation:
        if s not in vocab:
            vocab.append(s)
    return vocab


class CharTokenizer:
    """Character tokenizer with unknown token fallback."""

    def __init__(self, vocab: list[str] | None = None):
        self.vocab = list(vocab) if vocab is not None else get_default_vocab()
        self.char_to_id = {ch: i for i, ch in enumerate(self.vocab)}
        self.id_to_char = {i: ch for i, ch in enumerate(self.vocab)}
        if "<unk>" not in self.char_to_id:
            raise ValueError(
                "Vocab must include a '<unk>' token as the first entry. "
                "Use get_default_vocab() to build a compatible vocabulary."
            )
        self.unk_id = self.char_to_id["<unk>"]

    def __len__(self) -> int:
        """Number of tokens in the vocabulary."""
        return len(self.vocab)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode_char(self, ch: str) -> int:
        return self.char_to_id.get(ch, self.unk_id)

    def decode_id(self, token_id: int) -> str:
        return self.id_to_char.get(token_id, "<unk>")

    def encode(self, text: str) -> list[int]:
        return [self.encode_char(c) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join([self.decode_id(i) for i in ids])

