"""
key_mapper.py - Character to Physical Key Mapping for Kreo Hive 65/75
====================================================================
Maps typed and predicted characters to the exact physical key names
expected by keyboardrgb.py and the keyboard profiles.
"""

# Map standard characters to Kreo Hive profile key names
CHAR_TO_KEY = {
    # Lowercase alphabets
    **{chr(c): chr(c) for c in range(ord('a'), ord('z') + 1)},
    
    # Uppercase alphabets (maps to the letter key, Shift is implicit)
    **{chr(c): chr(c).lower() for c in range(ord('A'), ord('Z') + 1)},
    
    # Digits
    **{str(d): str(d) for d in range(10)},
    
    # Whitespace & Control
    ' ': 'space',
    '\t': 'tab',
    '\n': 'enter',
    '\r': 'enter',
    '\b': 'backspace',
    
    # Punctuation & Symbols (Unshifted)
    '-': 'minus',
    '=': 'equal',
    '[': 'lbracket',
    ']': 'rbracket',
    '\\': 'backslash',
    ';': 'semicolon',
    "'": 'quote',
    ',': 'comma',
    '.': 'period',
    '/': 'slash',
    '`': 'grave',
    
    # Shifted Symbols (Map to base key on keyboard)
    '!': '1',
    '@': '2',
    '#': '3',
    '$': '4',
    '%': '5',
    '^': '6',
    '&': '7',
    '*': '8',
    '(': '9',
    ')': '0',
    '_': 'minus',
    '+': 'equal',
    '{': 'lbracket',
    '}': 'rbracket',
    '|': 'backslash',
    ':': 'semicolon',
    '"': 'quote',
    '<': 'comma',
    '>': 'period',
    '?': 'slash',
    '~': 'grave',
}

# Special modifier indicators (if a character requires Shift)
SHIFTED_CHARS = set('~!@#$%^&*()_+{}|:"<>?' + 'ABCDEFGHIJKLMNOPQRSTUVWXYZ')


def char_to_key_name(ch: str) -> str | None:
    """
    Given a character (e.g. 'a', ' ', '\n', '?'), return the physical key name
    defined in the keyboard profile (e.g. 'a', 'space', 'enter', 'slash').
    Returns None if character does not exist on keyboard.
    """
    return CHAR_TO_KEY.get(ch, None)


def is_shifted(ch: str) -> bool:
    """Returns True if typing this character requires pressing the Shift key."""
    return ch in SHIFTED_CHARS


def get_default_vocab() -> list[str]:
    """
    Builds the standard character vocabulary:
    - Special tokens: <unk>
    - Space and control chars: ' ', '\n', '\t'
    - Digits '0'-'9'
    - Lowercase 'a'-'z'
    - Uppercase 'A'-'Z'
    - Punctuation symbols
    Total tokens: approx 96-100.
    """
    vocab = ["<unk>", " ", "\n", "\t"]
    
    # Add digits
    vocab.extend([str(d) for d in range(10)])
    
    # Add lowercase letters
    vocab.extend([chr(c) for c in range(ord('a'), ord('z') + 1)])
    
    # Add uppercase letters
    vocab.extend([chr(c) for c in range(ord('A'), ord('Z') + 1)])
    
    # Add punctuation
    symbols = list("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
    for s in symbols:
        if s not in vocab:
            vocab.append(s)
            
    return vocab


class CharTokenizer:
    """
    Character-level tokenizer with forward and reverse mapping.
    Handles unknown characters cleanly using <unk>.
    """
    def __init__(self, vocab: list[str] | None = None):
        if vocab is None:
            vocab = get_default_vocab()
            
        self.vocab = list(vocab)
        self.char_to_id = {ch: idx for idx, ch in enumerate(self.vocab)}
        self.id_to_char = {idx: ch for idx, ch in enumerate(self.vocab)}
        self.unk_token = "<unk>"
        self.unk_id = self.char_to_id.get(self.unk_token, 0)
        
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
        
    def encode_char(self, ch: str) -> int:
        """Encode a single character to token ID."""
        return self.char_to_id.get(ch, self.unk_id)
        
    def decode_id(self, token_id: int) -> str:
        """Decode a token ID to character."""
        return self.id_to_char.get(token_id, self.unk_token)
        
    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of token IDs."""
        return [self.encode_char(ch) for ch in text]
        
    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs into text."""
        return "".join([self.decode_id(i) for i in ids])
