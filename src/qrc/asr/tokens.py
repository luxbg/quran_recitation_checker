from dataclasses import dataclass
from pathlib import Path


@dataclass
class TokenTable:
    id_to_symbol: dict[int, str]
    symbol_to_id: dict[str, int]
    blank_id: int


def load_tokens(tokens_path: Path | str) -> TokenTable:
    id_to_symbol: dict[int, str] = {}
    with open(tokens_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            symbol, id_str = line.rsplit(" ", 1)
            id_to_symbol[int(id_str)] = symbol

    symbol_to_id = {sym: tid for tid, sym in id_to_symbol.items()}
    blank_id = symbol_to_id.get("<blank>")
    if blank_id is None:
        raise ValueError(f"No <blank> symbol found in {tokens_path}")

    return TokenTable(id_to_symbol=id_to_symbol, symbol_to_id=symbol_to_id, blank_id=blank_id)


def detect_space_symbol(table: TokenTable) -> str | None:
    """Return a whitespace-like symbol in the token table, if one exists.

    For zipformer_p-arabic-v3 this returns None: the 251-symbol alphabet has
    no word-boundary token, which is why the aligner uses character-level
    (Option B) alignment rather than buffering between space emissions.
    """
    for symbol in table.symbol_to_id:
        if symbol != "<blank>" and symbol.strip() == "":
            return symbol
    return None
