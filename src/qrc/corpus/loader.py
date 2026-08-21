import json
from pathlib import Path

from qrc.corpus.models import AyahEntry, AyahRef


def load_ayahs(ordered_quran_phonemes_path: Path | str) -> list[AyahEntry]:
    """Load ordered_quran_phonemes.json and return ayahs sorted by (surah, ayah).

    Don't trust the JSON's key insertion order -- sort explicitly.
    """
    with open(ordered_quran_phonemes_path, encoding="utf-8") as f:
        raw: dict[str, dict] = json.load(f)

    entries: list[AyahEntry] = []
    for key, value in raw.items():
        surah_str, ayah_str = key.split(":")
        entries.append(
            AyahEntry(
                ref=AyahRef(surah=int(surah_str), ayah=int(ayah_str)),
                aya_text=value["aya_text"],
                aya_phoneme=value["aya_phoneme"],
                words=tuple(value["aya_phonemes_list"]),
            )
        )

    entries.sort(key=lambda e: (e.ref.surah, e.ref.ayah))
    return entries
