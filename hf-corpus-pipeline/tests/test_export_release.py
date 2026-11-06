import importlib.util
from pathlib import Path

import polars as pl

_SPEC = importlib.util.spec_from_file_location(
    "export_release", Path(__file__).resolve().parents[1] / "scripts" / "export_release.py"
)
export_release = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(export_release)


def test_release_keeps_agri_drops_milu_and_flags_samanantar():
    sentences = pl.DataFrame(
        {
            "sentence_id": ["d", "c", "b", "a", "e"],
            "label": ["AGRI", "AGRI", "AGRI", "NON_AGRI", "AGRI"],
            "source_dataset": [
                "ai4bharat/sangraha",
                "ai4bharat/MILU",
                "shivam/marathi_samanantar_processed",
                "ai4bharat/sangraha",
                "ai4bharat/Kathbath",
            ],
        }
    )
    out = export_release.select_release_rows(sentences)
    assert out["sentence_id"].to_list() == ["b", "d", "e"]  # AGRI only, no MILU, sorted
    notes = dict(zip(out["sentence_id"], out["license_note"], strict=True))
    assert notes["b"].startswith("unclear")
    assert notes["d"] is None and notes["e"] is None
