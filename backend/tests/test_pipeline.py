import json

from dcp import pipeline
from dcp.metrics import classification_report


def test_end_to_end(tmp_path):
    (tmp_path / "a.txt").write_text(
        "शेतकऱ्यांनी सोयाबीनची पेरणी केली आहे। क्रिकेट सामना भारताने जिंकला आहे।\n"
        "शेतकऱ्यांनी सोयाबीनची पेरणी केली आहे।",
        encoding="utf-8",
    )
    (tmp_path / "b.html").write_text(
        "<html><body><nav>Menu</nav><p>किसानों को फसल बीमा योजना का लाभ मिलेगा।</p></body></html>",
        encoding="utf-8",
    )
    (tmp_path / "c.jsonl").write_text(json.dumps({"id": 1, "body": "Drip irrigation saves water for farmers."}) + "\n")

    sents = pipeline.run([tmp_path])
    by_text = {s.text: s for s in sents}

    sowing = [s for s in sents if "पेरणी" in s.text]
    assert len(sowing) == 2 and [s.quality["duplicate"] for s in sowing] == [False, True]
    assert sowing[0].labels["agri"]["label"] == "agri"
    assert sowing[0].labels["agri_subdomain"]["label"] == "crop_cultivation"
    assert sowing[0].lang == "mr"

    assert by_text["क्रिकेट सामना भारताने जिंकला आहे।"].labels["agri"]["label"] == "non_agri"

    scheme = by_text["किसानों को फसल बीमा योजना का लाभ मिलेगा।"]
    assert scheme.lang == "hi" and scheme.labels["agri_subdomain"]["label"] == "schemes_policy"
    assert "Menu" not in by_text

    assert by_text["Drip irrigation saves water for farmers."].labels["agri_subdomain"]["label"] == "soil_water"


def test_report():
    r = classification_report([("agri", "agri"), ("agri", "non_agri"), ("non_agri", "non_agri")], ["agri", "non_agri"])
    assert r["accuracy"] == 0.667
    assert r["confusion"]["non_agri"]["agri"] == 1
    assert r["per_class"]["agri"]["precision"] == 0.5
