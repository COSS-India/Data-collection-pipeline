from dcp.normalize import normalize
from dcp.segment import split_sentences


def test_danda_and_full_stop():
    text = "शेतकऱ्यांनी पेरणी केली आहे। पाऊस चांगला झाला. आता पीक वाढेल!"
    assert split_sentences(text) == [
        "शेतकऱ्यांनी पेरणी केली आहे।",
        "पाऊस चांगला झाला.",
        "आता पीक वाढेल!",
    ]


def test_abbreviations_not_split():
    text = "डॉ. पाटील यांनी सांगितले की ५.५ टन उत्पादन मिळाले. Dr. Rao agreed."
    assert split_sentences(text) == [
        "डॉ. पाटील यांनी सांगितले की ५.५ टन उत्पादन मिळाले.",
        "Dr. Rao agreed.",
    ]


def test_soft_wrapped_lines_are_joined():
    text = "सोयाबीनच्या पिकावर\nअळीचा प्रादुर्भाव दिसून येत आहे।\n- फवारणी करावी"
    assert split_sentences(text) == [
        "सोयाबीनच्या पिकावर अळीचा प्रादुर्भाव दिसून येत आहे।",
        "फवारणी करावी",
    ]


def test_normalize_pipe_as_danda():
    assert normalize("पाऊस आला |  ") == "पाऊस आला।"


def test_broken_pdf_encoding_flagged():
    from dcp.quality import assess

    q = assess("खरʟप हंगामात सोयाबीनची पेरणी जून महəďयात करावी.", "Devanagari")
    assert "broken_pdf_encoding" in q["flags"] and not q["usable"]
    assert assess("शेतकऱ्यांनी “पेरणी” केली आहे…", "Devanagari")["usable"]
