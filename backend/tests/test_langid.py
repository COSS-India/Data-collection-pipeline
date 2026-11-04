from dcp.langid import detect_language


def test_marathi():
    assert detect_language("शेतकऱ्यांनी कापसाची लागवड केली आहे आणि पाऊस चांगला झाला.") == ("mr", "Devanagari")


def test_hindi():
    assert detect_language("किसानों ने गेहूं की बुवाई की है और बारिश अच्छी हुई।") == ("hi", "Devanagari")


def test_english():
    assert detect_language("Farmers sowed cotton this season.") == ("en", "Latin")


def test_marathi_by_suffix_only():
    assert detect_language("खरीप हंगामात सोयाबीनची पेरणी जून महिन्यात करावी.")[0] == "mr"
