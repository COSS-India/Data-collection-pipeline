import pytest

from marathi_agri import text as T


class TestClean:
    def test_nfc_and_whitespace(self):
        assert T.clean_text("  शेतकरी   आणि\t पीक  ") == "शेतकरी आणि पीक"

    def test_removes_asr_tags_but_keeps_marathi_brackets(self):
        assert T.clean_text("आज [noise] पाऊस <unk> पडला (laugh)") == "आज पाऊस पडला"
        assert T.clean_text("कांदा [सुमारे] २० रुपये") == "कांदा [सुमारे] २० रुपये"

    def test_removes_urls_and_emails(self):
        assert (
            T.clean_text("अधिक माहिती https://krishi.gov.in वर व a@b.com वर")
            == "अधिक माहिती वर व वर"
        )

    def test_keeps_zwj_and_drops_zero_width_space(self):
        assert T.clean_text("र्‍या​") == "र्‍या"

    def test_danda_normalization(self):
        assert T.clean_text("पीक चांगले आले | दर वाढले ॥") == "पीक चांगले आले। दर वाढले।"

    def test_empty(self):
        assert T.clean_text(None) == ""
        assert T.clean_text("  \n [noise] ") == ""


class TestSplit:
    def test_danda_and_question(self):
        assert T.split_sentences("पीक चांगले आले। दर किती आहे? माहित नाही!") == [
            "पीक चांगले आले।",
            "दर किती आहे?",
            "माहित नाही!",
        ]

    def test_period_boundary(self):
        assert T.split_sentences("पाऊस पडला. पेरणी सुरू झाली.") == ["पाऊस पडला.", "पेरणी सुरू झाली."]

    @pytest.mark.parametrize(
        "text",
        [
            "डॉ. पाटील यांनी सल्ला दिला.",
            "श्री. देशमुख शेतात गेले.",
            "इ.स. २०२० मध्ये उत्पादन वाढले.",
            "हे अंतर १०० कि. मी. आहे.",
            "1. बियाणे निवडा.",
        ],
    )
    def test_abbreviations_do_not_split(self, text):
        assert T.split_sentences(text) == [text]

    def test_decimal_not_split(self):
        assert T.split_sentences("उत्पादन 2.5 टन झाले.") == ["उत्पादन 2.5 टन झाले."]

    def test_ellipsis_splits(self):
        assert T.split_sentences("पाऊस आला... पीक वाचले.") == ["पाऊस आला...", "पीक वाचले."]

    def test_no_punctuation_returns_whole(self):
        assert T.split_sentences("शेतकऱ्यांनी वेळेवर पेरणी करावी") == ["शेतकऱ्यांनी वेळेवर पेरणी करावी"]

    def test_newline_is_boundary_and_quotes_stay_attached(self):
        assert T.split_sentences('ते म्हणाले, "पीक चांगले आहे." नंतर गेले\nदुसरी ओळ') == [
            'ते म्हणाले, "पीक चांगले आहे."',
            "नंतर गेले",
            "दुसरी ओळ",
        ]


class TestMeasure:
    def test_devanagari_ratio(self):
        assert T.devanagari_ratio("शेतकरी") == 1.0
        assert T.devanagari_ratio("farmer") == 0.0
        assert T.devanagari_ratio("१२३ ...") == 0.0
        assert 0.4 < T.devanagari_ratio("शेत farm") < 0.7

    def test_end_punct(self):
        assert T.has_end_punct("पीक आले।")
        assert T.has_end_punct('तो म्हणाला "हो."')
        assert not T.has_end_punct("पीक आले")


class TestDedupKey:
    def test_punctuation_and_spacing_insensitive(self):
        a = T.dedup_key("पीक चांगले आले।")
        b = T.dedup_key("पीक  चांगले, आले")
        assert a == b
        assert T.sentence_id(a) == T.sentence_id(b)

    def test_different_text_different_id(self):
        assert T.sentence_id(T.dedup_key("पीक आले")) != T.sentence_id(T.dedup_key("पीक गेले"))
