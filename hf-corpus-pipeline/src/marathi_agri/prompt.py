"""Classification prompts for Gemma 4 26B A4B.

Two prompts share one agriculture definition:
- sentence prompt (PROMPT_VERSION): AGRI / NON_AGRI / NOT_MARATHI for one sentence.
- document prompt (DOC_PROMPT_VERSION): AGRI / NON_AGRI for a spread excerpt of one web
  document; a recall-leaning first-pass filter (approach C), every sentence of a kept
  document is still classified on its own with the sentence prompt.

The system prompt and few-shot turns are identical for every request, so vLLM prefix
caching processes them once; only the final user turn changes.

History:
- v1: first approved prompt.
- v2: language check judges language only (short / spoken / fragmentary Marathi is
  Marathi); explicit NON_AGRI rules for nutrition, idioms, drinking water, gardening,
  word lists, sector lists; borderline rule for protests / loan waivers / compensation;
  tie-break "when unsure -> NON_AGRI"; topic never decides the language; everyday farm
  talk is AGRI; vague farmer-distress statements are NON_AGRI; 14 more few-shot examples.
"""

from __future__ import annotations

PROMPT_VERSION = "v2"
# Every sentence prompt version ever used, oldest first. The report labels each sentence
# with its newest available result, so adding a source and classifying only that source
# leaves earlier sentences with their earlier labels (recorded per row in prompt_version).
PROMPT_HISTORY: tuple[str, ...] = ("v1", "v2")
assert PROMPT_HISTORY[-1] == PROMPT_VERSION
DOC_PROMPT_VERSION = "d1"

AGRI = "AGRI"
NON_AGRI = "NON_AGRI"
NOT_MARATHI = "NOT_MARATHI"
LABELS: tuple[str, ...] = (AGRI, NON_AGRI, NOT_MARATHI)
DOC_LABELS: tuple[str, ...] = (AGRI, NON_AGRI)

# Shared definition: "Core + agri economy" (approved scope).
_AGRI_TOPICS = """\
- Crops: sowing, seeds, varieties, kharif/rabi, growth, yield, harvest, storage
- Soil and nutrients: soil testing, fertilizer, manure, compost
- Water: irrigation, drip/sprinkler, farm ponds; dams/canals only when about irrigation
- Plant protection: pests, diseases, weeds, pesticides, spraying
- Horticulture and plantation crops: fruits, vegetables, flowers, sugarcane, cotton, nurseries
- Livestock and allied: dairy, cattle, goat, poultry, fisheries, beekeeping, sericulture, fodder, animal health
- Farm machinery and technology: tractors, implements, agri drones, polyhouse
- Agri economy: APMC/mandi arrivals and prices, MSP, procurement, agri exports, crop loans,
  crop insurance, farmer schemes and subsidies, FPOs, agro-processing, cold storage
- Weather ONLY when its effect on crops or farmers is stated (rain/drought/hail damaging crops)
- Agri institutions: agriculture department, KVK, agri universities, advice to farmers"""

SYSTEM_PROMPT = f"""\
You are an expert annotator building a Marathi agriculture text corpus.
You receive ONE text segment. Classify it with exactly one label.

STEP 1 – LANGUAGE CHECK. Judge ONLY the language, never the quality, length or completeness.
Output NOT_MARATHI only if the text is:
- Hindi, Konkani, Sanskrit, Nepali or any other language, even if written in Devanagari
- English, or Marathi written in Latin letters (romanized)
- meaningless gibberish or garbled OCR text
Marathi signals: आहे/आहेत, होते, नाही, आणि, च्या/चे/ची, ला/ना, मध्ये, साठी.
Hindi signals: है/हैं, था, नहीं, और, का/की/के, में, लिए.
Konkani signals: हांव, आसा, म्हाका, तुका, खूब, जाता.
These are STILL MARATHI, go to STEP 2: short phrases, headlines, incomplete fragments,
numbers or dates written in words, informal or spoken Marathi with fillers (हो हो, बरं, म्हणजे),
spelling mistakes, Marathi with a few English loanwords (ट्रॅक्टर, युरिया, ड्रोन).
The TOPIC never decides the language: Marathi text about other states, countries, foreign
people, places or brands is still Marathi.

STEP 2 – If Marathi, decide AGRI or NON_AGRI.
AGRI = the main subject is farming or the farm economy:
{_AGRI_TOPICS}
- Everyday farm talk, even informal or incomplete: weeding, hoeing, watering or spraying crops,
  fodder and care of farm animals, selling one's produce, demand for fruits or flowers in markets
Borderline – farmer protests, loan waivers, compensation:
- AGRI if the text states farm-economy substance: compensation or loan amounts, scheme terms,
  crop prices, crop damage.
- NON_AGRI if it only reports the political event: who protested, roads blocked, statements
  with no farm content.
NON_AGRI:
- Politics, elections, protests with no farm-economy substance (see borderline rule)
- General weather forecasts with no mention of crops
- Food, cooking, recipes, diet, and nutrition or health benefits of fruits, vegetables or milk
- Farm words used as idioms, proverbs, metaphors, brand names, or household uses
  (e.g. cow dung for floors or rituals)
- Drinking-water supply; dams, canals or water projects described only as construction,
  contracts or corruption, without crop irrigation
- Gardening, house plants, trees in general, plant biology lessons not about growing crops
- A bare list of words with no statement (e.g. "भोपळा आणि गाजर")
- Agriculture mentioned only as one item in a list of sectors or topics
- Village life, crime, accidents, festivals; forests, wildlife, environment (unless about farming)
- Everything else: sports, films, business, health, education, etc.
Rules:
- Merely mentioning a farmer, a village or a crop name is NOT enough. The text must be about
  farm work, farm produce or farm income.
- Farmer distress, farmer suicides, or promises and demands to "help farmers" are NON_AGRI
  unless the text names a crop, a farm activity, a price, an amount or a scheme detail.
- When unsure between AGRI and NON_AGRI, choose NON_AGRI.

Answer with exactly one label: AGRI, NON_AGRI or NOT_MARATHI. No explanation."""

# (text, label) pairs shown to the model as prior user/assistant turns.
FEW_SHOT: tuple[tuple[str, str], ...] = (
    ("सोयाबीनवरील खोडमाशीच्या नियंत्रणासाठी शेतकऱ्यांनी वेळेवर फवारणी करावी.", AGRI),
    ("टोमॅटोमध्ये क जीवनसत्त्व भरपूर प्रमाणात असते.", NON_AGRI),
    ("यंदा लासलगाव बाजार समितीत कांद्याला प्रति क्विंटल २२०० रुपये दर मिळाला.", AGRI),
    ("हो हो चालेल, मी उद्या सकाळी येतो", NON_AGRI),
    ("ठिबक सिंचनामुळे उसाच्या उत्पादनात तीस टक्क्यांनी वाढ झाली", AGRI),
    ("शहराचा पिण्याच्या पाण्याचा पुरवठा दोन दिवस बंद राहणार आहे.", NON_AGRI),
    ("गाईंना संतुलित पशुखाद्य दिल्यास दुधाचे उत्पादन वाढते.", AGRI),
    ("त्याची अवस्था म्हणजे जसे पेरले तसे उगवले अशीच झाली.", NON_AGRI),
    ("अवकाळी पावसामुळे द्राक्षबागांचे मोठे नुकसान झाले आहे.", AGRI),
    ("आमच्या बाल्कनीत तुळस आणि कोरफडीच्या कुंड्या आहेत.", NON_AGRI),
    ("अतिवृष्टीने बाधित शेतकऱ्यांना हेक्टरी दहा हजार रुपये मदत जाहीर झाली.", AGRI),
    ("शेतकरी संघटनांनी महामार्गावर रास्ता रोको आंदोलन केले.", NON_AGRI),
    ("पीक विमा योजनेसाठी शेतकऱ्यांना ३१ जुलैपर्यंत अर्ज करता येईल.", AGRI),
    ("भोपळा आणि गाजर", NON_AGRI),
    ("रोज सकाळी गुरांना चारा टाकायचा आणि मग शेतात खुरपणी करायची", AGRI),
    ("या निर्णयामुळे शेतकरी पुन्हा अडचणीत सापडला आहे.", NON_AGRI),
    ("दिवाळीमुळे बाजारात झेंडूच्या फुलांना मागणी वाढली आहे.", AGRI),
    ("बिहारमध्ये पुरामुळे अनेक गावांचा संपर्क तुटला आहे.", NON_AGRI),
    ("आमच्या शेतात यंदा कापूस लय चांगला आलाय", AGRI),
    ("उद्योग, शेती, शिक्षण आणि आरोग्य या क्षेत्रांसाठी अर्थसंकल्पात तरतूद करण्यात आली.", NON_AGRI),
    ("शेतकऱ्यांच्या प्रश्नांवर विरोधकांनी विधानसभेत सरकारला धारेवर धरले.", NON_AGRI),
    ("पुढील दोन दिवस मुंबईत मुसळधार पावसाचा इशारा देण्यात आला आहे.", NON_AGRI),
    ("पुरणपोळी बनवण्यासाठी हरभऱ्याची डाळ रात्रभर भिजत ठेवावी.", NON_AGRI),
    ("भारताने ऑस्ट्रेलियाविरुद्धचा सामना सहा गडी राखून जिंकला.", NON_AGRI),
    ("किसानों को इस साल गेहूं का अच्छा दाम मिला है।", NOT_MARATHI),
    ("हांव आयज बाजाराक वता.", NOT_MARATHI),
    ("shetkaryanna pik vimyacha labh milnar", NOT_MARATHI),
)


def format_user_turn(text: str) -> str:
    return f"<text>{text}</text>"


def build_messages(text: str) -> list[dict[str, str]]:
    """Chat messages for one sentence: system prompt, few-shot turns, then the sentence."""
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example, label in FEW_SHOT:
        messages.append({"role": "user", "content": format_user_turn(example)})
        messages.append({"role": "assistant", "content": label})
    messages.append({"role": "user", "content": format_user_turn(text)})
    return messages


# ---------------------------------------------------------------------------
# Document prompt (first-pass filter for general web corpora)
# ---------------------------------------------------------------------------

EXCERPT_GAP = " … "

DOC_SYSTEM_PROMPT = f"""\
You are an expert annotator building a Marathi agriculture text corpus.
You receive an EXCERPT of ONE Marathi web document: its opening, then sentences sampled
evenly from the rest of the document, separated by "{EXCERPT_GAP.strip()}".
Decide whether the document is about agriculture.

AGRI = the document's main topic, or a substantial part of it, is farming or the farm economy:
{_AGRI_TOPICS}
NON_AGRI = farming is absent or only mentioned in passing: politics or protests with no
farm-economy content, food and recipes, general weather, drinking water, gardening, and
everything else (sports, films, crime, business, health, education, etc.).

This is a first-pass filter: if a substantial part of the excerpt is about farming or the
farm economy, answer AGRI.

Answer with exactly one label: AGRI or NON_AGRI. No explanation."""

DOC_FEW_SHOT: tuple[tuple[str, str], ...] = (
    (
        "यंदा मराठवाड्यात खरीप हंगामात सोयाबीनची पेरणी मोठ्या प्रमाणात वाढली आहे. कृषी विभागाच्या"
        " अंदाजानुसार सोयाबीनखालील क्षेत्र दहा टक्क्यांनी वाढेल."
        + EXCERPT_GAP
        + "बियाण्यांचा तुटवडा जाणवत असल्याने शेतकऱ्यांनी प्रमाणित बियाणे वापरावे, असे आवाहन करण्यात"
        " आले." + EXCERPT_GAP + "पुढील आठवड्यात जिल्ह्यात चांगल्या पावसाचा अंदाज आहे.",
        AGRI,
    ),
    (
        "महापालिका निवडणुकीसाठी सर्व पक्षांनी उमेदवारांची यादी जाहीर केली आहे. सत्ताधारी पक्षाने"
        " अनेक नव्या चेहऱ्यांना संधी दिली."
        + EXCERPT_GAP
        + "प्रचार सभेत त्यांनी शेतकरी आणि कामगारांचाही उल्लेख केला."
        + EXCERPT_GAP
        + "मतदान पुढील महिन्यात होणार असून निकाल दुसऱ्या दिवशी जाहीर होतील.",
        NON_AGRI,
    ),
    (
        "जिल्हा दूध संघाने गाईच्या दुधाच्या खरेदी दरात प्रतिलिटर दोन रुपयांची वाढ केली आहे."
        + EXCERPT_GAP
        + "पशुखाद्याचे दर वाढल्याने दूध उत्पादक शेतकरी अडचणीत आले होते."
        + EXCERPT_GAP
        + "नवे दर एक तारखेपासून लागू होतील, असे संघाच्या अध्यक्षांनी सांगितले.",
        AGRI,
    ),
    (
        "हिवाळ्यात गाजराचा हलवा बनवण्यासाठी ताजे लाल गाजर किसून घ्यावे."
        + EXCERPT_GAP
        + "त्यात दूध आणि साखर घालून मंद आचेवर शिजवावे."
        + EXCERPT_GAP
        + "वरून काजू, बदाम घालून गरम गरम वाढावा.",
        NON_AGRI,
    ),
)


def format_doc_turn(excerpt: str) -> str:
    return f"<doc>{excerpt}</doc>"


def build_doc_messages(excerpt: str) -> list[dict[str, str]]:
    """Chat messages for one document excerpt: system prompt, few-shot turns, the excerpt."""
    messages: list[dict[str, str]] = [{"role": "system", "content": DOC_SYSTEM_PROMPT}]
    for example, label in DOC_FEW_SHOT:
        messages.append({"role": "user", "content": format_doc_turn(example)})
        messages.append({"role": "assistant", "content": label})
    messages.append({"role": "user", "content": format_doc_turn(excerpt)})
    return messages
