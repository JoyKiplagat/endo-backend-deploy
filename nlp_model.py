"""
EndoScan — NLP Path Model Loader
===================================
Loads the trained BioBERT NER model, the alias/semantic matching layers,
the ESI scoring engine, and configures Gemini — then defines
run_full_pipeline(), the function assess_patient.py actually calls.

WHERE THE MODELS GET CALLED:
  - BioBERT NER call        : inside predict(), search for `model(input_ids=...)`
  - Semantic mapper call    : inside map_span_to_id_semantic(), `sem_model.encode(...)`
  - ESI scoring (no model)  : inside calculate_esi() — this is the hand-tuned
                               rules formula, not a trained model (see the
                               calibration TODO in assess_patient.py)
  - Gemini extraction/explanation calls: inside extract_with_gemini() and
                               generate_explanation()

Load this ONCE at backend startup (importing this module runs the loading
code below immediately) — BioBERT and the sentence-transformer are both
too slow to reload per-request.

CHANGED FROM THE COLAB NOTEBOOK: the original notebook read the Gemini key
via `google.colab.userdata.get(...)`, which only works inside Colab. A
backend server needs the environment-variable pattern instead — see
GEMINI_API_KEY below. Set it with `export GEMINI_API_KEY=...` before
starting the backend, same as the scan-router notebook.
"""

import os
import json
from pathlib import Path
from difflib import SequenceMatcher

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# ── UPDATE THESE to wherever your NLP notebook actually saved these ────
DATA_DIR = Path('./data')
NER_MODEL_DIR = Path('./endoscan_ner_model')
# ─────────────────────────────────────────────────────────────────────

ALIASES_PATH = DATA_DIR / 'symptom_aliases.json'
SYMPTOMS_PATH = DATA_DIR / 'symptoms.json'
SCORING_RULES_PATH = DATA_DIR / 'scoring_rules.json'

for p in [NER_MODEL_DIR, ALIASES_PATH, SYMPTOMS_PATH, SCORING_RULES_PATH]:
    assert Path(p).exists(), f"Required NLP artifact not found: {p}"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── THE ACTUAL BIOBERT MODEL LOADING ────────────────────────────────────
model = AutoModelForTokenClassification.from_pretrained(str(NER_MODEL_DIR)).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(str(NER_MODEL_DIR))
model.eval()
print(f"BioBERT NER model loaded ✓  (device={DEVICE})")

label2id = {'O': 0, 'B-SYMPTOM': 1, 'I-SYMPTOM': 2}
id2label = {v: k for k, v in label2id.items()}

with open(ALIASES_PATH, 'r', encoding='utf-8') as f:
    alias_data = json.load(f)

with open(SYMPTOMS_PATH, 'r', encoding='utf-8') as f:
    symptoms_data = json.load(f)

with open(SCORING_RULES_PATH, 'r', encoding='utf-8') as f:
    scoring_rules = json.load(f)
print(f"Aliases loaded ✓  ({len(alias_data['aliases'])} entries)")

# id_to_name lookup + symptom registry (weights, body system)
id_to_name = {}
symptom_registry = {}
for cat_key, cat_val in symptoms_data['symptom_categories'].items():
    for s in cat_val['symptoms']:
        id_to_name[s['symptom_id']] = s['symptom_name']
        symptom_registry[s['symptom_id']] = {
            'name': s['symptom_name'],
            'base_weight': s['base_weight'],
            'body_system': s['body_system'],
            'category': cat_key,
        }

# ── THE ACTUAL SEMANTIC MODEL LOADING ───────────────────────────────────
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

sem_model = SentenceTransformer('all-MiniLM-L6-v2')
alias_phrases = [a['alias'].lower() for a in alias_data['aliases']]
alias_ids = [a['symptom_id'] for a in alias_data['aliases']]
alias_matrix = sem_model.encode(alias_phrases, show_progress_bar=False)
print(f"Semantic matcher ready ✓  ({alias_matrix.shape[0]} aliases embedded)")


# ── Layer 3 — ESI scoring engine (hand-tuned rules, NOT a trained model) ─
SEVERITY_MODIFIERS = {'mild': 0.5, 'moderate': 1.0, 'severe': 1.5, 'critical': 2.0}
DURATION_MODIFIERS = {e['id']: e['modifier'] for e in scoring_rules['duration_modifiers']['modifiers']}
HISTORY_MODIFIERS = {e['id']: e['modifier'] for e in scoring_rules['history_modifiers']['modifiers']}
MULTI_SYSTEM_BONUS = {e['systems_affected']: e['bonus_modifier'] for e in scoring_rules['multi_system_bonus']['thresholds']}
AUTO_ESCALATION = scoring_rules['auto_escalation_rules']['rules']
ESI_TIERS = scoring_rules['esi_tiers']
TIER_ORDER = ['LOW', 'MODERATE', 'HIGH', 'CRITICAL']
MAX_SCORE = 400

# Tier boundaries, calibrated to the observed score distribution (see the
# structured-vs-NLP scale-mismatch discussion — these are NOT on the same
# footing as the structured path's even quartiles, hence tier-level fusion)
ESI_TIERS['LOW']['score_range']      = {'min': 0,    'max': 2.5}
ESI_TIERS['MODERATE']['score_range'] = {'min': 2.6,  'max': 12.5}
ESI_TIERS['HIGH']['score_range']     = {'min': 12.6, 'max': 37.5}
ESI_TIERS['CRITICAL']['score_range'] = {'min': 37.6, 'max': 100}


def get_tier(score):
    for tier_name, tier_data in ESI_TIERS.items():
        if tier_data['score_range']['min'] <= score <= tier_data['score_range']['max']:
            return tier_name
    return 'CRITICAL'


def check_auto_escalation(detected_symptom_ids, severity_map):
    minimum_tier = None
    for rule in AUTO_ESCALATION:
        for sid in rule['trigger_symptom_ids']:
            if sid in detected_symptom_ids:
                trigger_sev = rule.get('trigger_severity', 'mild')
                detected_sev = severity_map.get(sid, 'moderate')
                sev_order = ['mild', 'moderate', 'severe', 'critical']
                if sev_order.index(detected_sev) >= sev_order.index(trigger_sev):
                    candidate = rule['minimum_tier']
                    if minimum_tier is None or TIER_ORDER.index(candidate) > TIER_ORDER.index(minimum_tier):
                        minimum_tier = candidate
    return minimum_tier


def calculate_esi(detected_symptoms, severity_map, duration_code=None, history_flags=None):
    """The hand-tuned clinical-rules scoring formula — NOT a trained model."""
    history_flags = history_flags or []

    if not detected_symptoms:
        return {'score': 0, 'base_score': 0, 'after_duration': 0, 'after_history': 0,
                'duration_modifier': 1.0, 'history_modifier': 1.0, 'system_bonus': 1.0,
                'tier': 'LOW', 'escalated': False, 'escalation_tier': None,
                'systems_affected': [], 'breakdown': [], 'auto_escalation': None}

    breakdown, systems_affected, base_score = [], set(), 0
    for sid in detected_symptoms:
        if sid not in symptom_registry:
            continue
        info = symptom_registry[sid]
        severity = severity_map.get(sid, 'moderate')
        sev_mod = SEVERITY_MODIFIERS.get(severity, 1.0)
        contrib = info['base_weight'] * sev_mod
        base_score += contrib
        systems_affected.add(info['body_system'])
        breakdown.append({'symptom_id': sid, 'name': info['name'], 'base_weight': info['base_weight'],
                           'severity': severity, 'sev_modifier': sev_mod, 'contribution': round(contrib, 3)})

    duration_mod = DURATION_MODIFIERS.get(duration_code, 1.0)
    after_duration = base_score * duration_mod

    history_mod = 1.0
    for hm in history_flags:
        if hm in HISTORY_MODIFIERS:
            history_mod *= HISTORY_MODIFIERS[hm]
    history_mod = min(history_mod, scoring_rules['history_modifiers']['cap'])
    after_history = after_duration * history_mod

    n_systems = min(len(systems_affected), 5)
    system_bonus = MULTI_SYSTEM_BONUS.get(n_systems, 1.0)
    raw_score = after_history * system_bonus
    final_score = round(min(max((raw_score / MAX_SCORE) * 100, 1), 100), 1)

    severity_by_id = {b['symptom_id']: b['severity'] for b in breakdown}
    escalation_tier = check_auto_escalation(set(detected_symptoms), severity_by_id)
    calculated_tier = get_tier(final_score)
    if escalation_tier and TIER_ORDER.index(escalation_tier) > TIER_ORDER.index(calculated_tier):
        final_tier, escalated = escalation_tier, True
    else:
        final_tier, escalated = calculated_tier, False

    return {'score': final_score, 'raw_score': round(raw_score, 2), 'base_score': round(base_score, 2),
            'after_duration': round(after_duration, 2), 'after_history': round(after_history, 2),
            'duration_modifier': duration_mod, 'history_modifier': round(history_mod, 2),
            'system_bonus': system_bonus, 'tier': final_tier, 'escalated': escalated,
            'escalation_tier': escalation_tier, 'systems_affected': list(systems_affected),
            'breakdown': breakdown, 'auto_escalation': escalation_tier}


# ── Layer 1 + 2 + 2b — alias matching, BioBERT NER, semantic mapping ────

def map_span_to_id_semantic(span_text, threshold=0.60):
    """THE SEMANTIC MODEL CALL — embeds the span, finds nearest alias."""
    span_embedding = sem_model.encode([span_text.lower()])
    similarities = cosine_similarity(span_embedding, alias_matrix)[0]
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]
    if best_score >= threshold:
        return alias_ids[best_idx], alias_phrases[best_idx], float(best_score)
    return None, None, float(best_score)


def predict(text: str) -> list:
    """
    Layer 1 (alias) + Layer 2 (BioBERT NER) + Layer 2b (semantic mapping).
    Returns a list of detection dicts with symptom_id, matched span, layer.
    """
    detections, l1_hits, matched_spans = [], [], set()
    text_lower = text.lower()

    # Layer 1 — alias matching
    all_aliases = sorted(alias_data['aliases'], key=lambda x: len(x['alias']), reverse=True)
    for entry in all_aliases:
        phrase = entry['alias'].lower()
        idx = text_lower.find(phrase)
        if idx != -1:
            span = (idx, idx + len(phrase))
            overlap = any(not (span[1] <= s[0] or span[0] >= s[1]) for s in matched_spans)
            if not overlap:
                l1_hits.append({'symptom_id': entry['symptom_id'], 'matched_phrase': phrase,
                                 'span': text[idx:idx + len(phrase)], 'layer': 'Layer 1 — Alias'})
                matched_spans.add(span)
    detections.extend(l1_hits)

    # Layer 2 — THE BIOBERT MODEL CALL
    encoding = tokenizer(text, max_length=128, padding='max_length', truncation=True,
                          return_offsets_mapping=True, return_tensors='pt')
    offset_mapping = encoding['offset_mapping'][0].tolist()
    input_ids_list = encoding['input_ids'][0].tolist()
    with torch.no_grad():
        outputs = model(input_ids=encoding['input_ids'].to(DEVICE),
                         attention_mask=encoding['attention_mask'].to(DEVICE))
    preds = torch.argmax(outputs.logits, dim=-1)[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids_list)

    current_span_tokens, current_span_start, ner_spans = [], None, []
    for idx, (pred, token, (tok_start, tok_end)) in enumerate(zip(preds, tokens, offset_mapping)):
        if tok_start == 0 and tok_end == 0:
            if current_span_tokens:
                ner_spans.append((current_span_start, current_span_tokens))
                current_span_tokens, current_span_start = [], None
            continue
        if pred == 1:
            if current_span_tokens:
                ner_spans.append((current_span_start, current_span_tokens))
            current_span_tokens, current_span_start = [token], tok_start
        elif pred == 2 and current_span_tokens:
            current_span_tokens.append(token)
        else:
            if current_span_tokens:
                ner_spans.append((current_span_start, current_span_tokens))
                current_span_tokens, current_span_start = [], None

    # Layer 2b — semantic mapping of NER spans not already caught by Layer 1
    l1_ids = {h['symptom_id'] for h in l1_hits}
    for span_start, span_tokens in ner_spans:
        span_text = tokenizer.convert_tokens_to_string(span_tokens).strip()
        span_lower = span_text.lower()
        already_caught = any(span_lower in h['matched_phrase'] or h['matched_phrase'] in span_lower
                              for h in l1_hits)
        if already_caught or len(span_text) < 3:
            continue
        sid, matched_alias, score = map_span_to_id_semantic(span_text)
        if sid and sid not in l1_ids:
            detections.append({'symptom_id': sid, 'matched_phrase': matched_alias, 'span': span_text,
                                'layer': f'Layer 2 — NER + Semantic ({score:.2f})'})
        elif not sid:
            detections.append({'symptom_id': 'UNKNOWN', 'matched_phrase': span_text, 'span': span_text,
                                'layer': f'Layer 2 — NER (unresolved, score: {score:.2f})'})

    return detections


# ── Gemini — extraction + explanation ───────────────────────────────────
import google.generativeai as genai

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
assert GEMINI_API_KEY, (
    "GEMINI_API_KEY not found in the environment. Set it with "
    "`export GEMINI_API_KEY=...` before starting the backend — "
    "never hardcode it in this file."
)
genai.configure(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-3.6-flash"

EXTRACTION_PROMPT = """
You are a clinical intake assistant for EndoScan, an endometriosis symptom assessment tool.
Your job is to extract structured information from a patient's free-text description.

You must return ONLY valid JSON. No explanation, no preamble.

Extract the following:

1. symptom_descriptions — list of symptom phrases exactly as the patient described them
2. severity_cues — for each symptom, infer severity from language:
   - "mild"     → a little, sometimes, occasionally, slight, minor
   - "moderate" → quite, fairly, regular, recurring, noticeable
   - "severe"   → very, really, bad, terrible, awful, significant, extreme
   - "critical" → unbearable, excruciating, cannot function, worst, debilitating, pass out, faint
3. duration — raw duration phrase if mentioned (e.g. "since I was 14", "for 3 years", null if not mentioned)
4. duration_code — map duration to one of: DM001 (<3 months), DM002 (3-6 months), DM003 (6-12 months), DM004 (1-2 years), DM005 (2-5 years), DM006 (>5 years), DM007 (since first period / since teenager / since adolescence), null if unknown
5. history_flags — list of applicable codes from:
   - HM001: mentions mother/sister/daughter has endometriosis
   - HM002: mentions being misdiagnosed or wrong diagnosis before
   - HM003: mentions laparoscopy that came back normal but pain continues
   - HM004: mentions tried the pill or hormones and they didn't help
   - HM005: mentions fertility problems or told cannot conceive
   - HM006: mentions already diagnosed with endometriosis
   - HM007: mentions chocolate cyst or endometrioma
   - HM008: mentions previous pelvic surgery or caesarean section
   - HM009: mentions autoimmune condition
   - HM010: symptoms started at first period or as a teenager
6. language — detected language: "en", "en-KE", or "sw"
7. needs_followup — list of what is still unknown: "duration", "severity", "history", or empty list

Patient input: "{user_input}"

Return JSON only. Example format:
{{
  "symptom_descriptions": ["painful periods", "pain during sex"],
  "severity_cues": {{"painful periods": "severe", "pain during sex": "moderate"}},
  "duration": "since I was 14",
  "duration_code": "DM007",
  "history_flags": ["HM010"],
  "language": "en-KE",
  "needs_followup": ["history"]
}}
"""

EXPLANATION_PROMPT = """
You are EndoScan, a compassionate health tool designed for people who may have endometriosis.
Many of the people using this tool have been dismissed by doctors and told their pain is normal.
Your tone is warm, validating, and clear. Never clinical or cold.

Based on the assessment below, write a patient-facing explanation.

Assessment:
- Symptoms identified: {symptom_list}
- Severity: {severity_summary}
- Duration: {duration}
- ESI Score: {score}
- Tier: {tier}
- Tier description: {tier_description}
- Systems affected: {systems}
- Recommended action: {recommended_action}
{escalation_note}

Write the explanation in this structure:
1. One sentence acknowledging what they shared (warm, not clinical)
2. What was identified — list the symptoms in plain language, not medical terms
3. Why the score is what it is — explain the duration and severity contribution briefly
4. What the tier means for them — what they should do next
5. One closing sentence that validates their experience

Rules:
- Maximum 200 words
- No bullet points — flowing paragraphs
- No scary language — empowering not alarming
- If CRITICAL tier, be urgent but calm — they need to act today, not panic
- Use plain English — no Latin terms
- Do not mention ESI score as a number — just describe the tier level
- Do not say "I" — you are EndoScan, a tool, not a person
"""


def extract_with_gemini(user_input: str) -> dict:
    """THE GEMINI EXTRACTION CALL."""
    gemini = genai.GenerativeModel(
        model_name=MODEL_NAME,
        generation_config={"temperature": 0.1, "response_mime_type": "application/json"}
    )
    prompt = EXTRACTION_PROMPT.format(user_input=user_input)
    response = gemini.generate_content(prompt)
    try:
        return json.loads(response.text)
    except json.JSONDecodeError:
        return {"symptom_descriptions": [user_input], "severity_cues": {}, "duration": None,
                "duration_code": None, "history_flags": [], "language": "en",
                "needs_followup": ["duration", "history"]}


def clean_extracted(extracted: dict) -> dict:
    severity_cues = extracted.get('severity_cues') or {}
    for key in severity_cues:
        if severity_cues[key] is None:
            severity_cues[key] = 'moderate'
    extracted['severity_cues'] = severity_cues
    extracted['needs_followup'] = [n for n in extracted.get('needs_followup', []) if n != 'severity']
    extracted.setdefault('symptom_descriptions', [])
    extracted.setdefault('duration', None)
    extracted.setdefault('duration_code', None)
    extracted.setdefault('history_flags', [])
    extracted.setdefault('language', 'en')
    return extracted


def generate_explanation(esi_result, extracted, detected_symptom_names, duration_raw) -> str:
    """THE GEMINI EXPLANATION CALL."""
    symptom_list = ", ".join(detected_symptom_names) if detected_symptom_names else "general symptoms"
    severity_summary = ", ".join(f"{k}: {v}" for k, v in (extracted.get('severity_cues') or {}).items()) or "moderate"
    tier_data = ESI_TIERS[esi_result['tier']]
    escalation_note = "- Note: Tier was escalated due to a clinical red flag symptom." if esi_result.get('escalated') else ""

    prompt = EXPLANATION_PROMPT.format(
        symptom_list=symptom_list, severity_summary=severity_summary,
        duration=duration_raw or "not specified", score=esi_result['score'], tier=esi_result['tier'],
        tier_description=tier_data['plain_language_explanation'],
        systems=", ".join(esi_result['systems_affected']) if esi_result['systems_affected'] else "one system",
        recommended_action=tier_data['recommended_action'], escalation_note=escalation_note
    )
    explanation_model = genai.GenerativeModel(model_name=MODEL_NAME, generation_config={"temperature": 0.7})
    return explanation_model.generate_content(prompt).text


# ── The actual entry point assess_patient.py calls ─────────────────────

def run_full_pipeline(user_input: str, duration_code=None, history_flags=None, verbose=False) -> dict:
    """
    Full NLP pipeline: Gemini extraction -> alias/NER/semantic detection
    -> ESI scoring -> Gemini explanation. This is what assess_patient.py
    calls as the NLP path.
    """
    extracted = clean_extracted(extract_with_gemini(user_input))

    if duration_code is None:
        duration_code = extracted.get('duration_code')
    if history_flags is None:
        history_flags = extracted.get('history_flags', [])

    all_detections, seen_ids = [], set()
    symptom_descriptions = extracted.get('symptom_descriptions', [user_input])
    for desc in list(set(symptom_descriptions + [user_input])):
        for d in predict(desc):
            sid = d['symptom_id']
            if sid and sid != 'UNKNOWN' and sid not in seen_ids:
                all_detections.append(d)
                seen_ids.add(sid)

    severity_cues = extracted.get('severity_cues', {})
    severity_map = {}
    for d in all_detections:
        sid = d['symptom_id']
        sev = 'moderate'
        for phrase, s in severity_cues.items():
            if phrase.lower() in d['span'].lower() or d['span'].lower() in phrase.lower():
                sev = s
                break
        severity_map[sid] = sev

    detected_ids = list(seen_ids)
    esi_result = calculate_esi(detected_ids, severity_map, duration_code, history_flags)

    detected_names = [id_to_name.get(sid, sid).replace('_', ' ') for sid in detected_ids if sid in symptom_registry]
    explanation = generate_explanation(esi_result, extracted, detected_names, extracted.get('duration'))

    if verbose:
        print(f"Detected {len(detected_ids)} symptoms | Tier: {esi_result['tier']} | Score: {esi_result['score']}")

    return {'extracted': extracted, 'detections': all_detections, 'severity_map': severity_map,
            'esi_result': esi_result, 'explanation': explanation}


if __name__ == '__main__':
    # Quick smoke test — run this file directly to confirm everything loads
    # and the pipeline produces a sane result before wiring it into the backend.
    test = run_full_pipeline(
        "I have had unbearable painful periods since I was 14, "
        "pain during sex, and my sister also has endometriosis",
        verbose=True
    )
    print(test['explanation'])
