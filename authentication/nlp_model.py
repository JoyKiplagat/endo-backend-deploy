import os
import json
import re
from pathlib import Path
import numpy as np
#import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
from google.genai import types
from django.conf import settings

BASE_DIR = Path(__file__).resolve().parent.parent
# ── Artifact Paths ───────────────────────────────────────────────────
DATA_DIR = BASE_DIR / 'data'
NER_MODEL_DIR = BASE_DIR / 'endoscan_ner_model'

ALIASES_PATH = DATA_DIR / 'symptom_aliases.json'
SYMPTOMS_PATH = DATA_DIR / 'symptoms.json'
SCORING_RULES_PATH = DATA_DIR / 'scoring_rules.json'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── 1. Load Data & Models ────────────────────────────────────────────
model = AutoModelForTokenClassification.from_pretrained(str(NER_MODEL_DIR)).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(str(NER_MODEL_DIR))
model.eval()

with open(ALIASES_PATH, 'r', encoding='utf-8') as f:
    alias_data = json.load(f)

with open(SYMPTOMS_PATH, 'r', encoding='utf-8') as f:
    symptoms_data = json.load(f)

with open(SCORING_RULES_PATH, 'r', encoding='utf-8') as f:
    scoring_rules = json.load(f)

# Plain-language mapping table (converts clinical jargon into patient-friendly language)
PLAIN_LANGUAGE_MAP = {
    "dysmenorrhea": "painful period cramps",
    "dyspareunia": "pain during sex",
    "dyschezia": "painful bowel movements",
    "dysuria": "pain during urination",
    "hematochezia": "blood in stool",
    "menorrhagia": "heavy period bleeding",
    "gastrointestinal": "Stomach & Digestion",
    "reproductive": "Pelvic & Reproductive Organs",
    "urinary": "Bladder & Urination",
    "neurological": "Nerves & Back Pain",
    "musculoskeletal": "Muscles & Joints"
}

symptom_registry = {}
for cat_key, cat_val in symptoms_data['symptom_categories'].items():
    for s in cat_val['symptoms']:
        raw_name = s['symptom_name'].lower().strip()
        clean_name = PLAIN_LANGUAGE_MAP.get(raw_name, raw_name.replace('_', ' ').title())
        
        raw_system = s.get('body_system', '').lower().strip()
        clean_system = PLAIN_LANGUAGE_MAP.get(raw_system, s.get('body_system', 'General').title())
        
        symptom_registry[s['symptom_id']] = {
            'name': clean_name,
            'base_weight': s.get('base_weight', 15.0),
            'body_system': clean_system,
            'category': cat_key,
        }

# Semantic Model for Local Resolution
sem_model = SentenceTransformer('all-MiniLM-L6-v2', device=str(DEVICE))
alias_phrases = [a['alias'].lower() for a in alias_data['aliases']]
alias_ids = [a['symptom_id'] for a in alias_data['aliases']]
alias_matrix = sem_model.encode(alias_phrases, show_progress_bar=False, batch_size=64)

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
ai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

SEVERITY_MODIFIERS = {'mild': 0.5, 'moderate': 1.0, 'severe': 1.5, 'critical': 2.0}
DURATION_MODIFIERS = scoring_rules.get('duration_modifiers', {'acute': 1.0, 'chronic': 1.25})
HISTORY_MODIFIERS = scoring_rules.get('history_modifiers', {})
MULTI_SYSTEM_BONUS = scoring_rules.get('multi_system_bonus', {1: 1.0, 2: 1.15, 3: 1.25, 4: 1.35, 5: 1.5})
MAX_SCORE = scoring_rules.get('max_expected_raw_score', 80.0)

# ── 2. Prompts ────────────────────────────────────────────────────────

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

FOLLOWUP_PROMPT = """
You are EndoScan, a compassionate symptom assessment assistant for endometriosis.
You are having a conversation with a patient.

Conversation so far:
{history}

Collected symptoms so far: {symptoms}
Duration known: {duration}
History known: {history_flags}
Still needed: {needs_followup}

Your job:
- If duration is unknown and not yet asked, ask gently about how long they have had these symptoms
- If history is unknown and not yet asked, ask ONE gentle question about family history or previous diagnosis
- If both are known, thank them and tell them you are calculating their assessment
- Never ask more than one question at a time
- Keep it warm and conversational — they may be distressed
- Maximum 2 sentences

Reply only with your next message to the patient. No JSON, no preamble.
"""

# ── 3. Helper & Local NER Functions ──────────────────────────────────
def get_tier_from_rules(score: float) -> str:
    import torch
    """Dynamically assign ESI tier based on structural JSON ranges."""
    esi_tiers = scoring_rules.get('esi_tiers', {})
    
    for tier_name, config in esi_tiers.items():
        score_range = config.get('score_range', {})
        min_val = score_range.get('min', 0)
        max_val = score_range.get('max', 100)
        
        if min_val <= score <= max_val:
            return tier_name

    if score <= 25.0: return 'LOW'
    elif score <= 50.0: return 'MODERATE'
    elif score <= 75.0: return 'HIGH'
    return 'CRITICAL'


def map_span_to_id_semantic(span_text: str, threshold=0.45):
    import torch
    """Maps free-text span to symptom ID via sentence transformer cosine similarity."""
    if not span_text or len(span_text.strip()) < 2:
        return None, None, 0.0

    span_embedding = sem_model.encode([span_text.lower()])
    similarities = cosine_similarity(span_embedding, alias_matrix)[0]
    best_idx = np.argmax(similarities)
    best_score = float(similarities[best_idx])
    
    if best_score >= threshold:
        return alias_ids[best_idx], alias_phrases[best_idx], best_score
    return None, None, best_score


def predict_single_pass(text: str) -> list:
    import torch
    """Run Layer 1 (Alias Match) & Layer 2 (BioBERT NER + Semantic Resolution) on text."""
    detections, l1_hits, matched_spans = [], [], set()
    text_lower = text.lower()

    # Layer 1 — Direct Alias Match
    all_aliases = sorted(alias_data['aliases'], key=lambda x: len(x['alias']), reverse=True)
    for entry in all_aliases:
        phrase = entry['alias'].lower()
        idx = text_lower.find(phrase)
        if idx != -1:
            span = (idx, idx + len(phrase))
            if not any(not (span[1] <= s[0] or span[0] >= s[1]) for s in matched_spans):
                l1_hits.append({
                    'symptom_id': entry['symptom_id'], 
                    'matched_phrase': phrase,
                    'span': text[idx:idx + len(phrase)], 
                    'layer': 'Layer 1 — Alias'
                })
                matched_spans.add(span)
    detections.extend(l1_hits)

    # Layer 2 — BioBERT NER Pass
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
        if pred == 1:  # B-SYMPTOM
            if current_span_tokens:
                ner_spans.append((current_span_start, current_span_tokens))
            current_span_tokens, current_span_start = [token], tok_start
        elif pred == 2 and current_span_tokens:  # I-SYMPTOM
            current_span_tokens.append(token)
        else:
            if current_span_tokens:
                ner_spans.append((current_span_start, current_span_tokens))
                current_span_tokens, current_span_start = [], None

    # Layer 2b — Semantic Resolution for unmatched NER spans
    l1_ids = {h['symptom_id'] for h in l1_hits}
    for span_start, span_tokens in ner_spans:
        span_text = tokenizer.convert_tokens_to_string(span_tokens).strip()
        span_lower = span_text.lower()
        if any(span_lower in h['matched_phrase'] or h['matched_phrase'] in span_lower for h in l1_hits) or len(span_text) < 3:
            continue
        sid, matched_alias, score = map_span_to_id_semantic(span_text)
        if sid and sid not in l1_ids:
            detections.append({
                'symptom_id': sid, 
                'matched_phrase': matched_alias, 
                'span': span_text,
                'layer': f'Layer 2 — NER + Semantic ({score:.2f})'
            })

    return detections


def calculate_esi(detected_symptoms, severity_map, duration_code=None, history_flags=None):
    import torch
    """Calculates ESI score (0-100) and assigns tier."""
    history_flags = history_flags or []
    if not detected_symptoms:
        return {'score': 0.0, 'tier': 'LOW', 'systems_affected': [], 'breakdown': []}

    breakdown, systems_affected, base_score = [], set(), 0
    for sid in detected_symptoms:
        if sid not in symptom_registry: 
            continue
        info = symptom_registry[sid]
        
        severity = severity_map.get(sid, severity_map.get(info['name'], 'moderate'))
        sev_mod = SEVERITY_MODIFIERS.get(severity, 1.0)
        contrib = info['base_weight'] * sev_mod
        base_score += contrib
        systems_affected.add(info['body_system'])
        
        # Enforce Plain Language in score breakdown items
        clean_item_name = PLAIN_LANGUAGE_MAP.get(info['name'].lower().strip(), info['name'])
        breakdown.append({
            'symptom_id': sid, 
            'name': clean_item_name, 
            'severity': severity, 
            'contribution': round(contrib, 3)
        })

    duration_mod = DURATION_MODIFIERS.get(duration_code, 1.0)
    after_duration = base_score * duration_mod

    history_mod = 1.0
    for hm in history_flags:
        if hm in HISTORY_MODIFIERS: 
            history_mod *= HISTORY_MODIFIERS[hm]
    
    cap_val = HISTORY_MODIFIERS.get('cap', 1.5) if isinstance(HISTORY_MODIFIERS, dict) else 1.5
    history_mod = min(history_mod, cap_val)

    n_systems = min(len(systems_affected), 5)
    system_bonus = MULTI_SYSTEM_BONUS.get(str(n_systems), MULTI_SYSTEM_BONUS.get(n_systems, 1.0))
    raw_score = after_duration * history_mod * system_bonus
    final_score = round(min(max((raw_score / MAX_SCORE) * 100, 0), 100), 1)

    calculated_tier = get_tier_from_rules(final_score)

    return {
        'score': final_score, 
        'tier': calculated_tier, 
        'systems_affected': list(systems_affected), 
        'breakdown': breakdown
    }

# ── 4. Main Pipeline Entrypoints ─────────────────────────────────────

def extract_extraction_data(user_input: str) -> dict:
    """Uses EXTRACTION_PROMPT via Gemini to extract structured JSON from user input."""
    if not ai_client:
        return {
            "symptom_descriptions": [user_input],
            "severity_cues": {},
            "duration": None,
            "duration_code": None,
            "history_flags": [],
            "language": "en",
            "needs_followup": ["duration", "history"]
        }

    formatted_prompt = EXTRACTION_PROMPT.format(user_input=user_input)
    try:
        response = ai_client.models.generate_content(
            model='gemini-3.6-flash',
            contents=formatted_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
                tools=[]
            )
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Extraction Prompt Gemini Error: {e}")
        return {
            "symptom_descriptions": [user_input],
            "severity_cues": {},
            "duration": None,
            "duration_code": None,
            "history_flags": [],
            "language": "en",
            "needs_followup": ["duration", "history"]
        }


def generate_explanation(assessment_data: dict) -> str:
    import torch
    """Uses EXPLANATION_PROMPT to generate patient-facing explanation."""
    if not ai_client:
        return "Thank you for sharing your symptoms. Based on what you shared, your symptoms show patterns that align with endometriosis. We recommend discussing these results with a healthcare specialist."

    formatted_prompt = EXPLANATION_PROMPT.format(
        symptom_list=", ".join(assessment_data.get('symptoms', [])),
        severity_summary=assessment_data.get('severity_summary', 'Moderate'),
        duration=assessment_data.get('duration', 'Not specified'),
        score=assessment_data.get('score', 0),
        tier=assessment_data.get('tier', 'MODERATE'),
        tier_description=assessment_data.get('tier_description', 'Moderate likelihood of Endometriosis'),
        systems=", ".join(assessment_data.get('systems', [])),
        recommended_action=assessment_data.get('recommended_action', 'Schedule a consultation with a specialist.'),
        escalation_note=assessment_data.get('escalation_note', '')
    )

    try:
        response = ai_client.models.generate_content(
            model='gemini-3.6-flash',
            contents=formatted_prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                tools=[]
            )
        )
        return response.text.strip()
    except Exception as e:
        print(f"Explanation Prompt Gemini Error: {e}")
        return "Thank you for sharing your symptoms. Based on what you shared, your symptoms show patterns that align with endometriosis. We recommend discussing these results with a healthcare specialist."


def generate_followup_question(history: str, symptoms: list, duration: str, history_flags: list, needs_followup: list) -> str:
    import torch
    """Uses FOLLOWUP_PROMPT to ask gentle follow-up questions."""
    if not ai_client:
        return "Thank you for sharing. Could you tell me roughly how long you have been experiencing these symptoms?"

    formatted_prompt = FOLLOWUP_PROMPT.format(
        history=history,
        symptoms=", ".join(symptoms),
        duration=duration or "Unknown",
        history_flags=", ".join(history_flags) if history_flags else "None",
        needs_followup=", ".join(needs_followup) if needs_followup else "None"
    )

    try:
        response = ai_client.models.generate_content(
            model='gemini-3.6-flash',
            contents=formatted_prompt,
            config=types.GenerateContentConfig(
                temperature=0.4,
                tools=[]
            )
        )
        return response.text.strip()
    except Exception as e:
        print(f"Followup Prompt Gemini Error: {e}")
        return "Thank you for sharing. Could you tell me roughly how long you have been experiencing these symptoms?"


def run_full_pipeline(user_input: str, history_text: str = "", duration_code=None, history_flags=None, **kwargs) -> dict:
    """
    Full pipeline integrating extraction, local semantic resolution, BioBERT fallback,
    ESI calculation, and compassionate explanation generation.
    """
    # 1. Extraction Pass via Gemini
    extracted = extract_extraction_data(user_input)

    symptom_phrases = extracted.get("symptom_descriptions", [])
    severity_cues = extracted.get("severity_cues", {})
    duration = extracted.get("duration")
    dur_code = duration_code or extracted.get("duration_code")
    hist_flags = history_flags or extracted.get("history_flags", [])
    needs_followup = extracted.get("needs_followup", [])

    # 2. Map Extracted Phrases to Registered Symptom IDs and Severity
    detected_ids = []
    mapped_severities = {}

    for phrase in symptom_phrases:
        sid, _, score = map_span_to_id_semantic(phrase, threshold=0.45)
        if sid:
            if sid not in detected_ids:
                detected_ids.append(sid)
            if phrase in severity_cues:
                mapped_severities[sid] = severity_cues[phrase]

    # 3. Fallback Pass: If semantic resolution yielded no IDs, run full local BioBERT/Alias pass
    if not detected_ids:
        local_hits = predict_single_pass(user_input)
        for h in local_hits:
            sid = h['symptom_id']
            if sid != 'UNKNOWN' and sid not in detected_ids:
                detected_ids.append(sid)
                mapped_severities[sid] = 'moderate'

    # ── SAFE SYMPTOM LIST RESOLUTION & SCOPE INITIALIZATION ─────────────────
    unique_symptoms = []
    for sid in detected_ids:
        if sid in symptom_registry:
            unique_symptoms.append(symptom_registry[sid]['name'])

    if not unique_symptoms and symptom_phrases:
        unique_symptoms = symptom_phrases

    # ── JARGON CLEANUP PASS ──────────────────────────────────────────────────
    # Clean up any leftover strings like "chronic_pelvic_pain" or "dysmenorrhea"
    unique_symptoms = [
        PLAIN_LANGUAGE_MAP.get(str(symptom).lower().strip(), str(symptom).replace('_', ' ').title()) 
        for symptom in unique_symptoms
    ]

    # 4. Calculate ESI Score
    esi_res = calculate_esi(
        detected_symptoms=detected_ids, 
        severity_map=mapped_severities, 
        duration_code=dur_code, 
        history_flags=hist_flags
    )

    # 5. Generate Patient Explanation
    assessment_payload = {
        'symptoms': unique_symptoms,
        'severity_summary': ", ".join([f"{k}: {v}" for k, v in severity_cues.items()]) or "Moderate",
        'duration': duration or "Not specified",
        'score': esi_res['score'],
        'tier': esi_res['tier'],
        'tier_description': f"{esi_res['tier'].title()} priority level based on reported symptoms.",
        'systems': esi_res['systems_affected'],
        'recommended_action': "Consult with a specialist or gynecologist.",
        'escalation_note': "If you experience unbearable pain or fainting, seek urgent care immediately." if esi_res['tier'] == 'CRITICAL' else ""
    }

    explanation_narrative = generate_explanation(assessment_payload)

    # 6. Check Follow-Up Needs
    next_question = None
    if needs_followup:
        next_question = generate_followup_question(
            history=history_text,
            symptoms=unique_symptoms,
            duration=duration,
            history_flags=hist_flags,
            needs_followup=needs_followup
        )

    # 7. Complete Payload Output
    return {
        'score': esi_res['score'],
        'tier': esi_res['tier'],
        'matched_symptoms': unique_symptoms,
        'extracted_phrases': symptom_phrases,
        'systems_affected': esi_res['systems_affected'],
        'breakdown': esi_res['breakdown'],
        'explanation': explanation_narrative,
        'needs_followup': needs_followup,
        'next_question': next_question,
        'extraction_data': extracted,
        'esi_result': esi_res
    }