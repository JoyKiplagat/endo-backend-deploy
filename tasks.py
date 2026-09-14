from celery import shared_task
from .models import QuestionnaireSubmission
from .nlp_model import run_full_pipeline
import logging

logger = logging.getLogger(__name__)

@shared_task
def process_nlp_questionnaire_task(submission_id, raw_answers):
    try:
        submission = QuestionnaireSubmission.objects.get(id=submission_id)
        submission.status = 'PROCESSING'
        submission.save()

        # Run NLP pipeline
        nlp_result = run_full_pipeline(raw_answers)
        
        print("="*60)
        print("NLP PIPELINE OUTPUT:", nlp_result)
        print("="*60)

        # Retrieve organ systems with fallbacks across pipeline keys
        extracted_systems = (
            nlp_result.get("organ_systems_flagged") or 
            nlp_result.get("systems_affected") or 
            (nlp_result.get("esi_result", {}).get("systems_affected") if isinstance(nlp_result.get("esi_result"), dict) else []) or 
            []
        )

        extracted_symptoms = nlp_result.get("matched_symptoms", [])
        explanation_text = (
            nlp_result.get("explanation") or 
            nlp_result.get("summary") or 
            nlp_result.get("clinical_insight") or 
            str(nlp_result)
        )

        # Build complete output dictionary aligned with TypeScript interfaces
        formatted_output = {
            "risk_level": nlp_result.get("tier") or nlp_result.get("esi_tier") or nlp_result.get("risk_level") or "Triage Complete",
            "matched_symptoms": extracted_symptoms,
            "organ_systems_flagged": extracted_systems,
            "systems_affected": extracted_systems,
            "explanation": explanation_text,
            "summary": explanation_text,
            "esi_result": {
                "tier": nlp_result.get("tier") or nlp_result.get("esi_tier") or nlp_result.get("risk_level") or "Triage Complete",
                "score": nlp_result.get("score") or nlp_result.get("esi_score") or 0,
                "adjusted_endo_score": nlp_result.get("score") or nlp_result.get("esi_score") or 0,
                "systems_affected": extracted_systems,
                "organ_systems_flagged": extracted_systems,
                "matched_symptoms": extracted_symptoms
            },
            "clinical_perspectives": nlp_result.get("clinical_perspectives") or {
                "endo_possibility": explanation_text,
                "differential_considerations": "Symptoms overlap with common pelvic or digestive health conditions.",
                "recommendations": "Consider scheduling a consultation with an Endometriosis Specialist or Gynecologist."
            }
        }

        submission.model_output = formatted_output
        submission.status = 'COMPLETED'
        submission.save()

    except Exception as e:
        logger.error(f"Error processing questionnaire task: {str(e)}")
        try:
            submission = QuestionnaireSubmission.objects.get(id=submission_id)
            submission.status = 'FAILED'
            submission.model_output = {"explanation": f"Model processing failed: {str(e)}"}
            submission.save()
        except Exception:
            pass


# from celery import shared_task
# from .models import QuestionnaireSubmission
# from .nlp_model import run_full_pipeline
# import logging

# logger = logging.getLogger(__name__)

# @shared_task
# def process_nlp_questionnaire_task(submission_id, raw_answers):
#     try:
#         submission = QuestionnaireSubmission.objects.get(id=submission_id)
#         submission.status = 'PROCESSING'
#         submission.save()

#         # Run your NLP pipeline
#         nlp_result = run_full_pipeline(raw_answers)
        
#         # LOG THIS TO YOUR CONSOLE TO SEE THE EXACT KEYS RETURNED BY YOUR MODEL
#         print("="*60)
#         print("NLP PIPELINE OUTPUT:", nlp_result)
#         print("="*60)

#         # Build clean output dictionary for React frontend
#         formatted_output = {
#             "esi_result": {
#                 "tier": nlp_result.get("tier") or nlp_result.get("esi_tier") or nlp_result.get("risk_level") or "Triage Complete",
#                 "score": nlp_result.get("score") or nlp_result.get("esi_score") or 0,
#                 "systems_affected": nlp_result.get("systems_affected", [])
#             },
#             # Map explanation to clinical summary, narrative, or raw dict string
#             "explanation": (
#                 nlp_result.get("explanation") or 
#                 nlp_result.get("summary") or 
#                 nlp_result.get("clinical_insight") or 
#                 nlp_result.get("gemini_explanation") or
#                 str(nlp_result)  # Fallback: display full raw output
#             )
#         }

#         submission.model_output = formatted_output
#         submission.status = 'COMPLETED'
#         submission.save()

#     except Exception as e:
#         logger.error(f"Error processing questionnaire task: {str(e)}")
#         submission = QuestionnaireSubmission.objects.get(id=submission_id)
#         submission.status = 'FAILED'
#         submission.model_output = {"explanation": f"Model processing failed: {str(e)}"}
#         submission.save()