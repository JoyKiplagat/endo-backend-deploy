from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth import authenticate
from django.contrib.auth.models import User
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication
import traceback
from .models import PatientProfile, SymptomLog, ScanRecord, QuestionnaireSubmission
from .serializers import RegisterSerializer, SymptomLogSerializer, ScanRecordSerializer

class PatientRegisterView(APIView):
    permission_classes = [permissions.AllowAny]
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            profile = user.profile
            
            # Save full display name cleanly to the PatientProfile record
            profile.name = request.data.get('name', 'Anonymous Patient')
            profile.save()
            
            # Catch initialization file attachments
            scan_file = request.FILES.get('scan_file')
            if scan_file:
                ScanRecord.objects.create(patient=profile, scan_file=scan_file)

            refresh = RefreshToken.for_user(user)
            return Response({
                "access": str(refresh.access_token),
                "refresh": str(refresh),
                "id": profile.id,
                "name": profile.name,
                "endometriosis_stage": profile.endometriosis_stage,
                "location": profile.location
            }, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class PatientLoginView(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request):
        email = request.data.get('email')
        password = request.data.get('password')
        
        user = authenticate(username=email, password=password)
        if user is not None:
            profile = user.profile
            refresh = RefreshToken.for_user(user)
            
            return Response({
                "access": str(refresh.access_token),
                "refresh": str(refresh),
                "id": profile.id,
                "name": profile.name,
                "endometriosis_stage": profile.endometriosis_stage,
                "location": profile.location,
                "diagnosis_date": profile.diagnosis_date
            }, status=status.HTTP_200_OK)
            
        return Response({"error": "Invalid verification credentials"}, status=status.HTTP_400_BAD_REQUEST)


class ForgotPasswordView(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request):
        email = request.data.get('email')
        new_password = request.data.get('new_password')
        
        if not email or not new_password:
            return Response({"error": "Both email and new_password are required."}, status=status.HTTP_400_BAD_REQUEST)
            
        try:
            user = User.objects.get(email=email)
            user.set_password(new_password)
            user.save()
            return Response({"message": "Password reset successfully completed!"}, status=status.HTTP_200_OK)
        except User.DoesNotExist:
            return Response({"error": "No account associated with this email address."}, status=status.HTTP_404_NOT_FOUND)


class PatientProfileUpdateView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, patient_id):
        try:
            profile = PatientProfile.objects.get(id=patient_id)
            return Response({
                "id": profile.id,
                "name": profile.name,
                "endometriosis_stage": profile.endometriosis_stage,
                "location": profile.location,
                "diagnosis_date": profile.diagnosis_date
            }, status=status.HTTP_200_OK)
        except PatientProfile.DoesNotExist:
            return Response({"error": "Profile not found"}, status=status.HTTP_404_NOT_FOUND)

    def put(self, request, patient_id):
        try:
            profile = PatientProfile.objects.get(id=patient_id)
        except PatientProfile.DoesNotExist:
            return Response({"error": "Profile not found"}, status=status.HTTP_404_NOT_FOUND)
            
        if 'name' in request.data:
            profile.name = request.data.get('name')
            
        profile.location = request.data.get('location', profile.location)
        profile.endometriosis_stage = request.data.get(
            'endometriosis_stage', 
            request.data.get('endometriosisStage', profile.endometriosis_stage)
        )
        profile.diagnosis_date = request.data.get(
            'diagnosis_date', 
            request.data.get('diagnosisDate', profile.diagnosis_date)
        )
        profile.save()

        return Response({
            "id": profile.id,
            "name": profile.name,
            "endometriosis_stage": profile.endometriosis_stage,
            "location": profile.location,
            "diagnosis_date": profile.diagnosis_date
        }, status=status.HTTP_200_OK)


class SymptomLogView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        patient_id = request.query_params.get('patient')
        if not patient_id:
            return Response([], status=status.HTTP_200_OK)
        logs = SymptomLog.objects.filter(patient_id=patient_id).order_by('-date')
        return Response(SymptomLogSerializer(logs, many=True).data, status=status.HTTP_200_OK)

    def post(self, request):
        serializer = SymptomLogSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


import os
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions
from rest_framework.parsers import MultiPartParser, FormParser
from .models import ScanRecord, PatientProfile
from .serializers import ScanRecordSerializer
# views.py
from .ml_router import route_and_explain # Import ML processing logic

class ScanRecordView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def get(self, request):
        patient_id = request.query_params.get('patient')
        if not patient_id:
            return Response([], status=status.HTTP_200_OK)
        scans = ScanRecord.objects.filter(patient_id=patient_id).order_by('-uploaded_at')
        serializer = ScanRecordSerializer(scans, many=True, context={'request': request})
        return Response(serializer.data, status=status.HTTP_200_OK)

    def post(self, request):
        try:
            patient_id = request.data.get('patient')
            try:
                patient_profile = PatientProfile.objects.get(id=patient_id)
            except (PatientProfile.DoesNotExist, ValueError):
                return Response({"patient": ["Invalid patient profile ID provided."]}, status=status.HTTP_400_BAD_REQUEST)

            scan_file = request.FILES.get('scan_file')
            if not scan_file:
                return Response({"scan_file": ["No file uploaded."]}, status=status.HTTP_400_BAD_REQUEST)

            # Save uploaded scan
            scan_instance = ScanRecord.objects.create(
                patient=patient_profile,
                scan_file=scan_file
            )

            # Run ML Vision Model logic using uploaded file path
            try:
                model_explanation = route_and_explain(scan_instance.scan_file.path)
                scan_instance.scan_note = model_explanation
                scan_instance.save()
            except Exception as ml_err:
                scan_instance.scan_note = f"File uploaded, but ML processing encountered issue: {str(ml_err)}"
                scan_instance.save()

            serializer = ScanRecordSerializer(scan_instance, context={'request': request})
            return Response(serializer.data, status=status.HTTP_201_CREATED)

        except Exception as err:
            print(f"CRITICAL 500 ERROR IN SCAN RECORD POST: {str(err)}")
            return Response({"error": "Internal server processing failure.", "details": str(err)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                        
import traceback
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication
from django.db import transaction

class QuestionnaireProcessView(APIView):
    """
    Authenticated endpoint. Receives raw user questionnaire responses,
    creates a QuestionnaireSubmission, passes data directly into the central nlp_model,
    stores the generated output, and returns the analysis back to React.
    """
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        patient_id = request.query_params.get('patient')
        
        # Scoped querying to guarantee data security across authenticated users
        if patient_id and (request.user.is_staff or getattr(request.user, 'is_doctor', False)):
            submissions = QuestionnaireSubmission.objects.filter(
                patient_id=patient_id
            ).order_by('-submitted_at')
        else:
            submissions = QuestionnaireSubmission.objects.filter(
                patient__user=request.user
            ).order_by('-submitted_at')

        data = [
            {
                "id": sub.id,
                "submitted_at": sub.submitted_at.isoformat() if getattr(sub, 'submitted_at', None) else None,
                "status": getattr(sub, 'status', 'COMPLETED'),
                "raw_responses": sub.raw_responses or {},
                "model_output": sub.model_output or {}
            }
            for sub in submissions
        ]

        return Response(data, status=status.HTTP_200_OK)

    def post(self, request):
        patient_profile, _ = PatientProfile.objects.get_or_create(user=request.user)
        raw_answers = request.data.get('raw_responses') or request.data.get('answers') or {}

        if not raw_answers or not isinstance(raw_answers, dict):
            return Response(
                {"error": "Invalid or missing raw_responses format. Must be a valid key-value object."},
                status=status.HTTP_400_BAD_REQUEST
            )

        # 1. Create initial record with 'PROCESSING' status
        submission = QuestionnaireSubmission.objects.create(
            patient=patient_profile,
            raw_responses=raw_answers,
            status='PROCESSING',
            model_output=None
        )

        try:
            # 2. Extract and format parameters for NLP inference
            text_segments = []
            duration_code = None
            history_flags = []

            for key, value in raw_answers.items():
                if not value:
                    continue

                if key == "q3_onset":
                    duration_code = value
                elif key in ["q11_family", "q12_mimic"] and isinstance(value, list):
                    history_flags.extend(value)

                if isinstance(value, list):
                    text_segments.append(f"Selected for {key}: {', '.join(map(str, value))}.")
                elif isinstance(value, str):
                    text_segments.append(f"{key}: {value}.")

            full_input_text = " ".join(text_segments)

            # 3. Execute NLP Pipeline
            pipeline_result = run_full_pipeline(
                user_input=full_input_text,
                duration_code=duration_code,
                history_flags=history_flags
            )

            # 4. Save analysis output back to database
            submission.model_output = pipeline_result
            submission.status = 'COMPLETED'
            submission.save()

            # 5. Return complete output payload to frontend
            return Response({
                "id": submission.id,
                "submitted_at": submission.submitted_at.isoformat() if getattr(submission, 'submitted_at', None) else None,
                "status": "COMPLETED",
                "raw_responses": submission.raw_responses,
                "model_output": pipeline_result
            }, status=status.HTTP_201_CREATED)

        except Exception as e:
            # Mark database entry as FAILED if error occurs mid-process
            submission.status = 'FAILED'
            submission.save()

            print("\n" + "="*60)
            print("CRITICAL BACKEND ERROR IN QUESTIONNAIRE POST:")
            traceback.print_exc()
            print("="*60 + "\n")

            return Response(
                {
                    "submission_id": submission.id,
                    "detail": "Internal model processing error.",
                    "error": str(e)
                }, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class QuestionnaireStatusView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request, submission_id):
        try:
            submission = QuestionnaireSubmission.objects.get(
                id=submission_id, 
                patient__user=request.user
            )
            return Response({
                "id": submission.id,
                "submitted_at": submission.submitted_at.isoformat() if getattr(submission, 'submitted_at', None) else None,
                "status": getattr(submission, 'status', 'COMPLETED'),
                "raw_responses": submission.raw_responses or {},
                "model_output": submission.model_output or {}
            }, status=status.HTTP_200_OK)
            
        except QuestionnaireSubmission.DoesNotExist:
            return Response({"error": "Submission not found or unauthorized."}, status=status.HTTP_404_NOT_FOUND)
        
                                     
from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
# In views.py
from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.permissions import AllowAny

from . import nlp_model  # Or your specific import setup

class ChatMessageViewSet(viewsets.ViewSet):
    permission_classes = [AllowAny]

    def list(self, request):
        """
        Handles GET requests. 
        Returns an empty list for anonymous, stateless sessions.
        """
        return Response([], status=status.HTTP_200_OK)

    def create(self, request, *args, **kwargs):
        user_message_text = request.data.get('message')

        if not user_message_text:
            return Response({"error": "Message body is required."}, status=status.HTTP_400_BAD_REQUEST)

        # Call your existing nlp_model function/method here
        bot_reply = nlp_model.predict(user_message_text) 

        return Response({
            "user_message": {
                "message": user_message_text,
                "sender": "user"
            },
            "bot_message": {
                "message": bot_reply,
                "sender": "bot"
            }
        }, status=status.HTTP_201_CREATED)
                                
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny
from rest_framework.parsers import MultiPartParser, FormParser
from .ml_router import route_and_explain
import os
from django.conf import settings

class AnonymousScanAnalyzeView(APIView):
    permission_classes = [AllowAny]
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        scan_file = request.FILES.get('scan_file')
        if not scan_file:
            return Response({"error": "No scan file uploaded."}, status=status.HTTP_400_BAD_REQUEST)

        temp_path = os.path.join(settings.MEDIA_ROOT, 'temp_scans', scan_file.name)
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)

        try:
            with open(temp_path, 'wb+') as destination:
                for chunk in scan_file.chunks():
                    destination.write(chunk)
            # Route scan through ML script (returns string text directly)
            result = route_and_explain(temp_path)
            return Response({
                "status": "success",
                "modality": "Pelvic Scan Analysis", # Default label
                "scan_note": str(result)            # Safely passes explanation string to frontend
            }, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": f"ML Analysis failed: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
   

             
# from rest_framework.views import APIView
# from rest_framework.permissions import AllowAny, IsAuthenticated
# from rest_framework.response import Response
# from rest_framework import status
# from .nlp_model import run_full_pipeline


# def format_questionnaire_input(raw_answers):
#     """
#     Helper function to parse structured or unstructured questionnaire responses
#     into full question-and-answer text statements for NLP analysis.
#     """
#     formatted_statements = []

#     if isinstance(raw_answers, dict):
#         for item in raw_answers.values():
#             # Matches structured format: { question_text, answer }
#             if isinstance(item, dict) and 'question_text' in item and 'answer' in item:
#                 q_text = item['question_text']
#                 ans = item['answer']
#                 if ans:
#                     ans_str = ", ".join(ans) if isinstance(ans, list) else str(ans)
#                     formatted_statements.append(f"{q_text}: {ans_str}")
#             # Fallback for plain key-value responses
#             elif isinstance(item, (str, list)) and item:
#                 ans_str = ", ".join(item) if isinstance(item, list) else str(item)
#                 formatted_statements.append(ans_str)

#         return "\n".join(formatted_statements)
    
#     return str(raw_answers)


# def extract_explanation_text(model_results):
#     """
#     Helper function to safely extract explanation/recommendation text from
#     various dictionary structures returned by the NLP/GenAI pipeline.
#     """
#     if isinstance(model_results, str):
#         return {"explanation": model_results}

#     if isinstance(model_results, dict):
#         # Inspect all possible key locations returned by GenAI
#         extracted_text = (
#             model_results.get('explanation') or 
#             model_results.get('recommendation') or 
#             model_results.get('clinical_summary') or 
#             model_results.get('summary') or 
#             model_results.get('details') or
#             (model_results.get('esi_result') if isinstance(model_results.get('esi_result'), str) else None) or
#             (model_results.get('esi_result', {}).get('explanation') if isinstance(model_results.get('esi_result'), dict) else None) or
#             (model_results.get('esi_result', {}).get('recommendation') if isinstance(model_results.get('esi_result'), dict) else None)
#         )
#         if extracted_text:
#             model_results['explanation'] = extracted_text

#     return model_results


# class AnonymousQuestionnaireView(APIView):
#     """
#     Public endpoint for anonymous questionnaire submissions.
#     No authentication required.
#     """
#     permission_classes = [AllowAny]
#     authentication_classes = [] 

#     def post(self, request):
#         try:
#             raw_answers = request.data.get('raw_responses') or request.data.get('answers') or {}

#             if not raw_answers:
#                 return Response(
#                     {"error": "No questionnaire responses provided."}, 
#                     status=status.HTTP_400_BAD_REQUEST
#                 )

#             text_input = format_questionnaire_input(raw_answers)

#             if not text_input.strip():
#                 return Response(
#                     {"error": "Questionnaire submitted with no completed answers."}, 
#                     status=status.HTTP_400_BAD_REQUEST
#                 )

#             # Pass full question & answer context to NLP pipeline
#             model_results = run_full_pipeline(user_input=text_input)
#             model_results = extract_explanation_text(model_results)

#             return Response({
#                 "status": "success",
#                 "model_output": model_results
#             }, status=status.HTTP_200_OK)

#         except Exception as e:
#             return Response(
#                 {"error": f"NLP evaluation failed: {str(e)}"}, 
#                 status=status.HTTP_500_INTERNAL_SERVER_ERROR
#             )


# class QuestionnaireProcessView(APIView):
#     """
#     Authenticated endpoint for logged-in users submitting questionnaire responses.
#     """
#     permission_classes = [IsAuthenticated]

#     def post(self, request):
#         try:
#             raw_answers = request.data.get('raw_responses') or request.data.get('answers') or {}

#             if not raw_answers:
#                 return Response(
#                     {"error": "No questionnaire responses provided."}, 
#                     status=status.HTTP_400_BAD_REQUEST
#                 )

#             text_input = format_questionnaire_input(raw_answers)

#             if not text_input.strip():
#                 return Response(
#                     {"error": "Questionnaire submitted with no completed answers."}, 
#                     status=status.HTTP_400_BAD_REQUEST
#                 )

#             model_results = run_full_pipeline(user_input=text_input)
#             model_results = extract_explanation_text(model_results)

#             return Response({
#                 "status": "success",
#                 "model_output": model_results
#             }, status=status.HTTP_200_OK)

#         except Exception as e:
#             return Response(
#                 {"error": f"NLP evaluation failed: {str(e)}"}, 
#                 status=status.HTTP_500_INTERNAL_SERVER_ERROR
#             )
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny
from .nlp_model import run_full_pipeline  # Ensure this points to your active model file

class AnonymousQuestionnaireView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        user_input = request.data.get('user_input', '').strip()
        
        if not user_input:
            return Response(
                {"error": "No symptom description provided."}, 
                status=status.HTTP_400_BAD_REQUEST
            )

        duration_code = request.data.get('duration_code', None)
        history_flags = request.data.get('history_flags', [])
        
class AnonymousQuestionnaireView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        # 1. Accept any standard key name sent by the React frontend
        user_input = (
            request.data.get('user_input') or 
            request.data.get('symptoms') or 
            request.data.get('text') or 
            request.data.get('raw_responses') or 
            ""
        ).strip()
        
        if not user_input:
            return Response(
                {"error": "No symptom description provided."}, 
                status=status.HTTP_400_BAD_REQUEST
            )

        duration_code = request.data.get('duration_code', None)
        history_flags = request.data.get('history_flags', [])
        history_text = request.data.get('history_text', "")

        try:
            # 2. Run the pipeline function
            pipeline_result = run_full_pipeline(
                user_input=user_input,
                history_text=history_text,
                duration_code=duration_code,
                history_flags=history_flags
            )

            # 3. Return payload structure expected by UI components
            return Response({
                "status": "success",
                "score": pipeline_result['score'],
                "raw_score": pipeline_result.get('esi_result', {}).get('score', pipeline_result['score']), 
                "tier": pipeline_result['tier'],
                "matched_symptoms": pipeline_result['matched_symptoms'],
                "extracted_phrases": pipeline_result['extracted_phrases'],
                "systems_affected": pipeline_result['systems_affected'],
                "explanation": pipeline_result['explanation'],
                "needs_followup": pipeline_result.get('needs_followup', []),
                "next_question": pipeline_result.get('next_question', "")
            }, status=status.HTTP_200_OK)

        except Exception as e:
            print(f"ANONYMOUS QUESTIONNAIRE ERROR: {str(e)}")
            traceback.print_exc()
            return Response(
                {"error": f"Questionnaire Analysis failed: {str(e)}"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )