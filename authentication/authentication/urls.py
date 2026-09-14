# from django.urls import path, include
# from rest_framework.routers import DefaultRouter

# from .views import (
#     PatientRegisterView, 
#     PatientLoginView, 
#     ForgotPasswordView, 
#     PatientProfileUpdateView, 
#     SymptomLogView, 
#     ScanRecordView,
#     QuestionnaireProcessView,
#     ChatMessageViewSet,
#     AnonymousQuestionnaireView, 
#     AnonymousScanAnalyzeView
# )

# urlpatterns = [
#     # Patient Authentication & Profile Endpoints
#     path('patients/register/', PatientRegisterView.as_view(), name='register'),
#     path('patients/login/', PatientLoginView.as_view(), name='login'),
#     path('patients/forgot-password/', ForgotPasswordView.as_view(), name='forgot-password'),
#     path('patients/profile/<int:patient_id>/', PatientProfileUpdateView.as_view(), name='profile-update'),
#     path('patients/logs/', SymptomLogView.as_view(), name='logs'),
#     path('patients/scans/', ScanRecordView.as_view(), name='scans'),

#     # Anonymous Scan Analysis Endpoints (Supports both /scans/ and /scan/ with or without /api/ prefix)
#     path('scans/anonymous-analyze/', AnonymousScanAnalyzeView.as_view(), name='anonymous-scans-analyze'),
#     path('scan/anonymous-analyze/', AnonymousScanAnalyzeView.as_view(), name='anonymous-scan-analyze'),
#     path('api/scans/anonymous-analyze/', AnonymousScanAnalyzeView.as_view(), name='anonymous-scans-analyze-alt'),
#     path('api/scan/anonymous-analyze/', AnonymousScanAnalyzeView.as_view(), name='anonymous-scan-analyze-alt'),
    
#     # Anonymous Questionnaire Analysis Endpoints (Supports both with or without /api/ prefix)
#     path('questionnaire/anonymous-analyze/', AnonymousQuestionnaireView.as_view(), name='anonymous-questionnaire-analyze'),
#     path('api/questionnaire/anonymous-analyze/', AnonymousQuestionnaireView.as_view(), name='anonymous-questionnaire-analyze-alt'),

#     # Chat Endpoint
#     path('patients/chat/', ChatMessageViewSet.as_view({
#         'get': 'list',
#         'post': 'create'
#     }), name='anonymous-chat'),  
# ]
from django.urls import path
from .views import (
    PatientRegisterView, 
    PatientLoginView, 
    ForgotPasswordView, 
    PatientProfileUpdateView, 
    SymptomLogView, 
    ScanRecordView,
    QuestionnaireProcessView,
    QuestionnaireStatusView,
    ChatMessageViewSet,
    AnonymousQuestionnaireView, 
    AnonymousScanAnalyzeView
)

urlpatterns = [
    # Authenticated Patient Auth & Management
    path('patients/register/', PatientRegisterView.as_view(), name='register'),
    path('patients/login/', PatientLoginView.as_view(), name='login'),
    path('patients/forgot-password/', ForgotPasswordView.as_view(), name='forgot-password'),
    path('patients/profile/<int:patient_id>/', PatientProfileUpdateView.as_view(), name='profile-update'),
    path('patients/logs/', SymptomLogView.as_view(), name='logs'),
    path('patients/scans/', ScanRecordView.as_view(), name='scans'),

    # Authenticated Questionnaire Endpoints (Matches your frontend call to POST /api/patients/questionnaire/)
    path('patients/questionnaire/', QuestionnaireProcessView.as_view(), name='patient-questionnaire-process'),
    path('patients/questionnaire/<int:submission_id>/', QuestionnaireStatusView.as_view(), name='patient-questionnaire-status'),
    path('api/patients/questionnaire/', QuestionnaireProcessView.as_view(), name='patient-questionnaire-process-alt'),
    path('api/patients/questionnaire/<int:submission_id>/', QuestionnaireStatusView.as_view(), name='patient-questionnaire-status-alt'),

    # Anonymous Questionnaire Endpoint
    path('questionnaire/anonymous-analyze/', AnonymousQuestionnaireView.as_view(), name='anonymous-questionnaire-analyze'),
    path('api/questionnaire/anonymous-analyze/', AnonymousQuestionnaireView.as_view(), name='anonymous-questionnaire-analyze-alt'),

    # Anonymous Scan Analysis Endpoints
    path('scans/anonymous-analyze/', AnonymousScanAnalyzeView.as_view(), name='anonymous-scans-analyze'),
    path('scan/anonymous-analyze/', AnonymousScanAnalyzeView.as_view(), name='anonymous-scan-analyze'),
    path('api/scans/anonymous-analyze/', AnonymousScanAnalyzeView.as_view(), name='anonymous-scans-analyze-alt'),
    path('api/scan/anonymous-analyze/', AnonymousScanAnalyzeView.as_view(), name='anonymous-scan-analyze-alt'),

    # Chat Endpoint
    path('patients/chat/', ChatMessageViewSet.as_view({
        'get': 'list',
        'post': 'create'
    }), name='anonymous-chat'),  
]