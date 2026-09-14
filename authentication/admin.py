from django.contrib import admin
from django.utils.html import format_html
from .models import PatientProfile, SymptomLog, ScanRecord, QuestionnaireSubmission
from django.utils.safestring import mark_safe
import json

admin.site.site_header = "Clinical Diagnostic Portal"
admin.site.site_title = "Endo Diagnostics Admin"
admin.site.index_title = "Patient Record & ML Pipeline Management"

@admin.register(PatientProfile)
class PatientProfileAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'get_email', 'location', 'endometriosis_stage', 'diagnosis_date')
    search_fields = ('name', 'user__email', 'location')
    list_filter = ('endometriosis_stage', 'location')
    ordering = ('id',)
    list_select_related = ('user',)  # Prevents N+1 query overhead for user.email

    def get_email(self, obj):
        return obj.user.email
    get_email.short_description = 'Account Email'

@admin.register(SymptomLog)
class SymptomLogAdmin(admin.ModelAdmin):
    list_display = ('id', 'patient', 'date', 'pain_level', 'bleeding', 'fatigue', 'questionnaire_summary')
    list_filter = ('pain_level', 'bleeding', 'fatigue', 'date')
    search_fields = ('patient__name', 'notes', 'questionnaire_summary')
    list_select_related = ('patient',)

@admin.register(ScanRecord)
class ScanRecordAdmin(admin.ModelAdmin):
    list_display = ('id', 'patient', 'uploaded_at', 'view_scan_link', 'scan_note')
    search_fields = ('patient__name', 'scan_note')
    list_select_related = ('patient',)

    def view_scan_link(self, obj):
        if obj.scan_file:
            return format_html('<a href="{}" target="_blank" style="font-weight: bold; color: #a41c3c;">View Scan Document ↗</a>', obj.scan_file.url)
        return "No File Attached"
    view_scan_link.short_description = 'Scan Document'
    
@admin.register(QuestionnaireSubmission)
class QuestionnaireSubmissionAdmin(admin.ModelAdmin):
    # Fields displayed on the main admin list panel view
    list_display = ('id', 'patient', 'submitted_at', 'model_output')
    list_filter = ('submitted_at', 'patient')
    
    # Moves editing fields into clear, readable, read-only blocks
    readonly_fields = ('submitted_at', 'pretty_raw_responses', 'pretty_model_output')
    exclude = ('raw_responses', 'model_output') # Hide the default ugly text boxes

    def pretty_raw_responses(self, instance):
        """Converts raw data maps into styled HTML code snippets inside Admin"""
        response_json = json.dumps(instance.raw_responses, indent=4)
        return mark_safe(f'<pre style="background: #f4f4f4; padding: 15px; border-radius: 6px;">{response_json}</pre>')
    
    def pretty_model_output(self, instance):
        """Formats ML engine output keys clearly"""
        if not instance.model_output:
            return "No ML parameters recorded."
        output_json = json.dumps(instance.model_output, indent=4)
        return mark_safe(f'<pre style="background: #eef9ff; padding: 15px; border-radius: 6px; border-left: 4px solid #3182ce;">{output_json}</pre>')

    # Change field titles within the panel view
    pretty_raw_responses.short_description = "Complete Survey Answers (All Captured Fields)"
    pretty_model_output.short_description = "Engine Evaluation & Risks Matrix Data"

from django.contrib import admin
from .models import Doctor

@admin.register(Doctor)
class DoctorAdmin(admin.ModelAdmin):
    # Columns shown in the list table
    list_display = ('name', 'location', 'specialty', 'contact_number', 'created_at')
    
    # Filter sidebar options
    list_filter = ('specialty', 'location')
    
    # Search bar targets
    search_fields = ('name', 'location', 'email', 'notes')
    
    # Organize form layout when adding/editing a doctor
    fieldsets = (
        ("Doctor Information", {
            'fields': ('name', 'specialty', 'location')
        }),
        ("Contact Information", {
            'fields': ('contact_number', 'email')
        }),
        ("Internal Notes", {
            'fields': ('notes',),
            'classes': ('collapse',),  # Collapsible section in admin
        }),
    )