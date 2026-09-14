from django.db import models
from django.contrib.auth.models import User

class PatientProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="profile")
    name = models.CharField(max_length=255, default="Anonymous")
    location = models.CharField(max_length=255)
    endometriosis_stage = models.CharField(max_length=100, default="Not Diagnosed / Unsure")
    diagnosis_date = models.CharField(max_length=100, default="N/A")
    
    def __str__(self):
        return self.name

class SymptomLog(models.Model):
    patient = models.ForeignKey(PatientProfile, on_delete=models.CASCADE, related_name="logs")
    date = models.DateField(auto_now_add=True)
    pain_level = models.IntegerField()
    bleeding = models.CharField(max_length=50)
    fatigue = models.CharField(max_length=50)
    notes = models.TextField(blank=True, null=True)
    questionnaire_summary = models.TextField(blank=True, null=True) 
    def __str__(self):
        return f"Symptom Log {self.id} for {self.patient.name}"
class ScanRecord(models.Model):
    patient = models.ForeignKey(PatientProfile, on_delete=models.CASCADE, related_name="scans")
    scan_file = models.FileField(upload_to="patient_scans/")
    uploaded_at = models.DateTimeField(auto_now_add=True)
    scan_note = models.TextField(blank=True, null=True) 
    def __str__(self):
        return f"Scan Record {self.id} for {self.patient.name}"

class QuestionnaireSubmission(models.Model):
    patient = models.ForeignKey(
    'PatientProfile', 
    on_delete=models.CASCADE, 
    related_name="questionnaire_submissions"
    )
    STATUS_CHOICES = (
    ('PENDING', 'Pending'),
    ('PROCESSING', 'Processing'),
    ('COMPLETED', 'Completed'),
    ('FAILED', 'Failed'),
    )

    submitted_at = models.DateTimeField(auto_now_add=True)
    
    # Stores raw responses from the frontend form (maps question IDs to answers)
    raw_responses = models.JSONField() 
    
    # Stores the final model prediction/output after processing raw_responses
    model_output = models.JSONField(blank=True, null=True) 
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING')
    def __str__(self):
        return f"Questionnaire {self.id} for {self.patient.name} on {self.submitted_at.strftime('%Y-%m-%d')}"
    
from django.db import models

class Doctor(models.Model):
    SPECIALTY_CHOICES = [
        ('gynecology', 'Gynecologist'),
        ('obstetrics', 'Obstetrician'),
        ('general', 'General Practitioner'),
        ('surgeon', 'Surgical Specialist'),
    ]

    name = models.CharField(max_length=255, verbose_name="Doctor's Name")
    location = models.CharField(max_length=255, help_text="city location")
    specialty = models.CharField(max_length=50, choices=SPECIALTY_CHOICES, default='gynecology')
    contact_number = models.CharField(max_length=20, blank=True, null=True)
    email = models.EmailField(blank=True, null=True)
    notes = models.TextField(blank=True, null=True, help_text="Internal administrative notes only")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Doctor"
        verbose_name_plural = "Doctors"
        ordering = ['name']

    def __str__(self):
        return f"Dr. {self.name} - {self.location}"