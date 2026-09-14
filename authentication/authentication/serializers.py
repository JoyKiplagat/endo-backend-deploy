from rest_framework import serializers
from django.contrib.auth.models import User
from .models import PatientProfile, SymptomLog, ScanRecord, QuestionnaireSubmission

class RegisterSerializer(serializers.ModelSerializer):
    location = serializers.CharField(write_only=True)
    endometriosis_stage = serializers.CharField(write_only=True, required=False)

    class Meta:
        model = User
        fields = ('email', 'password', 'location', 'endometriosis_stage')
        extra_kwargs = {'password': {'write_only': True}}

    def create(self, validated_data):
        user = User.objects.create_user(
            username=validated_data['email'], # Email doubles as username key
            email=validated_data['email'],
            password=validated_data['password']
        )
        PatientProfile.objects.create(
            user=user,
            location=validated_data['location'],
            endometriosis_stage=validated_data.get('endometriosis_stage', 'Not Diagnosed / Unsure')
        )
        return user

class SymptomLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = SymptomLog
        fields = '__all__'

from rest_framework import serializers
from .models import ScanRecord

class ScanRecordSerializer(serializers.ModelSerializer):
    # Generates a full media URL when request context is provided
    scan_file = serializers.FileField(read_only=True)

    class Meta:
        model = ScanRecord
        fields = ['id', 'patient', 'scan_file', 'scan_note', 'uploaded_at']
        read_only_fields = ['id', 'scan_note', 'uploaded_at']
        
from rest_framework import serializers
from .models import QuestionnaireSubmission, PatientProfile

class QuestionnaireSubmissionSerializer(serializers.ModelSerializer):
    # Ensures patient IDs can be passed and validated natively as integers
    patient = serializers.PrimaryKeyRelatedField(queryset=PatientProfile.objects.all())
    
    # Using JSONField explicitly forces DRF to accept any nested dictionary structures completely intact
    raw_responses = serializers.JSONField()
    model_output = serializers.JSONField(required=False, allow_null=True)

    class Meta:
        model = QuestionnaireSubmission
        fields = ['id', 'patient', 'submitted_at', 'raw_responses', 'model_output']

