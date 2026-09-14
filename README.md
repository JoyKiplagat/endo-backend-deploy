# endo_backend

A Django REST API backend for an endometriosis patient management platform. It handles patient authentication, symptom tracking, medical scan uploads, and AI-powered questionnaire processing — designed to serve a React/Vite frontend.

---

## Table of Contents

- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Data Models](#data-models)
- [API Endpoints](#api-endpoints)
- [Setup & Installation](#setup--installation)
- [Environment Configuration](#environment-configuration)
- [Running the Server](#running-the-server)
- [Notes for Production](#notes-for-production)

---

## Overview

`endo_backend` provides the server-side logic for a health app built specifically for endometriosis patients. Core features include:

- **Patient registration & login** using email/password with JWT authentication
- **Symptom logging** — tracking pain level, bleeding, fatigue, and notes over time
- **Medical scan uploads** — patients can upload MRI/ultrasound images, which are stored and annotated with an AI-generated analysis note
- **Questionnaire processing** — a structured clinical questionnaire that maps responses to symptom log entries with AI-generated risk summaries
- **Profile management** — patients can view and update their profile details (name, location, endometriosis stage, diagnosis date)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Framework | Django 6.1 |
| API | Django REST Framework |
| Authentication | JWT via `djangorestframework-simplejwt` |
| Database | PostgreSQL |
| CORS | `django-cors-headers` |
| File Storage | Django media files (local filesystem) |
| Runtime | Python 3.12 |

---

## Project Structure

```
endo_backend/
├── backend/                  # Django project config
│   ├── settings.py           # App settings, DB, JWT, CORS config
│   ├── urls.py               # Root URL routing
│   ├── asgi.py
│   └── wsgi.py
├── authentication/           # Main app — all patient logic lives here
│   ├── models.py             # PatientProfile, SymptomLog, ScanRecord
│   ├── serializers.py        # DRF serializers for all models
│   ├── views.py              # All API view classes
│   ├── urls.py               # App-level URL patterns
│   ├── admin.py              # Django admin registrations
│   └── migrations/           # Database migration history
├── media/
│   └── patient_scans/        # Uploaded scan files stored here
└── manage.py
```

---

## Data Models

### `PatientProfile`
Extends Django's built-in `User` model via a one-to-one relationship.

| Field | Type | Description |
|---|---|---|
| `user` | OneToOneField | Linked Django auth user |
| `name` | CharField | Patient's display name |
| `location` | CharField | Patient's location |
| `endometriosis_stage` | CharField | Diagnosis stage (e.g. Stage I–IV or "Not Diagnosed / Unsure") |
| `diagnosis_date` | CharField | Date of diagnosis or "N/A" |

### `SymptomLog`
A daily log entry submitted by the patient.

| Field | Type | Description |
|---|---|---|
| `patient` | ForeignKey | Linked `PatientProfile` |
| `date` | DateField | Auto-set on creation |
| `pain_level` | IntegerField | Pain rating (e.g. 1–10) |
| `bleeding` | CharField | Bleeding status |
| `fatigue` | CharField | Fatigue level |
| `notes` | TextField | Optional free-text notes |
| `questionnaire_summary` | TextField | AI-generated risk summary from the questionnaire |

### `ScanRecord`
A medical image uploaded by the patient.

| Field | Type | Description |
|---|---|---|
| `patient` | ForeignKey | Linked `PatientProfile` |
| `scan_file` | FileField | Uploaded file stored in `media/patient_scans/` |
| `uploaded_at` | DateTimeField | Auto-set on upload |
| `notes` | TextField | AI-generated analysis note |

---

## API Endpoints

All routes are prefixed with `/api/`.

### Authentication

| Method | Endpoint | Auth Required | Description |
|---|---|---|---|
| `POST` | `/api/patients/register/` | No | Register a new patient account |
| `POST` | `/api/patients/login/` | No | Login and receive JWT tokens |
| `POST` | `/api/patients/forgot-password/` | No | Reset password by email |

**Register** — accepts `multipart/form-data`:
```json
{
  "email": "patient@example.com",
  "password": "securepassword",
  "name": "Jane Doe",
  "location": "Nairobi, Kenya",
  "endometriosis_stage": "Stage II",
  "scan_file": "<optional file>"
}
```

**Login** — returns JWT `access` and `refresh` tokens along with patient profile fields.

---

### Patient Profile

| Method | Endpoint | Auth Required | Description |
|---|---|---|---|
| `GET` | `/api/patients/profile/<patient_id>/` | Yes | Fetch profile details |
| `PUT` | `/api/patients/profile/<patient_id>/` | Yes | Update profile details |

---

### Symptom Logs

| Method | Endpoint | Auth Required | Description |
|---|---|---|---|
| `GET` | `/api/patients/logs/?patient=<id>` | Yes | List all logs for a patient |
| `POST` | `/api/patients/logs/` | Yes | Create a new symptom log |

---

### Medical Scans

| Method | Endpoint | Auth Required | Description |
|---|---|---|---|
| `GET` | `/api/patients/scans/?patient=<id>` | Yes | List all scans for a patient |
| `POST` | `/api/patients/scans/` | Yes | Upload a new scan (triggers AI analysis) |

Upload accepts `multipart/form-data` with a `scan_file` field. After saving, the backend runs an AI analysis step and stores the result in the `notes` field of the `ScanRecord`.

---

### Questionnaire

| Method | Endpoint | Auth Required | Description |
|---|---|---|---|
| `POST` | `/api/patients/questionnaire/` | Yes | Submit questionnaire answers |

**Request body:**
```json
{
  "patient_id": 1,
  "date": "2026-09-03",
  "answers": {
    "pain_level": 7,
    "bleeding": "Heavy",
    "fatigue": "Severe",
    "pain_frequency": "Daily"
  }
}
```

Creates or updates a `SymptomLog` for the given date and stores an AI-generated risk summary in `questionnaire_summary`.

---

## Setup & Installation

### Prerequisites

- Python 3.12+
- PostgreSQL running locally
- `pip`

### Steps

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd endo_backend
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install django djangorestframework djangorestframework-simplejwt django-cors-headers psycopg2-binary
   ```

4. **Set up the PostgreSQL database**
   ```sql
   CREATE DATABASE endo_db;
   CREATE USER endo_user WITH PASSWORD 'your_password';
   GRANT ALL PRIVILEGES ON DATABASE endo_db TO endo_user;
   ```

5. **Apply migrations**
   ```bash
   python manage.py migrate
   ```

6. **Create a superuser** (for Django admin access)
   ```bash
   python manage.py createsuperuser
   ```

---

## Environment Configuration

Before running the server, update the following values in `backend/settings.py`:

| Setting | Location | Action Required |
|---|---|---|
| `SECRET_KEY` | `settings.py` | Replace the insecure default with a strong random key |
| `DATABASES['PASSWORD']` | `settings.py` | Set your actual PostgreSQL password |
| `SIMPLE_JWT['SIGNING_KEY']` | `settings.py` | Set a secure signing key separate from `SECRET_KEY` |
| `ALLOWED_HOSTS` | `settings.py` | Add your server's domain/IP for production |
| `CORS_ALLOWED_ORIGINS` | `settings.py` | Add your frontend's production URL |
| `DEBUG` | `settings.py` | Set to `False` in production |

> **Security note:** The current `settings.py` contains hardcoded credentials. These must be moved to environment variables (e.g. via `python-decouple` or `django-environ`) before deploying to any environment beyond local development.

---

## Running the Server

```bash
python manage.py runserver
```

The API will be available at `http://localhost:8000/api/`.

The Django admin panel is at `http://localhost:8000/admin/`.

Media files (uploaded scans) are served at `http://localhost:8000/media/` in development mode.

---

## Notes for Production

- Set `DEBUG = False` and configure `ALLOWED_HOSTS` properly.
- Move all secrets and credentials to environment variables — never commit them to version control.
- Configure a proper media file storage solution (e.g. AWS S3) instead of local filesystem storage.
- Use a production WSGI/ASGI server such as Gunicorn or Uvicorn behind Nginx.
- Enable HTTPS and update `CORS_ALLOWED_ORIGINS` and `CSRF_TRUSTED_ORIGINS` to HTTPS URLs.
- The `ForgotPasswordView` currently resets passwords without any email verification step — add a token-based email flow before going live.
- The AI scan analysis and questionnaire risk evaluation in `views.py` are currently stubbed with placeholder strings. Integrate your actual vision/inference model at the marked comment blocks.
