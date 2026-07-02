from django.urls import path
from .views import PredictionUploadView, PredictionHistoryView

urlpatterns = [
    path('upload/', PredictionUploadView.as_view(), name='prediction_upload'),
    path('history/', PredictionHistoryView.as_view(), name='prediction_history'),
]
