from rest_framework import serializers
from .models import PredictionRecord

class PredictionRecordSerializer(serializers.ModelSerializer):
    """
    Serializer for the PredictionRecord model.
    Handles serialization of prediction results and list items.
    """
    class Meta:
        model = PredictionRecord
        fields = ('id', 'image', 'predicted_class', 'confidence', 'created_at')
        read_only_fields = ('id', 'predicted_class', 'confidence', 'created_at')
