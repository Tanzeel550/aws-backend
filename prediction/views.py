from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from .models import PredictionRecord
from .serializers import PredictionRecordSerializer
from .model_loader import predict_digit

class PredictionUploadView(APIView):
    """
    API view to upload an image, run it through the PyTorch MNIST model,
    save the prediction history under the user, and return the result.
    """
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        serializer = PredictionRecordSerializer(data=request.data)
        if serializer.is_valid():
            uploaded_image = request.FILES.get('image')
            if not uploaded_image:
                return Response(
                    {"error": "No image file provided."}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            try:
                # Run inference using the preprocessed image
                predicted_class, confidence = predict_digit(uploaded_image)
            except Exception as e:
                return Response(
                    {"error": f"Model inference failed: {str(e)}"}, 
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
            
            # Save the record in the database
            record = serializer.save(
                user=request.user,
                predicted_class=predicted_class,
                confidence=confidence
            )
            
            # Serialize the complete saved record to return to client
            response_serializer = PredictionRecordSerializer(record, context={'request': request})
            return Response(response_serializer.data, status=status.HTTP_201_CREATED)
            
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class PredictionHistoryView(APIView):
    """
    API view to retrieve the classification history of the authenticated user.
    """
    def get(self, request, *args, **kwargs):
        predictions = PredictionRecord.objects.filter(user=request.user)
        serializer = PredictionRecordSerializer(predictions, many=True, context={'request': request})
        return Response(serializer.data, status=status.HTTP_200_OK)
