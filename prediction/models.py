from django.db import models
from django.conf import settings

class PredictionRecord(models.Model):
    """
    Model storing historical digit classifications made by users.
    """
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, 
        on_delete=models.CASCADE, 
        related_name='predictions'
    )
    image = models.ImageField(upload_to='mnist_uploads/')
    predicted_class = models.IntegerField()
    confidence = models.FloatField() # Store as percentage value, e.g. 98.54
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"User: {self.user.email} - Class: {self.predicted_class} ({self.confidence:.2f}%)"
