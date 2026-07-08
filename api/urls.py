from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.http import JsonResponse

def health_check(request):
    return JsonResponse({"status": "healthy"}, status=200)

urlpatterns = [
    path('', health_check, name='root_health_check'),
    path('admin/', admin.site.urls),
    path('health/', health_check, name='health_check'),
    path('api/users/', include('users.urls')),
    path('api/predict/', include('prediction.urls')),
]

# Serve media files in development/testing mode
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
