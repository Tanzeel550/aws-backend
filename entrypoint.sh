#!/bin/sh
set -e

# Generate database migrations for custom apps since migration folders are not committed
echo "Creating database migrations..."
python manage.py makemigrations users prediction

# Apply migrations
echo "Applying database migrations..."
python manage.py migrate --noinput

# Start the Django application using exec to handle signals properly
echo "Starting Django development server..."
exec python manage.py runserver 0.0.0.0:8000
