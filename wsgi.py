"""WSGI entry point.

PythonAnywhere does not read the Procfile -- it imports `application` from the
WSGI file configured in the Web tab. Point that file at this one.
"""
from app import app as application
