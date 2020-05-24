from django.urls import path
from . import views

urlpatterns = [
    path('', views.rec_face, name='facerec')
]
