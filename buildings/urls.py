from django.urls import path
from . import views

urlpatterns = [
    path('', views.index_view, name='index'),
    path('predict/', views.predict, name='predict'),  # Add predict URL
    path('building/', views.building_view, name='building_view'),
]