from django.urls import path
from . import views

urlpatterns = [
    path('', views.index_view, name='index'),
    # path('upload/', views.upload_image, name='upload_image'),
    path('building/', views.building_view, name='building_view'),
]
