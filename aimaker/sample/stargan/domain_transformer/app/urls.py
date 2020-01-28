from django.conf.urls import url
from . import views

urlpatterns = [
        url(r'^templates/', views.index_template, name='index_template'),
]
