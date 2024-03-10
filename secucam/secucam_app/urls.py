from django.urls import path
from . import views

urlpatterns = [
	path('', views.index, name='index'), # path to index.html
	path('footage/<int:pk>', views.footage, name='footage'), # path taken when click on a footage from the footage list on the left-hand side of web page
	path('delete/<int:pk>', views.delete, name='delete'), # path taken when click on delete under selected footage
]