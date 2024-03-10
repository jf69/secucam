from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from .models import *
from pathlib import Path

# main page logic
def index(request):
	# Select all rows from the Footage table
	footage_list = Footage.objects.all()
	# render the footage list in the index template
	return render(request, 'secucam_app/index.html', {'footage_list': footage_list})

# function for when a footage is selected from the list
def footage(request, pk):
	# Select the row that corresponds to the pk of the footage in the Footage table
	footage = Footage.objects.get(pk=pk)
	# Select all rows from the Footage table
	footage_list = Footage.objects.all()
	# render the footage and the footage list in the footage template
	return render(request, 'secucam_app/footage.html', {'footage': footage, 'footage_list': footage_list})

# function for when a selected footage is deleted
def delete(request, pk):
	# Select the row that corresponds to the pk of the footage in the Footage table
	footage = Footage.objects.get(pk=pk)
	# Create a path to the footage whose name corresponds to the pk of the footage in the Footage database
	p = Path(f'./secucam_app/footage/{footage}')
	# Delete the footage file from the directory
	p.unlink()
	# Also delete the row that refers to that footage from the Footage table
	Footage.objects.filter(pk=pk).delete()
	# Redirect to index.html
	return HttpResponseRedirect("/")