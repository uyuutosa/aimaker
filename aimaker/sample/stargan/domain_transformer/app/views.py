from django.shortcuts import render
from django.http.response import HttpResponse

# Create your views here.
def index_template(request):
    return render(request, 'index.html')
