from django.shortcuts import render
from django.http import HttpResponse
from .forms import CSVUploadForm
import csv


# Create your views here.

def index_action(request):
    if request.method == "GET":
        context = {"status": "Welcome to Kopper Capstone Project."}
        return render(request, "index.html", context)

    return


def dashboard_admin_action(request):
    if request.method == "GET":
        context = {"status": "Welcome to Kopper Capstone Project."}
        return render(request, "dashboard-admin.html", context)

    return


def new_calculation_action(request):
    if request.method == 'POST':
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Get the uploaded CSV file and read it
            csv_file = form.cleaned_data['csv_file']
            
            # Process CSV file
            csv_data = []
            decoded_file = csv_file.read().decode('utf-8')
            csv_reader = csv.reader(decoded_file.splitlines())
            for row in csv_reader:
                csv_data.append(row)
            
            # Get the data from the text fields
            text_box_1 = form.cleaned_data['text_box_1']
            text_box_2 = form.cleaned_data['text_box_2']
            dropdown_box_1 = form.cleaned_data['dropdown_box_1']
            dropdown_box_2 = form.cleaned_data['dropdown_box_2']
            text_box_5 = form.cleaned_data['text_box_5']
            text_box_6 = form.cleaned_data['text_box_6']
            
            # Do some calculations with the data
            # ...
            
            # Pass the results to the template
            context = {
                'text_box_1': text_box_1,
                'text_box_2': text_box_2,
                'dropdown_box_1': dropdown_box_1,
                'dropdown_box_2': dropdown_box_2,
                'text_box_5': text_box_5,
                'text_box_6': text_box_6,
                'csv_data': csv_data,
            }
            return render(request, 'calculation-result.html', context)
    else:
        form = CSVUploadForm()
    return render(request, 'new-calculation.html', {'form': form})
    # if request.method == "GET":
    #     context = {"status": "Welcome to Kopper Capstone Project."}
    #     return render(request, "new-calculation.html", context)

    # return


def add_new_ties_action(request):
    if request.method == "GET":
        context = {"status": "Welcome to Kopper Capstone Project."}
        return render(request, "add-new-ties.html", context)

    return


def calculation_result_action(request):
    if request.method == "GET":
        context = {"status": "Welcome to Kopper Capstone Project."}
        return render(request, "calculation-result.html", context)
    else:
        # todo: process with the form data.
        context = {"status": "Welcome to Kopper Capstone Project."}
        return render(request, "calculation-result.html", context)


