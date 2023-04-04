from django.shortcuts import render
from django.http import HttpResponse
from .forms import CSVUploadForm
import csv
# from .Opt2_1D import Tie, Railcar, multicar_optimize
from optimize import optimize, Tie, Railcar

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
            tie_list = []
            tie_width = 0
            tie_thickness = 0
            for index, value in enumerate(csv_data):
                print(value)
                if index == 0:
                    continue

                # if (tie_thickness != 0 and tie_thickness != int(value[1])) or (tie_width != 0 and tie_width != int(value[0])):
                #     return

                tie_width = int(value[0])
                tie_thickness = int(value[1])

                tie_list.append(Tie(length=float(value[2]), width=float(value[1]), thickness=float(value[0]),
                                    quantity=int(value[3]), weight_per_tie=int(text_box_6)))

            railcar_list = []
            for i in range(int(text_box_1)):
                railcar_list.append(Railcar(length=61, height=124, width=50, loading=100000))

            for i in range(int(text_box_2)):
                railcar_list.append(Railcar(length=73, height=124, width=50, loading=100000))



            # todo: v 和 h 是否对应box2 和 box1
            result = optimize(railcar_list=railcar_list, tie_list=tie_list, bundle_v=int(dropdown_box_2), bundle_h=int(dropdown_box_1),
                                       weight_diff=0.1, tie_width=tie_width, tie_thickness=tie_thickness)
            print(result)


            
            # Pass the results to the template
            context = {
                'text_box_1': text_box_1,
                'text_box_2': text_box_2,
                'dropdown_box_1': dropdown_box_1,
                'dropdown_box_2': dropdown_box_2,
                'text_box_5': text_box_5,
                'text_box_6': text_box_6,
                'csv_data': csv_data,
                'max_loading': str(result[0]),
                'layout_result': result[1],
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


