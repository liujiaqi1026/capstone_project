from django.shortcuts import render
from django.http import HttpResponse
from .forms import CSVUploadForm
from django.urls import reverse
import csv
# from .Opt2_1D import Tie, Railcar, multicar_optimize
from .optimize import optimize, Tie, Railcar


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
            tie_length = 0
            for index, value in enumerate(csv_data):
                if index == 0:
                    continue

                # if (tie_thickness != 0 and tie_thickness != int(value[1])) or (tie_width != 0 and tie_width != int(value[0])):
                #     return

                tie_width = float(value[1])
                tie_thickness = float(value[0])
                # tie_length = int(value[2])

                tie_list.append(Tie(length=float(value[2]), width=float(value[1]), thickness=float(value[0]),
                                    quantity=int(value[3]), weight_per_tie=float(text_box_6) *
                                                                           float(value[2]) * float(value[1])
                                                                           * float(value[0]) / 12))

            railcar_list = []
            for i in range(int(text_box_1)):
                railcar_list.append(Railcar(length=60, height=110, width=50, loading=10000000000000))

            for i in range(int(text_box_2)):
                railcar_list.append(Railcar(length=73, height=110, width=50, loading=10000000000000))

            railcar_list = sorted(railcar_list, key=lambda x: x.railcar_length, reverse=True)

            # todo: v 和 h 是否对应box2 和 box1
            result = optimize(railcar_list=railcar_list, tie_list=tie_list, bundle_v=int(dropdown_box_2),
                              bundle_h=int(dropdown_box_1),
                              weight_diff=0.1, tie_width=tie_width, tie_thickness=tie_thickness)

            temp = result[0]
            cars = temp['layout']

            # Pass the results to the template
            response = HttpResponse(content_type='text/csv')
            response['Content-Disposition'] = 'attachment; filename="my_file.csv"'
            writer = csv.writer(response)

            # create the result csv file.
            writer.writerow(["ORDER"])
            row_count = 0
            for indexOfCar, car in enumerate(cars):
                writer.writerow([str(railcar_list[indexOfCar].railcar_length) + ' ' + 'Centerbeam'])
                writer.writerow([])

                writer.writerow(['SIDE 1', '', '', '', '', '', '', 'SIDE 2', '', '', '', '', ''])

                for indexOfLayer, layer in enumerate(car[0][0]):
                    row_count += 1
                    writer.writerow(['ROW' + str(indexOfLayer + 1), '', '', '', '', '', '',
                                     'ROW' + str(indexOfLayer + 1), '', '', '', '', ''])
                    writer.writerow(['PCS', 'TH', 'W', 'L', 'Wt', 'QUANTITY', '',
                                     'PCS', 'TH', 'W', 'L', 'Wt', 'QUANTITY'])

                    for indexOfTie, leftSideTie in enumerate(layer):
                        rightSideTie = car[0][1][indexOfLayer][indexOfTie]

                        writer.writerow([str(int(dropdown_box_2) * int(dropdown_box_1)),
                                         tie_list[indexOfTie].thickness,
                                         tie_list[indexOfTie].width,
                                         tie_list[indexOfTie].length,
                                         float(dropdown_box_2) * float(dropdown_box_1) * float(tie_list[indexOfTie].weight_per_tie) * leftSideTie,
                                         leftSideTie,
                                         '',
                                         str(int(dropdown_box_2) * int(dropdown_box_1)),
                                         tie_list[indexOfTie].thickness,
                                         tie_list[indexOfTie].width,
                                         tie_list[indexOfTie].length,
                                         float(dropdown_box_2) * float(dropdown_box_1) * float(tie_list[indexOfTie].weight_per_tie) * rightSideTie,
                                         rightSideTie])

                    # wait for the total weight and total length
                    writer.writerow([])
                    writer.writerow([])

                writer.writerow([])

            writer.writerow(["small pieces: "])
            cars = result[1]["layout"]

            for indexOfCar, car in enumerate(cars):
                writer.writerow([str(railcar_list[indexOfCar].railcar_length) + ' ' + 'Centerbeam'])
                writer.writerow([])

                writer.writerow(['SIDE 1', '', '', '', '', '', '', 'SIDE 2', '', '', '', '', ''])

                for indexOfLayer, layer in enumerate(car[0][0]):
                    row_count += 1
                    writer.writerow(['ROW' + str(row_count), '', '', '', '', '', '',
                                     'ROW' + str(row_count), '', '', '', '', ''])
                    writer.writerow(['PCS', 'TH', 'W', 'L', 'Wt', 'QUANTITY', '',
                                     'PCS', 'TH', 'W', 'L', 'Wt', 'QUANTITY'])

                    for indexOfTie, leftSideTie in enumerate(layer):
                        rightSideTie = car[0][1][indexOfLayer][indexOfTie]

                        writer.writerow([str(int(dropdown_box_2)),
                                         tie_list[indexOfTie].thickness,
                                         tie_list[indexOfTie].width,
                                         tie_list[indexOfTie].length,
                                         float(dropdown_box_2) * float(tie_list[indexOfTie].weight_per_tie) * leftSideTie,
                                         leftSideTie,
                                         '',
                                         str(int(dropdown_box_2)),
                                         tie_list[indexOfTie].thickness,
                                         tie_list[indexOfTie].width,
                                         tie_list[indexOfTie].length,
                                         float(dropdown_box_2) * float(tie_list[indexOfTie].weight_per_tie) * rightSideTie,
                                         rightSideTie])

                    # wait for the total weight and total length
                    writer.writerow([])
                    writer.writerow([])

                writer.writerow([])


            csv_content = response.content.decode('utf-8')

            context = {
                'text_box_1': text_box_1,
                'text_box_2': text_box_2,
                'dropdown_box_1': dropdown_box_1,
                'dropdown_box_2': dropdown_box_2,
                'text_box_5': text_box_5,
                'text_box_6': text_box_6,
                'csv_data': csv_data,
                'max_loading': str(result[0]["load"] + result[1]["load"]),
                'layout_result': temp["layout"],
                'csv_content': csv_content,
            }
            return render(request, 'calculation-result.html', context)
    else:
        form = CSVUploadForm()
    return render(request, 'new-calculation.html', {'form': form})


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
