from django.shortcuts import render
from django.http import HttpResponse
from .forms import CSVUploadForm
from django.urls import reverse
import csv
# from .Opt2_1D import Tie, Railcar, multicar_optimize
from .optimize import optimize, Tie, Railcar
from datetime import datetime
import copy
import time


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
    start_time = time.time()
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
            radio = form.cleaned_data['radio']
            is_iterate = True if radio == "maximum" else False


            # Do some calculations with the data
            # ...
            tie_list = []
            temp_list = []
            tie_width = 0
            tie_thickness = 0
            tie_length = 0
            for index, value in enumerate(csv_data):
                if index == 0:
                    continue

                if int(value[3]) == 0:
                    continue

                tie_width = float(value[1])
                tie_thickness = float(value[0])
                # tie_length = int(value[2])

                tie_list.append(Tie(length=float(value[2]), width=float(value[1]), thickness=float(value[0]),
                                    quantity=int(value[3]), weight_per_tie=float(text_box_6) *
                                                                           float(value[2]) * float(value[1])
                                                                           * float(value[0]) / 12))
                temp_list = copy.deepcopy(tie_list)

            railcar_list = []
            for i in range(int(text_box_1)):
                railcar_list.append(Railcar(length=60, height=124, width=50, loading=10000000000000))

            for i in range(int(text_box_2)):
                railcar_list.append(Railcar(length=73, height=124, width=50, loading=10000000000000))

            railcar_list = sorted(railcar_list, key=lambda x: x.railcar_length, reverse=True)

            # todo: v 和 h 是否对应box2 和 box1
            result = optimize(railcar_list=railcar_list, tie_list=tie_list, bundle_v=int(dropdown_box_2),
                              bundle_h=int(dropdown_box_1),
                              weight_diff=0.01, tie_width=tie_width, tie_thickness=tie_thickness,isIterate=is_iterate)

            end_time = time.time()
            tie_list = temp_list

            temp = result[0]
            cars = temp['layout']

            # Pass the results to the template
            response = HttpResponse(content_type='text/csv')
            response['Content-Disposition'] = 'attachment; filename="my_file.csv"'
            writer = csv.writer(response)

            # create the result csv file.
            writer.writerow(["ORDER"])
            writer.writerow([str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))])
            writer.writerow([("weight factor: " + str(text_box_6))])
            writer.writerow([("number of 60 Centerbeam: " + str(text_box_1))])
            writer.writerow([("number of 73 Centerbeam: " + str(text_box_2))])
            writer.writerow([("note: " + str(text_box_5))])
            writer.writerow(["total loading: " + str(result[0]["load"])])
            writer.writerow(["running time: " + str(end_time - start_time)])
            writer.writerow([])

            for indexOfCar, car in enumerate(cars):
                writer.writerow([str(railcar_list[indexOfCar].railcar_length) + ' ' + 'Centerbeam No.' + str(indexOfCar + 1)])
                writer.writerow([])

                writer.writerow(['SIDE 1', '', '', '', '', '', '', 'SIDE 2', '', '', '', '', ''])

                total_left_side_weight = 0
                total_right_side_weight = 0

                for indexOfLayer, layer in enumerate(car[0][0]):
                    writer.writerow(['ROW' + str(indexOfLayer + 1), '', '', '', '', '', '',
                                     'ROW' + str(indexOfLayer + 1), '', '', '', '', ''])
                    writer.writerow(['PCS', 'TH', 'W', 'L', 'Wt', 'QTY of BUN', '',
                                     'PCS', 'TH', 'W', 'L', 'Wt', 'QTY of BUN'])

                    total_left_length = 0
                    total_right_length = 0
                    total_left_weight = 0
                    total_right_weight = 0

                    left_pcs = layer['pcs']

                    for indexOfTie, leftSideTie in enumerate(layer['layer']):
                        rightSideTie = car[0][1][indexOfLayer]['layer'][indexOfTie]
                        right_pcs = car[0][1][indexOfLayer]['pcs']

                        current_tie = tie_list[indexOfTie]
                        current_tie.quantity = current_tie.quantity - left_pcs * leftSideTie - right_pcs * rightSideTie

                        total_left_length += tie_list[indexOfTie].length * leftSideTie
                        total_right_length += tie_list[indexOfTie].length * rightSideTie
                        total_left_weight += left_pcs * float(tie_list[indexOfTie].weight_per_tie) * leftSideTie
                        total_right_weight += right_pcs * float(tie_list[indexOfTie].weight_per_tie) * rightSideTie

                        writer.writerow([
                                        # str(int(dropdown_box_2) * int(dropdown_box_1)),
                                        left_pcs,
                                         tie_list[indexOfTie].thickness,
                                         tie_list[indexOfTie].width,
                                         tie_list[indexOfTie].length,
                                         "-" if leftSideTie == 0 else left_pcs * float(tie_list[indexOfTie].weight_per_tie) * leftSideTie,
                                         "-" if leftSideTie == 0 else leftSideTie,
                                         '',
                                         # str(int(dropdown_box_2) * int(dropdown_box_1)),
                                        right_pcs,
                                         tie_list[indexOfTie].thickness,
                                         tie_list[indexOfTie].width,
                                         tie_list[indexOfTie].length,
                                         "-" if rightSideTie == 0 else right_pcs * float(tie_list[indexOfTie].weight_per_tie) * rightSideTie,
                                         "-" if rightSideTie == 0 else rightSideTie])

                    total_left_side_weight += total_left_weight
                    total_right_side_weight += total_right_weight

                    writer.writerow(['', '', '', 'Tot.L', 'Tot.W', '', '',
                                     '', '', '', 'Tot.L', 'Tot.W', ''])
                    writer.writerow(['', '', '', total_left_length, total_left_weight, '', '',
                                     '', '', '', total_right_length, total_right_weight, ''])

                    # wait for the total weight and total length
                    if indexOfLayer < len(car[0][0]) - 1:
                        writer.writerow([])
                        writer.writerow([])

                writer.writerow(['', '', '', '', 'SIDE 1 Tot.W', '', '',
                                 '', '', '', '', 'SIDE 2 Tot.W', ''])
                writer.writerow(['', '', '', '', total_left_side_weight, '', '',
                                 '', '', '', '', total_right_side_weight, ''])

                # output the remain ties.
                writer.writerow(["remain ties: "])
                writer.writerow(["Thickness", "Width", "Length", "Quantity"])

                remaining_weight = 0
                truck_needed = 0
                for tie in tie_list:
                    writer.writerow([str(tie.thickness), str(tie.width), str(tie.length), str(tie.quantity)])
                    remaining_weight += tie.weight_per_tie * tie.quantity
                truck_needed = remaining_weight / 45000

                writer.writerow(["remaining total weight: " + str(remaining_weight)])
                writer.writerow(["truck needed: " + str(truck_needed)])
                writer.writerow([])
                writer.writerow([])


            # writer.writerow(["SMALL PIECES: "])
            # cars = result[1]["layout"]
            #
            # for indexOfCar, car in enumerate(cars):
            #     writer.writerow([str(railcar_list[indexOfCar].railcar_length) + ' ' + 'Centerbeam No.' + str(indexOfCar + 1)])
            #     writer.writerow([])
            #
            #     writer.writerow(['SIDE 1', '', '', '', '', '', '', 'SIDE 2', '', '', '', '', ''])
            #
            #     row_count = row_count_map[str(indexOfCar)]
            #     total_left_side_weight = total_left_side_weight_map[str(indexOfCar)]
            #     total_right_side_weight = total_right_side_weight_map[str(indexOfCar)]
            #
            #     for indexOfLayer, layer in enumerate(car[0][0]):
            #         row_count += 1
            #         writer.writerow(['ROW' + str(row_count), '', '', '', '', '', '',
            #                          'ROW' + str(row_count), '', '', '', '', ''])
            #         writer.writerow(['PCS', 'TH', 'W', 'L', 'Wt', 'QTY of BUN', '',
            #                          'PCS', 'TH', 'W', 'L', 'Wt', 'QTY of BUN'])
            #
            #         total_left_length = 0
            #         total_right_length = 0
            #         total_left_weight = 0
            #         total_right_weight = 0
            #
            #         for indexOfTie, leftSideTie in enumerate(layer):
            #             rightSideTie = car[0][1][indexOfLayer][indexOfTie]
            #
            #             total_left_length += tie_list[indexOfTie].length * leftSideTie
            #             total_right_length += tie_list[indexOfTie].length * rightSideTie
            #             total_left_weight += float(dropdown_box_2) * float(tie_list[indexOfTie].weight_per_tie) * leftSideTie
            #             total_right_weight += float(dropdown_box_2) * float(tie_list[indexOfTie].weight_per_tie) * rightSideTie
            #
            #             writer.writerow([str(int(dropdown_box_2)),
            #                              tie_list[indexOfTie].thickness,
            #                              tie_list[indexOfTie].width,
            #                              tie_list[indexOfTie].length,
            #                              "-" if leftSideTie == 0 else float(dropdown_box_2) * float(tie_list[indexOfTie].weight_per_tie) * leftSideTie,
            #                              "-" if leftSideTie == 0 else leftSideTie,
            #                              '',
            #                              str(int(dropdown_box_2)),
            #                              tie_list[indexOfTie].thickness,
            #                              tie_list[indexOfTie].width,
            #                              tie_list[indexOfTie].length,
            #                              "-" if rightSideTie == 0 else float(dropdown_box_2) * float(tie_list[indexOfTie].weight_per_tie) * rightSideTie,
            #                              "-" if rightSideTie == 0 else rightSideTie])
            #
            #         total_left_side_weight += total_left_weight
            #         total_right_side_weight += total_right_weight
            #
            #         writer.writerow(['', '', '', 'Tot.L', 'Tot.W', '', '',
            #                          '', '', '', 'Tot.L', 'Tot.W', ''])
            #         writer.writerow(['', '', '', total_left_length, total_left_weight, '', '',
            #                          '', '', '', total_right_length, total_right_weight, ''])
            #
            #         if indexOfLayer < len(car[0][0]) - 1:
            #             writer.writerow([])
            #             writer.writerow([])
            #
            #     row_count_map[str(indexOfCar)] = row_count
            #     total_left_side_weight_map[str(indexOfCar)] = total_left_side_weight
            #     total_right_side_weight_map[str(indexOfCar)] = total_right_side_weight
            #
                # writer.writerow(['', '', '', '', 'SIDE 1 Tot.W', '', '',
                #                  '', '', '', '', 'SIDE 2 Tot.W', ''])
                # writer.writerow(['', '', '', '', total_left_side_weight, '', '',
                #                  '', '', '', '', total_right_side_weight, ''])
            #     writer.writerow([])


            csv_content = response.content.decode('utf-8')

            context = {
                'text_box_1': text_box_1,
                'text_box_2': text_box_2,
                'dropdown_box_1': dropdown_box_1,
                'dropdown_box_2': dropdown_box_2,
                'text_box_5': text_box_5,
                'text_box_6': text_box_6,
                'csv_data': csv_data,
                'max_loading': str(result[0]["load"]),
                'layout_result': temp["layout"],
                'csv_content': csv_content,
                "time": str(end_time - start_time),
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
