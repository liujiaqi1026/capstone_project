import sys
from typing import List
from gekko import GEKKO
import math
import pandas as pd
import numpy as np
import copy
import math


## MINLP: Mixed Integer Nonlinear Program
## APOPT : Advanced Process Optimization

'''
    Some Object definition.
'''
'''
    Tie is the fundamental element, has some attributes
'''
class Tie:
    width: float
    thickness: float
    length: float
    quantity: int
    weight_per_tie: float

    def __init__(self, width: float, thickness: float, length: float, quantity: int, weight_per_tie: float):
        self.width = width
        self.thickness = thickness
        self.length = length
        self.quantity = quantity
        self.weight_per_tie = weight_per_tie

    def __str__(self):
        return "length: " + str(self.length) + " width: " + str(self.width) \
               + " thickness: " + str(self.thickness) + " quantity: " + str(self.quantity) + " weight_per_tie: "+str(self.weight_per_tie)


'''
    tie_list: a list of tie Object, each tie response to a type of bundle, this bundle is made of same tie type.
              usually the list contains all the tie types.
    bundle_nums: the counterpart number of bundles

    For example:
    The task need to load four types
    the tie_list is [TieA, TieB, TieC, TieD]
    the bundle_nums is [1, 2, 3, 0]
    This means the layer contains 1 bundle of A, 2 bundles of B, 3 bundles of C, not for D
'''
class Layer:
    layer_height: float
    tie_list: List[Tie] = []
    bundle_nums: List[int]

    def init_tie(self, tie_list_: List[Tie]):
        self.tie_list = tie_list_.copy()
        self.bundle_nums = [0 for _ in range(len(tie_list_))]

'''
    Side contains a list of layer.
'''
class Side:
    side_width: float
    layers: List[Layer]

    def init(self, tie_list: List[Tie], layer_num: int):
        self.layers = [Layer() for _ in range(layer_num)]
        for layer in self.layers:
            layer.init_tie(tie_list)

'''
    Railcar has some basic attributes including length, width, length
    And it's layout is represented by 2 side.
'''
class Railcar:
    railcar_length: float
    railcar_height: float
    railcar_width: float
    railcar_loading: float
    side_a: Side = Side()
    side_b: Side = Side()

    def __init__(self, length: float, height: float, width: float, loading: float):
        self.railcar_length = length
        self.railcar_height = height
        self.railcar_width = width
        self.railcar_loading = loading

    def init(self, tie_list: List[Tie], layer_num: int):
        self.side_a.init(tie_list, layer_num)
        self.side_b.init(tie_list, layer_num)

    def __str__(self):
        return "length: " + str(self.railcar_length)+" height: "+ str(self.railcar_height)+" width: "+ str(self.railcar_width)+" loading: "+ str(self.railcar_loading)


'''
    optimize() is the exposed API to view.py
    Depends on isIterater parameter, it uses different fix_optimized() method.
    
    railcar_list: a list of railcar. Each is one specific railcar (same type of car is redeemed different)
    tie_list: a list of tie. Each is one type of tie. The quantity is one attribute of tie
    bundle_v: bundle vertical tie num. That represents how many stacks can make one bundle.
    bundle_h: bundle horizontal tie num. That represents a stack is composed of how many ties are placed side by side
        For example, the bundle is consisted of 5*4=20 ties, that means bundle_h=4, bundle_v=4.
    tie_width: the width of all ties
    tie_thickness: the thickness of all ties
        Because all the ties should have same width and thickness, these two parameters is used for checking
    weight_diff: desired weight difference of two side. weight_diff = abs(left-right)/(left + right)
        By default this value is 0.01, and we do not support customized parameter for this.
    isIterate: a boolean parameter to determine whether to iterate all bundle size solution to find the best one.
        if it is False, optimze() is the same with fixed_optimize(). It only uses the fixed bundle size assigned by bundle_v and bundle_h
        if it is True, it will try to use big bundles and small bundles to fill in the railcar. The program will use loop to iterately call fixed_optimize()
            The bundle_h will always be like 5, but bundle_v is varaible:
                the big bundle range will be [2, bundle_v]
                the small bundle range will be [2, bigbundle)
    debug: a boolean paramter used for GEKKO. If it is True, all the constrains will be strictly followed. Otherwise, it looses some constraints to get better solution.
    
'''
def optimize(railcar_list: List[Railcar], tie_list: List[Tie], bundle_v: int, bundle_h: int, tie_width: float,
             tie_thickness: float, weight_diff: float, isIterate: bool, debug: bool):
    # print the inputs for checking
    print("Ties:")
    for tie in tie_list:
        print(tie)
    print("Cars:")
    for car in railcar_list:
        print(car)
    print("bundle_v:" + str(bundle_v))
    print("bundle_h:" + str(bundle_h))
    print("weight_diff: " + str(weight_diff))
    print("tie_width:" + str(tie_width))
    print("tie_thickness:"+str(tie_thickness))

    # reverse the tie list to diversify the sorting of ties. That sometimes make some effects in GEKKO. It's OK not to do that
    if not debug:
        tie_list.reverse()

    # iterate the different bundle_v and bundle_h to find best combination
    for tie in tie_list:
        print(tie)
    if isIterate:
        res_list = []
        for v in range(2, bundle_v+1):
            for small_v in range(1, v):
                railcar_list_copy = copy.deepcopy(railcar_list)
                tie_list_copy = copy.deepcopy(tie_list)
                solution = fixed_optimize(railcar_list_copy, tie_list_copy, v, bundle_h, tie_width, tie_thickness, weight_diff, small_v, debug)
                # if debug is False, reverse back the tie list to make it seems like the same with input
                if not debug:
                    layout = solution[0]["layout"]
                    for i in range(len(layout)):
                        tmp1 = layout[i]
                        for j in range(len(tmp1)):
                            tmp2 = tmp1[j]
                            for k in range(len(tmp2)):
                                tmp3 = tmp2[k]
                                for p in range(len(tmp3)):
                                    dict = tmp3[p]
                                    layer = dict['layer']
                                    layer.reverse()
                res_list.append(solution)

        res = None
        for res_i in res_list:
            # the strategy is to find the best solution sorting by (load, layer nums)
            if res is None or (res_i[0]["load"] >= res[0]["load"] and res_i[0]["layer_num"] <= res[0]["layer_num"]):
                res = res_i
        return res

    else:
        railcar_list_copy = copy.deepcopy(railcar_list)
        tie_list_copy = copy.deepcopy(tie_list)
        return fixed_optimize(railcar_list_copy, tie_list_copy, bundle_v, bundle_h, tie_width, tie_thickness, weight_diff, 0, debug)
'''
    fixed_optimize() has almost the same parameter of optimize()
    It uses big bundle and small bundle to fill the railcar.
    The strategy is firstly load big bundle as much as possible,
    Then uses small bundle to fill the rest space.
    parameters: bundle_v and bundle_h is the shape of big bundle,
        small_bundle_v is the number of tie for small bundle on vertical direction. The small_bundle_h should be same with bundle_h because the width should be same
'''
def fixed_optimize(railcar_list: List[Railcar], tie_list: List[Tie], bundle_v: int, bundle_h: int, tie_width: float,
             tie_thickness: float, weight_diff: float, small_bundle_v: int, debug: bool):
    result = []
    # calculate the solution for big bundle
    solution1 = multicar_optimize(railcar_list, tie_list, bundle_v, bundle_h, tie_width, tie_thickness, weight_diff, debug)
    # if fixed wants no small bundle, it will then organize the result
    if small_bundle_v == 0:
        solution_combined = combineSolution(solution1, {"layout":[]}, tie_list, bundle_v, bundle_h, small_bundle_v)
        result.append(solution_combined)
        result = reformat(result, tie_list)
        return result

    tie_list = solution1["tie_list"]
    cnt_car = solution1["num_of_car"]

    # start to calculate small bundle solution. Copy the object to avoid heap object pollution
    railcar_list2 = copy.deepcopy(railcar_list)[:cnt_car]
    # the solution 1 return the height already occupied
    occupied_height = solution1["occupied_height"]

    # subtract the height already taken in big bundle
    for i in range(len(railcar_list2)):
        railcar = railcar_list2[i]
        railcar.railcar_height -= occupied_height[i]
    solution2 = multicar_optimize(railcar_list2, tie_list, small_bundle_v, bundle_h, tie_width, tie_thickness, weight_diff, debug)
    solution_combined = combineSolution(solution1, solution2, tie_list, bundle_v, bundle_h, small_bundle_v)
    result.append(solution_combined)
    result = reformat(result, tie_list)
    return result

'''
    combineSolution receives two solutions. Each solution has the information of 1. loading 2. layout
    The solution of one car should contains both big bundle and small bundle. 
    The input is two seperated solution, we want the method to bind them together
'''
def combineSolution(solution1, solution2, tie_list, bundle_v, bundle_h, small_bundle_v):
    layout1 = solution1["layout"]
    layout2 = solution2["layout"]
    # pad the solution2's layout. Sometimes only some car uses small bundles and others not. For those not using small bundle, put zero list
    for i in range(len(layout1) - len(layout2)):
        car = []
        side1 = []
        side2 = []
        layer1 = [0 for _ in range(len(tie_list))]
        layer2 = [0 for _ in range(len(tie_list))]
        side1.append(layer1)
        side2.append(layer2)
        car.extend([side1, side2])
        layout2.append([car])
    solution2["layout"] = layout2
    solution_combined = {}
    layout_combined = []


    # for each solution, fetch the big bundle layer(layer1) and small bundle layer(layer2)
    # combine them into a dictionary. dict["pcs"] = bundle_v * bundle_h or small_bundle_v * small bundle_h
    for i in range(len(layout1)):
        tmp = []
        for car_idx in range(len(layout1[i])):
            car = []
            for side_idx in range(len(layout1[i][car_idx])):
                side = []
                for layer_idx in range(len(layout1[i][car_idx][side_idx])):
                    layer = {}
                    layer["pcs"] = bundle_h * bundle_v
                    layer["layer"] = layout1[i][car_idx][side_idx][layer_idx]
                    side.append(layer)

                for layer_idx in range(len(layout2[i][car_idx][side_idx])):
                    if bundle_h * small_bundle_v == 0:
                        break
                    layer = {}
                    layer["pcs"] = bundle_h * small_bundle_v
                    layer["layer"] = layout2[i][car_idx][side_idx][layer_idx]
                    side.append(layer)

                car.append(side)
            tmp.append(car)
        layout_combined.append(tmp)
    solution_combined["layout"] = layout_combined
    return solution_combined

'''
    calculate the load using the load. Reformat method is to avoid error calculated by GEKKO. The answer is stored together with layout
'''
def reformat(result, tie_list):
    load = 0
    solution = result[0]
    layout = solution["layout"]
    layer_num = 0
    for i in range(len(layout)):
        for car in layout[i]:
            for side in car:
                for layer in side:
                    layer_num += 1
                    pcs = layer["pcs"]
                    x = layer["layer"]
                    for j in range(len(x)):
                        load += x[j] * pcs * tie_list[j].weight_per_tie
    result[0]["load"] = load
    result[0]["layer_num"] = layer_num
    return result

'''
    multiple_optimize() only deal with one type of bundle size. The strategy is to approximate the global solution by loading car by car
    For example, the available railcar are put in the railcar_list. It will fetch the first one and try to load as much as possible
    Then the tie quantity will decrease according to how many first car loads
    
'''
def multicar_optimize(railcar_list: List[Railcar], tie_list: List[Tie], bundle_v: int, bundle_h: int, tie_width: float,
                      tie_thickness: float, weight_diff: float, debug: bool):
    result = []
    obj = 0
    cnt = 0

    # because method calculate the solution car by car, if the previous car has the same size as current one, and the tie quantity is big enough, 
    # the algorithm will directly reuse the previous car solution
    last_railcar = Railcar(length=-1, height=124, width=50, loading=10000000000000)
    last_res_x = []
    last_obj = 0

    for railcar in railcar_list:
        # check if the last strategy is suitable for the next railcar
        if railcar.railcar_length == last_railcar.railcar_length:
            tie_list_copy = copy.deepcopy(tie_list)
            success = True
            # try to reuse the previous solution. If all the quantity is positive, it is valid to reuse it.
            for car_id in range(len(last_res_x)):
                for layer_id in range(len(last_res_x[car_id][0])):
                    for tie_id in range(len(last_res_x[car_id][0][layer_id])):
                        tie_list_copy[tie_id].quantity -= last_res_x[car_id][0][layer_id][tie_id] * bundle_h * bundle_v
                        if tie_list_copy[tie_id].quantity < 0:
                            success = False
                for layer_id in range(len(last_res_x[car_id][1])):
                    for tie_id in range(len(last_res_x[car_id][1][layer_id])):
                        tie_list_copy[tie_id].quantity -= last_res_x[car_id][1][layer_id][tie_id] * bundle_h * bundle_v
                        if tie_list_copy[tie_id].quantity < 0:
                            success = False
            if success:
                print("Follow previous solution")
                tie_list = tie_list_copy
                result.append(copy.deepcopy(last_res_x))
                cnt += 1
                obj += last_obj
                continue


        print("-------------------------------------------------------------")
        # check the quantity. If any quantity is negative, raise an exception at once
        found = False
        for tie in tie_list:
            if tie.quantity > 0:
                found = True
                break
        if not found:
            break


        # put the car in the list. The list only has one element.
        # that's weird because previous version we want to use all car's variables to calculate the solution
        # but DOF is too large. So we finally input the cars one by one
        railcar_list_tmp = [railcar]
        try:
            res = singlecar_optimize(railcar_list_tmp, tie_list, bundle_v, bundle_h, tie_width, tie_thickness,
                                     weight_diff, debug)
        except Exception as e:
            print(e)
            break



        res_x = res["x"]
        result.append(res_x)
        cnt += 1

        # update mark
        last_railcar = copy.deepcopy(railcar)
        last_res_x = copy.deepcopy(res_x)
        last_obj = obj



        print("=====================")
        print(res_x)
        print("=====================")


        # res_x is the solution calculated. Now update the tie quantity.
        for car_id in range(len(res_x)):
            for layer_id in range(len(res_x[car_id][0])):
                for tie_id in range(len(res_x[car_id][0][layer_id])):
                    tie_list[tie_id].quantity -= res_x[car_id][0][layer_id][tie_id] * bundle_h * bundle_v
            for layer_id in range(len(res_x[car_id][1])):
                for tie_id in range(len(res_x[car_id][1][layer_id])):
                    tie_list[tie_id].quantity -= res_x[car_id][1][layer_id][tie_id] * bundle_h * bundle_v
        print("remain: ")
        for tie in tie_list:
            print(tie)

    print("-----------------------------------------------final----------------------------------------------------")
    print("Number of railcar: " + str(cnt))
    print("remain: ")
    for tie in tie_list:
        print(tie)
    print("Max Load:" + str(int(obj)))

    # display the result of layout. That's for better show our temporary outcome to clients
    # actually we won't use it any longger
    df = pd.DataFrame(columns=["Tie" + str(i + 1) for i in range(len(tie_list))])
    # print(df)
    carId = 1
    indexs = []
    for i in range(len(result)):
        for car in result[i]:
            sideId = 1
            for side in car:
                layerId = 1
                for layer in side:
                    df.loc[df.shape[0]] = np.array(layer).round().astype(int)
                    indexs.append("Car" + str(carId) + " Side" + str(sideId) + " Layer" + str(layerId))
                    layerId += 1
                sideId += 1
            carId += 1
    df["Locations"] = indexs
    df = df.set_index("Locations")
    print("--------------------------------------------------")
    print(df)

    # collect the results
    report = {}
    # result
    report["layout"] = result
    # df
    report["df"] = df
    # tie_list remain
    report["tie_list"] = tie_list
    # railcar occupied height. That's for calculate the availble space for small bundle
    occupied_height = [0 for _ in range(len(railcar_list))]
    for i in range(len(result)):
        for car in result[i]:
            max_side_height = 0
            for side in car:
                cnt = 0
                for layer in side:
                    if max(layer) > 0:
                        cnt += 1
                h = cnt * bundle_v * tie_thickness
                max_side_height = max(max_side_height, h)
            # occupied_height.append(max_side_height)
            occupied_height[i] = max_side_height
    report["occupied_height"] = occupied_height
    report["num_of_car"] = cnt
    return report

'''
    singlecar_optimize() calculate the best layout for the car. This objective is approximated by optimized each layer
    railcar_list always has only one element.
'''
def singlecar_optimize(railcar_list: List[Railcar], tie_list: List[Tie], bundle_v: int, bundle_h: int, tie_width: float,
                       tie_thickness: float, weight_diff: float, debug: bool):
    # layer_num_limit calculate the upper limit of layer num. +2 is to incorporate the mat thickness
    layer_nums_limit = int(railcar_list[0].railcar_height // (bundle_v * tie_thickness + 2))
    # if the space is too small to put one more layer, we put zero arrays to return
    if layer_nums_limit == 0:
        x = [[[[0 for tie_id in range(len(tie_list))]],
              [[0 for tie_id in range(len(tie_list))]]]
             for car_id in range(len(railcar_list))]
        return {"obj": 0, "x": x}


    res = {}
    ## copy the object to avoid pollution
    railcar_list_copy = copy.deepcopy(railcar_list)
    tie_list_copy = copy.deepcopy(tie_list)
    # initialize the layout framework
    layout = [[[],[]] for _ in range(len(railcar_list))]
    # for each layer try to load as much as possible, until no more layer can be loaded
    for idx in range(layer_nums_limit):
        try:
            # calculate the layer optimization
            res_layer = singlelayer_optimize(railcar_list_copy, tie_list_copy, bundle_v, bundle_h, tie_width, tie_thickness, weight_diff, debug)
            x = res_layer["x"]
            # update the quantity. Maybe its useless.....
            for car_id in range(len(x)):
                for layer_id in range(len(x[car_id][0])):
                    layout[car_id][0].append(x[car_id][0][layer_id])
                    for tie_id in range(len(x[car_id][0][layer_id])):
                        tie_list_copy[tie_id].quantity -= x[car_id][0][layer_id][tie_id]*bundle_h*bundle_v
                for layer_id in range(len(x[car_id][1])):
                    layout[car_id][1].append(x[car_id][1][layer_id])
                    for tie_id in range(len(x[car_id][1][layer_id])):
                        tie_list_copy[tie_id].quantity -= x[car_id][1][layer_id][tie_id] * bundle_h * bundle_v


        except Exception as e:
            print(e)
            break

    # store the car result
    for car_id in range(len(railcar_list)):
        if len(layout[car_id][0]) == 0:
            layout[car_id][0].append([0 for _ in range(len(tie_list))])
        if len(layout[car_id][1]) == 0:
            layout[car_id][1].append([0 for _ in range(len(tie_list))])
    res["x"] = layout
    return res


'''
    singlelayer_optimize() does the basic optimization calculation. It uses GEKKO to iterate to the best point
    All the constraints will be defined there
'''
def singlelayer_optimize(railcar_list: List[Railcar], tie_list: List[Tie], bundle_v: int, bundle_h: int, tie_width: float,
                       tie_thickness: float, weight_diff: float, debug: bool):
    # check parameters
    if weight_diff > 1:
        raise Exception("Weight Difference can not be larger than 100%")
    for tie in tie_list:
        if not math.isclose(tie.width, tie_width, rel_tol=1e-9, abs_tol=1e-9) or not math.isclose(tie.thickness,
                                                                                                  tie_thickness,
                                                                                                  rel_tol=1e-9,
                                                                                                  abs_tol=1e-9):
            raise Exception("All tie should have same width and thickness")

    # GEKKO
    m = GEKKO()

    m.options.SOLVER = 1  # APOPT is an MINLP solver

    tie_num = len(tie_list)
    car_num = len(railcar_list)
    layer_nums = [1 for car in railcar_list]
    layer_nums_limit = [int(car.railcar_height // (bundle_v * tie_thickness + 2)) for car in railcar_list]

    # initialzie the layout framework
    if layer_nums_limit[0] == 0:
        x = [[[[0 for tie_id in range(tie_num)]],
              [[0 for tie_id in range(tie_num)]]]
             for car_id in range(car_num)]

        return {"obj": 0, "x": x}

    # calculate the upper limit for the bundle numbers
    # ensure the total length sum is with in the car
    x_limit = [[[[int(railcar_list[car_id].railcar_length // tie_list[tie_id].length) for tie_id in range(tie_num)] for
                 layer_id in range(layer_nums[car_id])],
                [[int(railcar_list[car_id].railcar_length // tie_list[tie_id].length) for tie_id in range(tie_num)] for
                 layer_id in range(layer_nums[car_id])]]
               for car_id in range(car_num)]
    # initialzie the layout framework
    layouts = [[[[m.Var(value=100, integer=True, lb=0, ub=x_limit[car_id][0][layer_id][tie_id]) for tie_id in
                  range(tie_num)] for layer_id in range(layer_nums[car_id])],
                [[m.Var(value=100, integer=True, lb=0, ub=x_limit[car_id][1][layer_id][tie_id]) for tie_id in
                  range(tie_num)] for layer_id in range(layer_nums[car_id])]]
               for car_id in range(car_num)]


    # set the core paramters of GEKKO
    # these parameters directly influence the performance.
    # you should pay more attention on minlp_maximum_iterations,minlp_max_iter_with_int_sol and nlp_maximum_iterations
    m.solver_options = ['minlp_maximum_iterations 5000',
                        # minlp iterations with integer solution
                        'minlp_max_iter_with_int_sol 2000',
                        # treat minlp as nlp
                        'minlp_as_nlp 0',
                        # nlp sub-problem max iterations
                        'nlp_maximum_iterations 500',
                        # 1 = depth first, 2 = breadth first
                        'minlp_branch_method 1',
                        # maximum deviation from whole number
                        'minlp_integer_tol 0.01',
                        # covergence tolerance
                        'minlp_gap_tol 0.01']

    # Constraint1: Total Length >= 0.9 Total Length
    for car_id in range(car_num):
        # side 1
        for layer_id in range(layer_nums[car_id]):
            x_list = layouts[car_id][0][layer_id]
            tmp = [m.Intermediate(x_list[i] * tie_list[i].length) for i in range(len(x_list))]
            m.Equation(
                (m.sum(tmp) + (m.sum(x_list) - 2) * 6/12 + 48/12 - railcar_list[car_id].railcar_length) >= 0)
        # side 2
        for layer_id in range(layer_nums[car_id]):
            x_list = layouts[car_id][1][layer_id]
            tmp = [m.Intermediate(x_list[i] * tie_list[i].length) for i in range(len(x_list))]
            m.Equation(
                (m.sum(tmp) + (m.sum(x_list) - 2) * 6/12 + 48/12 - railcar_list[car_id].railcar_length) >= 0)

    # Constraint2: Total Length
    for car_id in range(car_num):
        # side 1
        for layer_id in range(layer_nums[car_id]):
            x_list = layouts[car_id][0][layer_id]
            tmp = [m.Intermediate(x_list[i] * tie_list[i].length) for i in range(len(x_list))]
            m.Equation(m.sum(tmp) <= railcar_list[car_id].railcar_length)
        # side 2
        for layer_id in range(layer_nums[car_id]):
            x_list = layouts[car_id][1][layer_id]
            tmp = [m.Intermediate(x_list[i] * tie_list[i].length) for i in range(len(x_list))]
            m.Equation(m.sum(tmp) <= railcar_list[car_id].railcar_length)

    # Constraint3: Total Quantity
    for tie_id in range(tie_num):
        tie_tmp = []
        for car_id in range(car_num):
            # side 1
            for layer_id in range(layer_nums[car_id]):
                tie_tmp.append(layouts[car_id][0][layer_id][tie_id] * bundle_h * bundle_v)
            # side 2
            for layer_id in range(layer_nums[car_id]):
                tie_tmp.append(layouts[car_id][1][layer_id][tie_id] * bundle_h * bundle_v)
        m.Equation(m.sum(tie_tmp) <= tie_list[tie_id].quantity)

    # Constraint4: Total Loading
    for car_id in range(car_num):
        loading_tmp = []
        # side1
        for layer_id in range(layer_nums[car_id]):
            x_list = layouts[car_id][0][layer_id]
            loading_tmp.extend([m.Intermediate(x_list[i] * bundle_h * bundle_v * tie_list[i].weight_per_tie) for i in
                                range(len(x_list))])
        # side2
        for layer_id in range(layer_nums[car_id]):
            x_list = layouts[car_id][1][layer_id]
            loading_tmp.extend([m.Intermediate(x_list[i] * bundle_h * bundle_v * tie_list[i].weight_per_tie) for i in
                                range(len(x_list))])
        m.Equation(m.sum(loading_tmp) <= railcar_list[car_id].railcar_loading)

    # Constraint5: Weight Difference
    for car_id in range(car_num):
        # side1
        side1_tmp = []
        for layer_id in range(layer_nums[car_id]):
            x_list = layouts[car_id][0][layer_id]
            side1_tmp.extend([m.Intermediate(x_list[i] * tie_list[i].length) for i in
                              range(len(x_list))])
        # side2
        side2_tmp = []
        for layer_id in range(layer_nums[car_id]):
            x_list = layouts[car_id][1][layer_id]
            side2_tmp.extend([m.Intermediate(x_list[i] * tie_list[i].length) for i in
                              range(len(x_list))])
        m.Equation(m.sum(side1_tmp) - m.sum(side2_tmp) <= weight_diff * (m.sum(side1_tmp) + m.sum(side2_tmp)))
        m.Equation(m.sum(side2_tmp) - m.sum(side1_tmp) <= weight_diff * (m.sum(side1_tmp) + m.sum(side2_tmp)))

    # Objective
    total_load_tmp = []
    for car_id in range(car_num):
        loading_tmp = []
        # side1
        for layer_id in range(layer_nums[car_id]):
            x_list = layouts[car_id][0][layer_id]
            total_load_tmp.extend([m.Intermediate(x_list[i] * tie_list[i].length) for i in
                                   range(len(x_list))])
        # side2
        for layer_id in range(layer_nums[car_id]):
            x_list = layouts[car_id][1][layer_id]
            total_load_tmp.extend([m.Intermediate(x_list[i] * tie_list[i].length) for i in
                                   range(len(x_list))])
    m.Maximize(m.sum(total_load_tmp))
    m.solve(disp=True, debug=debug)

    # all belows are for checking
    for car_id in range(car_num):
        for layer_id in range(layer_nums[car_id]):
            l = 0
            for tie_id in range(tie_num):
                l = l + tie_list[tie_id].length * layouts[car_id][0][layer_id][tie_id].VALUE[0]
            print(l)
        print("------------------")
        for layer_id in range(layer_nums[car_id]):
            l = 0
            for tie_id in range(tie_num):
                l = l + tie_list[tie_id].length * layouts[car_id][1][layer_id][tie_id].VALUE[0]
            print(l)
    x = [[[[int(round(layouts[car_id][0][layer_id][tie_id].VALUE[0])) for tie_id in range(tie_num)] for layer_id in
           range(layer_nums[car_id])],
          [[int(round(layouts[car_id][1][layer_id][tie_id].VALUE[0])) for tie_id in range(tie_num)] for layer_id in
           range(layer_nums[car_id])]]
         for car_id in range(car_num)]
    # sort the layers according to the occupied length.
    # that's useless when there is only one layer
    for car in x:
        for j in range(len(car)):
            car[j] = sorted(car[j], key=lambda blist: sum(tie_list[i].length * num for i, num in enumerate(blist)),
                            reverse=True)

    # check the quantity of ties to make sure they are all positive
    load = 0
    tie_list_copy = copy.deepcopy(tie_list)
    success = True
    for car_id in range(len(x)):
        for layer_id in range(len(x[car_id][0])):
            for tie_id in range(len(x[car_id][0][layer_id])):
                tie_list_copy[tie_id].quantity -= x[car_id][0][layer_id][tie_id] * bundle_h * bundle_v
                load += tie_list[tie_id].weight_per_tie * x[car_id][0][layer_id][tie_id] * bundle_h * bundle_v
                if tie_list_copy[tie_id].quantity < 0:
                    success = False
        for layer_id in range(len(x[car_id][1])):
            for tie_id in range(len(x[car_id][1][layer_id])):
                tie_list_copy[tie_id].quantity -= x[car_id][1][layer_id][tie_id] * bundle_h * bundle_v
                load += tie_list[tie_id].weight_per_tie * x[car_id][0][layer_id][tie_id] * bundle_h * bundle_v
                if tie_list_copy[tie_id].quantity < 0:
                    success = False
    if not success:
        raise Exception("Load quantity exceeds")
    if load == 0:
        raise Exception("Empty layout")
    res = {"obj": load, "x": x}
    return res

