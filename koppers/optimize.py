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
               + " thickness: " + str(self.thickness) + " quantity: " + str(self.quantity)


class Layer:
    layer_height: float
    tie_list: List[Tie] = []
    bundle_nums: List[int]

    def init_tie(self, tie_list_: List[Tie]):
        self.tie_list = tie_list_.copy()
        self.bundle_nums = [0 for _ in range(len(tie_list_))]


class Side:
    side_width: float
    layers: List[Layer]

    def init(self, tie_list: List[Tie], layer_num: int):
        self.layers = [Layer() for _ in range(layer_num)]
        for layer in self.layers:
            layer.init_tie(tie_list)


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
        return "length: " + str(self.railcar_length)
'''
    solution = multiple_optimize(...)
    big-bundle = solution[0]
    small-bundle = solution[1]
    
    big-bundle is a diction including:
        "load": total laoding
        "layout":  carlist:[single[side:[layer:[bundle(int), ],],] ,]
        "df": index: carID_sideID_layerID . column:[tie1, tie2, tie3.....]
'''
def optimize(railcar_list: List[Railcar], tie_list: List[Tie], bundle_v: int, bundle_h: int, tie_width: float, tie_thickness: float, weight_diff: float):
    result = []
    railcar_list = sorted(railcar_list, key=lambda x: x.railcar_length, reverse=True)
    solution1 = multicar_optimize(railcar_list, tie_list, bundle_v, bundle_h, tie_width, tie_thickness, weight_diff)
    result.append(solution1)
    tie_list = solution1["tie_list"]
    railcar_list2 = copy.deepcopy(railcar_list)
    occupied_height = solution1["occupied_height"]
    for i in range(len(railcar_list2)):
        railcar = railcar_list2[i]
        railcar.railcar_height -= occupied_height[i]
    solution2 = multicar_optimize(railcar_list2, tie_list, 1, bundle_h, tie_width, tie_thickness, weight_diff)
    result.append(solution2)
    return result




def multicar_optimize(railcar_list: List[Railcar], tie_list: List[Tie], bundle_v: int, bundle_h: int, tie_width: float,
                      tie_thickness: float, weight_diff: float):
    result = []
    obj = 0
    cnt = 0
    railcar_list = sorted(railcar_list, key=lambda x: x.railcar_length, reverse=True)

    last_railcar = Railcar(length=-1, height=124, width=50, loading=10000000000000)
    last_res_x = []
    last_obj = 0

    for railcar in railcar_list:
        # check if the last strategy is suitable for the next railcar
        if railcar.railcar_length == last_railcar.railcar_length:
            tie_list_copy = copy.deepcopy(tie_list)
            success = True
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
        found = False
        for tie in tie_list:
            if tie.quantity > 0:
                found = True
                break
        if not found:
            break
        railcar_list_tmp = [railcar]
        try:
            res = singlecar_optimize(railcar_list_tmp, tie_list, bundle_v, bundle_h, tie_width, tie_thickness, weight_diff)
        except Exception as e:
            print(e)
            break
        obj += res["obj"]
        print(res["obj"])
        res_x = res["x"]
        result.append(res_x)
        cnt += 1

        # update mark
        last_railcar = copy.deepcopy(railcar)
        last_res_x = copy.deepcopy(res_x)
        last_obj = obj


        # tie_list_copy = copy.deepcopy(tie_list)
        print("=====================")
        print(res_x)
        print("=====================")
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
    print("Max Load:" + str(int(-obj)))
    # print(result)

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
    report = {}
    # obj: max_load
    report["load"] =  -obj
    # result
    report["layout"] = result
    # df
    report["df"] = df
    # tie_list remain
    report["tie_list"] = tie_list
    # railcar occupied height
    occupied_height = [0 for _ in range(len(railcar_list))]
    for i in range(len(result)):
        for car in result[i]:
            max_side_height = 0
            for side in car:
                cnt = 0
                for layer in side:
                    if max(layer) > 0:
                        cnt += 1
                h = cnt*bundle_v*tie_thickness
                max_side_height = max(max_side_height, h)
            # occupied_height.append(max_side_height)
            occupied_height[i] = max_side_height
    report["occupied_height"] = occupied_height
    return report




def singlecar_optimize(railcar_list: List[Railcar], tie_list: List[Tie], bundle_v: int, bundle_h: int, tie_width: float,
                       tie_thickness: float, weight_diff: float):
    # check
    if weight_diff > 1:
        raise Exception("Weight Difference can not be larger than 100%")
    for tie in tie_list:
        print(str(tie.width) + " " + str(tie_width))
        if not math.isclose(tie.width, tie_width, rel_tol=1e-9, abs_tol=1e-9) or not math.isclose(tie.thickness, tie_thickness, rel_tol=1e-9, abs_tol=1e-9):
            raise Exception("All tie should have same width and thickness")

    m = GEKKO()

    m.options.SOLVER = 1  # APOPT is an MINLP solver

    tie_num = len(tie_list)
    car_num = len(railcar_list)
    layer_nums = [int(car.railcar_height // (bundle_v * tie_thickness + 2)) for car in railcar_list]

    if layer_nums[0] == 0:
        x = [[[[0 for tie_id in range(tie_num)]],
              [[0 for tie_id in range(tie_num)]]]
             for car_id in range(car_num)]



        return {"obj": 0, "x": x}


    x_limit = [[[[int(railcar_list[car_id].railcar_length // tie_list[tie_id].length) for tie_id in range(tie_num)] for
                 layer_id in range(layer_nums[car_id])],
                [[int(railcar_list[car_id].railcar_length // tie_list[tie_id].length) for tie_id in range(tie_num)] for
                 layer_id in range(layer_nums[car_id])]]
               for car_id in range(car_num)]

    layouts = [[[[m.Var(value=100, integer=True, lb=0, ub=x_limit[car_id][0][layer_id][tie_id]) for tie_id in
                  range(tie_num)] for layer_id in range(layer_nums[car_id])],
                [[m.Var(value=100, integer=True, lb=0, ub=x_limit[car_id][1][layer_id][tie_id]) for tie_id in
                  range(tie_num)] for layer_id in range(layer_nums[car_id])]]
               for car_id in range(car_num)]
    m.solver_options = ['minlp_maximum_iterations 5000',
                        # minlp iterations with integer solution
                        'minlp_max_iter_with_int_sol 100',
                        # treat minlp as nlp
                        'minlp_as_nlp 0',
                        # nlp sub-problem max iterations
                        'nlp_maximum_iterations 50',
                        # 1 = depth first, 2 = breadth first
                        'minlp_branch_method 1',
                        # maximum deviation from whole number
                        'minlp_integer_tol 0.05',
                        # covergence tolerance
                        'minlp_gap_tol 0.01']

    # Constraint1: Total Length >= 0.9 Total Length
    for car_id in range(car_num):
        # side 1
        for layer_id in range(layer_nums[car_id]):
            x_list = layouts[car_id][0][layer_id]
            tmp = [m.Intermediate(x_list[i] * tie_list[i].length) for i in range(len(x_list))]
            m.Equation(
                m.sum(tmp) * (m.sum(tmp) + (m.sum(x_list) - 2) * 6 + 18 - railcar_list[car_id].railcar_length) >= 0)
        # side 2
        for layer_id in range(layer_nums[car_id]):
            x_list = layouts[car_id][1][layer_id]
            tmp = [m.Intermediate(x_list[i] * tie_list[i].length) for i in range(len(x_list))]
            m.Equation(
                m.sum(tmp) * (m.sum(tmp) + (m.sum(x_list) - 2) * 6 + 18 - railcar_list[car_id].railcar_length) >= 0)

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
            side1_tmp.extend([m.Intermediate(x_list[i] * bundle_h * bundle_v * tie_list[i].weight_per_tie) for i in
                              range(len(x_list))])
        # side2
        side2_tmp = []
        for layer_id in range(layer_nums[car_id]):
            x_list = layouts[car_id][1][layer_id]
            side2_tmp.extend([m.Intermediate(x_list[i] * bundle_h * bundle_v * tie_list[i].weight_per_tie) for i in
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
            total_load_tmp.extend([m.Intermediate(x_list[i] * bundle_h * bundle_v * tie_list[i].weight_per_tie) for i in
                                   range(len(x_list))])
        # side2
        for layer_id in range(layer_nums[car_id]):
            x_list = layouts[car_id][1][layer_id]
            total_load_tmp.extend([m.Intermediate(x_list[i] * bundle_h * bundle_v * tie_list[i].weight_per_tie) for i in
                                   range(len(x_list))])
    m.Maximize(m.sum(total_load_tmp))
    m.solve(disp=False,debug=False)

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
    # print(layouts)
    x = [[[[int(round(layouts[car_id][0][layer_id][tie_id].VALUE[0])) for tie_id in range(tie_num)] for layer_id in
           range(layer_nums[car_id])],
          [[int(round(layouts[car_id][1][layer_id][tie_id].VALUE[0])) for tie_id in range(tie_num)] for layer_id in
           range(layer_nums[car_id])]]
         for car_id in range(car_num)]
    res = {"obj": m.options.OBJFCNVAL, "x": x}

    tie_list_copy = copy.deepcopy(tie_list)
    success = True
    for car_id in range(len(x)):
        for layer_id in range(len(x[car_id][0])):
            for tie_id in range(len(x[car_id][0][layer_id])):
                tie_list_copy[tie_id].quantity -= x[car_id][0][layer_id][tie_id] * bundle_h * bundle_v
                if tie_list_copy[tie_id].quantity < 0:
                    success = False
        for layer_id in range(len(x[car_id][1])):
            for tie_id in range(len(x[car_id][1][layer_id])):
                tie_list_copy[tie_id].quantity -= x[car_id][1][layer_id][tie_id] * bundle_h * bundle_v
                if tie_list_copy[tie_id].quantity < 0:
                    success = False
    if not success:
        raise Exception()
    return res