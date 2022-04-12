import itertools

# data = [
#     [
#         [854, 882, 89, 623],
#         [943, 890, 64, 698],
#         [1032, 924, 59, 789],
#         [1129, 992, 63, 870]
#     ],
#     [
#         [1087, 968, 38, 872],
#         [1130, 1024, 41, 925],
#         [1034, 1048, 45, 1002],
#         [1142, 1091, 54, 984]
#     ],
#     [
#         [818, 746, 43, 591],
#         [894, 769, 52, 682],
#         [940, 795, 58, 728],
#         [978, 684, 59, 784]
#     ],
#     [
#         [605, 825, 14, 400],
#         [680, 952, 31, 512],
#         [812, 1023, 30, 501],
#         [927, 1038, 38, 580]
#     ]
# ]

# #roll up on location (all 4 cities)

# roll_up_on_cities = []

# for i in range(len(data[0])):
#     temp = []
#     for j in range(len(data[0][0])):
#         sum_val = 0
#         for k in range(len(data)):
#             sum_val+=data[k][i][j]
#         temp.append(sum_val)

#     roll_up_on_cities.append(temp)       


# print(roll_up_on_cities)

"""
Created on Tue Apr 12 06:15:40 2022

@author: mmritha
"""


def roll_up(data, time, country, cities):

    roll_up_data = {}
    add = lambda x: sum(x)
    roll_up_data[country] = {}
    for i in time:
        temp = list(zip(data[cities[0]][i], data[cities[1]][i]))
        roll_up_data[country][i] = [add(i) for i in temp]

    return (roll_up_data)

def drill_down(data, time, location):
    drill_down_data = []

    for i in time:
        temp  = [round((j /3),2) for j in data[location][i]]
        #print(temp)
        for i in range(3):
            drill_down_data.append(temp)
    return drill_down_data        



def dice(data, location, time, item):
    dice_data = {}
    for i in location:
        dice_data[i] = {}
        for value in time:
            
            temp = data[i][value]
            dice_data[i][value]  = [ data[i][value][x] for x in item]
            
    return (dice_data)     


def slice(data, att, location):
    slice_data = {}
    for i in location:
        slice_data[i] = data[i][att]

    return slice_data   





data = {"Chicago": {"Q1":[854, 882, 89, 623], "Q2":[943, 890, 64, 698], "Q3":[1032, 924, 59, 789], "Q4":[1129, 992, 63, 870]}, "New York": {"Q1":[1087, 968, 38, 872], "Q2":[1130, 1024, 41, 925], "Q3":[1034, 1048, 45, 1002], "Q4":[1142, 1091, 54, 984]}, "Toronto": {"Q1":[818, 746, 43, 591], "Q2":[894, 769, 52, 682], "Q3":[940, 795, 58, 728], "Q4":[978, 864, 59, 784]}, "Vancouver": {"Q1":[605, 825, 14, 400], "Q2":[680, 952, 31, 512], "Q3":[812, 1023, 30, 501], "Q4":[927, 1038, 38, 580]}}


location = ["Chicago", "New York", "Toronto", "Vancouver"]

time = ["Q1", "Q2", "Q3", "Q4"]


########################################################

#print(roll_up(data, time, "Canada", ["Vancouver", "Toronto"]))
#print(roll_up(data, time, "USA", ["Chicago", "New York"]))


########################################################
# drill_down_data = {}
# for i in location:

#     drill_down_data[i] = (drill_down(data, time,"Chicago"))

# print(drill_down_data)  


########################################################


#print(dice(data, ["Vancouver", "Toronto"], ["Q1", "Q2"], [0,1]))


########################################################

print(slice(data, "Q1", location))

