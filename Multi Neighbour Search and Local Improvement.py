import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd
import bisect
from amplpy import AMPL
import time
import csv
import itertools


def load_data(file_path):
    num_machines_list = []
    num_orders_list = []
    data_matrices = []

    with open(file_path, 'r') as file:
        num_instances = int(file.readline())

        for _ in range(num_instances):
            instance_number = int(file.readline())
            num_orders = int(file.readline())
            num_machines = int(file.readline())

            num_orders_list.append(num_orders)
            num_machines_list.append(num_machines)

            data_matrix = np.zeros((num_machines, num_orders + 1), dtype=int)
            # Set instance number in the first column:
            data_matrix[:, 0] = instance_number  

            for order in range(1, num_orders+1):
                for machine in range(num_machines):
                    time_on_machine = int(file.readline())
                    data_matrix[machine, order] = time_on_machine

            # Add the data matrix to the list
            data_matrices.append(data_matrix)
            # Print instance data
            print(f"Instance Number: {instance_number}")
            print("Data Matrix:")
            print(data_matrix)
            print()

    return num_instances, num_machines_list, num_orders_list, data_matrices

def make_span(instance_data, sequence, num_machines, num_orders):
    t = 0
    end_time = [[0 for _ in range(num_orders)] for _ in range(num_machines)]
    for machine in range(num_machines):
        if machine == 0:
            for j in sequence:
                t += instance_data[machine][j]
                end_time[machine][j] = t
        else:
            t = end_time[machine-1][sequence[0]] 
            for j in sequence:
                print(instance_data)
                sys.exit()
                if t < end_time[machine-1][j]:
                    t= end_time[machine-1][j]+ instance_data[machine][j]
                else:
                    t += instance_data[machine][j]
                end_time[machine][j] = t
    #print(t)
    return t


#LIHMN 2
def first_neighbourhood(schedule):
    for i in range(len(schedule) - 1):
        new_schedule = schedule[:]
        new_schedule[i], new_schedule[i+1] = new_schedule[i+1], new_schedule[i]
        #print(schedule)
        #print("jadide")
        #print(new_schedule)
        #print("---")
        yield new_schedule

def second_neighbourhood(schedule):
    for i, j in itertools.combinations(range(len(schedule)), 2):
        new_schedule = schedule[:]
        new_schedule[i], new_schedule[j] = new_schedule[j], new_schedule[i]
        yield new_schedule

def third_neighbourhood(schedule):
    for i, j, k in itertools.combinations(range(len(schedule)), 3):
        new_schedule = schedule[:]
        new_schedule[i], new_schedule[j], new_schedule[k] = new_schedule[j], new_schedule[k], new_schedule[i]
        yield new_schedule
'''
#LIHMN 1
def first_neighbourhood(schedule):
    for i in range(len(schedule) - 1):
        new_schedule = schedule[:]
        new_schedule[i], new_schedule[i+1] = new_schedule[i+1], new_schedule[i]
        #print(schedule)
        #print("jadide")
        #print(new_schedule)
        #print("---")
        yield new_schedule

def second_neighbourhood(schedule):
    for i in range(len(schedule) - 3):
        new_schedule = schedule[:]
        new_schedule[i:i+2], new_schedule[i+2:i+4] = new_schedule[i+2:i+4], new_schedule[i:i+2]
        yield new_schedule

def third_neighbourhood(schedule):
    for i, j in itertools.combinations(range(len(schedule)), 2):
        new_schedule = schedule[:]
        new_schedule[i], new_schedule[j] = new_schedule[j], new_schedule[i]
        yield new_schedule
'''
def apply_mnlh(data_matrix):
    num_instances = len(data_matrix)
    
    results = []
    
    for i in range(num_instances):
        start_time = time.time()
        
        data = data_matrix[i][:,1:]

        # Initialize the schedule with a random permutation of the jobs
        schedule = list(range(len(data[0])))
        np.random.shuffle(schedule)
        
        makespan = make_span(data, schedule, 5, 20)
        
        # Apply the multi-neighbourhood local improvement heuristic
        while True:
            improved = False
            
            # First Neighbourhood
            for new_schedule in first_neighbourhood(schedule):
                #print("first schedule:" + str(new_schedule))
                #new_makespan = calculate_makespan(new_schedule, data)
                new_makespan = make_span(data, new_schedule, 5, 20)
                if new_makespan < makespan:
                    schedule = new_schedule
                    makespan = new_makespan
                    improved = True
                    break  # Go back and search in the first neighbourhood again
            
            # Second Neighbourhood (only if no improvement in the first neighbourhood)
            if not improved:
                for new_schedule in second_neighbourhood(schedule):
                    #new_makespan = calculate_makespan(new_schedule, data)
                    new_makespan = make_span(data, new_schedule, 5, 20)
                    if new_makespan < makespan:
                        schedule = new_schedule
                        makespan = new_makespan
                        improved = True
                        break  # Go back and search in the first neighbourhood again
            
            # Third Neighbourhood (only if no improvement in the first and second neighbourhoods)
            if not improved:
                
                for new_schedule in third_neighbourhood(schedule):
                    #new_makespan = calculate_makespan(new_schedule, data)
                    new_makespan = make_span(data, new_schedule, 5, 20)
                    if new_makespan < makespan:
                        schedule = new_schedule
                        makespan = new_makespan
                        improved = True
                        break  # Go back and search in the first neighbourhood again
            
            if not improved:
                break
        
        end_time = time.time()
        
        solving_time = end_time - start_time

        results.append((makespan, solving_time,schedule))
    
    return results

#save the results to a CSV file
def save_results_to_csv(results, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Instance", "Makespan", "Solving Time"])
        for i in range(len(results)):
            writer.writerow([i+1, results[i][0], results[i][1]])

# Load data
num_instances, num_machines_list, num_orders_list, data_matrix = load_data('Fs_20.txt')
print(data_matrix[0]) #first instance data

results = apply_mnlh(data_matrix)
save_results_to_csv(results, "mnlh.csv")

for i in range(len(results)):
    print(f"Instance {i+1}: makespan = {results[i][0]}, solving time = {results[i][1]} seconds  -  Sequence = {results[i][2]}")

