import numpy as np
import time
import csv
import matplotlib.pyplot as plt

class SimulatedAnnealing:
    def __init__(self, data_matrix, T=-1, alpha=0.995, stopping_iter=100000, stopping_m = 100, N_t=10):
        self.data_matrix = data_matrix
        self.T = len(data_matrix) if T == -1 else T # When T is high, worse solutions can be accepted more frequently, which allows the algorithm to explore the solution space widely. 
        self.T_save = self.T  # save inital T to reset if batch annealing is used
        self.alpha = alpha #alpha is the cooling rate. It's a factor by which the temperature T is decreased in each iteration. The value of alpha is between 0 and 1. A smaller alpha means a slower cooling schedule, which can potentially lead to better solutions but at the cost of longer computation time.
        self.stopping_iter = stopping_iter
        self.stopping_m = stopping_m  # number of neighbors to visit without improvement
        self.iteration = 1
        self.best_fitness = float('inf')
        self.no_improve = 0  # counter for number of neighbors visited without improvement
        self.N_t = N_t  # In each tempreture how many solutions will check

    def calculate_fitness(self, sequence, instance_data):
        t = 0
        num_orders = 20
        num_machines = 5
        end_time = [[0 for _ in range(num_orders)] for _ in range(num_machines)]
        for machine in range(num_machines):
            if machine == 0:
                for j in sequence:
                    t += instance_data[machine][j]
                    end_time[machine][j] = t
            else:
                t = end_time[machine-1][sequence[0]] 
                for j in sequence:
                    if t < end_time[machine-1][j]:
                        t= end_time[machine-1][j]+ instance_data[machine][j]
                    else:
                        t += instance_data[machine][j]
                    end_time[machine][j] = t
        return t

    def anneal(self, sequence):
        fitness_over_time = []
        while self.no_improve < self.stopping_m and self.iteration < self.stopping_iter:
            for _ in range(self.N_t):  # Added loop for N_t neighbors
                # Generate a neighbor sequence by swapping two jobs
                i, j = np.random.choice(len(sequence), 2)
                neighbor = sequence.copy()
                neighbor[i], neighbor[j] = sequence[j], sequence[i]
                # Calculate its fitness
                fitness = self.calculate_fitness(neighbor, self.instance)
                # If it's better than the current best fitness, update the best fitness and reset the no_improve counter
                if fitness < self.best_fitness:
                    sequence = neighbor
                    self.best_fitness = fitness
                    self.no_improve = 0
                else:
                    # If the new solution is worse, accept it with a probability that depends on the temperature and the difference in fitness
                    if np.random.rand() < np.exp((self.best_fitness - fitness) / self.T):
                        sequence = neighbor
                    self.no_improve += 1
                self.iteration += 1
                fitness_over_time.append(self.best_fitness)  # Added this line
            self.T *= self.alpha
        return fitness_over_time  # Added this line

    def run(self):
        results = []
        all_fitness_over_time = []

        for instance in self.data_matrix:
            self.instance = instance[:, 1:]  # exclude the instance number
            sequence = np.random.permutation(20)  # Generates a random sequence from 1 to 20
            self.best_fitness = float('inf')  # Reset best_fitness for each instance
            self.no_improve = 0  # Reset no_improve for each instance
            start_time = time.time()
            fitness_over_time = self.anneal(sequence)  # Modified this line
            all_fitness_over_time.append(fitness_over_time)
            end_time = time.time()
            solving_time = end_time - start_time
            results.append((instance[0, 0], self.best_fitness, solving_time))  # instance number, makespan, solving time

        # Save results to CSV and TXT files
        with open('results.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Instance", "Makespan", "Solving Time"])
            writer.writerows(results)

        with open('results.txt', 'w') as txtfile:
            for result in results:
                txtfile.write(f"Instance {result[0]}: makespan = {result[1]}, solving time = {result[2]}\n")

        return results, all_fitness_over_time

    def plot_all_fitness_over_time(self, all_fitness_over_time):  # Added this method
        num_plots = len(all_fitness_over_time) // 10
        for i in range(num_plots):
            plt.figure(i)
            for j in range(10):
                index = i * 10 + j
                fitness_over_time = all_fitness_over_time[index]
                plt.plot(fitness_over_time, label=f'Instance {index+1}')
            plt.title(f'Best Fitness Over Time for Instances {i*10+1} to {(i+1)*10}')
            plt.xlabel('Iteration')
            plt.ylabel('Best Fitness')
            plt.legend()
            plt.savefig(str(i) + '.png')
        plt.show()

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
            #print(f"Instance Number: {instance_number}")
            #print("Data Matrix:")
            #print(data_matrix)
            #print()

    return num_instances, num_machines_list, num_orders_list, data_matrices

def main():
    # Load data
    num_instances, num_machines_list, num_orders_list, data_matrix = load_data('Fs_20.txt')
    #print(data_matrix[0]) #first instance data

    sa = SimulatedAnnealing(data_matrix, 100, 0.995, 100000, 1000, 1)
    start = time.time()
    results, all_fitness_over_time = sa.run()
    end = time.time()
    sa.plot_all_fitness_over_time(all_fitness_over_time)
    

    for instance_number, makespan, solving_time in results:
        print(f"Instance {instance_number}: makespan = {makespan}, solving time = {solving_time}")

    print('Time elapsed: ', end - start)

main()