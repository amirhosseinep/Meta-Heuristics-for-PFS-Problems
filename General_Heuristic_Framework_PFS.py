#MNLIH works but need csv and better output.

from abc import ABC, abstractmethod
import time
import time as tm
import csv
import itertools
import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd
import random

class DataLoader:
    def load_data(self, file_path,num_machines,num_orders):
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


class MakespanCalculator:
    def calculate_makespan(self, instance_data, sequence, num_machines, num_jobs):
        #print("$$$")
        #print(sequence)
        t = 0
        num_orders = num_jobs
        num_machines = num_machines
        end_time = [[0 for _ in range(num_orders)] for _ in range(num_machines)]
        for machine in range(num_machines):
            if machine == 0:
                for j in sequence:
                    t += instance_data[machine][j]
                    end_time[machine][j] = t
            else:
                t = end_time[machine-1][sequence[0]] 
                for j in sequence:
                    #print(str(j)+"  "+str(machine))
                    if t < end_time[machine-1][j]:
                        t = end_time[machine-1][j] + instance_data[machine][j]
                    else:
                        t += instance_data[machine][j]
                    end_time[machine][j] = t
        return t

class Heuristic(ABC):
    @abstractmethod
    def execute(self, data_matrix):
        pass

class GeneticAlgorithm(Heuristic):
    def __init__(self):
        self.ga = self._GeneticAlgorithm()

    class _GeneticAlgorithm:
        def __init__(self, problem_type="PFSP", population_size=100, stopping_criteria=5, mutation_rate=0.1, crossover_rate=0.8):
            self.problem_type = problem_type
            self.stopping_criteria = stopping_criteria if stopping_criteria else 1000
            self.mutation_rate = mutation_rate
            self.crossover_rate = crossover_rate
            self.population_size = population_size

        def initialize_population(self, num_individuals, num_jobs):
            """Initialize the population with random permutations."""
            population = []
            for _ in range(num_individuals):
                individual = np.random.permutation(num_jobs)
                population.append(individual)
            return population

        def roulette_wheel_selection(self, population, fitness_values):
            """Select individuals based on roulette wheel selection."""
            total_fitness = sum(fitness_values)
            probabilities = [fitness / total_fitness for fitness in fitness_values]
            selected_indices = np.random.choice(len(population), size=len(population), p=probabilities)
            selected_individuals = [population[i] for i in selected_indices]
            return selected_individuals

        def order_crossover_2children(self, parent1, parent2):
            """Perform Order Crossover (OX) to produce two offspring."""
            #print(parent1)
            #print(parent2)
            num_jobs = len(parent1)
            child1, child2 = [-1]*num_jobs, [-1]*num_jobs

            # Generate two random crossover points
            cx_point1 = np.random.randint(0, num_jobs)
            cx_point2 = np.random.randint(0, num_jobs)
            if cx_point2 < cx_point1:
                cx_point1, cx_point2 = cx_point2, cx_point1

            # Copy the sub-sequences from parents to children
            child1[cx_point1:cx_point2] = parent1[cx_point1:cx_point2]
            child2[cx_point1:cx_point2] = parent2[cx_point1:cx_point2]

            # Fill the remaining positions with the genes from the other parent in their original order
            for i in list(range(cx_point2, num_jobs)) + list(range(0, cx_point2)):
                if parent2[i] not in child1:
                    child1[child1.index(-1)] = parent2[i]
                if parent1[i] not in child2:
                    child2[child2.index(-1)] = parent1[i]

            #print("offspring: ")
            ##print(child1)
            #print(child2)
            #print("--")
            #sys.exit()
            return child1, child2

        def mutate(self, individual):
            if np.random.rand() < self.mutation_rate:
                idx1, idx2 = np.random.choice(len(individual), size=2, replace=False)
                individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
            return individual

        def order_crossover(self, parent1, parent2):
            """Perform Order Crossover to produce offspring."""
            num_jobs = len(parent1)
            # Randomly select the segment indices
            start_idx = random.randint(0, num_jobs - 1)
            end_idx = random.randint(start_idx, num_jobs - 1)
            
            # Initialize offspring with segment from parent1
            child = [-1] * num_jobs
            child[start_idx:end_idx + 1] = parent1[start_idx:end_idx + 1]
            
            # Fill in remaining positions with genes from parent2
            parent2_genes = [gene for gene in parent2 if gene not in child]
            child_idx = end_idx + 1
            for gene in parent2_genes:
                if child_idx == num_jobs:
                    child_idx = 0
                if child[child_idx] == -1:
                    child[child_idx] = gene
                    child_idx += 1
            return child

        def evaluate_fitness(self, population, instance_data):
            """Evaluate fitness of each individual in the population."""
            fitness_values = []
            for individual in population:
                makespan = MakespanCalculator.calculate_makespan(individual, instance_data)
                fitness_values.append(makespan)
            return fitness_values

        def run(self, instance_data):
            """Run the Genetic Algorithm."""
            population = self.initialize_population(self.population_size, num_jobs=len(instance_data[0]))
            
            best_fitness = float('inf')
            no_improvement_count = 0
            best_individual = None
            fitness_over_time = []

            for generation in range(1, self.stopping_criteria + 1):
                fitness_values = self.evaluate_fitness(population, instance_data)
                min_fitness = min(fitness_values)
                best_index = fitness_values.index(min_fitness)

                if min_fitness < best_fitness:
                    best_fitness = min_fitness
                    best_individual = population[best_index]
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                fitness_over_time.append(best_fitness)
                
                if no_improvement_count >= self.stopping_criteria:
                    break

                # Selection
                selected_individuals = self.roulette_wheel_selection(population, fitness_values)
                #print("after selection wheel")
                '''
                # Crossover using order_crossover with 1 child
                offspring_population = []
                for parent1, parent2 in zip(selected_individuals[::2], selected_individuals[1::2]):
                    if np.random.rand() < self.crossover_rate:
                        child1, child2 = self.order_crossover(parent1, parent2)
                        offspring_population.extend([child1, child2])
                    else:
                        offspring_population.extend([parent1, parent2])
                '''
                # Crossover using order_crossover with 2 children
                offspring_population = []
                for parent1, parent2 in zip(selected_individuals[::2], selected_individuals[1::2]):
                    if np.random.rand() < self.crossover_rate:
                        child = self.order_crossover_2children(parent1, parent2)
                        offspring_population.append(child)
                    else:
                        offspring_population.extend([parent1, parent2])

                mutated_population = [self.mutate(list(individual)) if isinstance(individual, tuple) else self.mutate(individual) for individual in offspring_population]
                
                # Flatten the data
                mutated_flattened_list = [list(item) if isinstance(item, list) else item.tolist() for item in mutated_population]
                flattened_list = [list(np.array(item).flatten()) for item in mutated_flattened_list]

                max_values_per_list = 20
                mutation_flat = []

                for sublist in flattened_list:
                    while sublist:
                        mutation_flat.append(sublist[:max_values_per_list])
                        sublist = sublist[max_values_per_list:]

                population = self.elitism(population, mutation_flat, instance_data)
                # Debug print: Uncomment if needed for debugging
                #print("Generation:", generation, "Best fitness:", best_fitness)
            #print("---")
            #print(fitness_over_time)
            #print("---")
            return best_individual, fitness_over_time, best_fitness

        def elitism(self, population, mutated_population, instance_data):
            """Apply elitism to select the best individuals for the next generation."""
            fitness_original = self.evaluate_fitness(population, instance_data)
            fitness_mutated = self.evaluate_fitness(mutated_population, instance_data)
            
            combined_population = list(zip(population, fitness_original)) + list(zip(mutated_population, fitness_mutated))
            combined_population.sort(key=lambda x: x[1])
            new_population = [individual for individual, _ in combined_population[:len(population)]]
            return new_population
    
    def execute(self, data_matrix):
        self.ga.run(data_matrix)

class SimulatedAnnealing(Heuristic):
    def execute(self, data_matrix):
        # Implement your Simulated Annealing logic here
        pass

class MNLI(Heuristic):
    def __init__(self):
        self.first_neighbourhood = self._first_neighbourhood
        self.second_neighbourhood = self._second_neighbourhood
        self.third_neighbourhood = self._third_neighbourhood

    def _first_neighbourhood(self, schedule):
        for i in range(len(schedule) - 1):
            new_schedule = schedule[:]
            new_schedule[i], new_schedule[i+1] = new_schedule[i+1], new_schedule[i]
            #print(schedule)
            #print("jadide")
            #print(new_schedule)
            #print("---")
            yield new_schedule

    def _second_neighbourhood(self, schedule):
        for i, j in itertools.combinations(range(len(schedule)), 2):
            new_schedule = schedule[:]
            new_schedule[i], new_schedule[j] = new_schedule[j], new_schedule[i]
            yield new_schedule

    def _third_neighbourhood(self, schedule):
        for i, j, k in itertools.combinations(range(len(schedule)), 3):
            new_schedule = schedule[:]
            new_schedule[i], new_schedule[j], new_schedule[k] = new_schedule[j], new_schedule[k], new_schedule[i]
            yield new_schedule

    def execute(self, data_matrix, num_machines, num_jobs):
        num_instances = len(data_matrix)
        results = []
        
        for i in range(num_instances):
            start_time = time.time()
            #print(data_matrix[i])
            #print(data_matrix)
            #sys.exit()
            data = data_matrix[i][:,1:]

            # Initialize the schedule with a random permutation of the jobs
            schedule = list(range(len(data[0])))
            np.random.shuffle(schedule)
            #data2 = data_matrix[i]
            makespan = MakespanCalculator.calculate_makespan(self, data, schedule, num_machines, num_jobs)
            
            # Apply the multi-neighbourhood local improvement heuristic
            while True:
                improved = False
                
                # First Neighbourhood
                for new_schedule in self._first_neighbourhood(schedule):
                    #print("first schedule:" + str(new_schedule))
                    #new_makespan = calculate_makespan(new_schedule, data)
                    new_makespan = MakespanCalculator.calculate_makespan(self , data, new_schedule, num_machines, num_jobs)
                    if new_makespan < makespan:
                        schedule = new_schedule
                        makespan = new_makespan
                        improved = True
                        break  # Go back and search in the first neighbourhood again
                
                # Second Neighbourhood (only if no improvement in the first neighbourhood)
                if not improved:
                    for new_schedule in self._second_neighbourhood(schedule):
                        #new_makespan = calculate_makespan(new_schedule, data)
                        new_makespan = MakespanCalculator.calculate_makespan(self, data, new_schedule, num_machines, num_jobs)
                        if new_makespan < makespan:
                            schedule = new_schedule
                            makespan = new_makespan
                            improved = True
                            break  # Go back and search in the first neighbourhood again
                
                # Third Neighbourhood (only if no improvement in the first and second neighbourhoods)
                if not improved:
                    
                    for new_schedule in self._third_neighbourhood(schedule):
                        new_makespan = MakespanCalculator.calculate_makespan(self, data, new_schedule, num_machines, num_jobs)
                        #new_makespan = make_span(data, new_schedule, 5, 20)
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
        print(results)
        return results

class MetaHeuristicFramework:
    def __init__(self, heuristic, file_path, num_machines, num_jobs):
        self.heuristic = heuristic
        self.num_machines = num_machines
        self.num_jobs = num_jobs
        self.file_path = file_path
        self.data_loader = DataLoader()
        self.makespan_calculator = MakespanCalculator()

    def execute(self):
        
        num_instances, num_machines_list, num_orders_list, data_matrix  = self.data_loader.load_data(self.file_path, self.num_machines, self.num_jobs)
        self.heuristic.execute(data_matrix, self.num_machines, self.num_jobs)
        # You might need to adjust the parameters for calculate_makespan based on your data structure
        #self.makespan_calculator.calculate_makespan(data_matrix, sequence, self.num_machines, self.num_jobs)

# Usage:
framework = MetaHeuristicFramework(MNLI(),'Fs_20.txt', num_machines = 5, num_jobs = 20)
framework.execute()
