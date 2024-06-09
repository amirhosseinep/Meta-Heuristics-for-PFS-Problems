import numpy as np
import random
import time as tm
import matplotlib.pyplot as plt
import pandas as pd
import sys 

class GeneticAlgorithm:
    def __init__(self, problem_type="PFSP", population_size=100, stopping_criteria=5, mutation_rate=0.1, crossover_rate=0.8):
        self.problem_type = problem_type
        self.stopping_criteria = stopping_criteria if stopping_criteria else 1000
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population_size = population_size
        self.num_jobs = 20
        self.num_machines = 5

    def initialize_population(self, num_individuals, num_jobs, num_instances):
        """Initialize the population with random permutations."""
        population = []
        for _ in range(num_individuals):
            individual = [np.random.permutation(num_jobs) for _ in range(num_instances)]
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
        num_instances = len(parent1)
        child1, child2 = [], []

        for i in range(num_instances):
            num_jobs = len(parent1[i])
            instance_child1, instance_child2 = [-1] * num_jobs, [-1] * num_jobs

            cx_point1 = np.random.randint(0, num_jobs)
            cx_point2 = np.random.randint(0, num_jobs)
            if cx_point2 < cx_point1:
                cx_point1, cx_point2 = cx_point2, cx_point1

            instance_child1[cx_point1:cx_point2] = parent1[i][cx_point1:cx_point2]
            instance_child2[cx_point1:cx_point2] = parent2[i][cx_point1:cx_point2]

            for j in list(range(cx_point2, num_jobs)) + list(range(0, cx_point2)):
                if parent2[i][j] not in instance_child1:
                    instance_child1[instance_child1.index(-1)] = parent2[i][j]
                if parent1[i][j] not in instance_child2:
                    instance_child2[instance_child2.index(-1)] = parent1[i][j]

            child1.append(instance_child1)
            child2.append(instance_child2)

        return child1, child2

    def mutate(self, individual):
        for i in range(len(individual)):
            if np.random.rand() < self.mutation_rate:
                idx1, idx2 = np.random.choice(len(individual[i]), size=2, replace=False)
                individual[i][idx1], individual[i][idx2] = individual[i][idx2], individual[i][idx1]
        return individual

    def order_crossover(self, parent1, parent2):
        """Perform Order Crossover to produce offspring."""
        num_instances = len(parent1)
        child = []

        for i in range(num_instances):
            num_jobs = len(parent1[i])
            start_idx = random.randint(0, num_jobs - 1)
            end_idx = random.randint(start_idx, num_jobs - 1)

            instance_child = [-1] * num_jobs
            instance_child[start_idx:end_idx + 1] = parent1[i][start_idx:end_idx + 1]

            parent2_genes = [gene for gene in parent2[i] if gene not in instance_child]
            child_idx = end_idx + 1
            for gene in parent2_genes:
                if child_idx == num_jobs:
                    child_idx = 0
                if instance_child[child_idx] == -1:
                    instance_child[child_idx] = gene
                    child_idx += 1

            child.append(instance_child)

        return child

    def evaluate_fitness(self, population, instance_data):
        """Evaluate fitness of each individual in the population."""
        fitness_values = []
        for individual in population:
            total_makespan = 0
            for i in range(len(individual)):
                makespan = self.calculate_makespan(individual[i], instance_data[i])
                total_makespan += makespan
            fitness_values.append(total_makespan)
        return fitness_values
        
    def calculate_makespan(self, sequence, instance_data):
        """Calculate makespan of a sequence."""
        t = 0
        num_machines = self.num_machines
        num_orders = self.num_jobs
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
                        t = end_time[machine-1][j] + instance_data[machine][j]
                    else:
                        t += instance_data[machine][j]
                    end_time[machine][j] = t
        return t

    def run(self, instance_data):
        """Run the Genetic Algorithm."""
        num_instances = len(instance_data)
        num_jobs = len(instance_data[0][0])
        population = self.initialize_population(self.population_size, num_jobs, num_instances)

        best_fitness_per_instance = [float('inf')] * num_instances
        no_improvement_count = 0
        best_individual_per_instance = [None] * num_instances
        fitness_over_time_per_instance = [[] for _ in range(num_instances)]

        for generation in range(1, self.stopping_criteria + 1):
            fitness_values = self.evaluate_fitness(population, instance_data)
            for i in range(num_instances):
                min_fitness = min(fitness_values[i::num_instances])
                best_index = fitness_values.index(min_fitness)

                if min_fitness < best_fitness_per_instance[i]:
                    best_fitness_per_instance[i] = min_fitness
                    best_individual_per_instance[i] = population[best_index // num_instances][i]
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                fitness_over_time_per_instance[i].append(best_fitness_per_instance[i])

            if no_improvement_count >= self.stopping_criteria:
                break

            selected_individuals = self.roulette_wheel_selection(population, fitness_values)

            offspring_population = []
            for parent1, parent2 in zip(selected_individuals[::2], selected_individuals[1::2]):
                if np.random.rand() < self.crossover_rate:
                    child = self.order_crossover(parent1, parent2)
                    offspring_population.append(child)
                else:
                    offspring_population.extend([parent1, parent2])

            mutated_population = [self.mutate(individual) for individual in offspring_population]

            population = self.elitism(population, mutated_population, instance_data)

        return best_individual_per_instance, fitness_over_time_per_instance, best_fitness_per_instance
    
    def elitism(self, population, mutated_population, instance_data):
        """Apply elitism to select the best individuals for the next generation."""
        fitness_original = self.evaluate_fitness(population, instance_data)
        fitness_mutated = self.evaluate_fitness(mutated_population, instance_data)

        combined_population = list(zip(population, fitness_original)) + list(zip(mutated_population, fitness_mutated))
        combined_population.sort(key=lambda x: x[1])
        new_population = [individual for individual, _ in combined_population[:len(population)]]
        return new_population

def load_data(file_path):
    """Load data from a file."""
    num_machines_list = []
    num_orders_list = []
    data_matrices = []
    df = pd.DataFrame()

    with open(file_path, 'r') as file:
        num_instances = int(file.readline())

        for _ in range(num_instances):
            instance_number = int(file.readline())
            num_orders = int(file.readline())
            num_machines = int(file.readline())

            num_orders_list.append(num_orders)
            num_machines_list.append(num_machines)

            data_matrix = np.zeros((num_machines, num_orders + 1), dtype=int)
            data_matrix[:, 0] = instance_number

            for order in range(1, num_orders+1):
                for machine in range(num_machines):
                    time_on_machine = int(file.readline())
                    data_matrix[machine, order] = time_on_machine

            data_matrices.append(data_matrix)
            df = pd.concat([df, pd.DataFrame(data_matrix)])

    df.to_csv('full_data.csv', index=True, header=False)
    return num_instances, num_machines_list, num_orders_list, data_matrices

def main():
    num_instances, num_machines_list, num_orders_list, data_matrix = load_data('Fs_20.txt')

    pfsp_ga = GeneticAlgorithm(problem_type="PFSP", population_size=100, stopping_criteria=100, mutation_rate=0.1, crossover_rate=0.9)

    instance_data_list = [data_matrix[i][:, 1:] for i in range(num_instances)]

    results = []
    for i in range(num_instances):
        total_time_starting = tm.time()
        best_permutation, fitness_over_time, best_fitness = pfsp_ga.run([instance_data_list[i]])
        total_time_ending = tm.time()

        solving_time = round((total_time_ending - total_time_starting), 3)

        makespan_i = pfsp_ga.calculate_makespan(best_permutation[0], instance_data=instance_data_list[i])
        results.append([i+1, makespan_i, solving_time])
        print("Instance " + str(i+1) + " - Makespan: " + str(makespan_i) + ", Solving Time: " + str(solving_time))

    # Save results to a CSV file
    df = pd.DataFrame(results, columns=['Instance', 'Makespan', 'Solving Time'])
    df.to_csv('results.csv', index=False)

if __name__ == '__main__':
    main()