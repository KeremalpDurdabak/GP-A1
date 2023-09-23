from modules.Individual import Individual

class Population:
    def __init__(self, problemDefinition):
        self.problemDefinition = problemDefinition
        self.individuals = []

    def create_population(self):
        # Initialize a population full of Individuals
        for _ in range(self.problemDefinition.population_count):
            new_individual = Individual(self.problemDefinition)
            new_individual.create_individual()
            self.individuals.append(new_individual)

    def addIndividual(self, new_individual):
        if len(self.individuals) < self.problemDefinition.population_count:
            self.individuals.append(new_individual)
        else:
            print("Cannot add more individuals. Population limit reached.")

    def removeIndividual(self, individual_to_remove):
        gap = self.problemDefinition.population_count - len(self.individuals)
        if gap < self.problemDefinition.gap_num:
            try:
                self.individuals.remove(individual_to_remove)
            except ValueError:
                print("Individual not found in the population.")
        else:
            print("Cannot remove more individuals. Gap limit reached.")

    def removePopulationGap(self):
        # Sort the individuals by their fitnessScore in ascending order
        self.individuals.sort(key=lambda x: x.fitnessScore)
        
        # Remove the worst individuals based on gap_num
        del self.individuals[:self.problemDefinition.gap_num]

    def replacePopulationGap(self, new_individuals):
        # Calculate the current gap size
        current_gap = self.problemDefinition.population_count - len(self.individuals)
        
        # Determine the number of new individuals to add
        num_new_individuals = min(current_gap, len(new_individuals))
        
        # Add the new individuals to fill the gap
        self.individuals.extend(new_individuals[:num_new_individuals])
