from generate_net import netw
from xtmqn import point
from torch import nn
import torch, json
import numpy as np

class network(netw):
    def __init__(self, deadlines, budgets, Rs, abilities, cost,\
                 providerN, taskN, providerNum, taskNum, edges, device,\
                taskTime, rep, deadline, providerAbility,\
                providerL, providerPrice, providerReliability,\
                    providerEnergyCost, budget, param, omega, lam):
        # super(network,self).__init__(providerN, taskN)
        self.device = device
        self.deadlines = deadlines
        self.budgets = budgets
        self.Rs = Rs
        self.abilities = abilities
        self.cost = cost
        self.providerNum = providerNum
        self.taskNum = taskNum
        self.edges = edges
        self.taskTime = taskTime
        self.rep = rep
        self.deadline = deadline
        self.providerAbility = providerAbility
        self.providerL = providerL
        self.providerPrice = providerPrice
        self.providerReliability = providerReliability
        self.providerEnergyCost = providerEnergyCost
        self.budget = budget
        self.param = param
        self.omega = omega
        self.lam = lam
        self.points = []
        self.x = torch.zeros((self.taskNum, self.providerNum)).to(self.device)
        self.pathe = [0 for i in range(taskNum)]
    
    def init_points(self):
        for i in range(self.taskNum):
            p = point()
            p.loc = i
            p.children = []
            p.finished = False
            p.deadline = self.deadlines[i]
            p.budget = self.budgets[i]
            p.Rs = self.Rs[i]
            p.abilities = self.abilities[i]
            p.cost = self.cost[i]
            p.provider = -1
            p.providerL = 0
            p.providerPrice = 0
            self.points.append(p)
        self.paths = [0 for i in range(self.taskNum)]
        return

    def objv(self): # calculate the objective value
        '''
        x: the final state of the network
        '''
        # calculate length of critical path
        self.critical_path = -1
        self.criticalpath(self.beginning, 0)
        budget = self.budget
        cost = 0
        for p in self.points:
            if p.finished:
                cost += self.providerPrice[p.provider][p.loc]
        if cost == 0:
            return 0, 0   
        self.cost = cost
        satisfaction = torch.tensor((budget - cost) / budget).to(self.device)
        if self.deadline > self.critical_path:
            satisfaction += torch.exp(torch.tensor(-self.lam * (self.deadline-self.critical_path))).to(self.device)
        satisfaction /= 2
        
        sat1 = satisfaction.clone()

        for p in self.points:
            if p.finished:
                sT = 0
                if p.deadline<self.paths[p.loc]:
                    sT = p.deadline/self.paths[p.loc]
                else:
                    sT = 1
                sP = p.budget/self.providerPrice[p.provider][p.loc] if p.budget>self.providerPrice[p.provider][p.loc] else 0
                satisfaction += self.omega[0] * sT + self.omega[1] * sP + self.omega[2] * (self.providerL[p.provider]-self.provider[p.provider])/self.providerL[p.provider]\
                    + self.omega[3] * self.rep[p.provider] + self.omega[4] *  (self.providerReliability[p.provider] + self.providerEnergyCost[p.provider])/2
                satisfaction += ((self.providerPrice[p.provider][p.loc]-p.cost)/self.providerPrice[p.provider][p.loc] + self.param[p.provider][0]+\
                    self.param[p.provider][1]+ self.param[p.provider][2]+self.param[p.provider][3])/5
        
        # satisfaction = self.add_punishment_to_objv(satisfaction, x)
        
        satisfactionS = nn.Sigmoid()(satisfaction)
        return satisfactionS, (sat1, satisfaction-sat1)

class geneticAlgorithm():
    def __init__(self, taskNum, providerNum, device):
        self.device = device
        self.taskNum = taskNum
        self.providerNum = providerNum
        self.population = []
        self.bestSolution = None
        self.size = 100
        self.finess = torch.zeros((self.size,)).to(self.device)
    
    def create_individual(self, deadlines, budgets, Rs, abilities, cost,\
                 providerN, taskN, providerNum, taskNum, edges, device,\
                     taskTime, rep, deadline, providerAbility,\
                     providerL, providerPrice, providerReliability,\
                     providerEnergyCost, budget, param, omega, lam):
        tempn = network(deadlines, budgets, Rs, abilities, cost,\
                    providerN, taskN, providerNum, taskNum, edges, device,\
                    taskTime, rep, deadline, providerAbility,\
                    providerL, providerPrice, providerReliability,\
                        providerEnergyCost, budget, param, omega, lam)
        tempn.init_points()
        tempn.beginning = point()
        tempn.beginning.loc = 0
        tempn.beginning.children = [tempn.points[0]]
        tempn.beginning.finished = True
        return tempn
    
    def initialize_population(self, size, deadlines, budgets, Rs, abilities, cost,\
                 providerN, taskN, providerNum, taskNum, edges, device,\
                     taskTime, rep, deadline, providerAbility,\
                     providerL, providerPrice, providerReliability,\
                     providerEnergyCost, budget, param, omega, lam):
        for _ in range(size):
            individual = self.create_individual(deadlines, budgets, Rs, abilities, cost,\
                    providerN, taskN, providerNum, taskNum, edges, device,\
                    taskTime, rep, deadline, providerAbility,\
                    providerL, providerPrice, providerReliability,\
                    providerEnergyCost, budget, param, omega, lam)
            self.population.append(individual)
            individual.init_points()
            individual.x = torch.zeros((self.taskNum, self.providerNum)).to(self.device)
            task_provider = torch.randint(0, self.providerNum+1, (self.taskNum,)).to(self.device)
            for i, p in enumerate(task_provider):
                if p == 0:
                    continue
                individual.x[i][p-1] = 1
            
    def fitness(self, individual):
        # Calculate the fitness of an individual
        # This is a placeholder, replace with actual fitness calculation
        return individual.objv()

    def process(self):
        for pop in self.population:
            for p in pop.points:
                for i,v in enumerate(pop.x[p.loc]):
                    if i == p.loc:
                        continue
                    if v == 1:
                        p.provider = i
    
    def calAllFitness(self):
        fitness_values = []
        for individual in self.population:
            fitness_values.append(self.fitness(individual))
        return fitness_values

    def selection(self):
        # Select individuals based on their fitness
        fitness_values = self.calAllFitness()
        alValue = zip(self.population, fitness_values)
        alValue = sorted(alValue, key=lambda x: x[1], reverse=True)
        selected = alValue[:self.size]
        self.population = [ind[0] for ind in selected]
        self.bestSolution = selected[0][0].x
        self.bestFitness = selected[0][1]
        return self.bestSolution, self.bestFitness
    
    def crossover(self, parent1, parent2):
        # Perform crossover between two parents to create a child
        x1 = parent1.x
        x2 = parent2.x
        cross_point = np.random.randint(0, self.taskNum)
        al = np.random.rand()
        
        x1_new = torch.cat((x1[:cross_point], x2[cross_point:]), dim=0)
        x2_new = torch.cat((x2[:cross_point], x1[cross_point:]), dim=0)

        child1 = self.create_individual(parent1.deadlines, parent1.budgets, parent1.Rs, parent1.abilities, parent1.cost,\
                    parent1.providerNum, parent1.taskNum, parent1.providerNum, parent1.taskNum, parent1.edges, self.device,\
                    parent1.taskTime, parent1.rep, parent1.deadline, parent1.providerAbility,\
                    parent1.providerL, parent1.providerPrice, parent1.providerReliability,\
                    parent1.providerEnergyCost, parent1.budget, parent1.param, parent1.omega, parent1.lam)
        child2 = self.create_individual(parent2.deadlines, parent2.budgets, parent2.Rs, parent2.abilities, parent2.cost,\
                    parent2.providerNum, parent2.taskNum, parent2.providerNum, parent2.taskNum, parent2.edges, self.device,\
                    parent2.taskTime, parent2.rep, parent2.deadline, parent2.providerAbility,\
                    parent2.providerL, parent2.providerPrice, parent2.providerReliability,\
                    parent2.providerEnergyCost, parent2.budget, parent2.param, parent2.omega, parent2.lam)
        child1.x = x1_new
        child2.x = x2_new
        child1.init_points()
        child2.init_points()
        return child1, child2
    
    def mutation(self, individual):
        # Perform mutation on an individual
        mutation_rate = 0.01
        for i in range(self.taskNum):
            if np.random.rand() < mutation_rate:
                individual.x[i,:] = 0
                individual.x[i, np.random.randint(0, self.providerNum)] = 1
    
    def init(self):
        for pop in self.population:
            pop.init_points()
            pop.critical_path = -1
            pop.pathe = [0 for i in range(self.taskNum)]
    def run(self, generations):
        for generation in range(generations):
            self.init()
            self.process()
            self.selection()
            new_population = []
            for i in range(0, len(self.population), 2):
                for j in range(i , len(self.population)):
                    if i == j:
                        continue
                    if np.random.rand() < (2 - pow(2,(((i+j)/2)/self.size))):
                        parent1 = self.population[i]
                        parent2 = self.population[j]
                        child1, child2 = self.crossover(parent1, parent2)
                        new_population.append(child1)
                        new_population.append(child2)
            
            for individual in new_population:
                self.mutation(individual)
            
            self.population = new_population
device = "cpu"
if __name__ == "__main__":
    torch.manual_seed(4244)
    np.random.seed(4244)
    # torch.autograd.set_detect_anomaly(True)
    det = json.load(open("data_5_3.json","r"))
    taskN = det["taskNum"]
    providerN = det["providerNum"]
    edges = det["edges"]
    
    deadlines = np.array(det["taskdeadlines"])
    budgets = np.array(det["taskbudgets"])
    Rs = np.array(det["taskResources"])
    abilities = np.array(det["taskabilities"])
    providerRep = np.array(det["providerRep"])
    cost = np.array(det["taskCost"])
    
    epoch = 800
    netsNum = 1
    # netsNum = 0

    taskTime = det["taskTime"]
    rep = det["providerRep"]
    deadline = det["deadline"]
    providerAbility = det["providerAbility"]
    providerL = torch.tensor(det["providerL"]).to(device)
    providerPrice = det["providerPrice"]
    providerReliability = det["providerReliability"]
    providerEnergyCost = det["providerEnergyCost"]
    budget = det["budget"]
    param = det["providerParam"]
    omega = [0.2, 0.2, 0.2, 0.2, 0.2]
    lam = 0.5
    

    model = geneticAlgorithm(taskN,providerN,device)
    
    model.initialize_population(100, deadlines, budgets, Rs, abilities, cost,\
                    providerN, taskN, providerN, taskN, edges, device,\
                    taskTime, rep, deadline, providerAbility,\
                    providerL, providerPrice, providerReliability,\
                    providerEnergyCost, budget, param, omega, lam)
    
    
    model.run(100)
    print("Best Solution: ", model.bestSolution)
    