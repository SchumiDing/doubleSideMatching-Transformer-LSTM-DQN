from xtmqn import point
import numpy as np

finded = []
to_be_finded = []

class netw:
    def __init__(self):
        self.points = []
    
    def add_point(self):
        p = point()
        self.points.append(p)
        p.loc = len(self.points)-1
        p.parents = []
        return p
    
    def add_edge(self, p1, p2):
        p1.children.append(p2)
        p2.parents.append(p1)
    
    

    def critical_path(self):
        for p in self.points:
            p.T = np.random.randint(1, 10)
            
        long_path = [0]*len(self.points)
        
        startp = point()
        startp.T = 0
        startp.children = [self.points[0]]
        startp.loc = -1
        
        po = [(startp,0)]
        

        
        while po != []:
            p,l = po.pop(0)
            for c in p.children:
                po.append((c, l+c.T))
            if p.loc != -1:
                long_path[p.loc] = max(long_path[p.loc], l+p.T)
        
        print(long_path)
        
        return
            
        

    
def dfs(p, finded, to_be_finded):
    num = 0
    finded.append(p)
    if p.children == []:
        return 1
    for c in p.children:
        num += dfs(c, finded, to_be_finded)
        
    return num+1

def find_end(p):
    if len(p.children) == 0:
        return p, 0
    for c in p.children:
        to_be_finded.append(c)
        return find_end(c)[0], find_end(c)[1]+1

def graph_generation(point_num):
    net = netw()
    for i in range(point_num):
        t = net.add_point()
    
    edges = []
    
    global finded, to_be_finded
    finded = []
    to_be_finded = []
    p = net.points[0]
    
    while dfs(p, finded, to_be_finded) < point_num:
        t1 = np.random.choice(finded)
        t2 = np.random.choice(net.points)
        while t2 in finded:
            t2 = np.random.choice(net.points)
        if t1 == t2:
            continue
        finded.append(t2)
        net.points.pop(net.points.index(t2))
        net.add_edge(t1, t2)
        edges.append((t1.loc, t2.loc))
        finded = []
        to_be_finded = []
    
    end_point, l = find_end(net.points[0])
    return edges, end_point, l, net



if __name__ == "__main__":
    task_num = 100
    provider_num = 100
    np.random.seed(4244)
    edges, _, _ , n= graph_generation(task_num)
    # n.critical_path()
    
    deadlines = np.random.randint(10, 100, task_num)
    budgets = np.random.randint(50, 100, task_num)
    Rs = np.random.randint(1, 10, task_num)
    abilities = np.random.randint(10, 100, task_num)
    providerRep = np.random.randint(0, 1000, provider_num)/1000
    
    deadline = 200
    providerAbility = np.random.randint(10, 100, provider_num)
    providerL = np.random.randint(50, 500, provider_num)
    taskTime = np.random.randint(1, 10, (provider_num, task_num))
    taskCost = np.random.randint(20, 70, task_num)
    
    providerPrice = np.random.randint(10, 70, (provider_num, task_num))
    providerTaskNum = np.random.randint(1, 10, provider_num)
    
    providerReliability = np.random.randint(0,1,provider_num)
    providerEnergyCost = np.random.randint(0, 1000, provider_num)/1000
    providerParam = np.random.randint(0, 1000, (provider_num,4))/1000
    
    budget = budgets.sum()
    
    det = {
        "taskNum": task_num, # Number of tasks
        "providerNum": provider_num, # Number of providers
        "providerRep": providerRep.tolist(), # Reputation of providers
        "providerAbility": providerAbility.tolist(), # Ability of providers
        "providerPrice": providerPrice.tolist(), # Price of providers
        "providerL": providerL.tolist(), # Resource Limitation of providers
        "providerReliability": providerReliability.tolist(), # Reliability of providers
        "providerEnergyCost": providerEnergyCost.tolist(), # Energy cost of providers
        "providerParam": providerParam.tolist(),
        "budget": int(budget), # Budget of the whole task network
        # '''
        # First column: Pc
        # Second column: Pt
        # Third column: Ce
        # Fourth column: Ig
        # '''
        "taskCost": taskCost.tolist(), # Cost of tasks
        "taskdeadlines": deadlines.tolist(), # Deadline of tasks
        "taskbudgets": budgets.tolist(), # Budget of tasks
        "taskResources": Rs.tolist(), # Resource occupied by each task
        "taskabilities": abilities.tolist(), # Ability requirements of each task
        "taskTime": taskTime.tolist(), # Time required for each provider to complete each task
        "edges": edges, # Edges of the task network
        "deadline": deadline # Deadline of the whole task network
    }
    
    import json
    json.dump(det, open(f"data_{task_num}_{provider_num}.json","w"), indent=4)