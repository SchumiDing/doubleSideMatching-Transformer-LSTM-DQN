import torch
from torch import nn
import sympy as sp
import numpy as np
from copy import deepcopy
from attNet import netAttention

class Net(nn.Module):
    def __init__(self, input_dim, output_dim, task_num, provider_num, net,device="cpu"): 
        '''
            input_dim = provider_num
            output_dim = task_num*provider_num
        '''
        super().__init__()

        # attention machine
        self.attentions = nn.Sequential(
            nn.MultiheadAttention(embed_dim=provider_num, num_heads=1),
            nn.MultiheadAttention(embed_dim=provider_num, num_heads=1),
            nn.MultiheadAttention(embed_dim=provider_num, num_heads=1),
            # netAttention(net, provider_num, (task_num, task_num), kernel_size=3, device=device),
            # netAttention(net, provider_num, (task_num, task_num), kernel_size=3, device=device),
            # netAttention(net, provider_num, (task_num, task_num), kernel_size=3, device=device),
        )
        
        self.klinear = nn.Sequential(
            nn.Linear(input_dim,input_dim).to(device),
            nn.Linear(input_dim,input_dim).to(device),
            nn.Linear(input_dim,input_dim).to(device),
        )
        
        self.vlinear = nn.Sequential(
            nn.Linear(input_dim,input_dim).to(device),
            nn.Linear(input_dim,input_dim).to(device),
            nn.Linear(input_dim,input_dim).to(device),
        )
        
        # T attention machine
        self.Tattentions = nn.Sequential(
            nn.MultiheadAttention(embed_dim=task_num, num_heads=1).to(device),
            nn.MultiheadAttention(embed_dim=task_num, num_heads=1).to(device),
            nn.MultiheadAttention(embed_dim=task_num, num_heads=1).to(device),
        )
        
        self.Tklinear = nn.Sequential(
            nn.Linear(task_num,task_num).to(device),
            nn.Linear(task_num,task_num).to(device),
            nn.Linear(task_num,task_num).to(device),
        )
        
        self.Tvlinear = nn.Sequential(
            nn.Linear(task_num,task_num).to(device),
            nn.Linear(task_num,task_num).to(device),
            nn.Linear(task_num,task_num).to(device),
        )
        
        # first linear layer
        self.linear = nn.Sequential(
            nn.Linear(provider_num*task_num,2*provider_num*task_num),
            nn.ReLU(),
            nn.Linear(2*provider_num*task_num,2*provider_num*task_num),
            nn.ReLU(),
            nn.Linear(2*provider_num*task_num,provider_num*task_num),
            nn.ReLU()
        )
        

        # degorate layer
        self.degorate = nn.Sequential(
            nn.Linear(provider_num*task_num,2*provider_num*task_num),
            nn.ReLU(),
            nn.Linear(2*provider_num*task_num,2*provider_num*task_num),
            nn.ReLU(),
            nn.Linear(2*provider_num*task_num,provider_num),
        )
        self.derograteT = nn.Sequential(
            nn.Linear(provider_num*task_num,2*provider_num*task_num),
            nn.ReLU(),
            nn.Linear(2*provider_num*task_num,2*provider_num*task_num),
            nn.ReLU(),
            nn.Linear(2*provider_num*task_num,task_num),
        )
        self.decision = nn.Sequential(
            nn.Linear(provider_num+task_num*2,(provider_num+task_num)*3//2),
            nn.ReLU(),
            nn.Linear((provider_num+task_num)*3//2,(provider_num+task_num)*3//2),
            nn.ReLU(),
            nn.Linear((provider_num+task_num)*3//2,(provider_num+task_num)*3//2),
            nn.ReLU(),
            nn.Linear((provider_num+task_num)*3//2,provider_num),
        )
        
        # final linear layer with current position added
        self.final = nn.Sequential(
            nn.Linear(provider_num+output_dim,output_dim),
            nn.Sigmoid()
        )
        
        # lstm components
        self.hidden_state = [torch.zeros(output_dim).unsqueeze(0).to(device)]
        self.cell_state = [torch.zeros(output_dim).unsqueeze(0).to(device)]
        self.forgetDoor = nn.Sequential(
            nn.Linear(provider_num+output_dim, output_dim),
            nn.Sigmoid()
        )
        self.inputDoor1 = nn.Sequential(
            nn.Linear(provider_num+output_dim, output_dim),
            nn.Sigmoid()
        )
        self.inputDoor2 = nn.Sequential(
            nn.Linear(provider_num+output_dim, output_dim),
            nn.Tanh()
        )
        self.outputDoor = nn.Sequential(
            nn.Linear(provider_num+output_dim, output_dim),
            nn.Sigmoid()
        )
        
        self.criteria = nn.MSELoss().to(device)
        self.task_num = task_num
        self.provider_num = provider_num
        
        # init weights
        for m in self.linear.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.degorate.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.final.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.forgetDoor.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.inputDoor1.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.inputDoor2.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.outputDoor.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.klinear.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.vlinear.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.attentions.modules():
            if isinstance(m, nn.MultiheadAttention):
                nn.init.xavier_normal_(m.in_proj_weight)
                nn.init.zeros_(m.in_proj_bias)
                nn.init.xavier_normal_(m.out_proj.weight)
                nn.init.zeros_(m.out_proj.bias)
        
    def attention(self, x):
        for at, kl, vl in zip(self.attentions, self.klinear, self.vlinear):
            k = kl(x)
            v = vl(x)
            x, _ = at(x, k, v)
        return x
    
    def Tattention(self, x):
        x = x.T
        for at, kl, vl in zip(self.Tattentions, self.Tklinear, self.Tvlinear):
            k = kl(x)
            v = vl(x)
            x, _ = at(x, k, v)
        x = x.T
        return x
    
    def forward(self,x,pos,p,prehc):
        x = self.attention(x)
        x_t = self.derograteT(x.flatten())
        x = self.Tattention(x)
        x = x.flatten()
        x = x.unsqueeze(0) 
        
        x = self.linear(x)
        
        x_p = self.degorate(x)
        
        x = self.decision(torch.cat([x_p,x_p,pos],dim=1))
        
        self.hidden_state.append(None)
        self.cell_state.append(None)
        
        remember = self.forgetDoor(torch.cat([x,self.hidden_state[prehc]],dim=1))
        input1 = self.inputDoor1(torch.cat([x,self.hidden_state[prehc]],dim=1))
        input2 = self.inputDoor2(torch.cat([x,self.hidden_state[prehc]],dim=1))
        output = self.outputDoor(torch.cat([x,self.hidden_state[prehc]],dim=1))
        
        self.cell_state[-1] = (remember * self.cell_state[prehc] + input1 * input2).clone().detach()
        out = output * torch.tanh(self.cell_state[-1]).clone().detach()
        self.hidden_state[-1] = out.clone().detach()
        hcV = len(self.hidden_state)-1
        x = self.final(torch.cat([x,out],dim=1))
        
        return x , hcV
    
    def choose_action(self, y):
        args = y[0].argmax()
        sense = ""
        arg = []
        if args < self.provider_num * self.task_num:
            sense = "Or"
            arg = y[0: self.provider_num * self.task_num]
        else:
            sense = "And"
            arg = y[self.provider_num * self.task_num:self.provider_num * self.task_num*2]
        return sense, arg
        
    
    def training_step(self,x,pos,y):
        y_hat = self(x,pos)
        loss = self.criteria(y_hat, y)
        return loss
    
    def configure_optimizers(self,optimizer,scheduler):
        self.optimizer = optimizer
        self.scheduler = scheduler
        return [self.optimizer], [self.scheduler]


class point:
    def __init__(self):
        self.parent = []
        self.children = []
        self.sense = "And" # AND, OR sense
        self.loc = 0
        self.provider = 0
        self.task = 0
        self.finished = False
        self.time = None


class network:
    def __init__(self, deadlines, budgets, Rs, abilities, cost,\
        input_dim, output_dim, provider_num, task_num, edges, device="cpu"):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.points = []
        for j in range(task_num):
            p = self.add_point(T=10)
            p.deadline = deadlines[j]
            p.budget = budgets[j]
            p.R = Rs[j]
            p.ability = abilities[j]
            p.cost = cost[j]
        for e in edges:
            self.add_edge(self.points[e[0]], self.points[e[1]], 1)
        self.edges = edges
        self.model = Net(input_dim, output_dim, task_num, provider_num, self,device).to(device)
        self.provider_num = provider_num
        self.task_num = task_num
        self.constrains = []
        self.device = device
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        self.model.configure_optimizers(optimizer, scheduler)
        self.critical_path = -1
        self.paths = [0 for i in range(task_num)]
        self.lam = 0.1
        self.lam = torch.tensor(self.lam).to(device)
        self.omega = [0.2,0.2,0.2,0.2,0.2]
        self.provider = torch.tensor([0 for i in range(provider_num)]).to(device)
        self.providerL = []
        # self.alpha = 0.5/self.task_num
        self.alpha = 0.1
        self.alphaC = 0.1
        self.alpha = torch.tensor(self.alpha).to(device)
        self.alphaC = torch.tensor(self.alphaC).to(device)
        self.providerW = torch.tensor([0 for i in range(provider_num)]).to(device)
        self.pathe = [0 for i in range(task_num)]
        self.edges = edges
        self.M = 1000000

    def add_point(self,T):
        p = point()
        self.points.append(p)
        p.T = T
        p.loc = len(self.points) - 1
        return p
    
    def set_beginning(self, point):
        self.beginning = point
        return point
    
    def forward(self, x, point, p, prehc):
        y, hc = self.model.forward(x, point, p, prehc)
        return y, hc
    
    def proceed(self, x, point): # calculate the choice of this step
        '''
        x: current state of the task match with providers
        point: current task
        '''
        if point.loc != -1:
            pos = torch.zeros(self.task_num)
            # pos[point.loc] = 1
            pos = pos.unsqueeze(0).to(self.device)
        else:
            pos = torch.zeros(self.task_num).unsqueeze(0).to(self.device)
        y, hcV = self.forward(x, pos, point, point.hc)
        sense, args = self.model.choose_action(y)
        provider = []
        task = []
        points = []
        choose_V = 0
        if sense == "Or":
            maxV = -1
            ag = None
            for c in point.children:
                if c.finished:
                    continue
                # for p in c.parent:
                #     if not p.finished:
                #         continue
                if maxV < torch.max(y[0][c.loc*self.provider_num:c.loc*self.provider_num+self.provider_num]):
                    maxV = torch.max(y[0][c.loc*self.provider_num:c.loc*self.provider_num+self.provider_num])
                    ag = c
            if ag == None:
                return None, None, None, None, None
            maxV = -1
            provid = 0
            for i in range(self.provider_num):
                if maxV < y[0][ag.loc*self.provider_num+i]:
                    maxV = y[0][ag.loc*self.provider_num+i]
                    provid = i
            if maxV == -1:
                return None, None, None, None, None
            provider = provid
            self.provider[provider] += int(ag.R)
            self.providerW[provider] += 1
            task = ag.loc
            points = ag
            ag.finished = True
            ag.provider = provider
            ag.sense = "Or"
            choose_V = [(y[0][self.provider_num*task+provider]).clone()]
        else:
            provider = []
            task = []
            points = []
            choose_V = []
            for c in point.children:
                if c.finished:
                    continue
                # for p in c.parent:
                #     if not p.finished:
                #         continue
                maxV = -1
                provid = 0
                for i in range(self.provider_num):
                    if maxV < y[0][self.provider_num*self.task_num+c.loc*self.provider_num+i]:
                        maxV = y[0][self.provider_num*self.task_num+c.loc*self.provider_num+i]
                        provid = i
                if maxV == -1:
                    continue
                provider.append(provid)
                self.provider[provider[-1]] += int(c.R)
                self.providerW[provider[-1]] += 1
                task.append(c.loc)
                points.append(c)
                c.provider = provider[-1]
                c.sense = "And"
                c.hc = hcV
                c.finished = True
                choose_V.append(y[0][self.provider_num*self.task_num+self.provider_num * task[-1] + provider[-1]].clone())
            if provider == []:
                return None, None, None, None, None
        return provider, task, points, sense, choose_V
    
    def train_proceed(self, x, point, epislon=0.9): # calculate the choice of this step
        '''
        x: current state of the task match with providers
        point: current task
        '''
        pos = torch.zeros(self.task_num).to(self.device).unsqueeze(0)
        pos[0][point.loc] = 1
        y, hcV = self.forward(x, pos, point, point.hc)
        sense, _ = self.model.choose_action(y)
        if np.random.rand() < epislon:
            orAnd = np.random.choice(["Or", "And"])
        else:
            orAnd = sense
            pass
        tasks = []
        providers = []
        choose = []
        points = []
        if orAnd == "Or":
            chs = []
            for c in point.children:
                chs.append(c)
            if chs == []:
                return None, None, None, None, None
            c = np.random.choice(chs)
            pros = list(range(self.provider_num))
            # for i in range(self.provider_num):
            #     if self.provider[i] + c.R >= self.providerL[i]:
            #         continue
            #     if self.providerPrice[c.provider][c.loc] > c.budget:
            #         continue
            #     pros.append(i)
            if pros == []:
                return None, None, None, None, None
            provider = np.random.choice(pros)
            self.provider[provider] += int(c.R)
            self.providerW[provider] += 1
            c.finished = True
            c.provider = provider
            c.sense = "Or"
            c.hc = hcV
            tasks = [c.loc]
            providers = [provider]
            points = c
            choose = [y[0][c.loc*self.provider_num+provider]]
        else:
            tasks = []
            providers = []
            choose = []
            for c in point.children:
                pros = list(range(self.provider_num))
                # for i in range(self.provider_num):
                #     if self.provider[i] + c.R >= self.providerL[i]:
                #         continue
                #     if self.providerPrice[c.provider][c.loc] > c.budget:
                #         continue
                #     pros.append(i)
                if pros == []:
                    continue
                provider = np.random.choice(pros)
                self.provider[provider] += int(c.R)
                self.providerW[provider] += 1
                c.finished = True
                c.provider = provider
                c.sense = "And"
                c.hc = hcV
                tasks.append(c.loc)
                providers.append(provider)
                points.append(c)
                choose.append(y[0][self.provider_num*self.task_num+c.loc*self.provider_num+provider])
            if tasks == []:
                return None, None, None, None, None
        return providers, tasks, points, orAnd, choose
    
    def search(self, startpoint): # search the whole network, choose with proceed()
        '''
        startpoint: the beginning point of the network, which is ahead of the point 0
        '''
        
        points = [startpoint]
        calcuated = []
        x_list = []
        x = torch.zeros(self.task_num, self.provider_num).to(self.device)
        x_list.append(x)
        temp = torch.clone(x)
        
        while len(points) > 0:
            p = points.pop(0)
            if p.children == []:
                continue
            provider, task, point, sense, choose = self.train_proceed(x_list[p.L], p)
            if provider == None:
                continue
            if sense == "Or":
                points.append(point)
                point.sense = "Or"
                point.provider = provider[0]
                point.L = len(x_list)
                temp = torch.clone(x)
                temp = temp.clone()
                temp[task[0], provider[0]] = 1
                x_list.append(temp)
                calcuated.append(choose)
            else:
                temp = torch.clone(x)
                temp = temp.clone()
                for pr, t, po in zip(provider, task, point):
                    temp = temp.clone()
                    points.append(po)
                    po.L = len(x_list)
                    po.sense = "And"
                    po.provider = pr
                    temp[t, pr] = 1
                    x_list.append(temp)
                    calcuated.append(choose)
            p.finished = True
        return calcuated, x_list

    def search_train(self, startpoint, epislon=0.9): # search the whole network, choose with train_proceed()
        '''
        startpoint: the beginning point of the network, which is ahead of the point 0
        '''
        points = [startpoint]
        calcuated = []
        x_list = []
        x = torch.zeros(self.task_num, self.provider_num).to(self.device)
        x_list.append(x)
        temp = torch.clone(x)
        
        while len(points) > 0:
            p = points.pop(0)
            if p.children == []:
                continue
            provider, task, point, sense, choose = self.train_proceed(x_list[p.L], p, epislon=epislon)
            if provider == None:
                continue
            if sense == "Or":
                points.append(point)
                point.sense = "Or"
                point.provider = provider[0]
                point.L = len(x_list)
                temp = torch.clone(x)
                temp = temp.clone()
                temp[task[0], provider[0]] = 1
                x_list.append(temp)
                calcuated.append(choose)
            else:
                temp = torch.clone(x)
                temp = temp.clone()
                for pr, t, po in zip(provider, task, point):
                    temp = temp.clone()
                    points.append(po)
                    po.L = len(x_list)
                    po.sense = "And"
                    po.provider = pr
                    temp[t, pr] = 1
                    x_list.append(temp)
                    calcuated.append(choose)
            p.finished = True
        return calcuated, x_list

    def criticalpath(self, p, time): # calculate the critical path
        pT = 0
        if p.loc != -1:
            if self.paths[p.loc] > time:
                self.paths[p.loc] = time
            # cR = (1-torch.exp(self.alpha*1)) if self.providerW[p.provider] > 1 else 1
            # pT = self.taskTime[p.provider][p.loc]*cR
            pT = self.taskTime[p.provider][p.loc]
            if self.pathe[p.loc] < time+pT:
                self.pathe[p.loc] = time+pT
        else:
            pT = 0
        for c in p.children:
            if c.finished:
                self.criticalpath(c, time+pT)
        if len(p.children) == 0:
            if self.critical_path < time+pT:
                self.critical_path = time+pT
        return
    
    def traceCriticalPath(self, p, time, critical_path): # trace the critical path
        pT = 0
        
        if p.loc != -1:
            if self.paths[p.loc] < time:
                self.paths[p.loc] = time
            # cR = (1-torch.exp(self.alpha*1)) if self.providerW[p.provider] > 1 else 1
            # pT = self.taskTime[p.provider][p.loc]*cR
            pT = self.taskTime[p.provider][p.loc]
            if self.pathe[p.loc] < time+pT:
                self.pathe[p.loc] = time+pT
        else:
            pT = 0
        critical_path.append((p.loc, p.provider, time, pT))
        if len(p.children) == 0:
            if self.critical_path < time+pT:
                self.critical_path = time+pT
                self.crPath = critical_path.copy()
        for c in p.children:
            if c.finished:
                self.traceCriticalPath(c, time+pT, critical_path)

        critical_path.pop(-1)
        return      
    
    def objv(self, x): # calculate the objective value
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

    def training_step(self, startpoint,state=0): # do one search with search_train, calculate objective value, and then calculate loss
        '''
        startpoint: the beginning point of the network, which is ahead of the point 0
        '''
        epislon = np.exp(-state/1000000)
        calcuated, x = self.search_train(startpoint, epislon)
        objv, obj = self.objv(x[-1])
        
        startpoint.finished = True
        startpoint.sense = "Or"
        loss_sum = 0
        self.model.optimizer.zero_grad()
        for cs in calcuated:
            l = torch.tensor(len(cs)).to(self.device)
            for c in cs:
                single_loss = self.model.criteria(c, objv)
                loss_sum += single_loss
        loss_sum.backward(retain_graph=True)
        self.model.optimizer.step()
        self.model.scheduler.step()

        return loss_sum, obj
        
    
    def add_restrition(self, new_constrainf):
        self.constrains.append(new_constrainf)
        return

    def add_punishment_to_objv(self, objv, x):
        x = torch.tensor(x).to(self.device)
        if sum(self.provider > self.providerL):
            objv -= self.M * (self.provider - self.providerL)
        cost = 0
        for p in self.points:
            if p.ability > self.providerAbility[p.provider]:
                objv -= self.M * (p.ability-self.providerAbility[p.provider])/len(self.points)
            if p.budget < self.providerPrice[p.provider][p.loc] and self.paths[p.loc] > self.deadline:
                objv -= self.M/len(self.points)
            if self.critical_path > self.deadline:
                objv -= self.M * (self.critical_path-self.deadline)/len(self.points)
            cost += self.providerPrice[p.provider][p.loc]
        
        
        return objv
    
    def add_edge(self, parent, child, edge):
        parent.children.append(child)
        child.parent.append(parent)
        return parent, child
    
    def return_children(self, parent):
        return parent.children, parent.children_edge
    
    def return_parent(self, child):
        return child.parent, child.parent_edge
    
    def init_points(self):
        for p in self.points:
            p.finished = False
            p.provider = -1
            p.sense = "And"
            p.time = None
        return

    def init_net(self):
        self.init_points()
        self.provider = torch.tensor([0 for i in range(self.provider_num)]).to(self.device)
        self.paths = [0 for i in range(self.task_num)]
        self.pathe = [0 for i in range(self.task_num)]
        self.providerW = [0 for i in range(self.provider_num)]
        self.critical_path = 0
        self.model.hidden_state = [torch.zeros(self.output_dim).unsqueeze(0).to(self.device)]
        self.model.cell_state = [torch.zeros(self.output_dim).unsqueeze(0).to(self.device)]
        
        return

device = "cpu"
datafile = "smalldata.json"
def process_bar(i, epoch, loss, taskN, objv):
    # os.system('cls' if os.name == 'nt' else 'clear')
    print("|"*int(i/epoch*100)+" "*int((epoch-i)/epoch*100)+"\r{}/{} loss:{}".format(i, epoch, loss/taskN))

if __name__ == '__main__':
    torch.manual_seed(4244)
    np.random.seed(4244)
    # torch.autograd.set_detect_anomaly(True)
    import numpy as np
    import json

        