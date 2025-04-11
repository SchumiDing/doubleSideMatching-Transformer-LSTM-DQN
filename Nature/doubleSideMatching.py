import torch,json
import numpy as np
from xtmqn import network, point
from copy import deepcopy

device = "cpu"
datafile = "data_100_100.json"

def process_bar(i, epoch, loss, taskN, objv, t):
    # os.system('cls' if os.name == 'nt' else 'clear')
    print("T"+"|"*int(i/epoch*100+0.5)+" "*int((epoch-i)/epoch*100+0.5)+f"||{i}/{epoch} loss:{loss:7.5} Time:{t:.5f}s")

if __name__ == "__main__":
    torch.manual_seed(4244)
    np.random.seed(4244)
    # torch.autograd.set_detect_anomaly(True)
    import numpy as np
    import json
    det = json.load(open(datafile,"r"))
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
    cnt = 10
    # net.model.load_state_dict(torch.load("test.pth")[0])
    nets = []
    draw = {
    }
    bestSolution = {
    }
    for i in range(netsNum):
        net_new = network(deadlines, budgets, Rs, abilities, cost,\
                          providerN,providerN*taskN*2,providerN,taskN, edges,device=device)
        net_new.taskTime = det["taskTime"]
        net_new.rep = det["providerRep"]
        net_new.deadline = det["deadline"]
        net_new.providerAbility = det["providerAbility"]
        net_new.providerL = torch.tensor(det["providerL"]).to(device)
        net_new.providerPrice = det["providerPrice"]
        net_new.providerReliability = det["providerReliability"]
        net_new.providerEnergyCost = det["providerEnergyCost"]
        net_new.budget = det["budget"]
        net_new.param = det["providerParam"]
        net_new.critical_path = -1
        net_new.paths = [0 for i in range(taskN)]
        
        startp = point()
        startp.loc = -1
        startp.children = [net_new.points[0]]
        net_new.init_points()
        startp.finished = True
        startp.L = 0
        startp.hc = 0
        net_new.set_beginning(startp)
        nets.append((net_new, startp))
        draw[i] = [[],[],[],[]]
        bestSolution[i] = [-1*net_new.M*100*taskN,-1*net_new.M*100*taskN,None]
    import time
    pretime = time.time()
    preTime = time.time()
    for j in range(epoch):
        if j % (100//(taskN**2)+1) == 0:
            pretime = time.time()
        for i, dt in enumerate(nets):
            net = dt[0]
            startp = dt[1]
            # print('-'*10+f"net {i}"+'-'*10)
            param = [x.clone for x in net.model.parameters()]
            loss, objv = net.training_step(startp,i)
            # print(loss/taskN, objv)
            # print(f"critical path:{net.critical_path}")
            draw[i][0].append((loss).item())

            
            if time.time()-pretime > 20 or True:
                startp.finished = False
                net.init_net()
                pretime = time.time()
                calculated, x = net.search(startp)
                objvt, objs = net.objv(x[-1])
                draw[i][1].append(objs[0].item())
                draw[i][2].append(objs[1].item())
                draw[i][3].append(net.critical_path)
                # torch.save((net.model.state_dict(),loss), "test.pth")
                # print(f"epoch:{j}/{epoch} loss:{loss} objv1:{objs[0]} objv2:{objs[1]}")
                if bestSolution[i][0] < objs[0].item() + objs[1].item():
                    if j == 473:
                        pass
                    # net.critical_path = -1
                    # net.traceCriticalPath(startp, 0, [])
                    bestSolution[i] = [objs[0].item() + objs[1].item(), objs, deepcopy(net.points), j, None , net.critical_path, net.cost]
            # process_bar(j, epoch, loss, taskN, objv)
            startp.finished = False
            net.init_net()
            
        if j % (100//(taskN**2)+1) == 0:
            # torch.save((net.model.state_dict(),loss), "test.pth")
            json.dump(draw, open("draw.json","w"), indent=4)
            process_bar(j, epoch, loss, taskN, objv, time.time()-pretime)
    print(time.time()-preTime)
    torch.save((net.model.state_dict(),loss), "test.pth")
    json.dump(draw, open("draw_attnet.json","w"), indent=4)
    for det in nets:
        net = det[0]
        startp = det[1]
        net.init_net()
        startp.finished = False
        calculated, x = net.search(startp)
        obj, objs = net.objv(x[-1])
        print(objs[0]+objs[1], objs)
        print(net.critical_path)
        for p in net.points:
            print(p.loc, p.provider, p.finished)
        print("----"*3)
    
    print("-"*50)
    print(" "*18+"Best Solution")
    print("-"*50)
    
    for b in bestSolution.keys():
        print(bestSolution[b][0], bestSolution[b][1])
        for p in bestSolution[b][2]:
            print(p.loc, p.provider, p.finished)
        print(f"epoch:{bestSolution[b][3]}")
        print(f"critical path:{bestSolution[b][5]}")
        print(f"cost:{bestSolution[b][6]}")
        print(f"path:{bestSolution[b][4]}")
        print("----"*3)