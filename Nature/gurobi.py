import gurobipy as gp
from gurobipy import GRB
import numpy as np

# ========================== 数据定义 ==========================
data = {
    "taskNum": 5,
    "providerNum": 3,
    "providerRep": [0.388, 0.833, 0.695],
    "providerAbility": [26, 62, 82],
    "providerPrice": [
        [12, 49, 59, 11, 37],
        [33, 29, 52, 29, 24],
        [63, 39, 53, 34, 34]
    ],
    "providerL": [487, 72, 120],
    "providerReliability": [0, 0, 0],
    "providerEnergyCost": [0.307, 0.042, 0.605],
    "providerParam": [
        [0.051, 0.827, 0.959, 0.577],
        [0.507, 0.873, 0.911, 0.83],
        [0.866, 0.146, 0.929, 0.263]
    ],
    "budget": 186.0,
    "taskCost": [64, 49, 33, 49, 62],
    "taskdeadlines": [32, 69, 63, 39, 99],
    "taskbudgets": [95, 99, 68, 62, 64],
    "taskResources": [6, 9, 1, 6, 7],
    "taskabilities": [95, 76, 85, 67, 94],
    "taskTime": [
        [2, 7, 9, 3, 5],
        [7, 2, 5, 2, 9],
        [8, 3, 1, 5, 9]
    ],
    "edges": [[0, 1], [0, 4], [0, 3], [4, 2]],
    "deadline": 200
}

tasks = range(data["taskNum"])
providers = range(data["providerNum"])
TB = sum(data["taskbudgets"])

# ========================== 预计算 A[i,j] ==========================
# A[i,j] = 0.2*(time_sat + cost_sat + providerRep + (providerReliability+providerEnergyCost)/2)
#         + 0.5 * (平均(providerParam))
A = {}
for i in tasks:
    d = data["taskdeadlines"][i]
    b = data["taskbudgets"][i]
    for j in providers:
        t = data["taskTime"][j][i]
        c = data["providerPrice"][j][i]
        time_sat = 1 if t <= d else d / t
        cost_sat = b / c if c <= b else 0
        rep = data["providerRep"][j]
        rel_eng = (data["providerReliability"][j] + data["providerEnergyCost"][j]) / 2
        param_sat = np.mean(data["providerParam"][j])
        A[(i, j)] = 0.2 * (time_sat + cost_sat + rep + rel_eng) + 0.5 * param_sat

# ========================== 构造模型 ==========================
model = gp.Model("TaskAssignment_MultiObj")

# 允许非凸（用于分段线性约束）
model.Params.NonConvex = 2

# 决策变量： x[i,j] = 1 表示任务 i 分配给服务商 j
x = model.addVars(tasks, providers, vtype=GRB.BINARY, name="x")

# 每个任务仅分配给一个服务商
model.addConstrs((gp.quicksum(x[i, j] for j in providers) == 1 for i in tasks), name="AssignTask")

# 服务商资源上限约束
model.addConstrs(
    (gp.quicksum(x[i, j] * data["taskResources"][i] for i in tasks) <= data["providerL"][j]
     for j in providers),
    name="Capacity"
)

# 网络约束
model.addConstrs(
    (gp.quicksum(x[i, j] for j in providers) <= 1 for i in tasks),
    name="Network"
)
model.addConstrs(
    (gp.quicksum(x[i, j] for i in tasks) <= 1 for j in providers),
    name="Network2"
)
model.addConstrs(
    (gp.quicksum(x[i, j] for i in data["edges"][k]) <= 1 for k in range(len(data["edges"]))),
    name="Network3"
)

# 必须完成前面的任务才能做后面的
for i in range(data["taskNum"]):
    for j in range(i + 1, data["taskNum"]):
        if (i, j) in data["edges"]:
            model.addConstr(x[i, j] <= x[j, i], name=f"Precedence_{i}_{j}")

# ========================== 定义满意度表达式 ==========================
# 分配部分满意度
assign_sat_expr = gp.quicksum(A[(i, j)] * x[i, j] for i in tasks for j in providers)

# 服务商剩余资源满意度：0.2 * (1 - (已分配资源 / providerL))
res_sat_expr = gp.quicksum(
    0.2 * (1 - (gp.quicksum(x[i, j] * data["taskResources"][i] for i in tasks) / data["providerL"][j]))
    for j in providers
)

total_satisfaction = assign_sat_expr + res_sat_expr

# ========================== 时间成本满意度 ==========================
# 总任务完成时间与总成本
T_expr = gp.quicksum(x[i, j] * data["taskTime"][j][i] for i in tasks for j in providers)
C_expr = gp.quicksum(x[i, j] * data["providerPrice"][j][i] for i in tasks for j in providers)

# 新增变量 T，并添加约束使 T == T_expr，方便分段线性近似
T = model.addVar(name="T")
model.addConstr(T == T_expr, name="TotalTimeDef")

# 使用分段线性近似 st = exp(-0.1*(deadline - T))（当 T<=deadline）
f = model.addVar(lb=0, ub=1, name="f")
T_min = sum(min(data["taskTime"][j][i] for j in providers) for i in tasks)
T_max = data["deadline"]
breakpoints_T = [T_min, (T_min+T_max)/3, (T_min+T_max)*2/3, T_max]
breakpoints_f = [np.exp(-0.1*(data["deadline"]-t)) for t in breakpoints_T]
model.addGenConstrPWL(T, f, breakpoints_T, breakpoints_f, name="PWL_st")

# 成本折扣
sc_expr = (TB - C_expr) / TB

satisfaction_tc = (f + sc_expr) / 2

# ========================== 设置多目标 ==========================
# 目标1: 最大化 total_satisfaction
model.ModelSense = GRB.MAXIMIZE
model.setObjectiveN(total_satisfaction, index=0, priority=2, name="TotalSatisfaction")

# 目标2: 最大化 satisfaction_tc
model.setObjectiveN(satisfaction_tc, index=1, priority=1, name="TimeCostSatisfaction")

# ========================== 求解 ==========================
model.optimize()

# ========================== 输出结果 ==========================
if model.status == GRB.Status.OPTIMAL or model.status == GRB.Status.SUBOPTIMAL:
    print("最优任务分配方案：")
    for i in tasks:
        for j in providers:
            if x[i, j].X > 0.5:
                print(f"任务 {i} 分配给服务商 {j}")
    print("\n目标函数值：")
    print(f"Total Satisfaction = {total_satisfaction.getValue()}")
    print(f"TimeCost Satisfaction = {satisfaction_tc.getValue()}")
    print(f"Total Time = {T_expr.getValue()}, Total Cost = {C_expr.getValue()}")
else:
    print("未找到最优解")
