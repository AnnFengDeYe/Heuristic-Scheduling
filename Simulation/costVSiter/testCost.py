import re

# 读取原始txt文件
with open('../originalDataset/ts_timetable_output.txt', 'r') as f:
    lines = f.readlines()

# 初始化列表
iterations = []
best_cost = []

# 定义正则表达式匹配Iter和Best Cost
iter_pattern = r"Iter=(\d+)"
best_cost_pattern = r"Best Cost: (\d+)"

# 初始化变量来存储最后一个最佳成本
last_best_cost = None
last_iteration = None

# 从每一行中提取数据
for line in lines:
    iter_match = re.search(iter_pattern, line)
    best_cost_match = re.search(best_cost_pattern, line)
    
    if iter_match and best_cost_match:
        current_iteration = int(iter_match.group(1))
        current_best_cost = int(best_cost_match.group(1))
        
        # 如果Best Cost变化了，就保存之前的值
        if current_best_cost != last_best_cost:
            if last_best_cost is not None:
                iterations.append(last_iteration)
                best_cost.append(last_best_cost)
        
        # 更新当前的Best Cost和Iteration
        last_best_cost = current_best_cost
        last_iteration = current_iteration

# 最后一行的数据也需要保存
if last_best_cost is not None:
    iterations.append(last_iteration)
    best_cost.append(last_best_cost)

# 保存到新的txt文件
with open('ts_cost_iter.txt', 'w') as f:
    f.write(f"iterations = {iterations}\n")
    f.write(f"BestCost = {best_cost}\n")

print("Data extraction and saving completed successfully!")