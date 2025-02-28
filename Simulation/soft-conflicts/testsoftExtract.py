import re

# 读取原始txt文件
with open('ts_soft.txt', 'r') as infile:
    lines = infile.readlines()

# 初始化存储迭代次数和软冲突数的列表
iterations2 = []
soft_conflicts2 = []

# 正则表达式提取Iter和Soft Conflicts值
for line in lines:
    match = re.search(r'Iter=(\d+),.*Soft Conflicts: (\d+)', line)
    if match:
        iterations2.append(int(match.group(1)))
        soft_conflicts2.append(int(match.group(2)))

# 将结果写入新文件
with open('ts_data_output.txt', 'w') as outfile:
    outfile.write(f"iterationsTS = {iterations2}\n")
    outfile.write(f"soft_conflictsTS = {soft_conflicts2}\n")

print("数据提取并保存完毕！")