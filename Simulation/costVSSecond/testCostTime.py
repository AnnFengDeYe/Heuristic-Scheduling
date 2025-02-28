import re

def extract_data(input_file, output_file):
    """
    从文本文件中提取数据，并按指定格式保存到另一个文件。

    Args:
        input_file: 输入的txt文件路径。
        output_file: 输出的txt文件路径。
    """

    second_list = []
    best_cost_list = []
    previous_time = -1  # 初始时间设置为-1，确保第一行数据一定会被记录
    previous_best_cost = None

    with open(input_file, 'r') as f_in:
        for line in f_in:
            # 使用正则表达式匹配 Best Cost 和 Time 字段
            match_best_cost = re.search(r'Best Cost: (\d+)', line)
            match_time = re.search(r'Time: (\d+\.\d+)s', line)

            if match_best_cost and match_time:
                best_cost = int(match_best_cost.group(1))
                time_seconds = round(float(match_time.group(1)))  # 四舍五入

                # 检查Best Cost或Time是否与前一行相同
                if best_cost != previous_best_cost or time_seconds != previous_time:
                    if len(second_list) > 0 and (best_cost == best_cost_list[-1] or time_seconds == second_list[-1]):
                        # 如果任何一个相同，覆盖上一行
                        second_list[-1] = time_seconds
                        best_cost_list[-1] = best_cost
                    else:
                        # 否则，添加新行
                        second_list.append(time_seconds)
                        best_cost_list.append(best_cost)
                    
                    previous_time = time_seconds
                    previous_best_cost = best_cost


    # 将数据写入输出文件
    with open(output_file, 'w') as f_out:
        f_out.write(f"Second = {second_list}\n")
        f_out.write(f"BestCost = {best_cost_list}\n")


# # 示例用法
# with open("input.txt", "w") as f:
#     f.write("Iter 1, Best Cost: 1021914, Hard Conflicts: 101, Soft Conflicts: 12015, Time: 0.16s\n")
#     f.write("Iter 2, Best Cost: 852009, Hard Conflicts: 84, Soft Conflicts: 12093, Time: 0.31s\n")
#     f.write("Iter 3, Best Cost: 810346, Hard Conflicts: 80, Soft Conflicts: 10426, Time: 0.43s\n")
#     f.write("Iter 4, Best Cost: 579890, Hard Conflicts: 57, Soft Conflicts: 9947, Time: 0.57s\n")
#     f.write("Iter 5, Best Cost: 514399, Hard Conflicts: 50, Soft Conflicts: 14449, Time: 0.70s\n")
#     f.write("Iter 6, Best Cost: 459733, Hard Conflicts: 45, Soft Conflicts: 9778, Time: 0.82s\n")
#     f.write("Iter 7, Best Cost: 399659, Hard Conflicts: 39, Soft Conflicts: 9698, Time: 0.96s\n")
#     f.write("Iter 8, Best Cost: 399517, Hard Conflicts: 39, Soft Conflicts: 9556, Time: 1.09s\n")

extract_data("../originalDataset/ts_timetable_output.txt", "ts_output.txt")


with open("ts_output.txt", "r") as f:
    print(f.read())