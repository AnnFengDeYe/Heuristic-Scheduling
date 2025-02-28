import re

# 定义处理文件的函数
def process_file(input_filename, output_filename):
    # 打开输入文件和输出文件
    with open(input_filename, 'r') as infile, open(output_filename, 'w') as outfile:
        last_soft_conflict = None  # 用于存储上一行的Soft Conflicts值

        for line in infile:
            # 检查行是否以'Iter'开头
            if line.startswith('Iter'):
                # 使用正则表达式提取Soft Conflicts的值
                match = re.search(r"Soft Conflicts: (\d+)", line)
                if match:
                    current_soft_conflict = int(match.group(1))  # 获取当前行的Soft Conflicts值

                    # 如果当前的Soft Conflicts与上一行相同，丢弃上一行数据，保留当前行
                    if current_soft_conflict == last_soft_conflict:
                        continue
                    
                    # 写入当前行到输出文件
                    outfile.write(line)

                    # 更新上一行的Soft Conflicts值
                    last_soft_conflict = current_soft_conflict
            else:
                # 如果行不以'Iter'开头，结束处理
                break

# 调用函数处理文件
input_filename = '../originalDataset/ts_timetable_output.txt'  # 输入文件名
output_filename = 'ts_soft.txt'  # 输出文件名
process_file(input_filename, output_filename)

print("文件处理完成，结果已输出到 output.txt")