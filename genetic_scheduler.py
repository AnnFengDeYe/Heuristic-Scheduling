from csvData.csv_strings import course_info_csv, student_courses_csv
import time
import pandas as pd
import random
import sys
import re
from collections import defaultdict
from io import StringIO

# --------------------------
# ga_scheduler
# --------------------------

# --------------------------
# 1) 教室名称映射
# --------------------------
classroom_mapping = {
    1: "Swann Lecture Theater",
    2: "Lecture Theater A - JCMB",
    3: "Lecture Theater B - JCMB",
    4: "Lecture Theater C - JCMB",
    5: "Alder Lecture Theater - Nucleus",
    6: "Elm Lecture Theater - Nucleus",
    7: "Oak Lecture Theater - Nucleus",
    8: "Larch Lecture Theater - Nucleus",
    9: "Yew Lecture Theater - Nucleus",
    10: "Lecture Theatre 1 - Ashworth Labs"
}


def get_workshop_name(ws_id):
    return f"ws{ws_id}"


# --------------------------
# 2) 读取CSV并解析
#   student_courses.csv + course_info.csv
# --------------------------
def parse_course_info(course_info_str):
    """
    从本地CSV文件读取数据:
      - student_courses.csv
      - course_info.csv
    """
    df = pd.read_csv(StringIO(course_info_str))
    lecturer_set = df['Lecturer'].unique().tolist()
    lecturer_to_id = {lec: i for i, lec in enumerate(lecturer_set)}

    course_dict = {}
    for idx, row in df.iterrows():
        cname = row['Course name'].strip()
        lecturer = row['Lecturer'].strip() if pd.notna(row['Lecturer']) else 'Unknown'
        lecture_str = row['Lecture']
        ws_str = row['ws']

        def parse_freq(freq_str):
            pattern = r'^(\d+)\*(\d+)(单|双)?'
            m = re.match(pattern, str(freq_str).strip())
            if m:
                t_per_wk = int(m.group(1))
                hrs = int(m.group(2))
                sd = m.group(3)
                if sd == '单':
                    weeks_type = 'single'
                elif sd == '双':
                    weeks_type = 'double'
                else:
                    weeks_type = 'every'
                return (t_per_wk, hrs, weeks_type)
            else:
                return (1, 1, 'every')

        lec_parsed = parse_freq(lecture_str)
        ws_parsed = parse_freq(ws_str)
        tid = lecturer_to_id.get(lecturer, -1)

        course_dict[cname] = {
            'lecturer': lecturer,
            'teacher_id': tid,
            'lecture_freq': lec_parsed,
            'ws_freq': ws_parsed,
            'orig_course': cname
        }
    return course_dict, lecturer_to_id

def parse_student_info(stu_info_str):
    df = pd.read_csv(StringIO(stu_info_str))
    import ast
    students_courses = []
    for idx, row in df.iterrows():
        sid = row['Student ID']
        course_list_str = row['Semester 1 Courses']
        try:
            course_list = ast.literal_eval(course_list_str)
        except:
            course_list = []
        students_courses.append({'id': sid, 'courses': course_list})
    return students_courses


# --------------------------
# 3) 常量设置
# --------------------------
SINGLE_WEEKS = [1, 3, 5, 7, 9, 11]
DOUBLE_WEEKS = [2, 4, 6, 8, 10]
lecture_classrooms = list(range(1, 11))
workshop_classrooms = list(range(1, 21))
LECTURE_SLOTS_PER_DAY = 3
WORKSHOP_SLOTS_PER_DAY = 5
DAYS_PER_WEEK = 5
HARD_CONFLICT_PENALTY = 9999
MAX_WS_CAPACITY = 30


# --------------------------
# 4) 构建课程->学生映射
# --------------------------
def build_course_to_students(students_courses):
    c2s = defaultdict(set)
    for sid, rec in enumerate(students_courses):
        for c in rec['courses']:
            c2s[c].add(sid)
    return c2s


# --------------------------
# 5) 并行Workshop拆分
# --------------------------
def split_workshop_sessions(course_dict, course2stu, max_capacity=30):
    new_cd = dict(course_dict)
    new_c2s = dict(course2stu)

    for c in list(course_dict.keys()):
        stud_count = len(course2stu.get(c, set()))
        ws_times, ws_hrs, ws_type = course_dict[c]['ws_freq']
        if stud_count > max_capacity and ws_times > 0:
            n_sessions = (stud_count // max_capacity)
            remainder = stud_count % max_capacity
            if remainder > 0:
                n_sessions += 1

            all_students = sorted(list(course2stu[c]))
            index = 0
            sessions_names = []
            base_lecturer = course_dict[c]['lecturer']
            base_tid = course_dict[c]['teacher_id']
            base_ws_freq = course_dict[c]['ws_freq']
            base_orig = course_dict[c]['orig_course']

            for i_sess in range(n_sessions):
                subname = f"{c}_WS{i_sess}"
                sessions_names.append(subname)
                chunk = all_students[index:index + max_capacity]
                index += max_capacity

                new_cd[subname] = {
                    'lecturer': base_lecturer,
                    'teacher_id': base_tid,
                    'lecture_freq': (0, 0, 'every'),
                    'ws_freq': base_ws_freq,
                    'orig_course': c
                }
                new_c2s[subname] = set(chunk)

            new_cd[c] = {
                'lecturer': base_lecturer,
                'teacher_id': base_tid,
                'lecture_freq': course_dict[c]['lecture_freq'],
                'ws_freq': (0, 0, 'every'),
                'orig_course': base_orig
            }
    return new_cd, new_c2s


# --------------------------
# 6) 生成初始解
# --------------------------
def random_initial_solution(course_dict, course2stu):
    sol = {'lecture': {}, 'workshop': {}}

    for c, info in course_dict.items():
        lec_times, lec_hrs, lec_wtype = info['lecture_freq']
        ws_times, ws_hrs, ws_wtype = info['ws_freq']

        # Lecture
        if lec_times > 0 and lec_hrs > 0:
            cr = random.choice(lecture_classrooms)
            slots = []

            # 如果是1小时课，确保安排合理
            if lec_hrs == 1:
                for i in range(lec_times):
                    d = random.randint(0, DAYS_PER_WEEK - 1)  # 随机选择一个星期几
                    s = random.randint(0, LECTURE_SLOTS_PER_DAY - 1)  # 随机选择一个时间段

                    # 确保两节课不在同一天
                    while any(slot[0] == d for slot in slots):
                        d = random.randint(0, DAYS_PER_WEEK - 1)  # 重新选择不同的星期几
                    slots.append((d, (s,)))

            # 如果是2小时课，确保不在同一天，并且时段是连续的
            elif lec_hrs == 2:
                for i in range(lec_times):
                    d = random.randint(0, DAYS_PER_WEEK - 1)
                    s = random.randint(0, LECTURE_SLOTS_PER_DAY - 2)  # 确保时段是连续的

                    # 确保两节课不在同一天
                    while any(slot[0] == d for slot in slots):
                        d = random.randint(0, DAYS_PER_WEEK - 1)  # 重新选择不同的星期几
                    slots.append((d, (s, s + 1)))

            # 如果是2*2课，确保不在同一天并且时段连续
            elif lec_hrs == 2 and lec_times == 2:
                for i in range(lec_times):
                    d = random.randint(0, DAYS_PER_WEEK - 1)
                    s = random.randint(0, LECTURE_SLOTS_PER_DAY - 2)  # 确保时段是连续的

                    # 确保两节课不在同一天
                    while any(slot[0] == d for slot in slots):
                        d = random.randint(0, DAYS_PER_WEEK - 1)  # 重新选择不同的星期几
                    slots.append((d, (s, s + 1)))

            sol['lecture'][c] = {
                'classroom': cr,
                'slots': slots,
                'weeks_type': lec_wtype,
                'hours': lec_hrs
            }

        # Workshop
        if ws_times > 0 and ws_hrs > 0:
            cr = random.choice(workshop_classrooms)
            slots = []
            for i in range(ws_times):
                if ws_hrs == 1:
                    d = random.randint(0, DAYS_PER_WEEK - 1)
                    s = random.randint(0, WORKSHOP_SLOTS_PER_DAY - 1)
                    slots.append((d, (s,)))
                else:
                    d = random.randint(0, DAYS_PER_WEEK - 1)
                    s = random.randint(0, WORKSHOP_SLOTS_PER_DAY - 2)
                    slots.append((d, (s, s + 1)))
            sol['workshop'][c] = {
                'classroom': cr,
                'slots': slots,
                'weeks_type': ws_wtype,
                'hours': ws_hrs
            }

    return sol


# --------------------------
# 7) 评估函数 (保持不变)
# --------------------------
def evaluate_solution(sol, course_dict, course2stu):
    hard_conflicts = 0
    soft_conflicts = 0

    usage = defaultdict(list)
    teacher_usage = defaultdict(list)

    def get_weeks(wt):
        if wt == 'single':
            return SINGLE_WEEKS
        elif wt == 'double':
            return DOUBLE_WEEKS
        else:
            return list(range(1, 12))

    for c, cinfo in sol['lecture'].items():
        if c not in course_dict:
            continue
        t_id = course_dict[c]['teacher_id']
        room = cinfo['classroom']
        wtype = cinfo['weeks_type']
        wlist = get_weeks(wtype)
        for (day, slot_tuple) in cinfo['slots']:
            for w in wlist:
                for st in slot_tuple:
                    usage[('lecture', w, day, st, room)].append(c)
                    teacher_usage[(w, day, st)].append((t_id, 'lecture', c))

    for c, cinfo in sol['workshop'].items():
        if c not in course_dict:
            continue
        t_id = course_dict[c]['teacher_id']
        room = cinfo['classroom']
        wtype = cinfo['weeks_type']
        wlist = get_weeks(wtype)
        for (day, slot_tuple) in cinfo['slots']:
            for w in wlist:
                for st in slot_tuple:
                    usage[('workshop', w, day, st, room)].append(c)
                    teacher_usage[(w, day, st)].append((t_id, 'workshop', c))

    for k, clist in usage.items():
        if len(clist) > 1:
            hard_conflicts += len(clist) * (len(clist) - 1) // 2

    for k, tlist in teacher_usage.items():
        count_map = defaultdict(int)
        for (tid, ttype, cname) in tlist:
            if tid >= 0:
                count_map[tid] += 1
        for tid, cnt in count_map.items():
            if cnt > 1:
                hard_conflicts += cnt * (cnt - 1) // 2

    course_times_lec = defaultdict(list)
    for c, cinfo in sol['lecture'].items():
        if c not in course_dict: continue
        lec_weeks = get_weeks(cinfo['weeks_type'])
        for (day, slot_tuple) in cinfo['slots']:
            for w in lec_weeks:
                for st in slot_tuple:
                    course_times_lec[c].append((w, day, st))

    course_times_ws = defaultdict(list)
    for c, cinfo in sol['workshop'].items():
        if c not in course_dict: continue
        ws_weeks = get_weeks(cinfo['weeks_type'])
        for (day, slot_tuple) in cinfo['slots']:
            for w in ws_weeks:
                for st in slot_tuple:
                    course_times_ws[c].append((w, day, st))

    for c_lec in sol['lecture'].keys():
        if c_lec not in course_dict:
            continue

        lec_set = set(course_times_lec[c_lec])

        if c_lec in course_times_ws:
            ws_set = set(course_times_ws[c_lec])
            hard_conflicts += len(lec_set.intersection(ws_set))

        for c_ws in sol['workshop'].keys():
            if c_ws not in course_dict:
                continue
            par = course_dict[c_ws]['orig_course']
            if par == c_lec:
                ws_set2 = set(course_times_ws[c_ws])
                hard_conflicts += len(lec_set.intersection(ws_set2))

    wds_map = defaultdict(list)
    for c, cinfo in sol['lecture'].items():
        if c not in course_dict:
            continue
        wlist = get_weeks(cinfo['weeks_type'])
        for (day, slots) in cinfo['slots']:
            for w in wlist:
                for s in slots:
                    wds_map[(w, day, s)].append(c)

    for c, cinfo in sol['workshop'].items():
        if c not in course_dict:
            continue
        wlist = get_weeks(cinfo['weeks_type'])
        for (day, slots) in cinfo['slots']:
            for w in wlist:
                for s in slots:
                    wds_map[(w, day, s)].append(c)

    for key, cList in wds_map.items():
        n = len(cList)
        if n > 1:
            cSets = [course2stu.get(cn, set()) for cn in cList]
            for i in range(n):
                for j in range(i + 1, n):
                    soft_conflicts += len(cSets[i].intersection(cSets[j]))

    return HARD_CONFLICT_PENALTY * hard_conflicts + soft_conflicts, hard_conflicts, soft_conflicts


# --------------------------
# 8) 遗传算法实现 (新增部分)
# --------------------------
class GeneticScheduler:
    def __init__(self, course_dict, student_mapping,
                 pop_size=50, generations=100,
                 crossover_prob=0.8, mutation_prob=0.2,
                 elite_num=2):
        self.courses = course_dict
        self.student_map = student_mapping
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.elite_num = elite_num
        self.start_time = time.time() # 记录开始时间

    def initialize_population(self):
        return [random_initial_solution(self.courses, self.student_map)
                for _ in range(self.pop_size)]

    def rank_population(self, population):
        return sorted(
            [(sol, evaluate_solution(sol, self.courses, self.student_map)) # 返回包含冲突信息的tuple
             for sol in population],
            key=lambda x: x[1][0] # 根据总成本排序
        )

    def select_parents(self, ranked_pop):
        weights = [1 / (i + 1) for i in range(len(ranked_pop))]
        total = sum(weights)
        probs = [w / total for w in weights]
        parents = random.choices(
            [item[0] for item in ranked_pop],
            weights=probs,
            k=2
        )
        return parents[0], parents[1]

    def crossover(self, parent1, parent2):
        child = {'lecture': {}, 'workshop': {}}
        for course in self.courses:
            if random.random() < 0.5:
                if course in parent1['lecture']:
                    child['lecture'][course] = parent1['lecture'][course].copy()
                if course in parent1['workshop']:
                    child['workshop'][course] = parent1['workshop'][course].copy()
            else:
                if course in parent2['lecture']:
                    child['lecture'][course] = parent2['lecture'][course].copy()
                if course in parent2['workshop']:
                    child['workshop'][course] = parent2['workshop'][course].copy()
        return child

    def mutate(self, individual):
        mutated = {
            'lecture': {c: info.copy() for c, info in individual['lecture'].items()},
            'workshop': {c: info.copy() for c, info in individual['workshop'].items()}
        }

        # 随机选择一个课程进行变异
        courses = list(self.courses.keys())
        target_course = random.choice(courses)

        # 变异lecture安排
        if target_course in mutated['lecture']:
            if random.random() < 0.5:
                mutated['lecture'][target_course]['classroom'] = random.choice(lecture_classrooms)
            if random.random() < 0.5:
                lec_info = self.courses[target_course]['lecture_freq']
                slots = []
                for _ in range(lec_info[0]):
                    day = random.randint(0, DAYS_PER_WEEK - 1)
                    start = random.randint(0, LECTURE_SLOTS_PER_DAY - lec_info[1])
                    slots.append((day, tuple(range(start, start + lec_info[1]))))
                mutated['lecture'][target_course]['slots'] = slots

        # 变异workshop安排
        if target_course in mutated['workshop']:
            if random.random() < 0.5:
                mutated['workshop'][target_course]['classroom'] = random.choice(workshop_classrooms)
            if random.random() < 0.5:
                ws_info = self.courses[target_course]['ws_freq']
                slots = []
                for _ in range(ws_info[0]):
                    day = random.randint(0, DAYS_PER_WEEK - 1)
                    start = random.randint(0, WORKSHOP_SLOTS_PER_DAY - ws_info[1])
                    slots.append((day, tuple(range(start, start + ws_info[1]))))
                mutated['workshop'][target_course]['slots'] = slots

        return mutated

    def evolve(self):
        population = self.initialize_population()
        best_solution = None
        best_fitness = float('inf')
        best_hard_conflicts = float('inf') # 记录最佳硬冲突
        best_soft_conflicts = float('inf') # 记录最佳软冲突


        for generation in range(self.generations):
            ranked = self.rank_population(population)
            current_best = ranked[0]
            current_fitness = current_best[1][0]
            current_hard_conflicts = current_best[1][1]
            current_soft_conflicts = current_best[1][2]


            if current_fitness < best_fitness:
                best_fitness = current_fitness
                best_solution = current_best[0]
                best_hard_conflicts = current_hard_conflicts
                best_soft_conflicts = current_soft_conflicts

            # 保留精英
            new_pop = [item[0] for item in ranked[:self.elite_num]]

            # 生成下一代
            while len(new_pop) < self.pop_size:
                parent_a, parent_b = self.select_parents(ranked)

                # 交叉
                if random.random() < self.crossover_prob:
                    child = self.crossover(parent_a, parent_b)
                else:
                    child = parent_a

                # 变异
                if random.random() < self.mutation_prob:
                    child = self.mutate(child)

                new_pop.append(child)

            population = new_pop
            elapsed_time = time.time() - self.start_time
            print(f"Generation {generation + 1}, Best Cost: {current_fitness}, Hard Conflicts: {current_hard_conflicts}, Soft Conflicts: {current_soft_conflicts}, Time: {elapsed_time:.2f}s")


        return best_solution, best_fitness, best_hard_conflicts, best_soft_conflicts


# --------------------------
# 9) 打印课表 (保持不变)
# --------------------------
def print_schedule(sol, course_dict):
    print("\n================= Final Timetable (Lecture) =================")
    day_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

    def lecture_slot_str(s):
        return ["10-11", "11-12", "12-13"][s]

    def workshop_slot_str(s):
        return ["9-10", "10-11", "11-12", "12-13", "13-14"][s]

    def weeks_type_str(wt):
        return {
            'single': "(Single Week Only)",
            'double': "(Double Week Only)",
            'every': "(Every Week)"
        }[wt]

    for c, cinfo in sol['lecture'].items():
        if c not in course_dict: continue
        cf = course_dict[c]
        if cinfo['hours'] <= 0: continue

        print(f"Course: {c} | Lecturer: {cf['lecturer']}")
        croom = classroom_mapping.get(cinfo['classroom'], f"Room {cinfo['classroom']}")
        print(f"  Classroom => {croom} {weeks_type_str(cinfo['weeks_type'])}")
        for (d, slots) in cinfo['slots']:
            day_str = day_name[d]
            time_str = " & ".join(lecture_slot_str(s) for s in slots)
            print(f"    {day_str}, {time_str}")
        print("-----")

    print("\n================= Final Timetable (Workshop) =================")
    for c, cinfo in sol['workshop'].items():
        if c not in course_dict: continue
        cf = course_dict[c]
        if cinfo['hours'] <= 0: continue

        print(f"Course: {c} | Instructor: {cf['lecturer']}")
        ws_name = get_workshop_name(cinfo['classroom'])
        print(f"  Workshop Room => {ws_name} {weeks_type_str(cinfo['weeks_type'])}")
        for (d, slots) in cinfo['slots']:
            day_str = day_name[d]
            time_str = " & ".join(workshop_slot_str(s) for s in slots)
            print(f"    {day_str}, {time_str}")
        print("-----")


# --------------------------
# 10) 主函数 (修改算法调用)
# --------------------------
def main():
    # Redirect stdout to a file
    original_stdout = sys.stdout
    with open('ga_timetable_output.txt', 'w', encoding='utf-8') as f: # Specify UTF-8 encoding
        sys.stdout = f

        # 解析数据
        course_dict, _ = parse_course_info(course_info_csv)
        students = parse_student_info(student_courses_csv)
        course_student_map = build_course_to_students(students)

        # 拆分workshop
        start_time = time.time()
        modified_courses, modified_mapping = split_workshop_sessions(course_dict, course_student_map)

        # 运行遗传算法
        scheduler = GeneticScheduler(
            modified_courses, modified_mapping,
            pop_size=100,
            generations=15000,  # 减少迭代次数以便更快地看到结果
            crossover_prob=0.7,
            mutation_prob=0.3,
            elite_num=3
        )
        best_sol, best_cost, best_hard, best_soft = scheduler.evolve()

        # 输出结果
        end_time = time.time()
        print(f"\nTotal running time: {end_time - start_time:.2f} seconds")
        print(f"Best solution cost: {best_cost}, Hard Conflicts: {best_hard}, Soft Conflicts: {best_soft}")
        print_schedule(best_sol, modified_courses)

    # Restore stdout
    sys.stdout = original_stdout
    print("Output saved to timetable_output.txt")

if __name__ == "__main__":
    main()