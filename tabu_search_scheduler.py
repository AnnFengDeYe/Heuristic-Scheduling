from csvData.csv_strings import course_info_csv, student_courses_csv
import time
import sys
import pandas as pd
import random
import re
from collections import defaultdict, deque
from io import StringIO

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
# --------------------------
def parse_course_info(course_info_str):
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

        # 每个课程的记录里再加一个 'orig_course' = cname (自身)
        # 以便后续 parallel session 也能记下这个来源
        course_dict[cname] = {
            'lecturer': lecturer,
            'teacher_id': tid,
            'lecture_freq': lec_parsed,
            'ws_freq': ws_parsed,
            'orig_course': cname  # 缺省是自己
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
    """
    对于 course2stu[c] > max_capacity 的课程, 拆分多个session:
     c+"_WS0", c+"_WS1", ...
    each sub-course:
      - 只保留 workshop freq
      - orig_course = c
      - lecture_freq=(0,0,'every')
      - student集合是对应30人/xx人
    并将c本身的ws_freq设为(0,0,'every')以禁用其原Workshop
    """
    new_cd = dict(course_dict)
    new_c2s = dict(course2stu)

    for c in list(course_dict.keys()):
        stud_count = len(course2stu.get(c, set()))
        ws_times, ws_hrs, ws_type = course_dict[c]['ws_freq']
        if stud_count > max_capacity and ws_times > 0:
            # need parallel
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
            base_orig = course_dict[c]['orig_course']  # c

            for i_sess in range(n_sessions):
                subname = f"{c}_WS{i_sess}"
                sessions_names.append(subname)
                chunk = all_students[index:index + max_capacity]
                index += max_capacity

                new_cd[subname] = {
                    'lecturer': base_lecturer,
                    'teacher_id': base_tid,
                    'lecture_freq': (0, 0, 'every'),  # no lecture
                    'ws_freq': base_ws_freq,
                    'orig_course': c  # 关键:表示它是c衍生
                }
                new_c2s[subname] = set(chunk)

            # disable old workshop
            old_lec = course_dict[c]['lecture_freq']
            old_orig = course_dict[c]['orig_course']
            new_cd[c] = {
                'lecturer': base_lecturer,
                'teacher_id': base_tid,
                'lecture_freq': old_lec,
                'ws_freq': (0, 0, 'every'),
                'orig_course': old_orig
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
# 7) 评估函数: 硬冲突+软冲突
# --------------------------
def evaluate_solution(sol, course_dict, course2stu):
    hard_conflicts = 0
    soft_conflicts = 0

    usage = defaultdict(list)  # (type, w, d, slot, room)->[courses]
    teacher_usage = defaultdict(list)  # (w,d,slot)->[(tid,type,course)]

    def get_weeks(wt):
        if wt == 'single':
            return SINGLE_WEEKS
        elif wt == 'double':
            return DOUBLE_WEEKS
        else:
            return list(range(1, 12))

    # --- Lecture
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

    # --- Workshop
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

    # (1) 教室冲突
    for k, clist in usage.items():
        if len(clist) > 1:
            n = len(clist)
            pairs = n * (n - 1) // 2
            hard_conflicts += pairs

    # (2) 教师冲突
    for k, tlist in teacher_usage.items():
        count_map = defaultdict(int)
        for (tid, ttype, cname) in tlist:
            if tid >= 0:
                count_map[tid] += 1
        for tid, cnt in count_map.items():
            if cnt > 1:
                pairs = cnt * (cnt - 1) // 2
                hard_conflicts += pairs

    # (3) 同门课 lecture-workshop 同时段
    #  a) EXACT same course name => old logic
    #  b) parallel session c' that has orig_course == c
    #       => 也不能跟 c 的 Lecture 同时段

    # 收集每个课程的 lecture times
    course_times_lec = defaultdict(list)
    for c, cinfo in sol['lecture'].items():
        if c not in course_dict: continue
        lec_weeks = get_weeks(cinfo['weeks_type'])
        for (day, slot_tuple) in cinfo['slots']:
            for w in lec_weeks:
                for st in slot_tuple:
                    course_times_lec[c].append((w, day, st))

    # 收集每个课程的 workshop times
    course_times_ws = defaultdict(list)
    for c, cinfo in sol['workshop'].items():
        if c not in course_dict: continue
        ws_weeks = get_weeks(cinfo['weeks_type'])
        for (day, slot_tuple) in cinfo['slots']:
            for w in ws_weeks:
                for st in slot_tuple:
                    course_times_ws[c].append((w, day, st))

    # 做一个 "which course is the parent"?
    # parallel session c' => 'orig_course'=X
    # check overlap with X's lecture
    # or if c' == c => old logic

    for c_lec in sol['lecture'].keys():
        if c_lec not in course_dict:
            continue

        lec_set = set(course_times_lec[c_lec])

        # 1) 先看 c_lec 是否有 workshop 同名 => old logic
        if c_lec in course_times_ws:
            ws_set = set(course_times_ws[c_lec])
            overlap = lec_set.intersection(ws_set)
            hard_conflicts += len(overlap)

        # 2) 再查**所有** workshop c_ws, 看 if c_ws's orig_course == c_lec
        for c_ws in sol['workshop'].keys():
            if c_ws not in course_dict:
                continue
            # 这里记录一下 c_ws 的 parent
            par = course_dict[c_ws]['orig_course']  # who is the original course
            if par == c_lec:
                # overlap => conflict
                ws_set2 = set(course_times_ws[c_ws])
                overlap2 = lec_set.intersection(ws_set2)
                hard_conflicts += len(overlap2)

    # 软冲突(学生撞课)
    wds_map = defaultdict(list)
    # fill wds_map from lecture
    for c, cinfo in sol['lecture'].items():
        if c not in course_dict:
            continue
        wlist = get_weeks(cinfo['weeks_type'])
        for (day, slots) in cinfo['slots']:
            for w in wlist:
                for s in slots:
                    wds_map[(w, day, s)].append(c)
    # fill wds_map from workshop
    for c, cinfo in sol['workshop'].items():
        if c not in course_dict:
            continue
        wlist = get_weeks(cinfo['weeks_type'])
        for (day, slots) in cinfo['slots']:
            for w in wlist:
                for s in slots:
                    wds_map[(w, day, s)].append(c)

    # pairwise common students
    for key, cList in wds_map.items():
        n = len(cList)
        if n > 1:
            cSets = [course2stu.get(cn, set()) for cn in cList]
            for i in range(n):
                for j in range(i + 1, n):
                    common = cSets[i].intersection(cSets[j])
                    soft_conflicts += len(common)

    cost = HARD_CONFLICT_PENALTY * hard_conflicts + soft_conflicts
    return cost, hard_conflicts, soft_conflicts


# --------------------------
# 8) 生成邻域(同原代码)
# --------------------------
def generate_neighbors(sol, course_dict, max_neighbors=10):
    neighbors = []
    all_courses = list(course_dict.keys())

    for _ in range(max_neighbors):
        ttype = random.choice(['lecture', 'workshop'])
        if len(all_courses) == 0:
            continue
        c = random.choice(all_courses)
        # ensure c exist in sol[ttype]
        if ttype == 'lecture':
            if c not in sol['lecture']:
                continue
            current_cinfo = sol['lecture'][c]
        else:
            if c not in sol['workshop']:
                continue
            current_cinfo = sol['workshop'][c]

        new_sol = {
            'lecture': {cc: dict(sol['lecture'][cc]) for cc in sol['lecture']},
            'workshop': {cc: dict(sol['workshop'][cc]) for cc in sol['workshop']}
        }
        new_cinfo = dict(current_cinfo)

        # maybe swap room
        if ttype == 'lecture':
            possible_rooms = lecture_classrooms
        else:
            possible_rooms = workshop_classrooms

        if random.random() < 0.5:
            new_cinfo['classroom'] = random.choice(possible_rooms)

        # maybe swap slots
        hours = new_cinfo['hours']
        slot_list = new_cinfo['slots']
        if random.random() < 0.5 and len(slot_list) > 0:
            new_slots = []
            for i in range(len(slot_list)):
                if hours == 1:
                    d = random.randint(0, DAYS_PER_WEEK - 1)
                    if ttype == 'lecture':
                        s = random.randint(0, LECTURE_SLOTS_PER_DAY - 1)
                        new_slots.append((d, (s,)))
                    else:
                        s = random.randint(0, WORKSHOP_SLOTS_PER_DAY - 1)
                        new_slots.append((d, (s,)))
                elif hours == 2:
                    d = random.randint(0, DAYS_PER_WEEK - 1)
                    if ttype == 'lecture':
                        s = random.randint(0, LECTURE_SLOTS_PER_DAY - 2)
                        new_slots.append((d, (s, s + 1)))
                    else:
                        s = random.randint(0, WORKSHOP_SLOTS_PER_DAY - 2)
                        new_slots.append((d, (s, s + 1)))
                else:
                    new_slots.append(slot_list[i])
            new_cinfo['slots'] = new_slots

        # update
        if ttype == 'lecture':
            new_sol['lecture'][c] = new_cinfo
        else:
            new_sol['workshop'][c] = new_cinfo

        neighbors.append(new_sol)
    return neighbors


# --------------------------
# 9) Tabu Search
# --------------------------
def tabu_search(course_dict, course2stu,
                max_iter=100,
                max_neighbors=10,
                tabu_tenure=10,
                seed=42):
    random.seed(seed)

    current_sol = random_initial_solution(course_dict, course2stu)
    best_sol = current_sol
    best_cost, best_hard, best_soft = evaluate_solution(best_sol, course_dict, course2stu)

    tabu_list = deque()
    tabu_set = set()

    def sol_signature(sol):
        s = []
        for c in sorted(sol['lecture'].keys()):
            ci = sol['lecture'][c]
            s.append(f"L:{c}:{ci['classroom']}:{ci['slots']}")
        for c in sorted(sol['workshop'].keys()):
            ci = sol['workshop'][c]
            s.append(f"W:{c}:{ci['classroom']}:{ci['slots']}")
        return '|'.join(s)

    cur_sig = sol_signature(current_sol)
    tabu_list.append(cur_sig)
    tabu_set.add(cur_sig)
    current_cost, current_hard, current_soft = best_cost, best_hard, best_soft

    start_time = time.time()  # 记录开始时间

    for it in range(max_iter):
        neighbors = generate_neighbors(current_sol, course_dict, max_neighbors)
        found_better = False
        best_neighbor = None
        best_neighbor_cost = float('inf')
        best_neighbor_hard = float('inf')  # 记录邻域最佳硬冲突
        best_neighbor_soft = float('inf')  # 记录邻域最佳软冲突

        for nb_sol in neighbors:
            nb_sig = sol_signature(nb_sol)
            cost_n, hard_n, soft_n = evaluate_solution(nb_sol, course_dict, course2stu)
            if nb_sig in tabu_set:
                # 破禁?
                if cost_n < best_cost:
                    best_neighbor = nb_sol
                    best_neighbor_cost = cost_n
                    best_neighbor_hard = hard_n
                    best_neighbor_soft = soft_n
                    found_better = True
            else:
                if cost_n < best_neighbor_cost:
                    best_neighbor = nb_sol
                    best_neighbor_cost = cost_n
                    best_neighbor_hard = hard_n
                    best_neighbor_soft = soft_n
                    found_better = True

        if found_better and best_neighbor is not None:
            current_sol = best_neighbor
            current_cost, current_hard, current_soft = best_neighbor_cost, best_neighbor_hard, best_neighbor_soft
            if current_cost < best_cost:
                best_cost = current_cost
                best_sol = current_sol
                _, best_hard, best_soft = evaluate_solution(best_sol, course_dict, course2stu)

            nb_sig2 = sol_signature(best_neighbor)
            tabu_list.append(nb_sig2)
            tabu_set.add(nb_sig2)
            if len(tabu_list) > tabu_tenure:
                old = tabu_list.popleft()
                if old in tabu_set:
                    tabu_set.remove(old)
        else:
            pass

        elapsed_time = time.time() - start_time
        print(f"Iter={it}, currentCost={current_cost}, Hard Conflicts: {current_hard}, Soft Conflicts: {current_soft}, Best Cost: {best_cost}, Time: {elapsed_time:.2f}s")

    return best_sol, best_cost, best_hard, best_soft # 返回最佳硬冲突和软冲突


# --------------------------
# 10) 打印课表
# --------------------------
def print_schedule(sol, course_dict):
    print("\n================= Final Timetable (Lecture) =================")
    day_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

    def lecture_slot_str(s):
        if s == 0:
            return "10-11"
        elif s == 1:
            return "11-12"
        else:
            return "12-13"

    def workshop_slot_str(s):
        if s == 0:
            return "9-10"
        elif s == 1:
            return "10-11"
        elif s == 2:
            return "11-12"
        elif s == 3:
            return "12-13"
        else:
            return "13-14"

    def weeks_type_str(wt):
        if wt == 'single':
            return "(Single Week Only)"
        elif wt == 'double':
            return "(Double Week Only)"
        else:
            return "(Every Week)"

    # Lecture
    for c, cinfo in sol['lecture'].items():
        if c not in course_dict: continue
        cf = course_dict[c]
        if cinfo['hours'] <= 0:
            continue
        lec_name = cf['lecturer']
        croom = cinfo['classroom']
        wt = cinfo['weeks_type']
        cls_name = classroom_mapping.get(croom, f"UnknownRoom_{croom}")
        wtype_label = weeks_type_str(wt)

        print(f"Course: {c} | Lecturer: {lec_name}")
        print(f"  Classroom => {croom} ({cls_name}) {wtype_label}")
        for (d, stuple) in cinfo['slots']:
            day_str = day_name[d]
            if len(stuple) == 1:
                slabel = lecture_slot_str(stuple[0])
            else:
                slabel = " & ".join([lecture_slot_str(xx) for xx in stuple])
            print(f"    {day_str}, {slabel}")
        print("-----")

    print("\n================= Final Timetable (Workshop) =================")
    for c, cinfo in sol['workshop'].items():
        if c not in course_dict: continue
        cf = course_dict[c]
        if cinfo['hours'] <= 0:
            continue
        lec_name = cf['lecturer']
        croom = cinfo['classroom']
        wt = cinfo['weeks_type']
        ws_n = get_workshop_name(croom)
        wtype_label = weeks_type_str(wt)

        print(f"Course: {c} | Workshop Teacher: {lec_name}")
        print(f"  WorkshopRoom => ws{croom} ({ws_n}) {wtype_label}")

        for (d, stuple) in cinfo['slots']:
            day_str = day_name[d]
            if len(stuple) == 1:
                slabel = workshop_slot_str(stuple[0])
            else:
                slabel = " & ".join([workshop_slot_str(xx) for xx in stuple])
            print(f"    {day_str}, {slabel}")
        print("-----")


def main():
    # Redirect stdout to a file
    original_stdout = sys.stdout
    with open('ts_timetable_output.txt', 'w', encoding='utf-8') as f: # Specify UTF-8 encoding
        sys.stdout = f

        # 1) parse
        cdict, lectid = parse_course_info(course_info_csv)
        stu_list = parse_student_info(student_courses_csv)
        c2s = build_course_to_students(stu_list)

        # start_time
        start_time = time.time()

        # 2) 并行Workshop拆分
        new_cd, new_c2s = split_workshop_sessions(cdict, c2s, max_capacity=30)

        # 3) tabu search
        best_sol, best_cost, best_hard, best_soft = tabu_search(new_cd, new_c2s, # 接收更多返回值
                                          max_iter=15000,  # 减少迭代次数以便更快地看到结果
                                          max_neighbors=40,
                                          tabu_tenure=10,
                                          seed=42)

        end_time = time.time()
        total_time = end_time - start_time

        print("\n=== Done Tabu Search ===")
        print("\nRunning time: {:.2f} seconds".format(total_time))
        print(f"Best cost: {best_cost}, Hard Conflicts: {best_hard}, Soft Conflicts:{best_soft}")

        # 4) print
        print_schedule(best_sol, new_cd)

    # Restore stdout
    sys.stdout = original_stdout
    print("Output saved to timetable_output.txt")

if __name__ == "__main__":
    main()