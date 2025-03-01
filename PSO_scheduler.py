# -*- coding: utf-8 -*-

from csvData.csv_strings import course_info_csv, student_courses_csv

import time
import pandas as pd
import sys
import random
import re
import math
import copy
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
            base_lecturer = course_dict[c]['lecturer']
            base_tid = course_dict[c]['teacher_id']
            base_ws_freq = course_dict[c]['ws_freq']
            base_orig = course_dict[c]['orig_course']

            for i_sess in range(n_sessions):
                subname = f"{c}_WS{i_sess}"
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

            if lec_hrs == 1:
                for i in range(lec_times):
                    d = random.randint(0, DAYS_PER_WEEK - 1)
                    while any(slot[0] == d for slot in slots):
                        d = random.randint(0, DAYS_PER_WEEK - 1)
                    s = random.randint(0, LECTURE_SLOTS_PER_DAY - 1)
                    slots.append((d, (s,)))

            elif lec_hrs == 2:
                for i in range(lec_times):
                    d = random.randint(0, DAYS_PER_WEEK - 1)
                    while any(slot[0] == d for slot in slots):
                        d = random.randint(0, DAYS_PER_WEEK - 1)
                    s = random.randint(0, LECTURE_SLOTS_PER_DAY - 2)
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
# 7) 评估函数
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

    # Lecture processing
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

    # Workshop processing
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

    # Classroom conflicts
    for k, clist in usage.items():
        if len(clist) > 1:
            n = len(clist)
            pairs = n * (n - 1) // 2
            hard_conflicts += pairs

    # Teacher conflicts
    for k, tlist in teacher_usage.items():
        count_map = defaultdict(int)
        for (tid, ttype, cname) in tlist:
            if tid >= 0:
                count_map[tid] += 1
        for tid, cnt in count_map.items():
            if cnt > 1:
                pairs = cnt * (cnt - 1) // 2
                hard_conflicts += pairs

    # Course time conflicts
    course_times_lec = defaultdict(list)
    for c, cinfo in sol['lecture'].items():
        if c not in course_dict:
            continue
        lec_weeks = get_weeks(cinfo['weeks_type'])
        for (day, slot_tuple) in cinfo['slots']:
            for w in lec_weeks:
                for st in slot_tuple:
                    course_times_lec[c].append((w, day, st))

    course_times_ws = defaultdict(list)
    for c, cinfo in sol['workshop'].items():
        if c not in course_dict:
            continue
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
            overlap = lec_set.intersection(ws_set)
            hard_conflicts += len(overlap)

        for c_ws in sol['workshop'].keys():
            if c_ws not in course_dict:
                continue
            par = course_dict[c_ws]['orig_course']
            if par == c_lec:
                ws_set2 = set(course_times_ws[c_ws])
                overlap2 = lec_set.intersection(ws_set2)
                hard_conflicts += len(overlap2)

    # Student conflicts
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
                    common = cSets[i].intersection(cSets[j])
                    soft_conflicts += len(common)

    # Additional constraints
    for c in sol['lecture']:
        if c not in course_dict:
            continue
        lec_times, lec_hrs, _ = course_dict[c]['lecture_freq']
        if lec_times > 1 and lec_hrs == 1:
            days_used = [day for (day, slot_tuple) in sol['lecture'][c]['slots']]
            if len(set(days_used)) < len(days_used):
                hard_conflicts += 1


    cost = HARD_CONFLICT_PENALTY * hard_conflicts + soft_conflicts
    return cost, hard_conflicts, soft_conflicts


# --------------------------
# 8) 粒子群算法
# --------------------------
def mutate_lecture(lecture_info, course_info):
    new_info = copy.deepcopy(lecture_info)
    lec_times, lec_hrs, _ = course_info['lecture_freq']
    if lec_times == 0 or lec_hrs == 0:
        return new_info

    if random.random() < 0.5:
        new_info['classroom'] = random.choice(lecture_classrooms)

    if random.random() < 0.5:
        new_slots = []
        for _ in range(lec_times):
            if lec_hrs == 1:
                d = random.randint(0, DAYS_PER_WEEK - 1)
                while any(slot[0] == d for slot in new_slots):
                    d = random.randint(0, DAYS_PER_WEEK - 1)
                s = random.randint(0, LECTURE_SLOTS_PER_DAY - 1)
                new_slots.append((d, (s,)))
            elif lec_hrs == 2:
                d = random.randint(0, DAYS_PER_WEEK - 1)
                while any(slot[0] == d for slot in new_slots):
                    d = random.randint(0, DAYS_PER_WEEK - 1)
                s = random.randint(0, LECTURE_SLOTS_PER_DAY - 2)
                new_slots.append((d, (s, s + 1)))
        new_info['slots'] = new_slots

    return new_info


def mutate_workshop(workshop_info, course_info):
    new_info = copy.deepcopy(workshop_info)
    ws_times, ws_hrs, _ = course_info['ws_freq']
    if ws_times == 0 or ws_hrs == 0:
        return new_info

    if random.random() < 0.5:
        new_info['classroom'] = random.choice(workshop_classrooms)

    if random.random() < 0.5:
        new_slots = []
        for _ in range(ws_times):
            if ws_hrs == 1:
                d = random.randint(0, DAYS_PER_WEEK - 1)
                s = random.randint(0, WORKSHOP_SLOTS_PER_DAY - 1)
                new_slots.append((d, (s,)))
            elif ws_hrs == 2:
                d = random.randint(0, DAYS_PER_WEEK - 1)
                s = random.randint(0, WORKSHOP_SLOTS_PER_DAY - 2)
                new_slots.append((d, (s, s + 1)))
        new_info['slots'] = new_slots

    return new_info


def particle_swarm_optimization(course_dict, course2stu,
                                num_particles=30,
                                max_iter=200,
                                c1=0.7, c2=0.3,
                                mutation_rate=0.1):
    particles = []
    for _ in range(num_particles):
        sol = random_initial_solution(course_dict, course2stu)
        cost, hc, sc = evaluate_solution(sol, course_dict, course2stu)
        particles.append({
            'position': copy.deepcopy(sol),
            'cost': cost,
            'hard_conflicts': hc,  # 记录硬冲突
            'soft_conflicts': sc,  # 记录软冲突
            'pbest': copy.deepcopy(sol),
            'pbest_cost': cost,
            'pbest_hard_conflicts': hc,  # 记录个体最佳硬冲突
            'pbest_soft_conflicts': sc  # 记录个体最佳软冲突

        })

    gbest = min(particles, key=lambda x: x['pbest_cost'])
    gbest_sol = copy.deepcopy(gbest['pbest'])
    gbest_cost = gbest['pbest_cost']
    gbest_hard = gbest['pbest_hard_conflicts'] # 记录全局最佳硬冲突
    gbest_soft = gbest['pbest_soft_conflicts'] # 记录全局最佳软冲突

    start_time = time.time() # 记录开始时间

    for iteration in range(max_iter):
        for particle in particles:
            current_sol = particle['position']
            pbest_sol = particle['pbest']
            new_sol = {'lecture': {}, 'workshop': {}}

            # Process lectures
            for c in course_dict:
                source = None
                r = random.random()
                if r < c1 and c in pbest_sol['lecture']:
                    source = pbest_sol['lecture'][c]
                elif r < (c1 + c2) and c in gbest_sol['lecture']:
                    source = gbest_sol['lecture'][c]
                elif c in current_sol['lecture']:
                    source = current_sol['lecture'][c]

                if source:
                    new_sol['lecture'][c] = copy.deepcopy(source)
                    if random.random() < mutation_rate:
                        new_sol['lecture'][c] = mutate_lecture(
                            new_sol['lecture'][c], course_dict[c]
                        )

            # Process workshops
            for c in course_dict:
                source = None
                r = random.random()
                if r < c1 and c in pbest_sol['workshop']:
                    source = pbest_sol['workshop'][c]
                elif r < (c1 + c2) and c in gbest_sol['workshop']:
                    source = gbest_sol['workshop'][c]
                elif c in current_sol['workshop']:
                    source = current_sol['workshop'][c]

                if source:
                    new_sol['workshop'][c] = copy.deepcopy(source)
                    if random.random() < mutation_rate:
                        new_sol['workshop'][c] = mutate_workshop(
                            new_sol['workshop'][c], course_dict[c]
                        )

            # Evaluate new solution
            new_cost, new_hard, new_soft = evaluate_solution(new_sol, course_dict, course2stu)

            # Update personal best
            if new_cost < particle['pbest_cost']:
                particle['pbest'] = copy.deepcopy(new_sol)
                particle['pbest_cost'] = new_cost
                particle['pbest_hard_conflicts'] = new_hard  # 更新个体最佳硬冲突
                particle['pbest_soft_conflicts'] = new_soft  # 更新个体最佳软冲突

            # Update global best
            if new_cost < gbest_cost:
                gbest_sol = copy.deepcopy(new_sol)
                gbest_cost = new_cost
                gbest_hard = new_hard # 更新全局最佳硬冲突
                gbest_soft = new_soft # 更新全局最佳软冲突


            particle['position'] = copy.deepcopy(new_sol)
            particle['cost'] = new_cost
            particle['hard_conflicts'] = new_hard  # 更新当前硬冲突
            particle['soft_conflicts'] = new_soft  # 更新当前软冲突

        elapsed_time = time.time() - start_time
        print(f"Iteration {iteration + 1}, Best Cost: {gbest_cost}, Hard Conflicts: {gbest_hard}, Soft Conflicts: {gbest_soft}, Time: {elapsed_time:.2f}s")

    return gbest_sol, gbest_cost, gbest_hard, gbest_soft


# --------------------------
# 9) 打印课表
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

    # Lecture
    for c, cinfo in sol['lecture'].items():
        if c not in course_dict:
            continue
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
        if c not in course_dict:
            continue
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


# --------------------------
# 主函数
# --------------------------
def main():
    # Redirect stdout to a file
    original_stdout = sys.stdout
    with open('pso_timetable_output.txt', 'w', encoding='utf-8') as f: # Specify UTF-8 encoding
        sys.stdout = f

        cdict, lectid = parse_course_info(course_info_csv)
        stu_list = parse_student_info(student_courses_csv)
        c2s = build_course_to_students(stu_list)

        start_time = time.time()

        new_cd, new_c2s = split_workshop_sessions(cdict, c2s, max_capacity=30)

        best_sol, best_cost, best_hard, best_soft = particle_swarm_optimization(new_cd, new_c2s,
                                                          num_particles=30,
                                                          max_iter=15000,  # 减少迭代次数
                                                          c1=0.7, c2=0.3,
                                                          mutation_rate=0.1)

        end_time = time.time()
        total_time = end_time - start_time

        print("\n=== Optimization Complete ===")
        print("\nRunning time: {:.2f} seconds".format(total_time))
        print(f"Best cost: {best_cost}, Hard Conflicts: {best_hard}, Soft Conflicts: {best_soft}")

        print_schedule(best_sol, new_cd)

    # Restore stdout
    sys.stdout = original_stdout
    print("Output saved to pso_timetable_output.txt")

if __name__ == "__main__":
    main()