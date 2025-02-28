import pandas as pd
import re

def parse_timetable(filename="sa_table.txt"):
    """
    Parses the timetable data from the given text file.

    Args:
        filename: The name of the text file containing the timetable data.

    Returns:
        A tuple containing two dictionaries: one for lectures and one for workshops.
        Each dictionary maps course names to a list of their respective data.
    """
    lectures = {}
    workshops = {}

    with open(filename, 'r') as f:
        lines = f.readlines()

    section = None  # Keep track of whether we're in Lecture or Workshop section
    for i, line in enumerate(lines):
        line = line.strip()

        if "Final Timetable (Lecture)" in line:
            section = "lecture"
            continue
        elif "Final Timetable (Workshop)" in line:
            section = "workshop"
            continue

        if section == "lecture":
            if line.startswith("Course:"):
                match = re.match(r"Course: (.*?)\s*\|", line)
                course_name = match.group(1).strip() if match else None

                classroom_line = lines[i + 1].strip()
                match = re.search(r"Classroom => \d+ \(([^)]+)\)", classroom_line)
                lecture_room = match.group(1).strip() if match else None

                lecture_slots = []
                j = i + 2
                while j < len(lines) and lines[j].strip() and not lines[j].strip().startswith("-----"):
                    slot_line = lines[j].strip()
                    if re.match(r"[A-Za-z]+day, \d+-\d+", slot_line) :
                       lecture_slots.append(slot_line)  # Collect time slots
                    j += 1
                    
                # Join multiple time slots into a single string
                lecture_slots_str = "; ".join(lecture_slots)


                if course_name and lecture_room:
                    lectures[course_name] = {
                        "Lecture Room": lecture_room,
                        "Lecture Slots": lecture_slots_str
                    }

        elif section == "workshop":
            if line.startswith("Course:"):
                match = re.match(r"Course: (.*?)\s*\|", line)
                course_name = match.group(1).strip() if match else None

                workshop_room_line = lines[i + 1].strip()
                match = re.search(r"WorkshopRoom => \w+ \(([^)]+)\)", workshop_room_line)
                workshop_room = match.group(1).strip() if match else None

                workshop_slots = []
                j = i + 2
                while j < len(lines) and lines[j].strip() and not lines[j].strip().startswith("-----"):
                    slot_line = lines[j].strip()
                    if re.match(r"[A-Za-z]+day, \d+-\d+", slot_line):
                        workshop_slots.append(slot_line)
                    j += 1

                workshop_slots_str = "; ".join(workshop_slots)

                if course_name and workshop_room:
                    # Remove _WS suffix and numbers
                    base_course_name = re.sub(r"_WS\d+$", "", course_name)

                    if base_course_name not in workshops:
                         workshops[base_course_name] = []

                    workshops[base_course_name].append({
                         "Workshop Room": workshop_room,
                         "Workshop Slots": workshop_slots_str
                    })

    return lectures, workshops



def create_timetable_excel(lectures, workshops, output_filename="sa_timetable.xlsx"):
    """
    Creates an Excel timetable from the parsed lecture and workshop data.

    Args:
        lectures: A dictionary of lecture data.
        workshops: A dictionary of workshop data.
        output_filename: The name of the output Excel file.
    """

    data = []
    for course_name, lecture_data in lectures.items():
        row = {
            "Course Name": course_name,
            "Lecture Room": lecture_data["Lecture Room"],
            "Lecture Slots": lecture_data["Lecture Slots"],
            "Workshop Room": "",
            "Workshop Slots": ""
        }

        if course_name in workshops:
            workshop_rooms = []
            workshop_slots = []
            for workshop_entry in workshops[course_name]: #workshops[course_name] is a list
               workshop_rooms.append(workshop_entry["Workshop Room"])
               workshop_slots.append(workshop_entry["Workshop Slots"])

            row["Workshop Room"] = "; ".join(workshop_rooms)
            row["Workshop Slots"] = "; ".join(workshop_slots)
        data.append(row)

    df = pd.DataFrame(data)
    df.to_excel(output_filename, index=False)


if __name__ == "__main__":
    lectures, workshops = parse_timetable()
    create_timetable_excel(lectures, workshops)
    print("Timetable created successfully in timetable.xlsx")