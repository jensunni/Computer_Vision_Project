"""Understand the overall workflow
Camera image + detected seat rectangles
# label_seats() #
Labels seats R1C1, R1C2, R2C1, etc.
# assign_persons() #
Matches people to seats.
# assign_objects() #
Matches objects to seats.
Save and load seat information
save_seats() and load_seats() use a JSON file."""
import json


# 1. Give each seat a row and column label
def label_seats(rectangles):
    if not rectangles:
        return []

    # Find the average seat height
    heights = [r[3] - r[1] for r in rectangles]
    heights.sort()
    median_height = heights[len(heights) // 2]

    # Sort seats from top to bottom
    rectangles.sort(key=lambda r: (r[1] + r[3]) / 2)

    rows = []
    current_row = [rectangles[0]]

    for seat in rectangles[1:]:
        seat_y = (seat[1] + seat[3]) / 2

        row_y = sum((r[1] + r[3]) / 2 for r in current_row) / len(current_row)

        if seat_y - row_y > 0.6 * median_height:
            rows.append(current_row)
            current_row = [seat]
        else:
            current_row.append(seat)

    rows.append(current_row)

    seats = []

    for row_number, row in enumerate(rows, start=1):
        row.sort(key=lambda r: (r[0] + r[2]) / 2)

        for col_number, rect in enumerate(row, start=1):
            seats.append({
                "id": f"R{row_number}C{col_number}",
                "row": row_number - 1,
                "col": col_number - 1,
                "rect": [int(v) for v in rect]
            })

    return seats


# 2. Save seats in a JSON file
def save_seats(rectangles, frame_size, path):
    seats = label_seats(rectangles)

    data = {
        "frame_size": list(frame_size),
        "seats": seats }

    with open(path, "w") as file:
        json.dump(data, file, indent=2)

    return seats


# 3. Load seats from a JSON file
def load_seats(path, frame_size=None):
    with open(path, "r") as file:
        data = json.load(file)

    seats = data["seats"]

    if frame_size and tuple(data["frame_size"]) != tuple(frame_size):
        old_width, old_height = data["frame_size"]
        new_width, new_height = frame_size

        x_scale = new_width / old_width
        y_scale = new_height / old_height

        for seat in seats:
            x1, y1, x2, y2 = seat["rect"]

            seat["rect"] = [
                int(x1 * x_scale),
                int(y1 * y_scale),
                int(x2 * x_scale),
                int(y2 * y_scale)
            ]

    return seats


# 4. Find the neighboring seat
def neighbor(seats, seat, direction):
    for other_seat in seats:
        same_row = other_seat["row"] == seat["row"]
        correct_column = other_seat["col"] == seat["col"] + direction

        if same_row and correct_column:
            return other_seat

    return None


# 5. Find the overlapping area of two rectangles
def overlap_area(box1, box2):
    width = min(box1[2], box2[2]) - max(box1[0], box2[0])
    height = min(box1[3], box2[3]) - max(box1[1], box2[1])

    width = max(0, width)
    height = max(0, height)

    return width * height


# 6. Assign people to seats
def assign_persons(seats, persons, min_cover=0.3):
    matches = []

    for person_index, person in enumerate(persons):
        box = person["box"]

        person_area = max(
            1,
            (box[2] - box[0]) * (box[3] - box[1])
        )

        for seat_index, seat in enumerate(seats):
            overlap = overlap_area(box, seat["rect"])
            score = overlap / person_area

            if score >= min_cover:
                matches.append((score, seat_index, person_index))

    # Consider the strongest matches first
    matches.sort(reverse=True)

    assigned = {}
    used_people = set()

    for score, seat_index, person_index in matches:
        seat_id = seats[seat_index]["id"]

        if seat_id not in assigned and person_index not in used_people:
            assigned[seat_id] = persons[person_index]
            used_people.add(person_index)

    unassigned = []

    for index, person in enumerate(persons):
        if index not in used_people:
            unassigned.append(person)

    return assigned, unassigned


# 7. Assign objects to seats
def assign_objects(seats, objects, margin=0.15):
    assignments = {}

    for obj in objects:
        x1, y1, x2, y2 = obj["box"]

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        best_seat = None
        shortest_distance = float("inf")

        for seat in seats:
            sx1, sy1, sx2, sy2 = seat["rect"]

            margin_x = (sx2 - sx1) * margin
            margin_y = (sy2 - sy1) * margin

            inside_seat = (
                sx1 - margin_x <= center_x <= sx2 + margin_x
                and sy1 - margin_y <= center_y <= sy2 + margin_y
            )

            if inside_seat:
                seat_x = (sx1 + sx2) / 2
                seat_y = (sy1 + sy2) / 2

                distance = (
                    (center_x - seat_x) ** 2
                    + (center_y - seat_y) ** 2
                )

                if distance < shortest_distance:
                    shortest_distance = distance
                    best_seat = seat

        if best_seat is not None:
            seat_id = best_seat["id"]

            if seat_id not in assignments:
                assignments[seat_id] = []

            assignments[seat_id].append(obj)

    return assignments