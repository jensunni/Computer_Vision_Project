"""Pure rule logic for the hall (no camera / model imports, so it is easy to unit-test)."""

import config as cfg
import seats as seatlib


# 1. Control when alerts should trigger
class Gate:
    def __init__(self):
        self.start = {}
        self.fired = {}

    def update(self, key, active, now, duration, cooldown):
        if not active:
            self.start.pop(key, None)
            return False

        if key not in self.start:
            self.start[key] = now

        active_time = now - self.start[key]
        last_alert = self.fired.get(key, -1000000000)
        time_since_alert = now - last_alert

        if active_time >= duration and time_since_alert >= cooldown:
            self.fired[key] = now
            return True

        return False


# 2. Store information about one seat
class SeatState:
    def __init__(self, seat):
        self.seat = seat

        self.total_ticks = 0
        self.occupied_ticks = 0

        self.samples = []
        self.base_yaw = 0.0
        self.base_off = 0.0

        self.expected = False
        self.track_id = None
        self.person_box = None

        self.last_seen = 0.0
        self.last_face_try = 0.0
        self.face_ts = -1000000000

        self.yaw = None
        self.off = None

        self.active = set()


# 3. Calculate each student's normal position
def finish_calibration(states):
    for state in states.values():

        # Was the seat occupied during calibration?
        if state.total_ticks > 0:
            occupied_ratio = state.occupied_ticks / state.total_ticks
            state.expected = occupied_ratio >= 0.6
        else:
            state.expected = False

        # Calculate the median head position
        if len(state.samples) >= 5:
            yaw_values = sorted(sample[0] for sample in state.samples)
            offset_values = sorted(sample[1] for sample in state.samples)

            middle = len(state.samples) // 2

            state.base_yaw = yaw_values[middle]
            state.base_off = offset_values[middle]


# 4. Detect head-turn behavior
def head_event(state, seats, states, timestamp):

    # Check whether a recent face measurement exists
    if state.yaw is None:
        return None, None

    if timestamp - state.face_ts > cfg.HALL_FACE_STALE:
        return None, None

    # Compare current head position with normal position
    yaw_change = state.yaw - state.base_yaw
    offset_change = state.off - state.base_off

    # Both changes must exceed their limits
    if abs(yaw_change) <= cfg.HALL_YAW_LIMIT:
        return None, None

    if abs(offset_change) <= cfg.HALL_OFFSET_LIMIT:
        return None, None

    # Determine the direction of the head turn
    if offset_change > 0:
        direction = 1
        side = "right"
    else:
        direction = -1
        side = "left"

    # Find the neighboring seat
    neighbor = seatlib.neighbor(seats, state.seat, direction)

    if neighbor is not None:
        neighbor_state = states[neighbor["id"]]

        if neighbor_state.expected:
            return "LOOK_NEIGHBOR", f"toward={neighbor['id']} yaw={yaw_change:+.0f}"

    return "HEAD_AWAY", f"yaw={yaw_change:+.0f} facing {side} (no occupied neighbor)"