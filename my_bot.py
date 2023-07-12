import random
import time
from enum import Enum
from math import hypot
from typing import Any, List, Optional

from lugo4py import orientation, lugo, specs, geo
from lugo4py.mapper import Mapper, Region
from lugo4py.rl.interfaces import BotTrainer
from lugo4py.rl.remote_control import RemoteControl
from lugo4py.snapshot import GameSnapshotReader

TRAINING_PLAYER_NUMBER = 5


class SensorArea(Enum):
    FRONT = 0
    FRONT_LEFT = 1
    FRONT_RIGHT = 2
    BACK = 3
    BACK_LEFT = 4
    BACK_RIGHT = 5


class MyBotTrainer(BotTrainer):
    def __init__(self, remote_control: RemoteControl):
        self.remote_control = remote_control
        self.Mapper = None

    def set_environment(self, data: Any):
        self.Mapper = Mapper(20, 10, lugo.TeamSide.HOME)

        for i in range(1, 12):
            self._random_player_pos(self.Mapper, lugo.TeamSide.HOME, i)
            self._random_player_pos(self.Mapper, lugo.TeamSide.AWAY, i)

        random_velocity = _create_velocity(0, orientation.NORTH)
        pos = self.Mapper.get_region(10, random.randint(2, 7)).get_center()
        self.remote_control.set_player_props(lugo.TeamSide.HOME, TRAINING_PLAYER_NUMBER, pos, random_velocity)

        ball_pos = self.Mapper.get_region(0, 0).get_center()
        ball_velocity = _create_velocity(0, orientation.NORTH)
        self.remote_control.set_game_props(1)
        return self.remote_control.set_ball_rops(ball_pos, ball_velocity).game_snapshot

    def get_state(self, snapshot: lugo.GameSnapshot):
        if snapshot is None:
            raise ValueError("got None as snapshot - something went wrong")
        reader = GameSnapshotReader(snapshot, lugo.TeamSide.HOME)
        me = self.get_me(reader)

        goal_position = reader.get_opponent_goal().get_center()

        return [
            abs(goal_position.x - me.position.x) / specs.FIELD_WIDTH,
            abs(goal_position.y - me.position.y) / specs.FIELD_HEIGHT,
            self.steps_to_obstacle_within_area(reader, SensorArea.FRONT),
            self.steps_to_obstacle_within_area(reader, SensorArea.FRONT_LEFT),
            self.steps_to_obstacle_within_area(reader, SensorArea.FRONT_RIGHT),
            self.steps_to_obstacle_within_area(reader, SensorArea.BACK),
            self.steps_to_obstacle_within_area(reader, SensorArea.BACK_LEFT),
            self.steps_to_obstacle_within_area(reader, SensorArea.BACK_RIGHT)
        ]

    def steps_to_obstacle_within_area(self, reader: GameSnapshotReader, sensor_area):
        # specs.PLAYER_MAX_SPEED is a step
        sensor_range = specs.PLAYER_MAX_SPEED * 18
        frontward_view = sensor_range
        sides_view = sensor_range
        backward_view = sensor_range

        my_pos = self.get_me(reader).position

        bot_point = [my_pos.x, my_pos.y]

        # Each region is a triangle where the start point is the bot position and the other two vertex
        # are defined by the sensor direction:

        # front
        point_a = [my_pos.x + frontward_view, my_pos.y + sides_view]
        point_b = [my_pos.x + frontward_view, my_pos.y - sides_view]
        if sensor_area == SensorArea.FRONT_LEFT:
            point_a = [my_pos.x, my_pos.y + sides_view]
            point_b = [my_pos.x + frontward_view, my_pos.y + sides_view]
        elif sensor_area == SensorArea.FRONT_RIGHT:
            point_a = [my_pos.x, my_pos.y - sides_view]
            point_b = [my_pos.x + frontward_view, my_pos.y - sides_view]
        elif sensor_area == SensorArea.BACK:
            point_a = [my_pos.x - backward_view, my_pos.y + sides_view]
            point_b = [my_pos.x - backward_view, my_pos.y - sides_view]
        elif sensor_area == SensorArea.BACK_LEFT:
            point_a = [my_pos.x, my_pos.y + sides_view]
            point_b = [my_pos.x - backward_view, my_pos.y + sides_view]
        elif sensor_area == SensorArea.BACK_RIGHT:
            point_a = [my_pos.x, my_pos.y - sides_view]
            point_b = [my_pos.x - backward_view, my_pos.y - sides_view]

        get_opponents = reader.get_team(reader.get_opponent_side()).players
        nearest_opponent_dist = None
        for opponent in get_opponents:
            opponent_point = [opponent.position.x, opponent.position.y]
            if is_point_in_polygon(opponent_point, [bot_point, point_a, point_b]):
                dist_to_bot = abs(geo.distance_between_points(opponent.position, my_pos))
                if nearest_opponent_dist is None or nearest_opponent_dist > dist_to_bot:
                    nearest_opponent_dist = dist_to_bot

        if nearest_opponent_dist is not None:
            return nearest_opponent_dist / sensor_range

        return 1

    def get_me(self, reader: GameSnapshotReader) -> lugo.Player:
        me = reader.get_player(lugo.TeamSide.HOME, TRAINING_PLAYER_NUMBER)
        if me is None:
            raise ValueError("did not find myself in the game")
        return me

    def play(self, order_set: lugo.OrderSet, snapshot: lugo.GameSnapshot, action: Any) -> lugo.OrderSet:
        # print(f"GOT ACTION -> {action}")
        reader = GameSnapshotReader(snapshot, lugo.TeamSide.HOME)
        direction = reader.make_order_move_by_direction(action)
        order_set.orders.extend([direction])
        return order_set

    def evaluate(self, previous_snapshot: lugo.GameSnapshot, new_snapshot: lugo.GameSnapshot) -> Any:
        reader_previous = GameSnapshotReader(previous_snapshot, lugo.TeamSide.HOME)
        reader = GameSnapshotReader(new_snapshot, lugo.TeamSide.HOME)
        me = self.get_me(reader)
        me_previously = self.get_me(reader_previous)

        opponent_goal = reader.get_opponent_goal().get_center()

        previous_dist = hypot(opponent_goal.x - me_previously.position.x,
                              opponent_goal.y - me_previously.position.y)
        actual_dist = hypot(opponent_goal.x - me.position.x,
                            opponent_goal.y - me.position.y)

        reward = (previous_dist - actual_dist)/specs.PLAYER_MAX_SPEED
        done = False

        if me.position.x > (specs.FIELD_WIDTH - specs.GOAL_ZONE_RANGE) * 0.90:  # positive end
            done = True
            reward = 10000
        elif new_snapshot.turn > 800 or me.position.x < specs.FIELD_WIDTH/3 or me.position.y == specs.MAX_Y_COORDINATE or me.position.y == 0:
            done = True
            reward = -5000
        else:  # negative end
            steps_to_closest_obstacle = min(
                self.steps_to_obstacle_within_area(reader, SensorArea.FRONT),
                self.steps_to_obstacle_within_area(reader, SensorArea.FRONT_LEFT),
                self.steps_to_obstacle_within_area(reader, SensorArea.FRONT_RIGHT),
                self.steps_to_obstacle_within_area(reader, SensorArea.BACK),
                self.steps_to_obstacle_within_area(reader, SensorArea.BACK_LEFT),
                self.steps_to_obstacle_within_area(reader, SensorArea.BACK_RIGHT)
            )

            if steps_to_closest_obstacle < 0.5:
                done = True
                reward = -30000

        return {'done': done, 'reward': reward}

    def _random_player_pos(self, mapper: Mapper, side: lugo.TeamSide, number: int) -> None:
        min_col = 7
        max_col = 17
        min_row = 1
        max_row = 8

        random_velocity = _create_velocity(0, orientation.NORTH)

        random_col = random.randint(min_col, max_col)
        random_row = random.randint(min_row, max_row)
        random_position = mapper.get_region(random_col, random_row).get_center()
        self.remote_control.set_player_props(side, number, random_position, random_velocity)

    def find_opponent(self, reader: GameSnapshotReader) -> List[List[bool]]:
        opponents = reader.get_team(
            reader.get_opponent_side()).players
        mapped_opponents = self._create_empty_mapped_opponents()

        for opponent in opponents:
            opponent_region = self.Mapper.get_region_from_point(
                opponent.position)
            col, row = opponent_region.get_col(), opponent_region.get_row()
            mapped_opponents[col][row] = True

        return mapped_opponents

    def _create_empty_mapped_opponents(self) -> List[List[Optional[bool]]]:
        mapped_opponents = []
        for _ in range(self.Mapper.get_num_cols()):
            mapped_opponents.append([None] * self.Mapper.get_num_rows())
        return mapped_opponents

    def has_opponent(self, mapped_opponents: List[List[bool]], region: Region) -> bool:
        col, row = region.get_col(), region.get_row()
        return mapped_opponents[col][row] is True


def _create_velocity(speed: float, direction) -> lugo.Velocity:
    velocity = lugo.new_velocity(direction)
    velocity.speed = speed
    return velocity


def delay(ms: float) -> None:
    time.sleep(ms / 1000)


def random_integer(min_val: int, max_val: int) -> int:
    return random.randint(min_val, max_val)


def is_point_in_polygon(point, polygon):
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i][0], polygon[i][1]
        xj, yj = polygon[j][0], polygon[j][1]

        intersect = ((yi > point[1]) != (yj > point[1])) and (point[0] < (xj - xi) * (point[1] - yi) / (yj - yi) + xi)
        if intersect:
            inside = not inside

        j = i

    return inside
