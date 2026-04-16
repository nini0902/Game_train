import time

import gymnasium as gym
import highway_env  # noqa: F401
import pygame


LANE_LEFT = 0
IDLE = 1
LANE_RIGHT = 2
FASTER = 3
SLOWER = 4


class HighwayPenaltyWrapper(gym.Wrapper):
	COLLISION_PENALTY = -1.0
	FRONT_TOO_FAR_PENALTY = -0.3
	FRONT_TOO_NEAR_PENALTY = -0.7
	NO_FRONT_DATA_BONUS = 0.2
	SPEED_BONUS = 0.1
	SPEED_THRESHOLD = 28.0
	CHASE_LANE_CHANGE_BONUS = 0.5
	LANE_CHANGE_FAILED_PENALTY = -0.2
	LANE_SAFETY_SUCCESS_BONUS = 0.2
	LEFT_BLOCKED_LANE_CHANGE_PENALTY = -0.9
	RIGHT_BLOCKED_LANE_CHANGE_PENALTY = -0.9

	FRONT_TOO_FAR_DISTANCE = 45.0
	FRONT_TOO_NEAR_DISTANCE = 30.0
	CHASE_DISTANCE_GAIN_THRESHOLD = 8.0

	LANE_WIDTH = 4.0
	SAME_LANE_Y_THRESHOLD = 2.0
	ADJACENT_LANE_Y_THRESHOLD = 1.5
	SIDE_LONGITUDINAL_CHECK = 30.0

	def __init__(self, env):
		super().__init__(env)
		self._last_obs = None

	def reset(self, **kwargs):
		obs, info = self.env.reset(**kwargs)
		self._last_obs = obs
		return obs, info

	def step(self, action):
		prev_obs = self._last_obs
		prev_lane_id = self._current_lane_id()
		prev_front_distance = self._front_distance_from_obs(prev_obs)
		left_occupied_before_change = self._adjacent_lane_has_vehicle(prev_obs, is_left=True)
		right_occupied_before_change = self._adjacent_lane_has_vehicle(prev_obs, is_left=False)

		obs, reward, terminated, truncated, info = self.env.step(action)
		self._last_obs = obs

		front_distance = self._front_distance_from_obs(obs)
		post_lane_id = self._current_lane_id()
		speed = float(info.get("speed", 0.0))
		is_lane_change_action = action in (LANE_LEFT, LANE_RIGHT)

		lane_change_success = self._lane_change_success(action, prev_lane_id, post_lane_id)
		pursuit_lane_change_success = self._pursuit_lane_change_success(
			lane_change_success,
			prev_front_distance,
			front_distance,
		)
		lane_safety_success = self._lane_safety_success(
			action,
			lane_change_success,
			left_occupied_before_change,
			right_occupied_before_change,
			pursuit_lane_change_success,
		)

		penalty = 0.0
		penalty_terms = {}

		collision_penalty = self._collision_penalty(info)
		if collision_penalty != 0.0:
			penalty += collision_penalty
			penalty_terms["collision"] = collision_penalty

		front_distance_reward = self._front_distance_reward(front_distance)
		if front_distance_reward != 0.0:
			penalty += front_distance_reward
			penalty_terms["front_distance"] = front_distance_reward

		speed_reward = self._speed_reward(speed)
		if speed_reward != 0.0:
			penalty += speed_reward
			penalty_terms["speed"] = speed_reward

		left_blocked_penalty, right_blocked_penalty = self._blocked_lane_change_penalty(
			action,
			left_occupied_before_change,
			right_occupied_before_change,
		)
		if left_blocked_penalty != 0.0:
			penalty += left_blocked_penalty
			penalty_terms["left_blocked_lane_change"] = left_blocked_penalty
		if right_blocked_penalty != 0.0:
			penalty += right_blocked_penalty
			penalty_terms["right_blocked_lane_change"] = right_blocked_penalty

		if pursuit_lane_change_success:
			penalty += self.CHASE_LANE_CHANGE_BONUS
			penalty_terms["pursuit_lane_change"] = self.CHASE_LANE_CHANGE_BONUS

		if lane_safety_success:
			penalty += self.LANE_SAFETY_SUCCESS_BONUS
			penalty_terms["lane_safety"] = self.LANE_SAFETY_SUCCESS_BONUS

		if is_lane_change_action and not pursuit_lane_change_success and not lane_safety_success:
			penalty += self.LANE_CHANGE_FAILED_PENALTY
			penalty_terms["lane_change"] = self.LANE_CHANGE_FAILED_PENALTY

		shaped_reward = reward + penalty

		info["base_reward"] = reward
		info["shaped_reward"] = shaped_reward
		info["penalty"] = penalty
		info["penalty_terms"] = penalty_terms
		info["front_distance"] = front_distance
		info["prev_front_distance"] = prev_front_distance
		info["left_occupied_before_change"] = left_occupied_before_change
		info["right_occupied_before_change"] = right_occupied_before_change
		info["lane_change_success"] = lane_change_success
		info["pursuit_lane_change_success"] = pursuit_lane_change_success
		info["lane_safety_success"] = lane_safety_success

		return obs, shaped_reward, terminated, truncated, info

	def _collision_penalty(self, info):
		crashed = bool(info.get("crashed", False))
		if crashed:
			return self.COLLISION_PENALTY
		return 0.0

	def _front_distance_reward(self, front_distance):
		if front_distance is None:
			return self.NO_FRONT_DATA_BONUS
		if front_distance > self.FRONT_TOO_FAR_DISTANCE:
			return self.FRONT_TOO_FAR_PENALTY
		if front_distance < self.FRONT_TOO_NEAR_DISTANCE:
			return self.FRONT_TOO_NEAR_PENALTY
		return 0.0

	def _speed_reward(self, speed):
		if speed >= self.SPEED_THRESHOLD:
			return self.SPEED_BONUS
		return 0.0

	def _blocked_lane_change_penalty(self, action, left_has_vehicle, right_has_vehicle):
		left_penalty = 0.0
		right_penalty = 0.0

		if action == LANE_LEFT and left_has_vehicle:
			left_penalty = self.LEFT_BLOCKED_LANE_CHANGE_PENALTY
		if action == LANE_RIGHT and right_has_vehicle:
			right_penalty = self.RIGHT_BLOCKED_LANE_CHANGE_PENALTY

		return left_penalty, right_penalty

	def _adjacent_lane_has_vehicle(self, obs, is_left):
		vehicles = self._valid_non_ego_rows(obs)
		lane_center = -self.LANE_WIDTH if is_left else self.LANE_WIDTH
		y_min = lane_center - self.ADJACENT_LANE_Y_THRESHOLD
		y_max = lane_center + self.ADJACENT_LANE_Y_THRESHOLD

		for row in vehicles:
			rel_x = row[1]
			rel_y = row[2]
			if abs(rel_x) <= self.SIDE_LONGITUDINAL_CHECK and y_min <= rel_y <= y_max:
				return True
		return False

	def _front_distance_from_obs(self, obs):
		vehicles = self._valid_non_ego_rows(obs)
		front_candidates = [row for row in vehicles if row[1] > 0 and abs(row[2]) <= self.SAME_LANE_Y_THRESHOLD]
		if not front_candidates:
			return None
		return min(row[1] for row in front_candidates)

	@staticmethod
	def _lane_change_success(action, prev_lane_id, post_lane_id):
		if prev_lane_id is None or post_lane_id is None:
			return False
		if action == LANE_LEFT:
			return post_lane_id < prev_lane_id
		if action == LANE_RIGHT:
			return post_lane_id > prev_lane_id
		return False

	def _pursuit_lane_change_success(self, lane_change_success, prev_front_distance, front_distance):
		if not lane_change_success:
			return False
		if prev_front_distance is None or front_distance is None:
			return False
		if prev_front_distance > self.FRONT_TOO_NEAR_DISTANCE:
			return False
		return (front_distance - prev_front_distance) >= self.CHASE_DISTANCE_GAIN_THRESHOLD

	@staticmethod
	def _lane_safety_success(
		action,
		lane_change_success,
		left_occupied_before_change,
		right_occupied_before_change,
		pursuit_lane_change_success,
	):
		if not lane_change_success or pursuit_lane_change_success:
			return False
		if action == LANE_LEFT:
			return not left_occupied_before_change
		if action == LANE_RIGHT:
			return not right_occupied_before_change
		return False

	def _current_lane_id(self):
		vehicle = getattr(self.env.unwrapped, "vehicle", None)
		if vehicle is None:
			return None
		lane_index = getattr(vehicle, "lane_index", None)
		if lane_index is None or len(lane_index) < 3:
			return None
		return int(lane_index[2])

	@staticmethod
	def _valid_non_ego_rows(obs):
		if obs is None or len(obs) <= 1:
			return []
		return [row for row in obs[1:] if row[0] > 0.5]


def get_action_from_keyboard() -> int:
	pressed = pygame.key.get_pressed()
	if pressed[pygame.K_UP]:
		return LANE_LEFT
	if pressed[pygame.K_DOWN]:
		return LANE_RIGHT
	if pressed[pygame.K_RIGHT]:
		return FASTER
	if pressed[pygame.K_LEFT]:
		return SLOWER
	return IDLE


def main() -> None:
	base_env = gym.make(
		"highway-v0",
		render_mode="human",
		config={"observation": {"type": "Kinematics", "normalize": False}},
	)
	env = HighwayPenaltyWrapper(base_env)

	observation, info = env.reset()
	env.render()

	clock = pygame.time.Clock()
	running = True

	while running:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False
			elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
				running = False

		action = get_action_from_keyboard()
		observation, reward, terminated, truncated, info = env.step(action)
		env.render()

		if terminated or truncated:
			observation, info = env.reset()
			env.render()

		clock.tick(15)
		time.sleep(0.001)

	env.close()


if __name__ == "__main__":
	main()

