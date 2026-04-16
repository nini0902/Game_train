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
	FRONT_IDEAL_BONUS = 0.8
	FRONT_TOO_FAR_PENALTY = -0.7
	FRONT_TOO_NEAR_PENALTY = -0.7
	FRONT_MISSING_BONUS = 0.2
	SPEED_BONUS = 0.1
	SPEED_THRESHOLD = 28.0
	CHASE_BONUS = 0.5
	CHASE_DISTANCE_GAIN = 8.0
	LANE_CHANGE_PENALTY = -0.2
	LANE_SAFETY_BONUS = 0.6
	LEFT_BLOCKED_LANE_CHANGE_PENALTY = -0.9
	RIGHT_BLOCKED_LANE_CHANGE_PENALTY = -0.9

	FRONT_TOO_FAR_DISTANCE = 45.0
	FRONT_TOO_NEAR_DISTANCE = 30.0
	IDEAL_FRONT_DISTANCE_MIN = 30.0
	IDEAL_FRONT_DISTANCE_MAX = 35.0

	SAME_LANE_Y_THRESHOLD = 2.0
	SIDE_LONGITUDINAL_CHECK = 30.0

	def reset(self, **kwargs):
		obs, info = self.env.reset(**kwargs)
		self._sync_previous_state(obs, info)
		return obs, info

	def step(self, action):
		ego_before = self.env.unwrapped.vehicle
		prev_lane_index = self._get_lane_index(ego_before)
		prev_front_distance = self.prev_front_distance
		left_occupied_before_change, right_occupied_before_change = self._lane_occupancy_before_change(ego_before, prev_lane_index)

		obs, reward, terminated, truncated, info = self.env.step(action)

		ego_after = self.env.unwrapped.vehicle
		current_lane_index = self._get_lane_index(ego_after)
		current_front_distance = self._front_distance_from_obs(obs)
		speed = float(info.get("speed", getattr(ego_after, "speed", 0.0)))
		crashed = bool(info.get("crashed", False))
		attempted_lane_change = action in (LANE_LEFT, LANE_RIGHT)
		lane_changed = prev_lane_index is not None and current_lane_index is not None and current_lane_index != prev_lane_index

		adjustment = 0.0
		bonus = 0.0
		penalty_terms = {}
		bonus_terms = {}

		if crashed or terminated:
			adjustment += self.COLLISION_PENALTY
			penalty_terms["collision"] = self.COLLISION_PENALTY

		if current_front_distance is None:
			adjustment += self.FRONT_MISSING_BONUS
			bonus_terms["front_distance_missing"] = self.FRONT_MISSING_BONUS
		elif current_front_distance > self.FRONT_TOO_FAR_DISTANCE:
			adjustment += self.FRONT_TOO_FAR_PENALTY
			penalty_terms["front_distance_too_far"] = self.FRONT_TOO_FAR_PENALTY
		elif current_front_distance < self.FRONT_TOO_NEAR_DISTANCE:
			adjustment += self.FRONT_TOO_NEAR_PENALTY
			penalty_terms["front_distance_too_near"] = self.FRONT_TOO_NEAR_PENALTY
		elif self.IDEAL_FRONT_DISTANCE_MIN <= current_front_distance <= self.IDEAL_FRONT_DISTANCE_MAX:
			adjustment += self.FRONT_IDEAL_BONUS
			bonus_terms["front_distance_ideal"] = self.FRONT_IDEAL_BONUS

		if speed >= self.SPEED_THRESHOLD:
			adjustment += self.SPEED_BONUS
			bonus_terms["speed_bonus"] = self.SPEED_BONUS

		if attempted_lane_change:
			if not lane_changed:
				adjustment += self.LANE_CHANGE_PENALTY
				penalty_terms["lane_change_failed_or_insufficient"] = self.LANE_CHANGE_PENALTY

			if action == LANE_LEFT:
				if left_occupied_before_change:
					adjustment += self.LEFT_BLOCKED_LANE_CHANGE_PENALTY
					penalty_terms["left_occupied_before_change"] = self.LEFT_BLOCKED_LANE_CHANGE_PENALTY
				elif lane_changed:
					adjustment += self.LANE_SAFETY_BONUS
					bonus_terms["lane_safety"] = self.LANE_SAFETY_BONUS

			if action == LANE_RIGHT:
				if right_occupied_before_change:
					adjustment += self.RIGHT_BLOCKED_LANE_CHANGE_PENALTY
					penalty_terms["right_occupied_before_change"] = self.RIGHT_BLOCKED_LANE_CHANGE_PENALTY
				elif lane_changed:
					adjustment += self.LANE_SAFETY_BONUS
					bonus_terms["lane_safety"] = self.LANE_SAFETY_BONUS

		if lane_changed and prev_front_distance is not None and attempted_lane_change:
			if current_front_distance is None:
				if prev_front_distance <= self.FRONT_TOO_NEAR_DISTANCE:
					adjustment += self.CHASE_BONUS
					bonus_terms["chase_lane_gain"] = self.CHASE_BONUS
			elif prev_front_distance <= self.FRONT_TOO_NEAR_DISTANCE and (current_front_distance - prev_front_distance) >= self.CHASE_DISTANCE_GAIN:
				adjustment += self.CHASE_BONUS
				bonus_terms["chase_lane_gain"] = self.CHASE_BONUS

		shaped_reward = reward + adjustment

		info["base_reward"] = reward
		info["shaped_reward"] = shaped_reward
		info["reward_adjustment"] = adjustment
		info["penalty_terms"] = penalty_terms
		info["bonus_terms"] = bonus_terms
		info["front_distance"] = current_front_distance
		info["prev_front_distance"] = prev_front_distance
		info["speed"] = speed
		info["crashed"] = crashed
		info["left_occupied_before_change"] = left_occupied_before_change
		info["right_occupied_before_change"] = right_occupied_before_change
		info["lane_changed"] = lane_changed

		self._sync_previous_state(obs, info)

		return obs, shaped_reward, terminated, truncated, info

	def _sync_previous_state(self, obs, info):
		self.prev_front_distance = self._front_distance_from_obs(obs)
		self.prev_lane_index = self._get_lane_index(getattr(self.env.unwrapped, "vehicle", None))
		self.prev_speed = float(info.get("speed", getattr(getattr(self.env.unwrapped, "vehicle", None), "speed", 0.0)))

	def _get_lane_index(self, vehicle):
		if vehicle is None or getattr(vehicle, "lane_index", None) is None:
			return None
		return vehicle.lane_index[2]

	def _front_distance_from_obs(self, obs):
		vehicles = self._valid_non_ego_rows(obs)
		front_candidates = [row for row in vehicles if row[1] > 0 and abs(row[2]) <= self.SAME_LANE_Y_THRESHOLD]

		if not front_candidates:
			return None

		return min(row[1] for row in front_candidates)

	def _lane_occupancy_before_change(self, ego_vehicle, ego_lane_index):
		if ego_vehicle is None or ego_lane_index is None or getattr(self.env.unwrapped, "road", None) is None:
			return False, False

		left_target_lane = ego_lane_index - 1
		right_target_lane = ego_lane_index + 1

		left_occupied = self._lane_has_vehicle(ego_vehicle, left_target_lane)
		right_occupied = self._lane_has_vehicle(ego_vehicle, right_target_lane)

		return left_occupied, right_occupied

	def _lane_has_vehicle(self, ego_vehicle, target_lane_index):
		if target_lane_index < 0:
			return False

		road = getattr(self.env.unwrapped, "road", None)
		if road is None:
			return False

		nearby = road.close_objects_to(
			ego_vehicle,
			self.SIDE_LONGITUDINAL_CHECK,
			count=20,
			see_behind=True,
			sort=True,
			vehicles_only=True,
		)

		for vehicle in nearby:
			if vehicle is ego_vehicle:
				continue
			if getattr(vehicle, "lane_index", None) is None:
				continue
			if vehicle.lane_index[2] != target_lane_index:
				continue
			rel = vehicle.to_dict(ego_vehicle)
			if abs(rel.get("x", 0.0)) <= self.SIDE_LONGITUDINAL_CHECK:
				return True

		return False

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

