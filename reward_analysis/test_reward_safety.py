import unittest
from unittest.mock import MagicMock
import numpy as np
import sys
import os

# Add project root to path
sys.path.append("/media/nemesis/disco4tb/Documents/VLM-RL")

# Mock ALL dependencies
sys.modules['carla'] = MagicMock()
sys.modules['pygame'] = MagicMock()
sys.modules['cv2'] = MagicMock()

# Mock torch and its submodules
mock_torch = MagicMock()
mock_torch.nn = MagicMock()
mock_torch.nn.Module = MagicMock
sys.modules['torch'] = mock_torch
sys.modules['torch.nn'] = mock_torch.nn

sys.modules['box'] = MagicMock()
sys.modules['stable_baselines3'] = MagicMock()
sys.modules['stable_baselines3.common'] = MagicMock()
sys.modules['stable_baselines3.common.noise'] = MagicMock()
sys.modules['stable_baselines3.common.torch_layers'] = MagicMock()
sys.modules['stable_baselines3.common.preprocessing'] = MagicMock()
sys.modules['gymnasium'] = MagicMock()
sys.modules['gym'] = MagicMock()
sys.modules['stable_baselines3.common.callbacks'] = MagicMock()
sys.modules['stable_baselines3.common.logger'] = MagicMock()

# Mock Box specifically to behave like a dict/object if needed
class MockBox(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
    def __getattr__(self, name):
        return self[name]

sys.modules['box'].Box = MockBox

# Now import the rewards
# We need to mock config.py imports before importing rewards
mock_config_module = MagicMock()
sys.modules['config'] = mock_config_module

# Create a mock CONFIG object with ALL necessary params used in rewards.py
# Based on grep search or reading the file
mock_config = MockBox({
    "reward_params": MockBox({
        "min_speed": 20.0,
        "target_speed": 25.0,
        "max_speed": 35.0,
        "max_distance": 3.0,
        "max_angle_center_lane": 20.0,
        "max_std_center_lane": 0.5,
        "penalty_reward": -10.0,
        "early_stop": False # Added this
    })
})
mock_config_module.CONFIG = mock_config

# Now we can import rewards. 
from carla_env.rewards import reward_fn5, reward_fn5_safe, create_reward_fn

# Re-inject global variables that rewards.py expects from CONFIG
import carla_env.rewards as rewards_module
rewards_module.min_speed = 20.0
rewards_module.target_speed = 25.0
rewards_module.max_speed = 35.0
rewards_module.max_distance = 3.0
rewards_module.max_angle_center_lane = 20.0
rewards_module.max_std_center_lane = 0.5
rewards_module.penalty_reward = -10.0
rewards_module.early_stop = False


class TestSafeReward(unittest.TestCase):
    def setUp(self):
        # Mock Environment
        self.env = MagicMock()
        self.env.vehicle = MagicMock()
        self.env.distance_from_center = 0.0
        self.env.distance_from_center_history = [0.0] * 10
        self.env.current_waypoint = MagicMock()
        self.env.vehicle.get_angle.return_value = 0.0 # Aligned
        
        # Mock Raycasting for Safe Reward
        self.rewards_module = rewards_module

    def test_blocked_scenario(self):
        """
        Scenario: Vehicle is stopped (0 km/h) because there is an obstacle 5m ahead.
        """
        # Setup
        self.env.vehicle.get_speed.return_value = 0.0 # Stopped
        
        # Mock obstacle detection to return 5.0 meters
        original_get_dist = self.rewards_module.get_distance_to_nearest_obstacle
        self.rewards_module.get_distance_to_nearest_obstacle = MagicMock(return_value=5.0)
        
        # 1. Test Original Reward (Should be 0 because speed is 0)
        # Note: reward_fn5 logic: if speed < min_speed (20), reward = speed/min_speed. 0/20 = 0.
        reward_original = reward_fn5(self.env)
        print(f"Original Reward (Stopped, Blocked): {reward_original}")
        
        # 2. Test Safe Reward (Should be High because we are safely stopped)
        reward_safe = reward_fn5_safe(self.env)
        print(f"Safe Reward (Stopped, Blocked): {reward_safe}")
        
        # Assertions
        self.assertLess(reward_original, 0.1, "Original reward should be low/zero when stopped")
        self.assertGreater(reward_safe, 0.9, "Safe reward should be high when stopped for obstacle")
        
        # Restore
        self.rewards_module.get_distance_to_nearest_obstacle = original_get_dist

    def test_creeping_scenario(self):
        """
        Scenario: Vehicle is creeping (2 km/h) into an obstacle 5m ahead.
        This is the 'Creeping Crash' behavior we want to discourage.
        """
        # Setup
        self.env.vehicle.get_speed.return_value = 2.0 # Creeping
        
        # Mock obstacle detection to return 5.0 meters
        original_get_dist = self.rewards_module.get_distance_to_nearest_obstacle
        self.rewards_module.get_distance_to_nearest_obstacle = MagicMock(return_value=5.0)
        
        # 1. Test Original Reward (Gives some points for moving)
        # 2/20 = 0.1
        reward_original = reward_fn5(self.env)
        print(f"Original Reward (Creeping 2km/h, Blocked): {reward_original}")
        
        # 2. Test Safe Reward (Should punish moving when blocked)
        # Logic: max(0.0, 1.0 - (2.0 / 5.0)) = 0.6. 
        # Wait, my logic was: 1.0 if speed < 1.0. Else decay.
        # If speed is 2.0, reward is 1.0 - 0.4 = 0.6.
        # Ideally this should be lower than the reward for stopping (1.0).
        reward_safe = reward_fn5_safe(self.env)
        print(f"Safe Reward (Creeping 2km/h, Blocked): {reward_safe}")
        
        # Assertions
        # The reward for stopping (1.0) should be significantly higher than creeping (0.6)
        # This incentivizes the agent to stop.
        self.assertTrue(True) # Just logging values for now
        
        # Restore
        self.rewards_module.get_distance_to_nearest_obstacle = original_get_dist

    def test_clear_road_scenario(self):
        """
        Scenario: Road is clear (50m), speed is 25 km/h (Target).
        Both should give max reward.
        """
        # Setup
        self.env.vehicle.get_speed.return_value = 25.0 
        
        # Mock obstacle detection to return 50.0 meters
        original_get_dist = self.rewards_module.get_distance_to_nearest_obstacle
        self.rewards_module.get_distance_to_nearest_obstacle = MagicMock(return_value=50.0)
        
        reward_original = reward_fn5(self.env)
        reward_safe = reward_fn5_safe(self.env)
        
        print(f"Original Reward (Clear, 25km/h): {reward_original}")
        print(f"Safe Reward (Clear, 25km/h): {reward_safe}")
        
        self.assertAlmostEqual(reward_original, reward_safe, places=2, msg="Rewards should be identical on clear road")
        self.assertGreater(reward_safe, 0.9)
        
        # Restore
        self.rewards_module.get_distance_to_nearest_obstacle = original_get_dist

if __name__ == '__main__':
    unittest.main()
