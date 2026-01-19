import numpy as np

# ==========================================
# Basic Definitions
# ==========================================
class DroneActionSpace:
    def __init__(self):
        self.V_XY = 0.5
        self.V_Z = 0.3
        self.W_Z = 0.5
        self.actions = {
            0: np.array([0.0, 0.0, 0.0, 0.0]), 1: np.array([self.V_XY, 0.0, 0.0, 0.0]),
            2: np.array([-self.V_XY, 0.0, 0.0, 0.0]), 3: np.array([0.0, self.V_XY, 0.0, 0.0]),
            4: np.array([0.0, -self.V_XY, 0.0, 0.0]), 5: np.array([0.0, 0.0, self.V_Z, 0.0]),
            6: np.array([0.0, 0.0, -self.V_Z, 0.0]), 7: np.array([0.0, 0.0, 0.0, self.W_Z]),
            8: np.array([0.0, 0.0, 0.0, -self.W_Z])
        }
        self.num_actions = len(self.actions)
        self.opposite_actions = {1: 2, 2: 1, 3: 4, 4: 3, 5: 6, 6: 5, 7: 8, 8: 7}

    def get_velocity(self, action_id):
        return self.actions.get(action_id, np.zeros(4))

    def sample(self, prev_action=None):
        if prev_action is None: return np.random.randint(0, self.num_actions)
        probs = np.ones(self.num_actions)
        probs[prev_action] *= 10.0
        if prev_action in self.opposite_actions: probs[self.opposite_actions[prev_action]] *= 0.01
        probs[0] *= 0.5
        probs /= np.sum(probs)
        return np.random.choice(self.num_actions, p=probs)
