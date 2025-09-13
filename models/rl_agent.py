import numpy as np
import pandas as pd
import pickle
import logging
from collections import defaultdict

class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: np.zeros(5))  # 5 pricing actions
        self.logger = logging.getLogger(__name__)
        
    def get_state_key(self, event_data):
        """Convert event data to discrete state"""
        demand_bin = min(int(event_data['demand_level'] * 10), 9)
        time_bin = min(int(event_data['time_to_event'] / 30), 11)  # 12 time bins
        seats_bin = min(int((event_data['seats_sold'] / event_data['seats_available']) * 10), 9) if event_data['seats_available'] > 0 else 0
        
        return f"{event_data['event_type']}_{demand_bin}_{time_bin}_{seats_bin}"
    
    def get_action(self, state_key, training=True):
        """Select action using epsilon-greedy policy"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, 5)  # Random action
        else:
            return np.argmax(self.q_table[state_key])  # Best action
    
    def get_price_multiplier(self, action):
        """Convert action to price multiplier"""
        multipliers = [0.8, 0.9, 1.0, 1.1, 1.2]  # Decrease, slight decrease, maintain, increase, big increase
        return multipliers[action]
    
    def update_q_table(self, state, action, reward, next_state):
        """Update Q-table using Q-learning update rule"""
        current_q = self.q_table[state][action]
        next_max_q = np.max(self.q_table[next_state])
        
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_max_q - current_q)
        self.q_table[state][action] = new_q
    
    def calculate_reward(self, event_data, new_price, old_price):
        """Calculate reward based on revenue and demand"""
        seats_sold = event_data['seats_sold']
        seats_available = event_data['seats_available']
        base_price = event_data['base_price']
        
        # Revenue calculation
        revenue_change = (new_price - old_price) * seats_sold
        
        # Demand sensitivity penalty
        price_change = (new_price - old_price) / old_price if old_price > 0 else 0
        demand_impact = -abs(price_change) * seats_sold * 0.1
        
        # Inventory pressure
        inventory_pressure = 0
        if seats_available > 0:
            inventory_ratio = seats_sold / seats_available
            if inventory_ratio > 0.8:  # High demand
                inventory_pressure = 50
            elif inventory_ratio < 0.3:  # Low demand
                inventory_pressure = -20
        
        total_reward = revenue_change + demand_impact + inventory_pressure
        return total_reward
    
    def save_model(self, filepath):
        """Save trained model"""
        with open(filepath, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model"""
        try:
            with open(filepath, 'rb') as f:
                self.q_table = defaultdict(lambda: np.zeros(5), pickle.load(f))
            self.logger.info(f"Model loaded from {filepath}")
        except FileNotFoundError:
            self.logger.warning(f"Model file {filepath} not found, starting with empty Q-table")