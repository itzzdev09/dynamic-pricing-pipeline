"""Tests for the Q-learning agent's action selection."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rl_agent import QLearningAgent  # noqa: E402


@pytest.fixture
def agent():
    return QLearningAgent()


def test_an_unseen_state_holds_price_rather_than_discounting(agent):
    """Regression: argmax over an all-zero row returns index 0, a 20% discount.

    Serving a state the agent had never trained on therefore cut the price by a
    fifth, silently.
    """
    action = agent.get_action('Concert_5_3_7', training=False)
    assert agent.get_price_multiplier(action) == 1.0


def test_inference_does_not_grow_the_q_table(agent):
    """Regression: indexing a defaultdict inserts, so every read added a row.

    The table grew unboundedly while serving, and save_model persisted the
    empty states.
    """
    for state in ['Concert_1_1_1', 'Sports_2_2_2', 'Theater_3_3_3']:
        agent.get_action(state, training=False)
    assert len(agent.q_table) == 0


def test_a_trained_state_uses_its_best_action(agent):
    agent.q_table['Concert_5_3_7'] = np.array([0.0, 0.0, 0.0, 9.0, 0.0])
    action = agent.get_action('Concert_5_3_7', training=False)
    assert action == 3
    assert agent.get_price_multiplier(action) == 1.1


def test_action_is_a_plain_int_so_it_indexes_and_pickles(agent):
    agent.q_table['Concert_5_3_7'] = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    assert type(agent.get_action('Concert_5_3_7', training=False)) is int

    exploring = QLearningAgent(epsilon=1.0)
    assert type(exploring.get_action('anything', training=True)) is int


def test_every_action_maps_to_a_multiplier(agent):
    assert [agent.get_price_multiplier(a) for a in range(5)] == [0.8, 0.9, 1.0, 1.1, 1.2]


def test_learning_still_records_states(agent):
    agent.update_q_table('Concert_5_3_7', 3, reward=10.0, next_state='Concert_5_4_7')
    assert agent.q_table['Concert_5_3_7'][3] > 0
