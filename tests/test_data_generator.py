"""Tests for the generated sample data."""
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_generator import DataGenerator  # noqa: E402


@pytest.fixture
def frame():
    random.seed(20260911)
    return DataGenerator(num_events=3000).generate_sample_data()


def test_never_sells_more_seats_than_the_venue_has(frame):
    """Regression: seats_available and seats_sold were drawn independently.

    seats_sold came from randint(0, 500) and seats_available from
    randint(50, 1000), so roughly 21% of rows sold more tickets than existed,
    by as much as 10x capacity.
    """
    oversold = frame[frame['seats_sold'] > frame['seats_available']]
    assert oversold.empty, f'{len(oversold)} rows sold more seats than were available'


def test_sold_ratio_stays_within_the_unit_interval(frame):
    """rl_agent bins and rewards on this ratio and assumes it is <= 1."""
    ratio = frame['seats_sold'] / frame['seats_available']
    assert ratio.min() >= 0.0
    assert ratio.max() <= 1.0


def test_generates_the_requested_number_of_events(frame):
    assert len(frame) == 3000
    assert frame['event_id'].nunique() == 3000


def test_core_columns_are_present_and_sane(frame):
    for column in ['event_id', 'event_type', 'base_price', 'seats_available', 'seats_sold']:
        assert column in frame.columns
    assert (frame['seats_available'] >= 50).all()
    assert (frame['seats_available'] <= 1000).all()
    assert (frame['seats_sold'] >= 0).all()
    assert (frame['demand_level'].between(0.1, 1.0)).all()
    assert (frame['time_to_event'].between(1, 365)).all()
    assert (frame['day_of_week'].between(0, 6)).all()
