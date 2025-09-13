import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import logging

class DataGenerator:
    def __init__(self, num_events=10000):
        self.num_events = num_events
        self.logger = logging.getLogger(__name__)
        
    def generate_sample_data(self):
        """Generate realistic event ticket data"""
        events = []
        
        # Event types and base prices
        event_types = ['Concert', 'Sports', 'Theater', 'Comedy', 'Conference']
        base_prices = {
            'Concert': 50,
            'Sports': 40,
            'Theater': 35,
            'Comedy': 25,
            'Conference': 100
        }
        
        for i in range(self.num_events):
            event_type = random.choice(event_types)
            base_price = base_prices[event_type]
            
            # Generate event data
            event_data = {
                'event_id': f'event_{i}',
                'event_type': event_type,
                'base_price': base_price,
                'current_price': base_price,
                'demand_level': random.uniform(0.1, 1.0),
                'time_to_event': random.randint(1, 365),
                'seats_available': random.randint(50, 1000),
                'seats_sold': random.randint(0, 500),
                'competitor_price': base_price * random.uniform(0.8, 1.2),
                'season_factor': random.uniform(0.8, 1.2),
                'day_of_week': random.randint(0, 6),
                'weather_factor': random.uniform(0.9, 1.1),
                'timestamp': datetime.now() - timedelta(days=random.randint(0, 365))
            }
            
            events.append(event_data)
        
        df = pd.DataFrame(events)
        return df
    
    def save_data(self, filepath='./data/sample_data.csv'):
        """Save generated data to CSV"""
        df = self.generate_sample_data()
        df.to_csv(filepath, index=False)
        self.logger.info(f"Generated {len(df)} sample events and saved to {filepath}")
        return df

if __name__ == "__main__":
    generator = DataGenerator(10000)
    df = generator.save_data()
    print(f"Generated {len(df)} events")
    print(df.head())