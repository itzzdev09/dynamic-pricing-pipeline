import os
from dataclasses import dataclass

@dataclass
class Config:
    # Kafka Settings
    KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
    KAFKA_TOPIC_EVENTS = 'event_tickets'
    KAFKA_TOPIC_PRICING = 'pricing_updates'
    
    # Redis Settings
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
    REDIS_DB = int(os.getenv('REDIS_DB', 0))
    
    # Model Settings
    LEARNING_RATE = 0.1
    DISCOUNT_FACTOR = 0.95
    EPSILON = 0.1
    BATCH_SIZE = 32
    
    # Simulation Settings
    NUM_EVENTS = 10000
    SIMULATION_DAYS = 30