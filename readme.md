#file structure

dynamic-pricing-pipeline/
├── config/
│   ├── __init__.py
│   └── settings.py
├── data/
│   ├── __init__.py
│   ├── data_generator.py
│   └── sample_data.csv
├── models/
│   ├── __init__.py
│   ├── rl_agent.py
│   ├── pricing_model.py
│   └── demand_forecast.py
├── pipeline/
│   ├── __init__.py
│   ├── kafka_producer.py
│   ├── kafka_consumer.py
│   └── redis_cache.py
├── dashboard/
│   ├── __init__.py
│   ├── powerbi_connector.py
│   └── kpi_calculator.py
├── utils/
│   ├── __init__.py
│   ├── logger.py
│   └── helpers.py
├── tests/
│   ├── __init__.py
│   ├── test_rl_agent.py
│   └── test_pricing_model.py
├── main.py
├── requirements.txt
├── README.md
└── .env