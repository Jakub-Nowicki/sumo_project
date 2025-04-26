# Sustainable Traffic Control with Reinforcement Learning

## SDG 11: Sustainable Cities and Communities

This experiment applies reinforcement learning to traffic signal control
with a focus on sustainability metrics relevant to SDG 11.

## Performance Metrics

### Traffic Efficiency
- Starting waiting time: 580.62 seconds
- Final waiting time: 239.50 seconds
- Improvement: 58.8%

### Environmental Impact
- Starting CO2 emissions: 76.68 g
- Final CO2 emissions: 69.67 g
- Reduction: 9.1%

### Learning Performance
- Starting average reward: -1286.62
- Final average reward: -686.61
- Improvement: 46.6%

## SDG 11 Relevance

This implementation addresses the following SDG 11 targets:

1. **Sustainable Transportation**: By optimizing traffic flow and reducing waiting times
2. **Environmental Sustainability**: By explicitly considering and reducing vehicle emissions
3. **Resource Efficiency**: By making better use of existing infrastructure

The multi-objective reward function balances these priorities with weights of:
- 80% for traffic efficiency (waiting times and queue length)
- 20% for environmental impact (CO2 emissions)

This weighting can be adjusted based on specific urban priorities and needs.
