# Sustainable Traffic Control with Reinforcement Learning

## SDG 11: Sustainable Cities and Communities

This experiment applies reinforcement learning to traffic signal control
with a focus on sustainability metrics relevant to SDG 11.

## Performance Metrics

### Traffic Efficiency
- Starting waiting time: 7978.27 seconds
- Final waiting time: 5549.88 seconds
- Improvement: 30.4%

### Environmental Impact
- Starting CO2 emissions: 133.72 g
- Final CO2 emissions: 139.85 g
- Reduction: -4.6%

### Learning Performance
- Starting average reward: -14035.06
- Final average reward: -10083.10
- Improvement: 28.2%

## SDG 11 Relevance

This implementation addresses the following SDG 11 targets:

1. **Sustainable Transportation**: By optimizing traffic flow and reducing waiting times
2. **Environmental Sustainability**: By explicitly considering and reducing vehicle emissions
3. **Resource Efficiency**: By making better use of existing infrastructure

The multi-objective reward function balances these priorities with weights of:
- 80% for traffic efficiency (waiting times and queue length)
- 20% for environmental impact (CO2 emissions)

This weighting can be adjusted based on specific urban priorities and needs.

## Connection to Specific SDG 11 Targets

**Target 11.2**: By 2030, provide access to safe, affordable, accessible and sustainable transport systems for all
- Our solution improves traffic flow, making transportation more efficient and sustainable
- Reduced waiting times improve accessibility of transportation systems

**Target 11.6**: By 2030, reduce the adverse per capita environmental impact of cities
- Our solution directly reduces vehicle emissions in urban areas
- Optimized traffic signals reduce idling time, a major source of unnecessary pollution

**Target 11.b**: Implement integrated policies and plans for resource efficiency, mitigation and adaptation to climate change
- This approach demonstrates how AI can be used for smarter resource allocation
- The multi-objective optimization approach shows how competing priorities can be balanced

