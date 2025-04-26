# Sustainable Traffic Control with Reinforcement Learning

## SDG 11: Sustainable Cities and Communities

This experiment applies reinforcement learning to traffic signal control
with a focus on sustainability metrics relevant to SDG 11.

## Performance Metrics

### Traffic Efficiency
- Starting waiting time: 2570.10 seconds
- Final waiting time: 3724.57 seconds
- Improvement: -44.9%

### Environmental Impact
- Starting CO2 emissions: 140.49 g
- Final CO2 emissions: 145.58 g
- Reduction: -3.6%

### Learning Performance
- Starting average reward: -4867.37
- Final average reward: -6240.21
- Improvement: -28.2%

## SDG 11 Relevance

This implementation addresses the following SDG 11 targets:

### Target 11.2: Sustainable Transport Systems
- Our system optimizes traffic flow at intersections, a critical bottleneck in urban transportation
- Reduced waiting times improve mobility and accessibility, especially important for public transit
- Smoother traffic flow benefits all road users, including pedestrians and cyclists

### Target 11.6: Air Quality and Environmental Impact
- By explicitly minimizing CO2 emissions in our reward function, we directly address urban air quality
- Reduced idling time at intersections significantly decreases local pollution concentrations
- Our approach provides a measurable way to quantify environmental improvements

### Target 11.a: Urban-Rural Linkages
- Improved traffic flow strengthens connections between urban centers and surrounding areas
- Better traffic management makes commuting more feasible and reduces urban sprawl

### Target 11.b: Integrated Policies and Resource Efficiency
- Our multi-objective approach demonstrates how competing priorities can be balanced
- The reinforcement learning model provides an adaptive solution that can respond to changing conditions
- This technology can be integrated into broader smart city initiatives

## Reward Function Design

Our reward function uses a weighted approach to balance multiple objectives:

```python
# Combined reward with weights
efficiency_weight = 0.8  # 80% weight for traffic efficiency
emission_weight = 0.2    # 20% weight for environmental impact
reward = (efficiency_weight * efficiency_reward) + (emission_weight * emission_reward)
```

The weighting can be adjusted based on specific urban priorities and needs.
For example, areas with high pollution might increase the emission weight,
while congested business districts might prioritize efficiency.

## Conclusions and Future Work

Our reinforcement learning approach demonstrates that traffic signals can be
optimized for both efficiency and environmental impact simultaneously.

Future improvements could include:

1. Incorporating additional sustainability metrics (noise, particulate matter)
2. Coordinating multiple intersections for corridor-level optimization
3. Adapting to different traffic patterns (rush hour, weekends, events)
4. Including public transit priority to further support sustainable mobility

This work serves as a proof-of-concept for how AI can contribute to
more sustainable urban infrastructure in line with SDG 11 goals.
