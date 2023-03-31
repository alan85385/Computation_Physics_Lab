# 2D Particles' Collisions
## Main Idea
### 1. Direct Calculation calculates all possible collsions' time (including the collisions of Balls-Walls and the collisions of Balls-Balls).
### 2. Choose the minimum value of them.
### 3. Evolve the system to the next collsion's time.
### 4. Based on the Direct Calculation, slice the time into dt and evolve the system with dt, that is Continuous Collision Detection.
### 5. Based on the Continuous Collision Detection, only check the overlaps on x or y directions, that is Sweep and Prune.
