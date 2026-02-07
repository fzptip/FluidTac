# FluidTac: A Vision-Based Arrayed Artificial Lateral Line Sensor for Underwater Ego-Motion Estimation

### Video:

### Design of FluidTac

- FluidTac transforms water flow disturbances into high-dimensional, dynamic features via a circularly arranged set of passive propellers. The system utilizes an internal camera to extract passive propeller rotation features and employs a neural network to establish a nonlinear mapping between hydrodynamic responses and the body's forward velocity.

<img width="1195" height="753" alt="3b960f6f767c3038733f1b6f55b8cc3a" src="https://github.com/user-attachments/assets/d64411b4-6f23-4e84-b116-87b97e5093b0" />

### Method

- This paper first extracts angular velocity data from the eight passive propellers via image processing algorithms to construct a sliding window sequence embedding spatiotemporal features. Subsequently, leveraging the feature extraction capabilities of ResNet, a non-linear mapping is established from the multi-dimensional rotational speed space to the robot's forward velocity. Finally, to mitigate the limitations of single-sensor modalities and enhance estimation frequency, a multi-rate Kalman filter is employed to fuse the low-frequency velocity observations provided by FluidTac with high-frequency inertial data derived from an IMU using Madgwick filtering.

<img width="1782" height="657" alt="0627556e661adf7889cace5990ae4e4c" src="https://github.com/user-attachments/assets/fc47f572-66a8-40cb-b882-7a969aa8630b" />

### Experiments

- First, flow sensing experiments were conducted in a controlled water tank environment. By quantitatively analyzing the response of FluidTac to varying incident flow directions and velocities, we validated the physical basis for its flow field feature fitting. Subsequently, an indoor testing platform was established utilizing a high-precision motion capture system to assess the performance of the sensor and the estimation method across two distinct trajectories. Finally, FluidTac was integrated into an AUV for field navigation experiments in an unstructured real-world lake environment, validating the system's engineering robustness and localization accuracy under actual flow conditions.

<img width="1409" height="1027" alt="756060dbc3b08d354d71c83547de41ab" src="https://github.com/user-attachments/assets/9c18a3e9-6c3e-4845-be43-adfed7951e35" />

