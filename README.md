# Fluid-Flow-Simulation-Using-Physics-Informed-Neural-Networks
🧠 Fluid Flow Simulation Using Physics-Informed Neural Networks (PINNs)

This MATLAB project simulates 2D incompressible Navier–Stokes flow using Physics-Informed Neural Networks (PINNs). It supports a variety of geometries and boundary conditions to model internal flows like in pipes, channels, nozzles, and around obstacles.

🚀 Features
- GUI-based Setup: Define grid size, time steps, inlet profile, and geometry using input dialogs.
- Multiple Geometries: Includes pipe, channel, nozzle, bend, t-junction, obstacle, and cylinder.
- Custom Inlet Profiles: Choose between constant or sinusoidal inflow velocity.
- PINN-based Solver: Uses deep learning to approximate velocity and pressure fields by minimizing Navier–Stokes residuals and enforcing boundary/initial conditions.
- Training Loss Tracking: Tracks PDE, boundary, wall, outlet, and initial losses during training.
- Interactive Visualization:
  - Prompt user for a specific time (t) to visualize final fields.
  - Animate velocity and pressure fields over a chosen time interval.

📊 Outputs
- Contour and Quiver Plots of velocity magnitude and streamlines.
- Surface Plot of pressure fields.
- Training Loss Curve over epochs.
- Real-time Animation of velocity and pressure fields evolving with time.

🛠️ Requirements
- MATLAB with Deep Learning Toolbox.
- No external dependencies—entirely self-contained.
