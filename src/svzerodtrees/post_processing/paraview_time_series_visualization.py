
#### ParaView Python Script: Load and Visualize Time Series of .vtu Files ####
# Usage: Run with `pvpython` or load into ParaView Python Shell

from paraview.simple import *

# ----------------------------
# File patterns for time series
preop_pattern = "path/to/preop_*.vtu"
postop_pattern = "path/to/postop_unadapted_*.vtu"
adapted_pattern = "path/to/postop_adapted_*.vtu"

# ----------------------------
# Load time series
preop = OpenDataFile(preop_pattern)
postop = OpenDataFile(postop_pattern)
adapted = OpenDataFile(adapted_pattern)

# Ensure time is enabled
animationScene1 = GetAnimationScene()
animationScene1.PlayMode = 'Snap To TimeSteps'

# Set up layout with three views
layout = CreateLayout("Side-by-Side Time Series Views")
view1 = CreateRenderView()
view2 = CreateRenderView()
view3 = CreateRenderView()
AssignViewToLayout(view=view1, layout=layout, index=0)
AssignViewToLayout(view=view2, layout=layout, index=1)
AssignViewToLayout(view=view3, layout=layout, index=2)

# Link cameras for synchronized view manipulation
LinkCamera(view1, view2)
LinkCamera(view1, view3)

# Visualization settings
for dataset, view, label in zip([preop, postop, adapted], [view1, view2, view3], ['Preop', 'Postop', 'Adapted']):
    display = Show(dataset, view)
    display.Representation = "Surface"
    ColorBy(display, ("POINT_DATA", "pressure"))
    display.SetScalarBarVisibility(view, True)
    RenameSource(label, dataset)

# Set consistent color map range
pressure_range = [20, 40]  # Customize based on data
pressure_lut = GetColorTransferFunction("pressure")
pressure_lut.RescaleTransferFunction(*pressure_range)

# Reset cameras
view1.ResetCamera()
view2.ResetCamera()
view3.ResetCamera()

# Loop through all timesteps and save screenshots
for t in animationScene1.TimeKeeper.TimestepValues:
    animationScene1.TimeKeeper.Time = t
    Render()
    SaveScreenshot(f"hemodynamics_frame_{t:.2f}.png", layout, ImageResolution=[2400, 800])
