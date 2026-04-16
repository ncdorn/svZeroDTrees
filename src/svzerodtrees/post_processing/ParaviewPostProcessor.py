
#### ParaView Python Script Template: Visualization of Preop, Postop, and Adapted Hemodynamics ####
# Usage: Run with `pvpython` or load into the ParaView Python Shell
import sys
sys.path.append("/Applications/ParaView-5.12.0-RC3.app/Contents/Python")
from vtkmodules.util.numpy_support import vtk_to_numpy
import glob
from paraview.simple import *
from svzerodtrees.preop.preop import ClinicalTargets
import os
from pathlib import Path
from PIL import Image

class ParaviewPostProcessor:
    def __init__(self, preop_vtu, postop_vtu, adapted_vtu, figures_dir, clinical_targets: ClinicalTargets):
        self.preop_vtu = sorted(glob.glob(preop_vtu))
        self.postop_vtu = sorted(glob.glob(postop_vtu))
        self.adapted_vtu = sorted(glob.glob(adapted_vtu))
        self.clinical_targets = clinical_targets
        self.figures_dir = Path(figures_dir)

        # Load the data
        self.preop = OpenDataFile(self.preop_vtu)
        self.postop = OpenDataFile(self.postop_vtu)
        self.adapted = OpenDataFile(self.adapted_vtu)

        self.animation_scene = GetAnimationScene()
        self.animation_scene.PlayMode = 'Snap To TimeSteps'


    def pressure_animation(self):

        # Set up layout with three views
        self.layout = CreateLayout("Side-by-Side Time Series Views")
        self.view1 = CreateRenderView()
        self.view2 = CreateRenderView()
        self.view3 = CreateRenderView()
        AssignViewToLayout(view=self.view1, layout=self.layout, hint=1)
        AssignViewToLayout(view=self.view2, layout=self.layout, hint=2)
        AssignViewToLayout(view=self.view3, layout=self.layout, hint=3)

        # Link cameras for synchronized view manipulation
        AddCameraLink(self.view1, self.view2)
        AddCameraLink(self.view1, self.view3)

        print("Generating pressure animation...")
        # Setup figures directory
        pressure_animation_dir = self.figures_dir / "pressure-animation"
        pressure_animation_dir.mkdir(parents=True, exist_ok=True)

        # Display setup
        for dataset, view, label in zip(
            [self.preop, self.postop, self.adapted],
            [self.view1, self.view2, self.view3],
            ['Preop', 'Postop', 'Adapted']
        ):
            display = Show(dataset, view)
            display.Representation = "Surface"
            ColorBy(display, ("POINTS", "Pressure"))
            display.SetScalarBarVisibility(view, True)
            RenameSource(label, dataset)

        # Set pressure colormap range based on clinical MPA mean pressure
        pressure_mean = self.clinical_targets.mpa_p[0]
        pressure_range = [max(0, pressure_mean - 5), pressure_mean + 10]
        lut = GetColorTransferFunction("Pressure")
        for view in [self.view1, self.view2, self.view3]:
            lutColorBar = GetScalarBar(lut, view)
            lutColorBar.Title = "Pressure (mmHg)"
            lutColorBar.TitleFontSize = 10
            lutColorBar.LabelFontSize = 8

        # Reset cameras
        self.view1.ResetCamera()
        self.view2.ResetCamera()
        self.view3.ResetCamera()

        # Render and save frame-by-frame screenshots
        for t in self.animation_scene.TimeKeeper.TimestepValues:
            self.animation_scene.TimeKeeper.Time = t
            Render()
            frame_path = pressure_animation_dir / f"hemodynamics_frame_{t:.2f}.png"
            SaveScreenshot(str(frame_path), self.layout, ImageResolution=[2400, 800])

        self.pngs2gif(pressure_animation_dir, self.figures_dir / "pressure_animation.gif")

        print(f"Pressure animation saved to {self.figures_dir / 'pressure_animation.gif'}")

    @staticmethod
    def pngs2gif(png_dir: Path, gif_path: Path, fps=50):
        # Collect and sort PNGs
        png_files = sorted([f for f in png_dir.iterdir() if f.suffix == '.png'])
        images = [Image.open(str(p)) for p in png_files]

        # Convert to GIF
        duration = 1000 // fps
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0,
            optimize=False
        )
        print(f"GIF saved at: {gif_path}")

    
    def plot_mean_pressure_comparison(self, screenshot_name="mean_pressure_comparison.png"):
        """
        Show pressure distribution and mean pressure annotation for all 3 models in a side-by-side layout.
        Saves a screenshot to the figures directory.
        """
        print("Generating 3-panel mean pressure comparison...")

        # Create a new layout and views
        layout = CreateLayout("Mean Pressure Comparison")
        view1 = CreateRenderView()
        AssignViewToLayout(view=view1, layout=layout)
        layout.SplitHorizontal(0, 0.33)
        view2 = CreateRenderView()
        AssignViewToLayout(view=view2, layout=layout, hint=1)
        layout.SplitHorizontal(1, 0.5)
        view3 = CreateRenderView()
        AssignViewToLayout(view=view3, layout=layout, hint=2)

        views = [view1, view2, view3]
        datasets = [self.preop, self.postop, self.adapted]
        labels = ["Preop", "Postop", "Adapted"]

        # Compute shared color range
        pressure_mean = self.clinical_targets.mpa_p[0]
        pressure_range = [max(0, pressure_mean - 5), pressure_mean + 10]
        pressure_lut = GetColorTransferFunction("pressure")
        pressure_lut.RescaleTransferFunction(*pressure_range)

        for data, view, label in zip(datasets, views, labels):
            SetActiveSource(data)
            display = Show(data, view)
            display.Representation = "Surface"
            ColorBy(display, ("POINTS", "pressure"))
            display.SetScalarBarVisibility(view, False)  # Hide in all but one

            # Compute mean pressure
            integrate = IntegrateVariables(Input=data)
            UpdatePipeline()
            result = servermanager.Fetch(integrate)
            field_data = result.GetFieldData()

            pressure_array = field_data.GetArray("Pressure")
            volume_array = field_data.GetArray("Volume")

            pressure = vtk_to_numpy(pressure_array)[0]
            volume = vtk_to_numpy(volume_array)[0]
            mean_pressure = pressure / volume if volume > 0 else float("nan")

            # Annotate view with mean pressure
            annotation = CreateText()
            annotation.Text = f"{label} Mean Pressure:\n{mean_pressure:.2f} mmHg"
            annotation_disp = Show(annotation, view)
            annotation_disp.WindowLocation = 'UpperCenter'
            annotation_disp.TextProperty.FontSize = 10

            # Set LUT and reset camera
            lutColorBar = GetScalarBar(pressure_lut, view)
            lutColorBar.Title = "Pressure (mmHg)"
            lutColorBar.TitleFontSize = 10
            lutColorBar.LabelFontSize = 8

            view.ResetCamera()

        # Show scalar bar only in last view
        GetDisplayProperties(self.adapted, view3).SetScalarBarVisibility(view3, True)

        # Render and save
        Render()
        screenshot_path = self.figures_dir / screenshot_name
        SaveScreenshot(str(screenshot_path), layout, ImageResolution=[2400, 800])
        print(f"Saved mean pressure comparison to {screenshot_path}")


    def render_temporal_average_pressure(self, dataset_name: str = "preop", screenshot_name: str = "avg_pressure.png"):
        """
        Compute and render the time-averaged pressure field on the model surface.
        Saves a screenshot to figures_dir.
        """
        print(f"Rendering temporal average pressure for {dataset_name}...")

        dataset_map = {
            "preop": self.preop,
            "postop": self.postop,
            "adapted": self.adapted
        }

        if dataset_name not in dataset_map:
            raise ValueError(f"Invalid dataset_name: {dataset_name}. Must be one of {list(dataset_map.keys())}")

        data = dataset_map[dataset_name]

        # Step 1: Compute temporal average
        temporal_stats = TemporalStatistics(Input=data)
        temporal_stats.UpdatePipeline()

        # Step 2: Set up a new render view
        view = CreateRenderView()
        SetActiveSource(temporal_stats)
        display = Show(temporal_stats, view)
        display.Representation = "Surface"

        # Step 3: Color by average pressure
        ColorBy(display, ("POINTS", "Pressure_average"))
        pressure_lut = GetColorTransferFunction("Pressure_average")
        display.SetScalarBarVisibility(view, True)

        # Adjust scalar bar style
        scalar_bar = GetScalarBar(pressure_lut, view)
        scalar_bar.Title = "Avg Pressure (mmHg)"
        scalar_bar.TitleFontSize = 10
        scalar_bar.LabelFontSize = 8

        # Step 4: Reset camera and render
        view.ResetCamera()
        Render()

        # Step 5: Save screenshot
        screenshot_path = self.figures_dir / screenshot_name
        SaveScreenshot(str(screenshot_path), view, ImageResolution=[1200, 800])
        print(f"Saved temporal average pressure rendering to {screenshot_path}")