import os
import re
import psutil
import warnings

import numpy as np
from scipy.signal import savgol_filter

from matplotlib import pyplot as plt
from matplotlib.colors import Colormap

import pyvista as pv

from snn.utils.data import SNNData
from snn.utils.visual.embedding import Embedder, SklearnEmbedder

ALPHA = 0.5


class Explorer:
    def __init__(
        self, data: SNNData, embedder: Embedder | SklearnEmbedder = None, fit: bool = False, cmap: str | Colormap = None
    ):
        self.data = data

        # transfer to CPU if necessary.
        self.data.buffer = self.data.buffer.to("cpu")

        # an embedding method is completely optional.
        if embedder is not None:
            self.embedder = embedder
            self.projection = self.fit() if fit else None

        else:
            # data will not be projected to a lower dimension, and calling fit() will do nothing.
            self.embedder = None
            self.projection = self.data

        self.cmap = plt.get_cmap("viridis") if cmap is None else cmap

        # check for jupyter context.
        self.is_notebook = self.is_notebook()

    def fit(self, *args, **kwargs) -> SNNData:
        """Project data to lower dimension using the given embedder."""
        if self.embedder is not None:
            self.projection = self.embedder(*args, **kwargs)

        return self.projection

    def scatter2d(self, key: str):
        fig = plt.figure()
        fig.add_subplot()

        color = self.projection.metadata[key][:] if key else None
        plt.scatter(self.projection[:, 0], self.projection[:, 1], c=color)

    def scatter3d(self, key: str):
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        color = self.projection.metadata[key][:] if key else None
        ax.scatter(self.projection[:, 0], self.projection[:, 1], self.projection[:, 2], c=color)

    def spline3d(self, key: str):
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        color = self.projection.metadata[key][:] if key else None
        for cls in set(self.projection.metadata[key]):
            d = self.projection.trim(self.projection.select(conditions=[f"{key}=={cls}"]))

            x_fine = savgol_filter(d[:, 0], 3, 2, mode="wrap")
            y_fine = savgol_filter(d[:, 1], 3, 2, mode="wrap")
            z_fine = savgol_filter(d[:, 2], 3, 2, mode="wrap")

            ax.plot(x_fine, y_fine, z_fine, alpha=ALPHA)

        ax.scatter(self.projection[:, 0], self.projection[:, 1], self.projection[:, 2], c=color)

    def surf3d(self, key: str):
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        for cls in set(self.projection.metadata[key]):
            d = self.projection.trim(self.projection.select(conditions=[f"{key}=={cls}"]))
            ax.plot_trisurf(d[:, 0], d[:, 1], d[:, 2], alpha=ALPHA, antialiased=True)

        plt.show()

    @staticmethod
    def is_notebook():
        return any(re.search("jupyter", x) for x in psutil.Process().parent().cmdline())

    def vista3d(self, key: str, projection: str = None, interactive: bool = True, path: str = "", **kwargs):
        # suppress warnings.
        warnings.filterwarnings("ignore", category=UserWarning)

        # create a connected point cloud representation by computing class-specific surfaces.
        wrap = "'" if isinstance(self.projection.metadata[key][0], str) else ""
        surfaces = [self.projection.select(f"{key}=={wrap}{i}{wrap}") for i in set(self.projection.metadata[key][:])]
        vtk_format = []

        title = kwargs.pop("title", "")

        for surf_indices in surfaces:
            # FIX np.ndarray has no index_select in projection.trim when axis != None.
            vtk_format.append(len(self.projection.trim(surf_indices)[:]))
            vtk_format.extend(surf_indices)

        cloud = pv.PolyData(np.array(self.projection[:]), vtk_format)

        # add labels and colors.
        cloud[key] = self.projection.metadata[key][:]
        _, colors = np.unique(self.projection.metadata[key][:], return_inverse=True)

        if projection == "sphere":
            surf = pv.Sphere(radius=200, theta_resolution=400, phi_resolution=400)
            surf = surf.interpolate(cloud, radius=1, strategy="closest_point", pass_point_data=True)
        elif projection == "torus":
            surf = pv.ParametricTorus(200, 20)
            surf = surf.interpolate(cloud, radius=1, strategy="closest_point", pass_point_data=True)
        elif projection == "diamond":
            surf = cloud.delaunay_3d()
            surf.interpolate(cloud, radius=1, strategy="closest_point", pass_point_data=True)
        else:
            surf = cloud

        # plot class surfaces.
        plotter = pv.Plotter(notebook=self.is_notebook, off_screen=not interactive, window_size=(1920, 1080))
        plotter.add_mesh(
            cloud,
            scalars=colors,
            cmap=self.cmap,
            scalar_bar_args={"title": key, "n_labels": len(set(cloud[key])), "fmt": "%.1f"},
            **kwargs,
        )

        plotter.add_title(title)
        plotter.add_mesh(surf, cmap=self.cmap, opacity=0.1)
        plotter.background_color = "white"

        image = plotter.show(jupyter_backend="ipygany", interactive=interactive, return_img=True)

        if not self.is_notebook:
            if not interactive:
                # show a static plot; redundant for notebooks.
                plt.imshow(image)

                plt.axis("off")
                plt.title(title)

            plt.show()

        if path and not interactive:
            plt.imsave(path, image)
