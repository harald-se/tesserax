import io
from pathlib import Path
from typing import BinaryIO, Literal
import imageio
import cairosvg
from .canvas import Canvas


class Scene:
    def __init__(
        self, canvas: Canvas, fps: int = 30, dpi: int = 96, background: str = "white"
    ) -> None:
        self.canvas = canvas
        self.fps = fps
        self.dpi = dpi
        self.background = background
        self._frames: list[bytes] = []

    def capture(self) -> None:
        """
        Snapshots the current state of the canvas into a PNG buffer.
        """
        # 1. Generate SVG String
        svg_string = self.canvas._build_svg()

        # 2. Rasterize to PNG bytes (In-Memory)
        # We assume standard sRGB output
        png_data: bytes = cairosvg.svg2png(
            bytestring=svg_string.encode("utf-8"),
            dpi=self.dpi,
            background_color=self.background,
        ) # type: ignore

        self._frames.append(png_data)

    def save(
        self,
        destination: str | Path | BinaryIO,
        format: Literal["gif", "mp4", None] = None,
    ) -> None:
        """
        Compiles captured frames and saves to the destination.

        Args:
            destination: File path (str/Path) or a file-like object.
            format: 'gif', 'mp4', etc. Optional if destination is a path with extension.
        """
        if not self._frames:
            print("No frames captured!")
            return

        # 1. Prepare Images
        images = []
        for png_bytes in self._frames:
            images.append(imageio.imread(io.BytesIO(png_bytes)))

        # 2. Determine Format
        if format is None:
            if isinstance(destination, (str, Path)):
                # Infer from extension (e.g. "video.mp4" -> "mp4")
                format = Path(destination).suffix.strip(".")
            else:
                format = "gif"  # Default fallback for file objects

        # 3. Generate to In-Memory Buffer
        buffer = io.BytesIO()

        # ImageIO args based on format
        kwargs = {"fps": self.fps}
        if format == "gif":
            kwargs["loop"] = 0  # Infinite loop
        elif format == "mp4":
            kwargs["quality"] = 8  # High quality

        # Write to the buffer
        imageio.mimsave(buffer, images, format=format, **kwargs)
        buffer.seek(0)

        # 4. Write to Destination
        if isinstance(destination, (str, Path)):
            with open(destination, "wb") as f:
                f.write(buffer.getbuffer())
        else:
            destination.write(buffer.getbuffer())

    def display(self):
        """
        Renders the animation as a GIF and returns an IPython Image object
        for inline display in Jupyter/Colab notebooks.
        """
        try:
            from IPython.display import Image, display
        except ImportError:
            raise ImportError("IPython is required for Scene.display()")

        buffer = io.BytesIO()
        self.save(buffer, format="gif")
        display(Image(data=buffer.getvalue(), format="gif"))
