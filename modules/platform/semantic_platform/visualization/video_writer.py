import time
from pathlib import Path
from datetime import datetime
from matplotlib.animation import FFMpegWriter
import matplotlib.pyplot as plt

class VideoWriter:
    """Simple video recorder."""
    def __init__(self, figure, output_path=None, fps=15):
        """
        Initialize video recorder.

        Args:
            figure: matplotlib figure object
            output_path: output path, auto-generated if None
            fps: frames per second
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"skill_execution_{timestamp}.mp4"
        
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        metadata = dict(
            title='Skill Execution Recording',
            artist='SkillExecutor',
            comment=f'Recorded at {fps}fps'
        )
        
        self.video = FFMpegWriter(fps=fps, metadata=metadata)
        self.video.setup(figure, str(self.output_path), dpi=300)
        self._frame_count = 0
        print(f"Video recording started: {self.output_path}")
    
    def update(self):
        """Capture current frame."""
        self.video.grab_frame()
        self._frame_count += 1

    def close(self):
        """Close video file."""
        self.video.finish()
        print(f"Video saved: {self.output_path} ({self._frame_count} frames)")