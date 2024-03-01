from pathlib import Path
import typer
from bonn import TUMDataset
from color_icp import ColorIcp
import numpy as np

docstring = "write your help here"
app = typer.Typer(rich_markup_mode="rich")

def get_fps(times):
    total_time_s = sum(times) * 1e-9
    return float(len(times) / total_time_s)


@app.command(help=docstring)
def color_icp_pipeline(
    data: Path = typer.Argument(..., help="your datset path", show_default=False),
    output_file: str = typer.Argument("tum_trajectory", help="output path "),
):
    dataset = TUMDataset(data)
    pipeline = ColorIcp(dataset, output_file)
    pipeline.run()
    pipeline.save_to_tum()
    np.savetxt(output_file + "_gt.txt", dataset.gt_poses.astype(np.float64))
    print(int(np.floor(get_fps(pipeline.times))))
    print(data)


if __name__ == "__main__":
    app()
