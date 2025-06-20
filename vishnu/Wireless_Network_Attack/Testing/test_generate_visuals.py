import os
import sys
import subprocess
import pytest

def test_generate_visuals_script_creates_images(tmp_path):
    # Create a temp folder for visuals
    output_dir = tmp_path / "visuals"
    output_dir.mkdir()

    # Path to your generate_visuals.py script
    script_path = os.path.join(os.path.dirname(__file__),"..", "generate_visuals.py")
    script_path = os.path.abspath(script_path)
    # Copy the original script to a temp file (unmodified)
    temp_script_path = tmp_path / "temp_generate_visuals.py"
    with open(script_path, "r") as f:
        content = f.read()
    with open(temp_script_path, "w") as f:
        f.write(content)

    # Set environment variable to point output_dir to temp folder
    env = os.environ.copy()
    env["VISUALS_OUTPUT_DIR"] = str(output_dir)

    # Run the script as a subprocess using the current python interpreter
    result = subprocess.run([sys.executable, str(temp_script_path)], capture_output=True, text=True, env=env)

    # Assert the script ran successfully
    assert result.returncode == 0, f"Script failed with error:\n{result.stderr}"

    # Check that all expected image files are created
    expected_files = [
        "enhanced_scatter_src_dst.png",
        "violinplot_duration.png",
        "styled_correlation_heatmap.png",
        "donut_attack_distribution.png"
    ]

    for filename in expected_files:
        file_path = output_dir / filename
        assert file_path.exists(), f"Expected image file not found: {file_path}"
