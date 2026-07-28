from src.cli import main
import pytest
from pathlib import Path

@pytest.fixture
def sample_image():
    """ Get first available image from data/raw/"""
    raw_dir = Path("data/raw")
    images = list(raw_dir.glob("*.jpg"))
    if not images:
        pytest.skip("No images found in data/raw/")
    return str(images[0])

def test_cli_creates_output_file(tmp_path, sample_image):
    output_path = tmp_path / "compressed.jpg"
    main([sample_image, "--rank", "10", "--output", str(output_path)])
    assert output_path.exists()

def test_cli_prints_metrics(tmp_path, capsys, sample_image):
    output_path = tmp_path / "compressed.jpg"
    main([sample_image, "--rank", "10", "--output", str(output_path)])

    captured = capsys.readouterr()
    assert "PSNR" in captured.out
    assert "SSIM" in captured.out

def test_cli_color_flag(tmp_path, sample_image):
    output_path = tmp_path / "compressed_color.jpg"
    main([sample_image, "--rank", "10", "--output", str(output_path), "--color"])

    assert output_path.exists()

def test_cli_randomized_flag(tmp_path, sample_image):
    output_path = tmp_path / "compressed_rand.jpg"
    main([sample_image, "--rank", "10", "--output", str(output_path), "--randomized"])
    assert output_path.exists()

def test_cli_tucker_randomized_rejected(tmp_path, sample_image):
    output_path = tmp_path / "t.jpg"
    with pytest.raises(SystemExit):
        main([sample_image, "--rank", "10", "--output", str(output_path),
              "--color", "--tucker", "--randomized"])
    assert not output_path.exists()

def test_cli_rejects_nonpositive_rank(tmp_path, sample_image):
    output_path = tmp_path / "t.jpg"
    with pytest.raises(SystemExit):
        main([sample_image, "--rank", "0", "--output", str(output_path)])
    assert not output_path.exists()