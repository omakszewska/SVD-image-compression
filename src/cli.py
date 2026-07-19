import argparse
def build_parser():
    parser = argparse.ArgumentParser(description="Compress an image using SVD")
    parser.add_argument("input_path", help="Path to the input image")
    parser.add_argument("--rank", type=int, required=True, help="Number of singular values to keep (k)")
    parser.add_argument("--output", required=True, help="Path to save the compressed image")
    parser.add_argument("--color", action="store_true", help="Compress in RGB instead of grayscale")
    parser.add_argument("--flattened", action="store_true", help="Use flattened RGB compression instead of per-channel")
    return parser

def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    from PIL import Image
    import numpy as np
    from src.compression import compress_grayscale, compress_rgb_per_channel, compress_rgb_flattened

    image = Image.open(args.input_path)
    array = np.array(image.convert("RGB" if args.color else "L"))

    if args.color:
        if args.flattened:
            compressed = compress_rgb_flattened(array, args.rank)
        else:
            compressed = compress_rgb_per_channel(array, args.rank)
    else:
        compressed = compress_grayscale(array, args.rank)

    compressed_image = Image.fromarray(np.clip(compressed, 0, 255).astype(np.uint8))
    compressed_image.save(args.output)
    print(f"Compressed image saved to {args.output}")


if __name__ == "__main__":
    main()