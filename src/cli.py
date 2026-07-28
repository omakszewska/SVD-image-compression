import argparse
def build_parser():
    parser = argparse.ArgumentParser(description="Compress an image using SVD")
    parser.add_argument("input_path", help="Path to the input image")
    parser.add_argument("--rank", type=positive_int, required=True,
                    help="Number of singular values to keep (k), must be >= 1")
    parser.add_argument("--output", required=True, help="Path to save the compressed image")
    parser.add_argument("--color", action="store_true", help="Compress in RGB instead of grayscale")
    parser.add_argument("--randomized", action="store_true", help="Use randomized SVD for compression")
    
    method_group = parser.add_mutually_exclusive_group()
    method_group.add_argument("--flattened", action="store_true", help="Use flattened RGB compression instead of per-channel")
    method_group.add_argument("--tucker", action="store_true", help="Use Tucker (HOSVD) tensor decomposition instead of per-channel")

    return parser

def positive_int(value):
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError(f"rank must be >= 1, got {n}")
    return n

def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.tucker and args.randomized:
        parser.error("--randomized does not apply to --tucker (Tucker uses its own decomposition)")

    from PIL import Image
    import numpy as np
    from src.compression import (
        compress_grayscale, 
        compress_rgb_per_channel, 
        compress_rgb_flattened,
        compress_rgb_tucker,
        reconstruct_matrix
    )
    from src.randomized_svd import randomized_svd
    from src.metrics import psnr, ssim, compression_ratio, tucker_compression_ratio

    def compress_2d(matrix, k):
        if args.randomized:
            U, S, Vt = randomized_svd(matrix.astype(np.float64), k)
            return reconstruct_matrix(U, S, Vt)
        return compress_grayscale(matrix, k)

    image = Image.open(args.input_path)
    array = np.array(image.convert("RGB" if args.color else "L"))

    if args.color:
        if args.tucker:
            ranks = [args.rank, args.rank, args.rank]
            compressed, ranks_used = compress_rgb_tucker(array, ranks)
            ratio = tucker_compression_ratio(array.shape, ranks_used)
        elif args.flattened:
            flattened_shape = (array.shape[0], array.shape[1] * 3)
            compressed = compress_2d(array.reshape(flattened_shape), args.rank).reshape(array.shape)
            ratio = compression_ratio(flattened_shape, args.rank)
        else:
            compressed = np.stack(
                [compress_2d(array[:, :, i], args.rank) for i in range(3)], axis=2
            )
            ratio = compression_ratio(array.shape, args.rank)
    else:
        compressed = compress_2d(array, args.rank)
        ratio = compression_ratio(array.shape, args.rank)

    compressed_image = Image.fromarray(np.clip(compressed, 0, 255).astype(np.uint8))
    compressed_image.save(args.output)
    print(f"Compressed image saved to {args.output}")

    compressed_clipped = np.clip(compressed, 0, 255).astype(np.uint8)
    psnr_value = psnr(array, compressed_clipped)
    ssim_value = ssim(array, compressed_clipped)
    
    print(f"PSNR: {psnr_value:.2f} dB")
    print(f"SSIM: {ssim_value:.4f}")
    print(f"Compression Ratio: {ratio:.2f}x")


if __name__ == "__main__":
    main()