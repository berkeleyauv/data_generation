import argparse
from generate import main as generate


def main(
    backgrounds_dir,
    real_targets_dir,
    fake_targets_dir,
    output_img_dir,
    output_yolo_dir,
    max_attempts,
    num_backgrounds=None,
):
    return generate(
        backgrounds_dir,
        real_targets_dir,
        fake_targets_dir,
        output_img_dir,
        output_yolo_dir,
        max_attempts,
        num_backgrounds,
        multiple=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--backgrounds_dir", required=True)
    parser.add_argument("--real_targets_dir", required=True)
    parser.add_argument(
        "--fake_targets_dir",
        default=None,
        help="Folder with fake target images (optional)",
    )

    parser.add_argument("--output_img_dir", required=True)
    parser.add_argument("--output_yolo_dir", required=True)

    parser.add_argument("--max_attempts", type=int, default=20)
    parser.add_argument(
        "--num_backgrounds",
        type=int,
        default=None,
        help="Number of backgrounds to process",
    )

    args = parser.parse_args()

    main(
        args.backgrounds_dir,
        args.real_targets_dir,
        args.fake_targets_dir,
        args.output_img_dir,
        args.output_yolo_dir,
        args.max_attempts,
        args.num_backgrounds,
    )
