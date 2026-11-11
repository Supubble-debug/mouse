"""mouse training entrypoint (stub)."""
import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="mouse training")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_ROOT", "output"))
    parser.add_argument("--data-dir", default=os.environ.get("DATA_ROOT", "data"))
    args = parser.parse_args()
    print(f"config={args.config} output_dir={args.output_dir} data_dir={args.data_dir}")
    raise NotImplementedError("training not implemented yet")


if __name__ == "__main__":
    main()
