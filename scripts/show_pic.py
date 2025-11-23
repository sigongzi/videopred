import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--num", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    with h5py.File(args.input, "r") as f:
        target = f["target"][:5]
        generated = f["generated"][:5]
        prompts = f["prompt"][:5].astype(str) if "prompt" in f else []

    n = min(len(target), args.num)
    print(f"n is {n}")
    fig, axes = plt.subplots(n, 2, figsize=(8, 3 * n))
    if n == 1:
        axes = np.array([axes])

    for i in range(n):
        axes[i, 0].imshow(target[i])
        axes[i, 0].axis("off")
        axes[i, 0].set_title("target")

        axes[i, 1].imshow(generated[i])
        axes[i, 1].axis("off")
        title = "generated"
        if i < len(prompts):
            title += f"\n{prompts[i]}"
        axes[i, 1].set_title(title)

    plt.tight_layout()
    if args.output:
        plt.savefig(args.output, dpi=200)
    plt.show()

if __name__ == "__main__":
    main()