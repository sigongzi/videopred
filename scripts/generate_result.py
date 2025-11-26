# /data1/sijiaqi/instructpix2pix/src/videopred/test_export_hdf5.py
import argparse
import os
import numpy as np
import torch
import h5py
from tqdm import tqdm

from videopred.dataset import VideoFramesDataset
from videopred.model.wrap_model import WrapInstructPix2Pix

def to_uint8(x):
    x = x.detach().cpu().numpy()
    x = (np.clip((x + 1.0) * 0.5, 0.0, 1.0) * 255.0).astype(np.uint8)
    return x

def sample_next_frame(model, original_img, cond, steps):
    with torch.no_grad():
        lat_orig = model.encode_image(original_img)
        latents = torch.randn_like(lat_orig)
        model.scheduler.set_timesteps(steps)
        for t in model.scheduler.timesteps:
            t_tensor = torch.tensor([t], device=model.device, dtype=torch.long)
            noise_pred = model.forward_unet(lat_orig, latents, t_tensor, cond)
            latents = model.scheduler.step(noise_pred, t, latents).prev_sample
        img = model.decode_latent(latents)
    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="/data1/sijiaqi/videopred/data/final_train_test/test_frames")
    parser.add_argument("--checkpoint_dir", type=str, default="/data1/sijiaqi/videopred/outputs/20251123-134058/final")
    parser.add_argument("--base_model", type=str,default="/data1/sijiaqi/videopred/model/instruct-pix2pix")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num-frames", type=int, default=20)
    parser.add_argument("--resolution", type=int, default=96)
    parser.add_argument("--max_number", type=int, required=False)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ds = VideoFramesDataset(
        root=args.data_root,
        sub_dir=["cover_object", "drop_object", "move_object"],
        num_frames=args.num_frames,
        resolution=args.resolution,
    )

    n_total = len(ds)
    n_samples = n_total
    assert n_samples > 0, "dataset is empty"

    model = WrapInstructPix2Pix(
        pretrained_model_name_or_path = args.base_model,
        checkpoint_dir = args.checkpoint_dir,
        device=args.device
    )
    model.eval()

    sample0 = ds[0]
    T = sample0["video"].shape[0]
    H = sample0["video"].shape[2]
    W = sample0["video"].shape[3]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    str_dt = h5py.string_dtype("utf-8")

    use_sample_number = args.max_number if args.max_number is not None else len(ds) 
    with h5py.File(args.output, "w") as f:
        #d_images = f.create_dataset("images", shape=(n_samples, T, H, W, 3), dtype=np.uint8)
        d_target = f.create_dataset("target", shape=(n_samples, H, W, 3), dtype=np.uint8)
        d_generated = f.create_dataset("generated", shape=(n_samples, H, W, 3), dtype=np.uint8)
        d_prompt = f.create_dataset("prompt", shape=(n_samples,), dtype=str_dt)

        for i in tqdm(range(use_sample_number), desc="export"):
            item = ds[i]
            video = item["video"].unsqueeze(0).to(args.device)
            target = item["target"].unsqueeze(0).to(args.device)
            prompt = item["prompt"]

            
            #print(f"The prompt is {prompt}")

            gen = model.predict(video, prompt)


            #print(f"max value of gen {gen.max().item()}")
            #print(f"min value of gen {gen.min().item()}")
            #print(f"min value of target {target.min().item()}")
            #print(f"max value of target {target.max().item()}")
            vid_np = to_uint8(item["video"].permute(0, 2, 3, 1))
            tgt_np = to_uint8(item["target"].permute(1, 2, 0).unsqueeze(0))[0]
            gen_np = to_uint8(gen[0].permute(1, 2, 0).unsqueeze(0))[0]

            #print(f"tgt_np:{tgt_np}")
            #print(f"gen_np:{gen_np}")
            #d_images[i] = vid_np
            d_target[i] = tgt_np
            d_generated[i] = gen_np
            d_prompt[i] = prompt

if __name__ == "__main__":
    main()