import os
import argparse
import logging
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from torchvision.utils import save_image
import lpips
import cv2

from data.dataset_video_train import DataLoaderTurbVideo
from model.TMT_DC import TMT_MS


def flow_warp(img, flow):
    assert flow.dim() == 4, f"flow must be [B, 2, H, W], but got {flow.shape}"
    B, C, H, W = img.size()
    grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    grid = torch.stack((grid_x, grid_y), 2).float().to(img.device)  # [H,W,2]
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # [B,H,W,2]

    vgrid = grid + flow.permute(0, 2, 3, 1)
    vgrid_x = 2.0 * vgrid[..., 0] / (W - 1) - 1.0
    vgrid_y = 2.0 * vgrid[..., 1] / (H - 1) - 1.0
    vgrid_norm = torch.stack((vgrid_x, vgrid_y), dim=-1)

    output = F.grid_sample(img, vgrid_norm, align_corners=True)
    return output


def percentile_clip(x, p=95, eps=1e-6):
    q = torch.quantile(x.view(x.shape[0], -1), p/100.0, dim=1, keepdim=True)
    q = q.clamp(min=eps)
    return (x / q.view(-1,1,1)).clamp(0, 1)


def sobel_edge_map(img):  # img: BxCxHxW in [0,1]
    gray = img.mean(dim=1, keepdim=True)
    kx = torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]], device=img.device, dtype=img.dtype)
    ky = torch.tensor([[[[-1,-2,-1],[0,0,0],[1,2,1]]]], device=img.device, dtype=img.dtype)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    mag = torch.sqrt(gx*gx + gy*gy)
    mag = mag / (mag.amax(dim=(2,3), keepdim=True) + 1e-6)
    return mag  # [0,1]

# def estimate_turb_level(frame, eps=1e-6):
#     if frame.dim() == 3:
#         frame = frame.unsqueeze(0)  

#     B, C, H, W = frame.shape
#     frame = frame.clamp(0, 1)

#     gray = frame.mean(dim=1, keepdim=True) 

#     lap_kernel = torch.tensor(
#         [[0.,  1., 0.],
#          [1., -4., 1.],
#          [0.,  1., 0.]],
#         device=frame.device,
#         dtype=frame.dtype
#     ).view(1, 1, 3, 3)

#     lap = F.conv2d(gray, lap_kernel, padding=1)
#     sharp = (lap ** 2).mean(dim=(1, 2, 3))         

#     sharp_norm = sharp / (sharp + 0.01)              
#     turb = 1.0 - sharp_norm                         
#     turb = turb.clamp(0.0, 1.0)

#     return turb.unsqueeze(1)    

def clip_turb_level(frame):
    with torch.no_grad():
        x = F.interpolate(frame, size=(224,224),
                          mode="bilinear", align_corners=False)

        x = (x - 0.5) / 0.5 
        
        img_feat = clip_model.encode_image(x)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        sims = img_feat @ text_feats.T 

        turb = (sims * TURB_LABELS).sum(dim=1) / sims.sum(dim=1)
        turb = turb.clamp(0,1)

        return turb.unsqueeze(1)              



@torch.no_grad()
def compute_motion_map(I_prev, I_t, flow_fw, flow_bw=None, a=0.5, b=0.5, c=0.0, d=0.3, ema_prev=None, beta=0.8):

    mag = torch.sqrt(flow_fw[:,0:1]**2 + flow_fw[:,1:2]**2)
    F_norm = percentile_clip(mag)
    
    Iprev_warp = flow_warp(I_prev, flow_fw)
    photometric = (I_t - Iprev_warp).abs().mean(dim=1, keepdim=True)
    E_norm = percentile_clip(photometric)

    if flow_bw is not None:
        fb = flow_warp(flow_bw, flow_fw) + flow_fw 
        C_norm = percentile_clip(torch.sqrt(fb[:,0:1]**2 + fb[:,1:2]**2))
    else:
        C_norm = torch.zeros_like(F_norm)

    Edge = sobel_edge_map(I_t)

    m = a*F_norm + b*E_norm + c*C_norm - d*Edge
    M = torch.tanh(m*2.0)          # [-1,1]
    M = (M + 1.0) * 0.5            # [0,1]

    if ema_prev is None:
        M_s = M
    else:
        M_s = (1.0 - beta) * M + beta * ema_prev

    S = (1.0 - M_s).clamp(0,1)  
    return S, M_s.clamp(0,1), Iprev_warp


def tensor_to_rgb_uint8(x):  # CxHxW, [0,1]
    x = x.clamp(0,1).permute(1,2,0).detach().cpu().numpy()
    return (x*255.0).astype(np.uint8)


def save_motion_overlay_video(frames_rgb, motions_01, out_path, fps=25):
    if len(frames_rgb) == 0: 
        return
    H, W, _ = frames_rgb[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    for img, m in zip(frames_rgb, motions_01):
        m_np = (m.detach().cpu().numpy()*255).astype(np.uint8)
        heat_bgr = cv2.applyColorMap(m_np, cv2.COLORMAP_JET)
        overlay_rgb = (0.6*img + 0.4*heat_bgr[:,:,::-1]).clip(0,255).astype(np.uint8)
        vw.write(overlay_rgb[:,:,::-1])
    vw.release()


def test_tensor_img(gt, output, lpips_fn):
    if gt.dim() == 4:
        gt = gt.unsqueeze(0)
    if output.dim() == 4:
        output = output.unsqueeze(0)
    
    B, T, C, H, W = gt.shape
    psnr_video = []
    ssim_video = []
    lpips_video = []
    
    for b in range(B):
        for t in range(T):
            gt_frame = gt[b, t].clamp(0, 1)
            out_frame = output[b, t].clamp(0, 1)
            
            gt_np = gt_frame.cpu().numpy().transpose(1, 2, 0)
            out_np = out_frame.cpu().numpy().transpose(1, 2, 0)
            
            psnr_val = compare_psnr(gt_np, out_np, data_range=1.0)
            ssim_val = compare_ssim(gt_np, out_np, data_range=1.0, channel_axis=-1, win_size=7)
            
            out_norm = (out_frame * 2 - 1).clamp(-1, 1).unsqueeze(0)
            gt_norm = (gt_frame * 2 - 1).clamp(-1, 1).unsqueeze(0)
            lpips_val = lpips_fn(out_norm, gt_norm).mean().item()
            
            psnr_video.append(psnr_val)
            ssim_video.append(ssim_val)
            lpips_video.append(lpips_val)
    
    return psnr_video, ssim_video, lpips_video


def collate_fn(batch):
    return tuple(zip(*batch))


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():
    parser = argparse.ArgumentParser(description="Recursive 2-frame restoration inference")
    parser.add_argument("--data_path", type=str, required=True, help="Root folder of turbulent videos")
    parser.add_argument("--label_dir", type=str, required=True, help="GT label dir")
    parser.add_argument("--ckpt", type=str, required=True, help="model checkpoint path")
    parser.add_argument("--save_dir", type=str, default="./infer_results")
    parser.add_argument("--num_frames", type=int, default=10)
    parser.add_argument("--patch_size", type=int, default=128, help="only used if dataset crops patches")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--tmt_dims", type=int, default=16)
    parser.add_argument("--matf", action="store_true")
    parser.add_argument("--matf_a", type=float, default=0.5)
    parser.add_argument("--matf_b", type=float, default=0.5)
    parser.add_argument("--matf_c", type=float, default=0.0)
    parser.add_argument("--matf_d", type=float, default=0.3)
    parser.add_argument("--matf_beta", type=float, default=0.8)
    parser.add_argument("--save_motion_vis", action="store_true")
    return parser.parse_args()

def main():
    args = get_args()
    device = get_device()

    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, "infer.log")
    logging.basicConfig(filename=log_path, level=logging.INFO)
    print(f"Logs -> {log_path}")

    lpips_loss_fn = lpips.LPIPS(net='alex').to(device)
    lpips_loss_fn.eval()

    # dataset
    val_dataset = DataLoaderTurbVideo(
        root_dir=args.data_path,
        label_dir=args.label_dir,
        num_frames=args.num_frames,
        patch_size=args.patch_size,
        noise=0.0001,
        is_train=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # model
    model = TMT_MS(
        num_blocks=[2, 3, 3, 4],
        num_refinement_blocks=2,
        warp_mode="all",
        n_frames=2,
        dim=args.tmt_dims,
        mambadef=True
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(
        {k: v for k, v in state_dict.items()
         if k in model.state_dict() and model.state_dict()[k].shape == v.shape},
        strict=False
    )
    print(f"✅ Loaded checkpoint: {args.ckpt}")
    model.eval()

    psnr_all, ssim_all, lpips_all = [], [], []
    total_restore_time = 0.0
    frame_count = 0
    static_consistency_list = []

    with torch.no_grad():
        for val_data_idx, val_data in enumerate(tqdm(val_loader)):
            full_seq, target_seq = val_data[0][0].to(device), val_data[1][0].to(device)
            O_prev = full_seq[0]

            video_outputs = []
            video_targets = []
            
            matf_ema_state = None
            motion_frames, motion_maps = [], []

            for t in range(1, full_seq.shape[0]):
                I_t = full_seq[t]
                T_t = target_seq[t]

                input_cat = torch.stack([O_prev, I_t], dim=0).unsqueeze(0).permute(0, 2, 1, 3, 4)

                start = datetime.now().timestamp()
                input_cat = torch.stack([O_prev, I_t], dim=0).unsqueeze(0).permute(0, 2, 1, 3, 4)
                turb_level = clip_turb_level(I_t.unsqueeze(0))  # I_t: [C,H,W] -> [1,C,H,W]
                turb_float = turb_level.item()
                print(f"[turb] Video {val_data_idx}, Frame {t:02d}: turb_level = {turb_float:.4f}")
                output, flows = model(input_cat, turb_level=turb_level)
                
                end = datetime.now().timestamp()
                total_restore_time += (end - start)
                frame_count += 1

                output = output.squeeze(0)
                if output.dim() == 4:
                    output = output[:, -1, :, :]

                if args.matf and (flows is not None) and (len(flows) > 0):
                    flow_lvl = flows[0]
                    if flow_lvl.dim() == 5:
                        flow_lvl = flow_lvl[:, :, 0]
                    flow_fw = F.interpolate(flow_lvl, size=I_t.shape[-2:], mode='bilinear', align_corners=True)

                    S_t, M_t, Iprev_warp = compute_motion_map(
                        O_prev.unsqueeze(0), I_t.unsqueeze(0), flow_fw, flow_bw=None,
                        a=args.matf_a, b=args.matf_b, c=args.matf_c, d=args.matf_d,
                        ema_prev=matf_ema_state, beta=args.matf_beta
                    )
                    matf_ema_state = M_t

                    S_ch = S_t.squeeze(0).expand_as(output)
                    O_hat = S_ch * Iprev_warp.squeeze(0) + (1.0 - S_ch) * output

                    static_consistency = F.l1_loss(S_ch * output, S_ch * Iprev_warp.squeeze(0)).item()
                    static_consistency_list.append(static_consistency)

                    if args.save_motion_vis and val_data_idx == 0:
                        motion_frames.append(tensor_to_rgb_uint8(I_t))
                        motion_maps.append(M_t.squeeze(0).squeeze(0))

                    O_prev = O_hat.detach()
                    out_eval = O_hat
                else:
                    O_prev = output.detach()
                    out_eval = output

                video_outputs.append(out_eval.unsqueeze(0))
                video_targets.append(T_t.unsqueeze(0))

                if val_data_idx == 0:
                    out_dir = os.path.join(args.save_dir, f"seq0")
                    os.makedirs(out_dir, exist_ok=True)
                    save_image(I_t, os.path.join(out_dir, f"frame{t:03d}_input.png"))
                    save_image(out_eval, os.path.join(out_dir, f"frame{t:03d}_output.png"))
                    save_image(T_t, os.path.join(out_dir, f"frame{t:03d}_gt.png"))

            if video_outputs:
                video_output = torch.stack(video_outputs, dim=1)   # [1,T,C,H,W]
                video_target = torch.stack(video_targets, dim=1)
                psnr_v, ssim_v, lpips_v = test_tensor_img(video_target, video_output, lpips_loss_fn)
                psnr_all.append(sum(psnr_v)/len(psnr_v))
                ssim_all.append(sum(ssim_v)/len(ssim_v))
                lpips_all.append(sum(lpips_v)/len(lpips_v))
                print(f"Video {val_data_idx}: PSNR={psnr_all[-1]:.4f}, SSIM={ssim_all[-1]:.4f}, LPIPS={lpips_all[-1]:.4f}")

            if args.save_motion_vis and args.matf and val_data_idx == 0 and len(motion_frames) > 0:
                out_mp4 = os.path.join(args.save_dir, f"seq0_motion_overlay.mp4")
                save_motion_overlay_video(motion_frames, motion_maps, out_mp4, fps=25)
                print(f"[VIS] motion overlay saved -> {out_mp4}")

    mean_psnr = np.mean(psnr_all) if len(psnr_all) else 0.0
    mean_ssim = np.mean(ssim_all) if len(ssim_all) else 0.0
    mean_lpips = np.mean(lpips_all) if len(lpips_all) else 0.0
    avg_restore_time = total_restore_time / max(frame_count, 1)

    if len(static_consistency_list) > 0:
        mean_static_cons = np.mean(static_consistency_list)
        print(f"[INFER] Static-Consistency(L1 on static) : {mean_static_cons:.6f}")

    print(f"[INFER] Overall - PSNR: {mean_psnr:.4f}, SSIM: {mean_ssim:.4f}, LPIPS: {mean_lpips:.4f}")
    print(f"[INFER] Avg Restore Time: {avg_restore_time:.4f} sec/frame")

    logging.info(f"[INFER] Overall - PSNR: {mean_psnr:.4f}, SSIM: {mean_ssim:.4f}, LPIPS: {mean_lpips:.4f}")
    logging.info(f"[INFER] Avg Restore Time: {avg_restore_time:.4f} sec/frame")


if __name__ == "__main__":
    main()
