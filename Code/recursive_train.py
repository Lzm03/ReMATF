import os
import argparse
import logging
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

from torchvision.utils import save_image
from data.dataset_video_train import DataLoaderTurbVideo
from model.TMT_DC import TMT_MS
from utils.general import create_log_folder
from utils.losses import CharbonnierLoss, dwtLoss
from ultralytics import YOLO

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from ultralytics.utils.ops import non_max_suppression
import lpips 
import time 

def visualize_flow(flow):
    if isinstance(flow, torch.Tensor):
        flow = flow.detach().cpu().squeeze(0).numpy()

    if flow.shape[0] == 2:  
        u, v = flow[0], flow[1]
    else:
        raise ValueError("Flow must have shape [2, H, W]")

    mag, ang = cv2.cartToPolar(u, v, angleInDegrees=True)

    hsv = np.zeros((u.shape[0], u.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ang / 2              
    hsv[..., 1] = 255                   
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX) 

    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return flow_rgb

def get_args():
    parser = argparse.ArgumentParser(description="Recursive 2-frame restoration training")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--log_path", type=str, default="./log")
    parser.add_argument("--run_name", type=str, default="Recursive2Frame")
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--alpha", type=float, default=0.9, help="Recursive blending weight for previous frame")
    parser.add_argument("--num_frames", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--tmt_dims", type=int, default=16)
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--quick_val", action="store_true")
    parser.add_argument("--lambda_dwt", type=float, default=0.02)
    parser.add_argument("--train_label_dir", type=str, required=True)
    parser.add_argument("--val_label_dir", type=str, required=True)
    parser.add_argument("--eval_only", action="store_true", help="Only run validation without training")
    parser.add_argument("--matf", action="store_true",
                        help="Enable Motion-Adaptive Temporal Fusion (MATF) that generates per-pixel motion-based weighting maps.")
    parser.add_argument("--matf_a", type=float, default=0.5,
                        help="Weight of optical flow magnitude term in motion composition (larger motion → more dynamic).")
    parser.add_argument("--matf_b", type=float, default=0.5,
                        help="Weight of reprojection error term in motion composition (larger warp error/occlusion → more dynamic).")
    parser.add_argument("--matf_c", type=float, default=0.0,
                        help="Weight of forward-backward consistency term in motion composition (larger inconsistency → more dynamic).")
    parser.add_argument("--matf_d", type=float, default=0.3,
                        help="Edge protection strength (reduces 'dynamic degree' near edges to prevent edge blurring).")
    parser.add_argument("--matf_beta", type=float, default=0.8,
                        help="EMA smoothing factor for motion map to suppress frame-wise jitter and improve temporal stability.")
    parser.add_argument("--lambda_temp", type=float, default=1.0,
                        help="Weight for static-aware temporal consistency loss; enforces stronger constraint in static regions.")
    parser.add_argument("--save_motion_vis", action="store_true",
                        help="Save visualization video with overlaid motion maps on the validation set (sequence 0).")
    return parser.parse_args()

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
    if len(frames_rgb) == 0: return
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
    out_frames = []
    
    for b in range(B):
        for t in range(T):
            gt_frame = gt[b, t].clamp(0, 1)
            out_frame = output[b, t].clamp(0, 1)
            
            gt_np = gt_frame.cpu().numpy().transpose(1, 2, 0)
            out_np = out_frame.cpu().numpy().transpose(1, 2, 0)
            
            psnr_val = compare_psnr(gt_np, out_np, data_range=1.0)
            psnr_video.append(psnr_val)
            
            ssim_val = compare_ssim(gt_np, out_np, data_range=1.0, channel_axis=-1, win_size=7)
            ssim_video.append(ssim_val)
            
            out_norm = (out_frame * 2 - 1).clamp(-1, 1).unsqueeze(0)
            gt_norm = (gt_frame * 2 - 1).clamp(-1, 1).unsqueeze(0)
            lpips_val = lpips_fn(out_norm, gt_norm).mean().item()
            lpips_video.append(lpips_val)
            
            out_frame_uint8 = (out_np * 255).astype(np.uint8)
            out_frames.append(out_frame_uint8)
    
    return out_frames, psnr_video, ssim_video, lpips_video


def collate_fn(batch):
    return tuple(zip(*batch))

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def blend_prev_frame(O_prev, T_t, alpha_t):
    return alpha_t * O_prev + (1 - alpha_t) * T_t.detach()

def main():
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

    args = get_args()
    device = get_device()
    
    lpips_loss_fn = lpips.LPIPS(net='alex').to(device)
    lpips_loss_fn.eval()

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    run_name = args.run_name + "_" + datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    run_path = os.path.join(args.log_path, run_name)
    result_img_path, path_ckpt = create_log_folder(run_path)
    logging.basicConfig(filename=os.path.join(run_path, "training.log"), level=logging.INFO)
    writer = SummaryWriter(log_dir=run_path)


    train_dataset = DataLoaderTurbVideo(
        root_dir=args.train_path,
        label_dir=args.train_label_dir,
        num_frames=args.num_frames,
        patch_size=args.patch_size,
        noise=0.0001,
        is_train=True,
        max_clips_per_video=1
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True, collate_fn=collate_fn)

    val_dataset = DataLoaderTurbVideo(
        root_dir=args.val_path,
        label_dir=args.val_label_dir,
        num_frames=args.num_frames,
        patch_size=args.patch_size,
        noise=0.0001,
        is_train=False,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=args.num_workers, drop_last=True, collate_fn=collate_fn)

    model = TMT_MS(
        num_blocks=[2, 3, 3, 4],
        num_refinement_blocks=2,
        warp_mode="all",
        n_frames=2,
        dim=args.tmt_dims,
        mambadef=True
    ).to(device)

    if args.resume_ckpt and os.path.exists(args.resume_ckpt):
        ckpt = torch.load(args.resume_ckpt, map_location=device)
        state_dict = ckpt.get("state_dict", ckpt)
        model.load_state_dict({k: v for k, v in state_dict.items() if k in model.state_dict() and model.state_dict()[k].shape == v.shape}, strict=False)
        print(f"✅ Loaded checkpoint: {args.resume_ckpt}")

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = CharbonnierLoss()
    wavelet_criterion = dwtLoss(loss_weight=args.lambda_dwt, wkd_level=1, wkd_basis='haar').to(device)
    temporal_criterion = nn.L1Loss()

    best_score = 0
    best_model_path = os.path.join(path_ckpt, "best_model.pth")
    latest_model_path = os.path.join(path_ckpt, "latest_model.pth")

    patience = 5
    no_improve_count = 0
    output_history = []
    charb_losses = []
    dwt_losses = []
    flow_losses = []
    total_losses = []
    
    
    if args.eval_only:
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

                    start = time.time()
                    output, flows = model(input_cat)
                    end = time.time()

                    total_restore_time += (end - start)
                    frame_count += 1

                    output = output.squeeze(0)
                    if output.dim() == 4:
                        output = output[:, -1, :, :]
                        
                    O_hat = output 
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

                        S_ch = S_t.squeeze(0).expand_as(output)  # CxHxW
                        O_hat = S_ch * Iprev_warp.squeeze(0) + (1.0 - S_ch) * output

                        static_consistency = F.l1_loss(S_ch * output, S_ch * Iprev_warp.squeeze(0)).item()
                        static_consistency_list.append(static_consistency)

                        if args.save_motion_vis and val_data_idx == 0:
                            motion_frames.append(tensor_to_rgb_uint8(I_t))
                            motion_maps.append(M_t.squeeze(0).squeeze(0))

                        O_prev = O_hat.detach()
                    else:
                        if hasattr(args, "alpha"):
                            O_prev = args.alpha * O_prev + (1.0 - args.alpha) * output.detach()
                        else:
                            O_prev = output.detach()

                    out_eval = O_hat if (args.matf) else output
                    
                    video_outputs.append(out_eval.unsqueeze(0))
                    video_targets.append(T_t.unsqueeze(0))

                    if val_data_idx == 0:
                        output_dir_epoch = os.path.join(result_img_path, f"eval_only")
                        os.makedirs(output_dir_epoch, exist_ok=True)
                        save_image(I_t, os.path.join(output_dir_epoch, f"seq{val_data_idx}_frame{t}_input.png"))
                        save_image(out_eval, os.path.join(output_dir_epoch, f"seq{val_data_idx}_frame{t}_output.png"))
                        save_image(T_t, os.path.join(output_dir_epoch, f"seq{val_data_idx}_frame{t}_gt.png"))

                if video_outputs:
                    video_output = torch.stack(video_outputs, dim=1)  # [1, T, C, H, W]
                    video_target = torch.stack(video_targets, dim=1)  # [1, T, C, H, W]
                    
                    out_frames, psnr_video, ssim_video, lpips_video = test_tensor_img(
                        video_target, video_output, lpips_loss_fn
                    )
                    
                    psnr_all.append(sum(psnr_video)/len(psnr_video))
                    ssim_all.append(sum(ssim_video)/len(ssim_video))
                    lpips_all.append(sum(lpips_video)/len(lpips_video))
                    
                    print(f"Video {val_data_idx}: PSNR={psnr_all[-1]:.4f}, SSIM={ssim_all[-1]:.4f}, LPIPS={lpips_all[-1]:.4f}")

                if args.save_motion_vis and args.matf and val_data_idx == 0 and len(motion_frames) > 0:
                    out_mp4 = os.path.join(result_img_path, f"eval_only_seq0_motion_overlay.mp4")
                    save_motion_overlay_video(motion_frames, motion_maps, out_mp4, fps=25)
                    print(f"[VIS] motion overlay saved -> {out_mp4}")

        mean_psnr = np.mean(psnr_all) if len(psnr_all) else 0.0
        mean_ssim = np.mean(ssim_all) if len(ssim_all) else 0.0
        mean_lpips = np.mean(lpips_all) if len(lpips_all) else 0.0
        avg_restore_time = total_restore_time / max(frame_count, 1)

        if len(static_consistency_list) > 0:
            mean_static_cons = np.mean(static_consistency_list)
            print(f"[EVAL ONLY] Static-Consistency(L1 on static) : {mean_static_cons:.6f}")

        print(f"[EVAL ONLY] Overall - PSNR: {mean_psnr:.4f}, SSIM: {mean_ssim:.4f}, LPIPS: {mean_lpips:.4f}")
        print(f"[EVAL ONLY] Avg Restore Time: {avg_restore_time:.4f} sec/frame")

        logging.info(f"[EVAL ONLY] Overall - PSNR: {mean_psnr:.4f}, SSIM: {mean_ssim:.4f}, LPIPS: {mean_lpips:.4f}")
        logging.info(f"[EVAL ONLY] Avg Restore Time: {avg_restore_time:.4f} sec/frame")
        return


    for epoch in range(args.epochs):
        model.train()
        train_loss_list = []

        for data in tqdm(train_loader):
            full_seq, target_seq, meta = data[0][0].to(device), data[1][0].to(device), data[2][0]
            O_prev = full_seq[0]

            matf_ema_state = None

            for t in range(1, full_seq.shape[0]):
                I_t = full_seq[t]
                T_t = target_seq[t]

                input_cat = torch.stack([O_prev, I_t], dim=0).unsqueeze(0).permute(0, 2, 1, 3, 4)
                output, flows = model(input_cat)
                output = output.squeeze(0)
                if output.dim() == 4:
                    output = output[:, -1, :, :]
                
                if t > 1:
                    flow_loss = 0.0
                    for k in range(1, min(len(output_history)+1, len(flows)+1)):
                        prev_out = output_history[-k]  
                        flow_k = flows[k-1][:, :, 0, :, :]
                        flow_k = F.interpolate(flow_k, size=output.shape[-2:], mode='bilinear', align_corners=True)
                        warped = flow_warp(prev_out.unsqueeze(0), flow_k)
                        weight = 0.5 ** k
                        flow_loss += weight * F.l1_loss(output.unsqueeze(0), warped)
                else:
                    flow_loss = 0.0

                output_history.append(output.detach())
                if len(output_history) > 3:
                    output_history.pop(0)
                    
                lambda_flow = 0.0
                
                charb_loss = criterion(output, T_t)
                dwt_loss = wavelet_criterion(output.unsqueeze(0), T_t.unsqueeze(0))
                
                if args.matf and (flows is not None) and (len(flows) > 0):
                    flow_lvl = flows[0]
                    if flow_lvl.dim() == 5:
                        flow_lvl = flow_lvl[:, :, 0]  # Bx2xhxw
                    flow_fw = F.interpolate(flow_lvl, size=I_t.shape[-2:], mode='bilinear', align_corners=True)

                    S_t, M_t, Iprev_warp = compute_motion_map(
                        O_prev.unsqueeze(0), I_t.unsqueeze(0), flow_fw, flow_bw=None,
                        a=args.matf_a, b=args.matf_b, c=args.matf_c, d=args.matf_d,
                        ema_prev=matf_ema_state, beta=args.matf_beta
                    )
                    matf_ema_state = M_t 

                    S_ch = S_t.squeeze(0).expand_as(output)  # CxHxW
                    O_hat = S_ch * Iprev_warp.squeeze(0) + (1.0 - S_ch) * output 
                    temp_loss = F.l1_loss(S_ch * output, S_ch * Iprev_warp.squeeze(0)) 
                else:
                    O_hat = output
                    temp_loss = torch.tensor(0.0, device=device)

            
                loss = charb_loss + dwt_loss + lambda_flow * flow_loss + args.lambda_temp * temp_loss

                train_loss_list.append(loss.item())
                charb_losses.append(charb_loss.item())
                dwt_losses.append(dwt_loss.item())
                flow_losses.append(flow_loss if isinstance(flow_loss, float) else flow_loss.item())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if args.matf and (flows is not None) and (len(flows) > 0):
                    O_prev = O_hat.detach()
        
            if args.quick_val:
                break
            
        mean_train_loss = np.mean(train_loss_list)
        mean_charb = np.mean(charb_losses)
        mean_dwt = np.mean(dwt_losses)
        mean_flow = np.mean(flow_losses)
        print(f"[Train@Epoch {epoch}] Avg charb: {mean_charb:.4f}, dwt: {mean_dwt:.4f}, flow: {mean_flow:.4f}, total: {mean_train_loss:.4f}")
        logging.info(f"[Train@Epoch {epoch}] Mean Loss: {mean_train_loss:.6f}")
        writer.add_scalar("Train/Loss", mean_train_loss, epoch)
        scheduler.step()


        model.eval()
        psnr_all, ssim_all, lpips_all = [], [], []
        with torch.no_grad():
            for val_data_idx, val_data in enumerate(tqdm(val_loader)):
                full_seq, target_seq = val_data[0][0].to(device), val_data[1][0].to(device)
                O_prev = full_seq[0]

                video_outputs = []
                video_targets = []
                
                motion_frames, motion_maps = [], []
                matf_ema_state = None

                for t in range(1, full_seq.shape[0]):
                    I_t = full_seq[t]
                    T_t = target_seq[t]

                    input_cat = torch.stack([O_prev, I_t], dim=0)  # [2, 3, H, W]
                    input_cat = input_cat.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, 3, 2, H, W]
                    output, _ = model(input_cat)
                    output = output.squeeze(0)
                    if output.dim() == 4:
                        output = output[:, -1, :, :]

                    if args.matf and (_ is not None) and (len(_) > 0):
                        flow_lvl = _[0]
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
                        O_prev = O_hat.detach()

                        if args.save_motion_vis and val_data_idx == 0:
                            motion_frames.append(tensor_to_rgb_uint8(I_t))
                            motion_maps.append(M_t.squeeze(0).squeeze(0))
                    else:
                        O_prev = output.detach()

                    video_outputs.append(output.unsqueeze(0))
                    video_targets.append(T_t.unsqueeze(0))

                    if val_data_idx == 0:
                        output_dir_epoch = os.path.join(result_img_path, f"epoch{epoch}")
                        os.makedirs(output_dir_epoch, exist_ok=True)

                        save_image(I_t, os.path.join(output_dir_epoch, f"seq{val_data_idx}_frame{t}_input.png"))
                        save_image(output, os.path.join(output_dir_epoch, f"seq{val_data_idx}_frame{t}_output.png"))
                        save_image(T_t, os.path.join(output_dir_epoch, f"seq{val_data_idx}_frame{t}_gt.png"))

                if video_outputs:
                    video_output = torch.stack(video_outputs, dim=1)  # [1, T, C, H, W]
                    video_target = torch.stack(video_targets, dim=1)  # [1, T, C, H, W]
                    
                    out_frames, psnr_video, ssim_video, lpips_video = test_tensor_img(
                        video_target, video_output, lpips_loss_fn
                    )
                    
                    psnr_all.append(sum(psnr_video)/len(psnr_video))
                    ssim_all.append(sum(ssim_video)/len(ssim_video))
                    lpips_all.append(sum(lpips_video)/len(lpips_video))
                    
                    print(f"Epoch {epoch}, Video {val_data_idx}: PSNR={psnr_all[-1]:.4f}, SSIM={ssim_all[-1]:.4f}, LPIPS={lpips_all[-1]:.4f}")

                if args.save_motion_vis and args.matf and val_data_idx == 0 and len(motion_frames) > 0:
                    out_mp4 = os.path.join(result_img_path, f"epoch{epoch}_seq0_motion_overlay.mp4")
                    save_motion_overlay_video(motion_frames, motion_maps, out_mp4, fps=25)
                    print(f"[VIS] motion overlay saved -> {out_mp4}")

        mean_psnr = np.mean(psnr_all) if len(psnr_all) else 0.0
        mean_ssim = np.mean(ssim_all) if len(ssim_all) else 0.0
        mean_lpips = np.mean(lpips_all) if len(lpips_all) else 0.0
        current_score = mean_psnr + mean_ssim

        print(f"[Val@Epoch {epoch}] Overall - PSNR: {mean_psnr:.4f}, SSIM: {mean_ssim:.4f}, LPIPS: {mean_lpips:.4f}")
        logging.info(f"[Val@Epoch {epoch}] Overall - PSNR: {mean_psnr:.4f}, SSIM: {mean_ssim:.4f}, LPIPS: {mean_lpips:.4f}")
        writer.add_scalar("Val/PSNR", mean_psnr, epoch)
        writer.add_scalar("Val/SSIM", mean_ssim, epoch)
        writer.add_scalar("Val/LPIPS", mean_lpips, epoch)

        if current_score > best_score:
            best_score = current_score
            torch.save({"epoch": epoch, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}, best_model_path)
            print(f"✅ Best model saved at epoch {epoch}")
            logging.info(f"✅ Best model saved at epoch {epoch}")
            
        torch.save({"epoch": epoch, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}, latest_model_path)
        logging.info(f"💾 Latest model saved for epoch {epoch}")

if __name__ == "__main__":
    main()
