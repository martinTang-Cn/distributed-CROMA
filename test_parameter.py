from train_croma_whu_distil import OpticalSatelliteClient, RadarSatelliteClient, GroundServer
from pretrain_croma import CROMA
import torch, argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Distillation 分割训练 - 卫星与地面服务器协同"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/featurize/data",
        help="WHU 光学-SAR 数据根目录",
    )
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # 分开设置不同组件的学习率
    parser.add_argument("--lr_optical", type=float, default=1e-4, help="光学编码器学习率")
    parser.add_argument("--lr_radar", type=float, default=1e-4, help="雷达编码器学习率")
    parser.add_argument("--lr_server", type=float, default=1e-4, help="服务器学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    
    parser.add_argument("--encoder_dim", type=int, default=768)
    parser.add_argument("--encoder_layers", type=int, default=12)
    parser.add_argument("--attention_heads", type=int, default=16)
    parser.add_argument("--decoder_dim", type=int, default=512)
    parser.add_argument("--decoder_layers", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=8)
    parser.add_argument(
        "--pretrained_ckpt",
        type=str,
        required=True,
        help="预训练 CROMA checkpoint 路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/featurize/work/CROMA_checkpoint/croma_whu_distil_checkpoints",
    )
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--num_ratio", type=float, default=1.0)
    parser.add_argument("--stride_ratio", type=float, default=0.9)
    parser.add_argument("--max_grad_norm", type=float, default=0.0)
    parser.add_argument("--distill_temperature", type=float, default=1.0)
    parser.add_argument("--dataset", type=str, choices=["whu", "bigearthnet"], default="whu",
                        help="选择使用的数据集：'whu' 使用 WHUOptSarPatchDataset，'bigearthnet' 使用 BigEarthNetDataset")
    return parser.parse_args()

def main():
    args = parse_args()
    num_patches = (120//8)**2

    croma = CROMA(
            patch_size=8,
            encoder_dim=args.encoder_dim,
            encoder_layers=args.encoder_layers,
            attention_heads=args.attention_heads,
            decoder_dim=args.decoder_dim,
            decoder_layers=args.decoder_layers,
            total_channels=10 + 2,
            num_patches=num_patches,
            opt_channels=10,
            radar_channels=2,
        )

    # 加载预训练权重
    ckpt = torch.load(args.pretrained_ckpt, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = croma.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Warning] Missing keys: {missing}")
    if unexpected:
        print(f"[Warning] Unexpected keys: {unexpected}")

    # 获取 attn_bias
    attn_bias = croma.attn_bias

    # 构建光学卫星客户端
    optical_client = OpticalSatelliteClient(
        optical_encoder=croma.optical_encoder,
        attn_bias=attn_bias,
        encoder_dim=args.encoder_dim,
        num_patches=num_patches,
        num_classes=args.num_classes,
    )

    # 构建雷达卫星客户端
    radar_client = RadarSatelliteClient(
        radar_encoder=croma.radar_encoder,
        attn_bias=attn_bias,
        encoder_dim=args.encoder_dim,
        num_patches=num_patches,
        num_classes=args.num_classes,
    )

    # 构建地面服务器
    ground_server = GroundServer(
        cross_encoder=croma.cross_encoder,
        encoder_dim=args.encoder_dim,
        num_patches=num_patches,
        num_classes=args.num_classes,
    )

    optical_client_params = sum(p.numel() for p in optical_client.parameters())
    radar_client_params = sum(p.numel() for p in radar_client.parameters())
    ground_server_params = sum(p.numel() for p in ground_server.parameters())
    print(f"optical_client_params: {optical_client_params/1e6:.2f}M")
    print(f"radar_client_params: {radar_client_params/1e6:.2f}M")
    print(f"ground_server_params: {ground_server_params/1e6:.2f}M")

if __name__ == "__main__":
    main()