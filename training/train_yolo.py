import torch
import wandb
from pathlib import Path
from ultralytics.utils.metrics import DetMetrics
from ultralytics import YOLO



def main(model_name: str = "yolo11n.pt",
         data_yml_path: Path = "dataset.yml",
         result_dir: Path = 'runs',
         epochs: int = 100,
         img_size: int = 640,
         batch_size: int = 16,
         device: str = "cpu",
         n_workers: int = 3,
         cache: bool = False,
         save_period: int = -1,
         patience: int = 10,
         optimizer: str = "auto",
         lr0: float = 0.01,
         tag: str = 'yolo11_ft') -> DetMetrics | None:
    """YOLO training."""
    model = YOLO(model_name)
    results = model.train(
        data=data_yml_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        workers=n_workers,
        cache=cache,
        save_period=save_period,
        patience=patience,
        optimizer=optimizer,
        lr0=lr0,
        project=result_dir,
        name=tag
    )
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="YOLO training pipeline.")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="YOLO model name or path")
    parser.add_argument("--data", type=Path, default="dataset.yml", help="Path to dataset YAML")
    parser.add_argument("--result-dir", type=Path, default="runs", dest="result_dir", help="Directory to save results")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--img-size", type=int, nargs="+", default=[640], dest="img_size")
    parser.add_argument("--batch-size", type=int, default=16, dest="batch_size", help="Batch size")
    parser.add_argument("--workers", type=int, default=3, help="Number of dataloader workers")
    parser.add_argument("--cache", action="store_true", help="Cache images for faster training")
    parser.add_argument("--save-period", type=int, default=-1, dest="save_period", help="Save checkpoint every N epochs (-1 to disable)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (epochs)")
    parser.add_argument("--optimizer", type=str, default="auto", help="Optimizer: auto, SGD, Adam, AdamW, etc.")
    parser.add_argument("--lr0", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, 0, 0,1, etc.")
    parser.add_argument("--tag", type=str, default="experiment", help="Current train loop launch tag")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="yolo11-finetune", dest="wandb_project")
    args = parser.parse_args()

    data = Path(args.data)
    result_dir = Path(args.result_dir)

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.tag,
            config=vars(args),
        )

    res = main(
        model_name=args.model,
        data_yml_path=data,
        result_dir=result_dir,
        epochs=args.epochs,
        img_size=max(args.img_size), # select the biggest side
        batch_size=args.batch_size,
        device=args.device,
        n_workers=args.workers,
        cache=args.cache,
        save_period=args.save_period,
        patience=args.patience,
        optimizer=args.optimizer,
        lr0=args.lr0,
        tag=args.tag,
    )

    if args.wandb:
        wandb.finish()