import torch.multiprocessing as mp
import os, sys, logging, time, numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from args import args
from init_dataloader import init_dataset
from model.surrogate_models import RegPointNet, RegDGCNN
from tqdm import tqdm

torch.set_num_threads(max(1, int(os.getenv("OMP_NUM_THREADS", "2"))))
torch.set_num_interop_threads(1)

def ckpt_path(saved_dir):
    return os.path.join(saved_dir, "checkpoint_last.pt")

def save_checkpoint(epoch, model, optimizer, scheduler, best_loss, best_epoch, saved_dir, rank):
    """Save/overwrite last checkpoint (rank-0 only), then sync."""
    if is_main(rank):
        state = {
            "epoch": epoch,
            "model_state_dict": (model.module if isinstance(model, DDP) else model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "best_loss": best_loss,
            "best_epoch": best_epoch,
        }
        fpath = ckpt_path(saved_dir)
        tmp = fpath + ".tmp"
        torch.save(state, tmp)        # write to temp
        os.replace(tmp, fpath)        # atomic replace on POSIX

    if dist.is_initialized():
        dist.barrier()
        
def resume_if_requested(args, model, optimizer, scheduler, saved_dir, device, rank, best_loss, best_epoch):
    """If args.resume and checkpoint exists: load & return start_epoch, best_loss, best_epoch."""
    start_epoch = 1
    if getattr(args, "resume", False):
        fpath = ckpt_path(saved_dir)
        if os.path.isfile(fpath):
            # Load to CPU for safety, then copy into model on current device
            ckpt = torch.load(fpath, map_location="cpu", weights_only=True)
            (model.module if isinstance(model, DDP) else model).load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)
            if ckpt.get("scheduler_state_dict") is not None and scheduler is not None:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            best_loss  = ckpt.get("best_loss", best_loss)
            best_epoch = ckpt.get("best_epoch", best_epoch)
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            if is_main(rank):
                logging.info(f"Resumed from {fpath} at epoch {start_epoch-1} "
                             f"(best {best_loss:.6f} @ {best_epoch})")
        else:
            if is_main(rank):
                logging.warning(f"args.resume=True, but no checkpoint at {fpath}. Starting fresh.")
    
    # Sync so all ranks agree on start_epoch
    if dist.is_initialized():
        dist.barrier()
    return start_epoch, best_loss, best_epoch

def setup_dist():
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    use_cuda = torch.cuda.is_available()
    if world_size > 1:
        if use_cuda:
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl" if use_cuda else "gloo", init_method="env://")
    return rank, world_size, local_rank

def is_main(rank): return rank == 0

def setup_logging(rank):
    if is_main(rank):
        logging.basicConfig(level=logging.INFO,
                            format="[%(asctime)s] %(levelname)s: %(message)s",
                            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)

def seed_all(seed, rank):
    s = seed + rank
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    np.random.seed(s)
    torch.backends.cudnn.benchmark = True

def main():
    rank, world_size, local_rank = setup_dist()
    setup_logging(rank)
    seed_all(args.seed, rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if is_main(rank):
        logging.info(f"Device = {device} | RANK {rank}/{world_size}")

    train_loader_in, val_loader_in, test_loader_in = init_dataset(
        args.data_path, args.pc_dir, args.stl_dir, args.augmentation
    )
    
    train_ds, val_ds, test_ds = train_loader_in.dataset, val_loader_in.dataset, test_loader_in.dataset

    train_sampler = DistributedSampler(train_ds, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if world_size > 1 else None

    num_workers = getattr(args, "num_workers", max(1, os.cpu_count() // 2))
    
    common_loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    
    if num_workers > 0:
        common_loader_kwargs.update(
            prefetch_factor=2,
            multiprocessing_context="spawn",
            timeout=120,
        )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        drop_last=False,
        **common_loader_kwargs,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        drop_last=True,
        **common_loader_kwargs,
    )

    if is_main(rank):
        logging.info(f"Samples Train/Val/Test: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")
        try:
            t0 = time.perf_counter()
            _data, _pc = next(iter(train_loader))
            dt = time.perf_counter() - t0
            print(f"[Sanity] Erster Batch kam in {dt:.2f}s, "
                  f"train num_workers={train_loader.num_workers}, "
                  f"prefetch_factor={getattr(train_loader, 'prefetch_factor', 'n/a')}")
            del _data, _pc
        except Exception as e:
            print("[Sanity] DataLoader-Fehler:", repr(e), flush=True)
            raise

    if args.model == "PN":
        model = RegPointNet(args)
        model_desc = "pointnet"
    elif args.model == "DGCNN":
        model = RegDGCNN(args)
        model_desc = "dgcnn"
    else:
        raise ValueError(f"Unknown Model: {args.model}")

    model = model.to(device)
    if world_size > 1 and device.type == "cuda":
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(
        model, 
        device_ids=[device.index] if device.type == "cuda" else None,
        find_unused_parameters=False,
    ) if world_size > 1 else model

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=20, factor=0.5)

    saved_dir = f"./saved_model/{model_desc}/"
    if is_main(rank):
        os.makedirs(saved_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(saved_dir, "runs")) if is_main(rank) else None

    BEST_LOSS = float("inf")
    BEST_EPOCH = 0
    
    start_epoch, BEST_LOSS, BEST_EPOCH = resume_if_requested(
        args, model, optimizer, scheduler, saved_dir, device, rank, BEST_LOSS, BEST_EPOCH
    )

    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None: train_sampler.set_epoch(epoch)
        if val_sampler is not None:   val_sampler.set_epoch(epoch)

        red_dev = device if device.type == "cuda" else torch.device("cpu")

        if dist.is_initialized():
            dist.barrier()
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_start = time.perf_counter()

        model.train()
        train_loss_sum = 0.0
        train_n = 0
        
        train_sse = 0.0
        train_sum_y = 0.0
        train_sum_y2 = 0.0
        
        for data, pc in tqdm(train_loader, disable=not is_main(rank)):
            pc = pc.permute(0, 2, 1).to(device, non_blocking=True)
            targets = data[:, -1].unsqueeze(1).to(device, non_blocking=True)

            if args.model in ("PN", "DGCNN"):
                q = data[:, 1].to(device, non_blocking=True)
                preds = model(pc, q)
            else:
                preds = model(pc)
            loss = F.mse_loss(preds, targets, reduction="sum")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            bs = targets.size(0)
            sse_batch = loss.detach().item()
            
            train_loss_sum += sse_batch
            train_n += targets.numel()

            train_sse += sse_batch
            train_sum_y += targets.sum().item()
            train_sum_y2 += (targets * targets).sum().item()

        if device.type == "cuda":
            torch.cuda.synchronize()
        train_time = time.perf_counter() - train_start

        if dist.is_initialized():
            train_time_tensor = torch.tensor(train_time, device=red_dev, dtype=torch.float64)
            dist.all_reduce(train_time_tensor, op=dist.ReduceOp.MAX)
            train_time = train_time_tensor.item()
        
        train_tot = torch.tensor([train_loss_sum, float(train_n)],
                                 device=red_dev, dtype=torch.float64)
                          
        if dist.is_initialized():
            dist.all_reduce(train_tot, op=dist.ReduceOp.SUM)
        
        mean_train_loss = (train_tot[0] / train_tot[1]).item()
        
        train_stats = torch.tensor([train_sse, train_sum_y, train_sum_y2, float(train_n)], device=red_dev, dtype=torch.float64)
        
        if dist.is_initialized():
            dist.all_reduce(train_stats, op=dist.ReduceOp.SUM)

        n_all_tr = train_stats[3].item()
        mean_y_tr = (train_stats[1] / max(n_all_tr, 1.0)).item()
        sst_tr = (train_stats[2] - n_all_tr * (mean_y_tr ** 2))
        global_train_r2 = 1.0 - (train_stats[0].item() / max(sst_tr.item(), 1e-12))
        
        model.eval()
        val_sse = 0.0
        val_sum_y = 0.0
        val_sum_y2 = 0.0
        val_n = 0.0
        with torch.no_grad():
            for data, pc in tqdm(val_loader, disable=not is_main(rank)):
                pc = pc.permute(0, 2, 1).to(device, non_blocking=True)
                targets = data[:, -1].unsqueeze(1).to(device, non_blocking=True)

                if args.model in ("PN", "DGCNN"):
                    q = data[:, 1].to(device, non_blocking=True)
                    preds = model(pc, q)
                else:
                    preds = model(pc)
                sse_batch = F.mse_loss(preds, targets, reduction="sum").item()
                
                val_sse += sse_batch
                val_n += targets.numel()
                val_sum_y += targets.sum().item()
                val_sum_y2 += (targets * targets).sum().item()

        val_stats = torch.tensor([val_sse, val_sum_y, val_sum_y2, val_n], device=red_dev, dtype=torch.float64)
                               
        if dist.is_initialized():
            dist.all_reduce(val_stats, op=dist.ReduceOp.SUM)
            
        mean_val_loss = (val_stats[0] / val_stats[3]).item()
        n_all = val_stats[3].item()
        mean_y = (val_stats[1] / max(n_all, 1.0)).item()
        sst = (val_stats[2] - n_all * (mean_y ** 2)).item()
        global_val_r2 = 1.0 - (val_stats[0].item() / max(sst, 1e-12))

        scheduler.step(mean_val_loss)

        if is_main(rank):
            if writer:
                writer.add_scalar("Loss/Train MSE", mean_train_loss, epoch)
                writer.add_scalar("Loss/Val MSE", mean_val_loss, epoch)
                writer.add_scalar("Loss/Train R2", global_train_r2, epoch)
                writer.add_scalar("Loss/Val R2", global_val_r2, epoch)
                writer.add_scalar("Time/Train Epoch (s)", train_time, epoch)

            if mean_val_loss < BEST_LOSS:
                fpath = os.path.join(saved_dir, "model.tar")
                state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": mean_val_loss,
                }, fpath)
                BEST_LOSS = mean_val_loss
                BEST_EPOCH = epoch
            
            logging.info(
                f"Epoch {epoch}/{args.epochs} | "
                f"Train MSE {mean_train_loss:.6f} | Val MSE {mean_val_loss:.6f} | "
                f"Train R2 {global_train_r2:.6f} | Val R2 {global_val_r2:.6f} | "
                f"Train Time {train_time:.2f}s | Best {BEST_LOSS:.6f} @ {BEST_EPOCH}"
            )
            
        save_checkpoint(epoch, model, optimizer, scheduler, BEST_LOSS, BEST_EPOCH, saved_dir, rank)

    if writer and is_main(rank):
        writer.close()

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
