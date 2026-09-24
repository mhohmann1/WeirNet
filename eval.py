import os
import time

import matplotlib.pyplot as plt
import numpy as np

from args import args
from init_dataloader import init_dataset
from dataloader import RowDataset, GNNDataset
from torch.utils.data import DataLoader
from utils import r2_score
from model.surrogate_models import *

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

def save_scatter_grid(plots, out_path, title=None):
    if not plots:
        return

    label_size = 14
    tick_size = 12
    title_size = 16

    item = plots[0]

    y_true = np.asarray(item["y_true"])
    y_pred = np.asarray(item["y_pred"])

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(
        y_true,
        y_pred,
        alpha=0.6,
        edgecolors="none"
    )

    min_val = float(np.min([y_true.min(), y_pred.min()]))
    max_val = float(np.max([y_true.max(), y_pred.max()]))

    if min_val == max_val:
        min_val -= 1.0
        max_val += 1.0

    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        color="black",
        linewidth=1
    )

    ax.set_xlabel(r"$c_D$", fontsize=label_size)
    ax.set_ylabel(r"$\hat{c}_D$", fontsize=label_size)

    ax.set_title(item.get("label", ""), fontsize=title_size)
    ax.tick_params(axis="both", labelsize=tick_size)
    ax.grid(True, linestyle="--", alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=title_size)

    fig.tight_layout()
    fig.savefig(
        out_path,
        format="pdf",
        bbox_inches="tight"
    )

    plt.close(fig)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

train_loader, val_loader, test_loader = init_dataset(args.data_path, args.pc_dir, args.stl_dir, args.augmentation)

print("Samples in Training Set:  ", len(train_loader.dataset))
print("Samples in Validation Set: ", len(val_loader.dataset))
print("Samples in Testing Set: ", len(test_loader.dataset))

if args.model == "PN":
    model = RegPointNet(args)
    model_desc = "pointnet"
elif args.model == "GNN":
    model = DragGNN_XL()
    model_desc = "gnn"
elif args.model == "DGCNN":
    model = RegDGCNN(args)
    model_desc = "dgcnn"
model = model.to(device)

SAVED_DIR = f"./saved_model/{model_desc}/"
os.makedirs(SAVED_DIR, exist_ok=True)

model_path = f"./saved_model/{model_desc}/model.tar"
checkpoint = torch.load(model_path, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
print(f"Saved model with Val Loss: {loss:.6f} at Epoch: {epoch}")

if torch.cuda.device_count() > 1:
    print("GPUs found: ", torch.cuda.device_count())
    model = nn.DataParallel(model)

model.eval()
tot_test_loss, tot_test_r2, tot_test_mae, tot_test_max_ae = 0, 0, 0, 0
tot_inference_time = 0.0
tot_samples = 0
y_true_all = []
y_pred_all = []
with torch.no_grad():
    if args.model == "GNN":
        for values, data in test_loader:
            data = data.to(device)
            targets = data.y.view(-1, 1).to(device)
            q = values[:, 1].to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            preds = model(data, q)
            if device.type == "cuda":
                torch.cuda.synchronize()
            tot_inference_time += time.perf_counter() - start_time
            tot_samples += values.size(0)
            y_true_all.append(targets.detach().cpu().numpy().ravel())
            y_pred_all.append(preds.detach().cpu().numpy().ravel())
            loss = F.mse_loss(preds, targets)
            r2 = r2_score(preds, targets)
            mae = F.l1_loss(preds, targets)
            max_ae = (preds - targets).abs().max()
            tot_test_loss += loss.item()
            tot_test_r2 += r2.item()
            tot_test_mae += mae.item()
            tot_test_max_ae = max(tot_test_max_ae, max_ae.item())

    else:
        for values, pc in test_loader:
            pc = pc.permute(0, 2, 1).to(device)
            targets = values[:, -1].unsqueeze(1).to(device)
            q = values[:, 1].to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            preds = model(pc, q)
            if device.type == "cuda":
                torch.cuda.synchronize()
            tot_inference_time += time.perf_counter() - start_time
            tot_samples += values.size(0)
            y_true_all.append(targets.detach().cpu().numpy().ravel())
            y_pred_all.append(preds.detach().cpu().numpy().ravel())
            loss = F.mse_loss(preds, targets)
            r2 = r2_score(preds, targets)
            mae = F.l1_loss(preds, targets)
            max_ae = (preds - targets).abs().max()
            tot_test_loss += loss.item()
            tot_test_r2 += r2.item()
            tot_test_mae += mae.item()
            tot_test_max_ae = max(tot_test_max_ae, max_ae.item())

mean_test_loss = tot_test_loss / len(test_loader)
mean_test_r2 = tot_test_r2 / len(test_loader)
mean_test_mae = tot_test_mae / len(test_loader)
mean_test_max_ae = tot_test_max_ae
mean_inference_time = tot_inference_time / tot_samples if tot_samples else 0.0

print(
    "Test Loss: "
    f"{mean_test_loss:.6f}, Test R2: {mean_test_r2:.6f}, "
    f"Test MAE: {mean_test_mae:.6f}, "
    f"Test Max AE: {mean_test_max_ae:.6f}, "
    f"Inference Time (s): {tot_inference_time:.6f}, "
    f"Avg Inference Time per Sample (s): {mean_inference_time:.6f}"
)

if y_true_all and y_pred_all:
    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    plot_path = os.path.join(SAVED_DIR, "test_scatter_grid.pdf")
    plots = [{"label": args.model, "y_true": y_true_all, "y_pred": y_pred_all}]
    save_scatter_grid(plots, plot_path, title=None)
    print(f"Saved scatter grid to {plot_path}")
