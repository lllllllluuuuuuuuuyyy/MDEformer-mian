import argparse
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import datetime
import time
from torchinfo import summary
import yaml
import json
import sys
import copy
import csv
from sklearn.preprocessing import StandardScaler
from torch.cuda.amp import autocast, GradScaler
from sklearn.decomposition import PCA
sys.path.append("..")
from lib.utils import (
    MaskedMAELoss,
    print_log,
    seed_everything,
    set_cpu_num,
    CustomJSONEncoder,
)
from lib.metrics import RMSE_MAE_MAPE
from lib.data_prepare import get_dataloaders_from_index_data
from model.MDEformer import MDEformer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import matplotlib.patheffects as path_effects
@torch.no_grad()
def eval_model(model, valset_loader, criterion):
    model.eval()
    batch_loss_list = []
    for x_batch, y_batch in valset_loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        out_batch = model(x_batch)
        out_batch = SCALER.inverse_transform(out_batch)
        loss = criterion(out_batch, y_batch)
        batch_loss_list.append(loss.item())

    return np.mean(batch_loss_list)

@torch.no_grad()
def predict(model, loader):
    model.eval()
    y = []
    out = []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        out_batch = model(x_batch)
        out_batch = SCALER.inverse_transform(out_batch)
        out_batch = out_batch.cpu().numpy()
        y_batch = y_batch.cpu().numpy()
        out.append(out_batch)
        y.append(y_batch)

    out = np.vstack(out).squeeze()  # (samples, out_steps, num_nodes)
    y = np.vstack(y).squeeze()

    return y, out

def train_one_epoch(
    model, trainset_loader, optimizer, scheduler, scaler, criterion, clip_grad, log=None
):
    global cfg, global_iter_count, global_target_length

    model.train()
    batch_loss_list = []
    for x_batch, y_batch in trainset_loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        with autocast():
            out_batch = model(x_batch)
            out_batch = SCALER.inverse_transform(out_batch)
            loss = criterion(out_batch, y_batch)
        batch_loss_list.append(loss.item())
        optimizer.zero_grad()
        # loss.backward()
        scaler.scale(loss).backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        # optimizer.step()
        scaler.step(optimizer)
        scaler.update()

    epoch_loss = np.mean(batch_loss_list)
    scheduler.step()

    return epoch_loss


def train(
    model,
    trainset_loader,
    valset_loader,
    optimizer,
    scheduler,
    scaler,
    criterion,
    clip_grad=0,
    max_epochs=200,
    early_stop=10,
    verbose=1,
    plot=False,
    log=None,
    save=None,
):
    model = model.to(DEVICE)
    wait = 0
    min_val_loss = np.inf
    train_loss_list = []
    val_loss_list = []
    for epoch in range(max_epochs):
        train_loss = train_one_epoch(
            model, trainset_loader, optimizer, scheduler, scaler, criterion, clip_grad, log=log
        )
        train_loss_list.append(train_loss)

        val_loss = eval_model(model, valset_loader, criterion)
        val_loss_list.append(val_loss)

        if (epoch + 1) % verbose == 0:
            print_log(
                datetime.datetime.now(),
                "Epoch",
                epoch + 1,
                " \tTrain Loss = %.5f" % train_loss,
                "Val Loss = %.5f" % val_loss,
                log=log,
            )

        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= early_stop:
                break

    model.load_state_dict(best_state_dict)

    # # 节假日编码可视化
    # warnings.filterwarnings('ignore')
    # # Define font path and properties
    # font_path = "tnw+simsun.ttf"
    # font_manager.fontManager.addfont(font_path)
    # prop = font_manager.FontProperties(fname=font_path)
    # # Update matplotlib configuration
    # config = {
    #     "font.family": 'sans-serif',
    #     "font.size": 12,
    #     "mathtext.fontset": 'stix',
    #     "font.sans-serif": prop.get_name(),
    # }
    # rcParams.update(config)
    #
    # # 加载嵌入权重
    # hol_embedding_weight = best_state_dict["hol_embedding.weight"]
    # print(f"hol_embedding.weight 数据形状: {hol_embedding_weight.shape}")
    # hol_embedding_weight_np = hol_embedding_weight.cpu().numpy()
    #
    # # 节假日标签定义
    # holiday_labels = [0, 1, 2, 3]  # 0: 工作日, 1: 周六, 2: 联邦假日和周天, 3: 特殊假日前/后一天
    # label_names = {
    #     0: 'Week.',
    #     1: 'Sat.',
    #     2: 'Sun/hol.',
    #     3: 'PHB&A.'
    # }
    #
    # # 颜色映射
    # colors = ['royalblue', 'limegreen', 'crimson', 'darkorange']  # 优化后的颜色：宝蓝、青柠绿、绯红、橙红
    #
    # # 针对小样本优化t-SNE参数
    # tsne = TSNE(n_components=2,
    #             perplexity=1.2,  # 低困惑度适应小样本
    #             early_exaggeration=24,  # 增强初始分离
    #             learning_rate=180,  # 中等学习率
    #             n_iter=3500,  # 增加迭代次数确保收敛
    #             random_state=42,  # 确保可复现
    #             init='pca',  # PCA初始化提高稳定性
    #             metric='cosine')  # 使用余弦距离更符合嵌入特性
    #
    # # 执行t-SNE降维
    # tsne_results = tsne.fit_transform(hol_embedding_weight_np)
    #
    # # 创建高质量图像
    # plt.figure(figsize=(7.5, 7.1), dpi=300)
    #
    # # 使用兼容的样式设置
    # plt.style.use('default')  # 重置为默认样式
    # plt.rcParams['axes.grid'] = True  # 开启网格
    # plt.rcParams['grid.linestyle'] = '-'  # 虚线网格
    # plt.rcParams['grid.alpha'] = 0.6  # 网格透明度
    # plt.rcParams['axes.facecolor'] = 'white'  # 白色背景
    # plt.rcParams['axes.edgecolor'] = 'black'  # 黑色边框
    # plt.rcParams['axes.linewidth'] = 1  # 边框宽度
    # plt.rcParams['font.family'] = 'sans-serif'  # 边框宽度
    # plt.rcParams['mathtext.fontset'] = 'stix'  # 边框宽度
    # plt.rcParams['font.sans-serif'] = prop.get_name()  # 边框宽度
    # # 设置全局字体大小（放大字体）
    # plt.rcParams['font.size'] = 16  # 基础字体大小
    # plt.rcParams['axes.labelsize'] = 20  # 坐标轴标签字体大小
    # plt.rcParams['xtick.labelsize'] = 16  # X轴刻度字体大小
    # plt.rcParams['ytick.labelsize'] = 16  # Y轴刻度字体大小
    # plt.rcParams['legend.fontsize'] = 16  # 图例字体大小
    #
    # # 设置刻度朝内
    # plt.rcParams['xtick.direction'] = 'in'
    # plt.rcParams['ytick.direction'] = 'in'
    #
    # # 绘制散点图
    # scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1],
    #                       c=colors, s=300,  # 增大点的大小
    #                       edgecolor='black', linewidth=2,
    #                       alpha=0.95, zorder=5)
    #
    # # 添加图例 - 直接使用标签名称
    # legend_elements = [
    #     plt.Line2D([0], [0], marker='o', color='w',
    #                label=label_names[i],
    #                markerfacecolor=colors[i],
    #                markersize=16)
    #     for i in range(4)
    # ]
    #
    # legend = plt.legend(handles=legend_elements, loc='best',
    #                     fontsize=16,  # 增大图例字体
    #                     frameon=True,
    #                     framealpha=0.95,
    #                     shadow=True)  # 增大标题字体
    #
    # # # 设置图例标题字体
    # # plt.setp(legend.get_title(), fontfamily='Times New Roman')
    #
    # # 添加坐标轴标签
    # plt.xlabel("t-SNE Dimension 1", fontsize=20, labelpad=12, fontfamily='Times New Roman')
    # plt.ylabel("t-SNE Dimension 2", fontsize=20, labelpad=12, fontfamily='Times New Roman')
    #
    # # 添加网格和边框美化
    # plt.grid(True, linestyle='-', alpha=0.6, linewidth=1)
    # plt.gca().spines['top'].set_visible(True)
    # plt.gca().spines['right'].set_visible(True)
    # plt.gca().spines['bottom'].set_linewidth(1)
    # plt.gca().spines['left'].set_linewidth(1)
    #
    # # 设置刻度方向朝内
    # plt.tick_params(axis='both', which='both', direction='in',
    #                 length=6, width=1.5)  # 调整刻度线长度和宽度
    #
    # # 计算并显示类别间距离（仅控制台输出）
    # # 计算并显示类别间距离
    # from scipy.spatial import distance
    # print("\n类别间余弦距离矩阵:")
    # cos_dist_matrix = distance.cdist(hol_embedding_weight_np, hol_embedding_weight_np, 'cosine')
    #
    # print("          " + "".join([f"{label_names[i][:5]:>10}" for i in range(4)]))
    # for i in range(4):
    #     print(f"{label_names[i][:10]:<10}", end="")
    #     for j in range(4):
    #         print(f"{cos_dist_matrix[i, j]:>10.3f}", end="")
    #     print()
    #
    # # 保存高清图像
    # plt.tight_layout()
    # plt.savefig("holiday_tsne_4x24.tiff", dpi=300,
    #             bbox_inches='tight', pad_inches=0.3)
    # plt.show()
    # holiday_labels = [0, 1, 2, 3]  # 0: 工作日, 1: 周六, 2: 联邦假日和周天, 3: 特殊假日前/后一天
    # pca = PCA(n_components=2)
    # reduced = pca.fit_transform(hol_embedding_weight_np)
    #
    # # 颜色映射表
    # colors = {0: 'royalblue', 1: 'limegreen', 2: 'crimson', 3: 'darkorange'}
    # labels = {0: 'Wek.', 1: 'Sat.', 2: 'Sun/hol.', 3: 'Pph.'}
    #
    # plt.figure(figsize=(8, 8))
    #
    # # 获取图的边界范围
    # x_min, x_max = np.min(reduced[:, 0]), np.max(reduced[:, 0])
    # y_min, y_max = np.min(reduced[:, 1]), np.max(reduced[:, 1])
    #
    # # 绘制点
    # for i, (x, y) in enumerate(reduced):
    #     color = colors[holiday_labels[i]]  # 根据编码设置颜色
    #     label = labels[holiday_labels[i]]  # 获取对应标签
    #     plt.scatter(x, y, color=color, s=150, label=label)  # 将标注通过图例添加到图中
    #
    # # 添加两点之间的连线并标注距离
    # for i in range(len(reduced)):
    #     for j in range(i + 1, len(reduced)):
    #         # 获取两点的坐标
    #         x1, y1 = reduced[i]
    #         x2, y2 = reduced[j]
    #         # 绘制两点之间的连线
    #         plt.plot([x1, x2], [y1, y2], color="black", linestyle="--", linewidth=1.5, alpha=0.6)
    #         # 计算欧几里得距离
    #         distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    #         # 标注距离（显示在线段中点位置）
    #         mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
    #         plt.text(mid_x, mid_y, f"{distance:.2f}", fontsize=12, color="purple")
    #
    # # 设置图标题和坐标轴标签
    # # plt.title("PCA Visualization of hol_embedding.weight", fontsize=20)
    # plt.xlabel("PCA Dimension 1", fontsize=18, labelpad=10)
    # plt.ylabel("PCA Dimension 2", fontsize=18, labelpad=10)
    #
    # # 设置网格线
    # plt.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.6)
    #
    # # 设置横纵轴刻度线朝里
    # plt.tick_params(axis='both', direction='in', length=6, width=1.5, labelsize=14)
    #
    # # 调整刻度值的字体大小
    # plt.xticks(fontsize=16)
    # plt.yticks(fontsize=16)
    #
    # # 添加图例
    # plt.legend(loc='upper right', fontsize=14, frameon=True)  # 设置图例位置和样式
    #
    # # 保存图片到指定地址
    # save_path = "hol_embedding_pca_with_legend.tiff"  # 修改为你想保存的路径
    # plt.savefig(save_path, dpi=300, bbox_inches='tight')  # dpi=300 提高图片分辨率，bbox_inches='tight' 去掉多余空白
    # plt.show()
    train_rmse, train_mae, train_mape = RMSE_MAE_MAPE(*predict(model, trainset_loader))
    val_rmse, val_mae, val_mape = RMSE_MAE_MAPE(*predict(model, valset_loader))

    out_str = f"Early stopping at epoch: {epoch+1}\n"
    out_str += f"Best at epoch {best_epoch+1}:\n"
    out_str += "Train Loss = %.5f\n" % train_loss_list[best_epoch]
    out_str += "Train RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
        train_rmse,
        train_mae,
        train_mape,
    )
    out_str += "Val Loss = %.5f\n" % val_loss_list[best_epoch]
    out_str += "Val RMSE = %.5f, MAE = %.5f, MAPE = %.5f" % (
        val_rmse,
        val_mae,
        val_mape,
    )
    print_log(out_str, log=log)

    if plot:
        plt.plot(range(0, epoch + 1), train_loss_list, "-", label="Train Loss")
        plt.plot(range(0, epoch + 1), val_loss_list, "-", label="Val Loss")
        plt.title("Epoch-Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    if save:
        torch.save(best_state_dict, save)
    return model


@torch.no_grad()
def test_model(model, testset_loader, scaler, log=None):
    model.eval()

    print_log("--------- Test ---------", log=log)

    start = time.time()
    y_true, y_pred = predict(model, testset_loader)
    end = time.time()
    # # 扩展到 (3566, 12, 170, 1)
    scaler1 = StandardScaler()
    y_true = np.expand_dims(y_true, axis=-1)
    y_pred = np.expand_dims(y_pred, axis=-1)
    truefile = 'data{08}_true.npy'
    predltfile = 'data{08}_pred.npy'
    np.save(truefile, y_true)
    np.save(predltfile, y_pred)

    rmse_all, mae_all, mape_all = RMSE_MAE_MAPE(y_true, y_pred)
    out_str = "All Steps RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
        rmse_all,
        mae_all,
        mape_all,
    )
    out_steps = y_pred.shape[1]
    for i in range(out_steps):
        rmse, mae, mape = RMSE_MAE_MAPE(y_true[:, i, :], y_pred[:, i, :])
        out_str += "Step %d RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
            i + 1,
            rmse,
            mae,
            mape,
        )

    print_log(out_str, log=log, end="")
    print_log("Inference time: %.2f s" % (end - start), log=log)


if __name__ == "__main__":
    # -------------------------- set running environment ------------------------- #
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="pems08")
    parser.add_argument("-g", "--gpu_num", type=int, default=0)
    args = parser.parse_args()
    seed = torch.randint(1000, (1,)) # set random seed here
    seed_everything(seed)
    set_cpu_num(1)
    GPU_ID = args.gpu_num
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU_ID}"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = args.dataset
    dataset = dataset.upper()
    data_path = f"../data/{dataset}"
    model_name = MDEformer.__name__
    with open(f"{model_name}.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg = cfg[dataset]

    # -------------------------------- load model -------------------------------- #
    model = MDEformer(**cfg["model_args"])

    # ------------------------------- make log file ------------------------------ #
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_path = f"../log08/"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log = os.path.join(log_path, f"{model_name}-{dataset}-{now}.log")
    log = open(log, "a")
    log.seek(0)
    log.truncate()

    # ------------------------------- load dataset ------------------------------- #
    print_log(dataset, log=log)
    (
        trainset_loader,
        valset_loader,
        testset_loader,
        SCALER,
    ) = get_dataloaders_from_index_data(
        data_path,
        tod=cfg.get("time_of_day"),
        dow=cfg.get("day_of_week"),
        batch_size=cfg.get("batch_size"),
        log=log,
    )
    print_log(log=log)

    # --------------------------- set model saving path -------------------------- #
    save_path = f"../saved_models/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save = os.path.join(save_path, f"{model_name}-{dataset}-{now}.pt")

    # ---------------------- set loss, optimizer, scheduler ---------------------- #
    if dataset in ("METRLA", "PEMSBAY"):
        criterion = MaskedMAELoss()
    elif dataset in ("PEMS03", "PEMS04", "PEMS07", "PEMS08"):
        criterion = nn.HuberLoss()
    else:
        raise ValueError("Unsupported dataset.")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0),
        eps=cfg.get("eps", 1e-8),
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg["milestones"],
        gamma=cfg.get("lr_decay_rate", 0.1),
        # verbose=False,
    )
    scaler = GradScaler()

    # --------------------------- print model structure -------------------------- #
    print_log("---------", model_name, "---------", log=log)
    print_log(
        json.dumps(cfg, ensure_ascii=False, indent=4, cls=CustomJSONEncoder), log=log
    )
    print_log(
        summary(
            model,
            [
                cfg["batch_size"],
                cfg["in_steps"],
                cfg["num_nodes"],
                next(iter(trainset_loader))[0].shape[-1],
            ],
            verbose=0,  # avoid print twice
        ),
        log=log,
    )
    print_log(log=log)

    # --------------------------- train and test model --------------------------- #
    print_log(f"Loss: {criterion._get_name()}", log=log)
    print_log(log=log)

    model = train(
        model,
        trainset_loader,
        valset_loader,
        optimizer,
        scheduler,
        scaler,
        criterion,
        clip_grad=cfg.get("clip_grad"),
        max_epochs=cfg.get("max_epochs", 200),
        early_stop=cfg.get("early_stop", 10),
        verbose=1,
        log=log,
        save=save,
    )

    print_log(f"Saved Model: {save}", log=log)
    test_model(model, testset_loader, scaler, log=log)
    log.close()
