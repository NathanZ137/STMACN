import torch
import numpy as np
import seaborn as sns
import random
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
from utils.trafficdataset import TrafficDataset
from utils.utils import load_config, Seq2Instance, count_parameters
from utils.utils import metrics
from trainer.base_trainer import BaseTrainer
from model.STMACN import STMACN
import time
# import ipdb
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# Set random seeds
def seed_env(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # multi-GPU
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_env(42)

class STMACNTrainer(BaseTrainer):
    def __init__(self, cfg_file):
        self.conf = load_config(cfg_file)
        self.device = self.load_device()
        self.SE = self.load_SE()
        

    def load_SE(self):
        SE_file = f"data/{self.conf['dataset_name']}/SE_{self.conf['dataset_name']}.txt"
        with open(SE_file, mode='r') as f:
            lines = f.readlines()
            # V=325,D=64
            num_vertex, dims = map(int, lines[0].split(' '))
            SE = torch.zeros((num_vertex, dims), dtype=torch.float32)
            for line in lines[1:]:
                parts = line.split(' ')
                # 顶点编号
                index = int(parts[0])
                values = list(map(float, parts[1:]))
                SE[index] = torch.tensor(values, dtype=torch.float32)
        
        return SE.to(self.device)


    def load_data(self):
        """
        data
            - trainX: (num_sample, num_his, num_vertex)
            - trainTE: (num_sample, num_his + num_pred, 2)
            - trainY: (num_sample, num_pred, num_vertex)
            - valX: (num_sample, num_his, num_vertex)
            - valTE: (num_sample, num_his + num_pred, 2)
            - valY: (num_sample, num_pred, num_vertex)
            - testX: (num_sample, num_his, num_vertex)
            - testTE: (num_sample, num_his + num_pred, 2)
            - testY: (num_sample, num_pred, num_vertex)
            - mean: float
            - std: float
        """
        data = {}

        # Get Traffic Data
        TE_file = f"data/{self.conf['dataset_name']}/TE_{self.conf['dataset_name']}.npz"
        traffic_file = f"data/{self.conf['dataset_name']}/{self.conf['dataset_name']}.npz"
        
        # [seq_len, num_vertex]
        traffic = np.load(traffic_file)['data']
        traffic_data = torch.from_numpy(traffic)
        
        # train/val/test Split
        seq_len = traffic.shape[0]
        train_step = round(self.conf['train_radio'] * seq_len)
        val_step = round(self.conf['val_radio'] * seq_len)
        test_step = round(self.conf['test_radio'] * seq_len)
        train = traffic_data[:train_step]
        val = traffic_data[train_step:train_step+val_step]
        test = traffic_data[-test_step:]

        # X,Y
        num_his = self.conf['num_his']
        num_pred= self.conf['num_pred']
        trainX, data['trainY'] = Seq2Instance(train, num_his, num_pred)
        valX, data['valY'] = Seq2Instance(val, num_his, num_pred)
        testX, data['testY'] = Seq2Instance(test, num_his, num_pred)

        # Normalization
        mean, std = torch.mean(trainX), torch.std(trainX)
        data['trainX'] = (trainX - mean) / std
        data['valX'] = (valX - mean) / std
        data['testX'] = (testX - mean) / std

        self.mean = mean.clone().detach().to(self.device)
        self.std = std.clone().detach().to(self.device)

        # Get TE initial
        time = np.load(TE_file)['data']
        time = torch.from_numpy(time)

        # train/val/test TE Split
        train = time[:train_step]
        val = time[train_step:train_step+val_step]
        test = time[-test_step:]
        # [num_sample, num_his+num_pred, 2]
        trainTE_his, trainTE_pred = Seq2Instance(train, num_his, num_pred)
        data['trainTE'] = torch.cat((trainTE_his, trainTE_pred), dim=1)
        valTE_his, valTE_pred = Seq2Instance(val, num_his, num_pred)
        data['valTE'] = torch.cat((valTE_his, valTE_pred), dim=1)
        testTE_his, testTE_pred = Seq2Instance(test, num_his, num_pred)
        data['testTE'] = torch.cat((testTE_his, testTE_pred), dim=1)

        # 加载数据集
        train_dataset = TrafficDataset(data['trainX'], data['trainY'], data['trainTE'])
        val_dataset = TrafficDataset(data['valX'], data['valY'], data['valTE'])
        test_dataset = TrafficDataset(data['testX'], data['testY'], data['testTE'])
        # dataloader
        self.train_loader = DataLoader(train_dataset,
                                       batch_size=self.conf['batch_size'],
                                       shuffle=True,
                                       num_workers=self.conf['num_workers'],
                                       pin_memory=True)
        self.val_loader = DataLoader(val_dataset,
                                       batch_size=self.conf['batch_size'],
                                       shuffle=False,
                                       num_workers=self.conf['num_workers'],
                                       pin_memory=True)
        self.test_loader = DataLoader(test_dataset,
                                       batch_size=self.conf['batch_size'],
                                       shuffle=False,
                                       num_workers=self.conf['num_workers'],
                                       pin_memory=True)
        

    # def visualize_random_node_predictions(self, y_true, y_pred, time_steps):

    #     num_nodes = y_true.shape[2]
    #     selected_nodes = random.sample(range(num_nodes), 3)

    #     num_steps = min(time_steps, y_true.shape[0])  # Ensure we don't exceed available time steps
    #     fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    #     for i, node in enumerate(selected_nodes):
    #         true_series = y_true[:time_steps, 0, node]
    #         pred_series = y_pred[:time_steps, 0, node]
    #         axes[i].plot(range(num_steps), true_series, label='True', color='blue')
    #         axes[i].plot(range(num_steps), pred_series, label='Predict', color='red')
    #         # axes[i].set_title(f'Node {node}')
    #         axes[i].set_xlabel('Time Steps')
    #         axes[i].set_ylabel('Traffic Flow')
    #         axes[i].legend()
    #         axes[i].grid(True)
    #         # Set y-axis limits based on data
    #         axes[i].set_ylim([min(np.min(y_true[:time_steps, 0, node]), np.min(y_pred[:time_steps, 0, node])),
    #                          max(np.max(y_true[:time_steps, 0, node]), np.max(y_pred[:time_steps, 0, node])) + 80])
            
    #         labels = ['(a)', '(b)', '(c)']
    #         label = labels[i]
    #         axes[i].text(0.5, -0.15, f'{label} Node {node} on PeMS08', 
    #                     ha='center', va='center', transform=axes[i].transAxes, fontsize=15)

    #     plt.tight_layout()
    #     plot_filename = os.path.join('./pic', f'plot_nodes_{selected_nodes[0]}_{selected_nodes[1]}_{selected_nodes[2]}_PeMS08.svg')
    #     plt.savefig(plot_filename, format='svg')
    #     plt.close()


    # def visualize_attention(self, attention_score, num_heads, batch_index, time_index, num_nodes):
    #     
    #     # 获取 batch_size，假设 attention_score 的第一维为 num_heads * batch_size
    #     batch_size = attention_score.shape[0] // num_heads
        
    #     # 重塑为 [batch_size, num_heads, num_step, num_vertex, num_vertex]
    #     att_score_reshaped = attention_score.view(batch_size, num_heads, *attention_score.shape[1:])
        
    #     # 在 head 维度上取平均, 得到 [batch_size, num_step, num_vertex, num_vertex]
    #     att_score_avg = att_score_reshaped.mean(dim=1)
        
    #     # 选择指定的样本和时间步的 attention 矩阵
    #     att_matrix = att_score_avg[batch_index, time_index]  # [num_vertex, num_vertex]
        
    #     # 对矩阵进行 L2 归一化：除以矩阵整体的 L2 范数
    #     norm = torch.norm(att_matrix, p=2)
    #     att_matrix_norm = att_matrix / (norm + 1e-8)
        
    #     # 截取前 num_nodes 个节点构成的子矩阵
    #     att_matrix_norm_sub = att_matrix_norm[:num_nodes, :num_nodes].detach().cpu().numpy()
        
    #     # 绘制热力图
    #     plt.figure(figsize=(8, 6))
    #     sns.heatmap(att_matrix_norm_sub, cmap="Reds", square=True, annot=False)
    #     plt.subplots_adjust(bottom=0.15)
    #     plt.xlabel("Nodes")
    #     plt.ylabel("Nodes")
    #     plt.figtext(0.48, 0.02, '(b) Spatial Attention Weight Matrix', ha="center", fontsize=15)
    #     # plt.show()
    #     plot_filename = os.path.join('./pic', 'Spatial Attention Weight Matrix.svg')
    #     plt.savefig(plot_filename, format='svg')
    #     plt.close()


    # def plot_node_similarity_heatmap(self, SE, num_nodes):
    #     
    #     # 选择前 num_nodes 个节点的嵌入
    #     embedding_subset = SE[:num_nodes].detach().cpu().numpy()
        
    #     # 计算余弦相似度矩阵
    #     similarity_matrix = cosine_similarity(embedding_subset)
        
    #     # 绘制热力图，图形尺寸和颜色映射固定
    #     plt.figure(figsize=(8, 6))
    #     sns.heatmap(similarity_matrix, cmap="Reds", square=True, annot=False)
    #     plt.subplots_adjust(bottom=0.15)
    #     plt.xlabel("Nodes")
    #     plt.ylabel("Nodes")
    #     plt.figtext(0.48, 0.02, '(a) Spatial Embedding Matrix', ha="center", fontsize=15)
    #     # plt.show()
    #     plot_filename = os.path.join('./pic', 'Spatial Embedding Matrix.svg')
    #     plt.savefig(plot_filename, format='svg')
    #     plt.close()
   
        
    def train_epoch(self, epoch):
        total_loss = 0
        self.model.train()
        t_begin = time.time()
        
        for batch_index, (x, y, te) in enumerate(self.train_loader):
            # ipdb.set_trace()
            X = x.to(self.device)
            Y = y.to(self.device)
            TE = te.to(self.device)
            Y_hat = self.model(X, TE, return_attention=False)
            Y_hat = Y_hat * self.std + self.mean

            loss_batch = self.loss_criterion(Y_hat, Y)
            self.optimizer.zero_grad()
            loss_batch.backward()
            self.optimizer.step()
            
            total_loss += loss_batch.item()
            # print loss
            if (batch_index + 1) % 10 == 0:
                print(f"[Training] Epoch:{epoch:<5}, Batch:{batch_index+1:<5}, MAE Loss:{loss_batch.item():.4f}")
        t_end = time.time() - t_begin

        return total_loss / len(self.train_loader), t_end
    

    def validate_epoch(self, epoch):
        total_loss = 0
        self.model.eval()
        t_begin = time.time()
        
        with torch.no_grad():
            for batch_index, (x, y, te) in enumerate(self.val_loader):
                X = x.to(self.device)
                Y = y.to(self.device)
                TE = te.to(self.device)
                Y_hat = self.model(X, TE, return_attention=False)
                Y_hat = Y_hat * self.std + self.mean

                loss_batch = self.loss_criterion(Y_hat, Y)
                total_loss += loss_batch.item()
                # print loss
                if (batch_index + 1) % 10 == 0:
                    print(f"[Valitate] Epoch:{epoch:<5}, Batch:{batch_index+1:<5}, MAE Loss:{loss_batch.item():.4f}")

        t_end = time.time() - t_begin

        val_loss = total_loss / len(self.val_loader)
        return (val_loss, t_end)
    
    
    def test_epoch(self):
        self.model.eval()
        t_begin = time.time()
        total_mae = 0
        total_rmse = 0
        total_mape = 0

        y_hat_list = []
        y_list = []
        x_list = []

        with torch.no_grad():
            for batch_index, (x, y, te) in enumerate(self.test_loader):
                X = x.to(self.device)
                Y = y.to(self.device)
                TE = te.to(self.device)
                Y_hat, att_score = self.model(X, TE, return_attention=True)
                Y_hat = Y_hat * self.std + self.mean

                y_hat = Y_hat.clone().detach().cpu()
                y = Y.clone().detach().cpu()

                newx = X * self.std + self.mean
                newx = newx.clone().detach().cpu()
                x_list.append(newx.numpy())
                y_list.append(y.numpy())
                y_hat_list.append(y_hat.numpy())

                mae, rmse, mape = metrics(y_hat, y)
                total_mae += mae
                total_rmse += rmse
                total_mape += mape
                # print loss
                if (batch_index + 1) % 10 == 0:
                    print(f"[Valitate]Batch:{batch_index+1:<4}, MAE Loss:{mae:.4f}, RMSE Loss:{rmse:.4f}, MAPE Loss:{mape*100:.4f}")

        t_end = time.time() - t_begin

        y = np.concatenate(y_list, axis=0)
        y_hat = np.concatenate(y_hat_list, axis=0)
        x = np.concatenate(x_list, axis=0)
        # self.visualize_random_node_predictions(y, y_hat, time_steps=1000)

        # 模型返回： output, att_score = model(X, TE, return_attention=True)
        # self.visualize_attention(att_score, num_heads=self.conf['num_heads'], batch_index=0, time_index=0, num_nodes=30)
        # self.plot_node_similarity_heatmap(SE=self.SE, num_nodes=30)

        n1 = len(self.test_loader)
        avg_mae = total_mae / n1
        avg_rmse = total_rmse / n1
        avg_mape = total_mape / n1
        return (avg_mae, avg_rmse, avg_mape, t_end)
    

    # def visualize_random_nodes(self, time_steps):
    
    #     # 获取训练集数据 (num_sample, num_his, num_vertex)
    #     traffic = self.train_loader.dataset.X  # 形状: (num_sample, num_his, num_vertex)
    #     traffic = traffic.to(self.device)
    #     traffic = traffic * self.std + self.mean  # 反归一化
    #     traffic = traffic.cpu().numpy()  # 转换为 numpy 方便绘制

    #     num_sample, num_his, num_vertex = traffic.shape

    #     # 随机选择两个节点
    #     random_nodes = random.sample(range(num_vertex), 3)

    #     # 如果数据时间步不足，则绘制全部时间步
    #     if num_sample < time_steps:
    #         time_steps = num_sample

    #     # 绘图
    #     time_axis = np.arange(time_steps)
    #     plt.figure(figsize=(8, 5))
    #     for node in random_nodes:
    #         plt.plot(time_axis, traffic[:time_steps, 0, node], label=f'Node {node}')
        
    #     plt.xlabel('Time Steps')
    #     plt.ylabel('Traffic Flow')
    #     plt.suptitle('(b) Traffic flow of three random nodes on PeMS08', y=0.05, fontsize=15)
    #     plt.subplots_adjust(bottom=0.15)
    #     plt.legend()
    #     plt.grid(True)
    #     plot_filename = os.path.join('./pic', 'PeMS08_TrafficFlow.svg')
    #     plt.savefig(plot_filename, format='svg')
    #     plt.close()
        

    def train(self):
        self.load_SE()
        self.load_data()
        self.model = STMACN(
                self.SE,
                self.conf
            ).to(self.device)
        self.setup_train()
        count_parameters(self.model)
        min_val_epoch = 0
        train_total_loss = []
        val_total_loss = []
        min_loss_val = np.Inf
        epochs = self.conf['max_epoch']
        T_begin = time.time()
        
        for epoch in range(1, epochs+1):
            # train
            epoch_train_loss, time_train_epoch = self.train_epoch(epoch)
            train_total_loss.append(epoch_train_loss)
            # valitate
            epoch_val_loss, time_val_epoch = self.validate_epoch(epoch)
            val_total_loss.append(epoch_val_loss)

            # train time log
            print("Epoch: %03d/%03d, Training time: %.1f Seconds, Inference time:%.1f Seconds" % 
                (epoch, epochs, time_train_epoch, time_val_epoch))
            # train loss log
            print(f"Training loss: {epoch_train_loss:.4f}, Validation loss: {epoch_val_loss:.4f}")

            self.scheduler.step()
        
            # 更小的loss_val,保存模型参数
            if epoch_val_loss < min_loss_val:
                min_loss_val = epoch_val_loss
                min_val_epoch = epoch
                model_file = f"./ckpt/STMACN_{self.conf['dataset_name']}_epoch{epochs}_min_val_epoch{min_val_epoch}.ckpt"
                torch.save(self.model.state_dict(), model_file)
                print('min_val_epoch: ', min_val_epoch)
                print('min_loss_val: ', min_loss_val)

        T_end = time.time() - T_begin

        print(f"Well Done! Total Cost {T_end/60:.2f} Minutes To Train!")
        print(f"model has been saved with min loss of {min_loss_val:.4f} at epoch {min_val_epoch}")

        # Save validation loss to a file
        # np.save(f'./loss/{self.conf["dataset_name"]}/val_loss_{self.conf["dataset_name"]}_origin.npy', np.array(val_total_loss))
        # np.save(f'./loss/{self.conf["dataset_name"]}/val_loss_{self.conf["dataset_name"]}_ablation2.npy', np.array(val_total_loss))

        # 在训练结束后调用可视化函数，查看一个样本中两个随机节点的真实值
        # self.visualize_random_nodes(time_steps=1000)


    def load_pretrained_model(self):
        ckpt_dir = './ckpt'
        ckpt_files = os.listdir(ckpt_dir)

        model_path = None
        
        # 只保留当前数据集有关模型文件
        model_files=[]
        for file in ckpt_files:
            if self.conf['dataset_name'] == file.split('_')[1]:
                model_files.append(file)
        def extract_epoch(filename):
            return int(filename.split("min_val_epoch")[-1].replace("epoch", "").replace(".ckpt", ""))
        
        min_val_loss_files = max(model_files, key=extract_epoch, default=None)

        model_path = os.path.join(ckpt_dir, min_val_loss_files)
        print(f"选中的模型: {min_val_loss_files}")
        
        # Load model
        loaded_model = torch.load(model_path, map_location=self.device)
        if isinstance(loaded_model, dict):
            self.load_SE()
            self.model = STMACN(
                    self.SE,
                    self.conf
                ).to(self.device)
            self.model.load_state_dict(loaded_model)
        else:
            self.model = loaded_model
      
      
        print(f"model restored from {model_path}, start inference...\n")


    def eval(self):
        self.load_data()
        self.load_pretrained_model()
        mae, rmse, mape, t_cost = self.test_epoch()
        print(f"Inference Time Cost: {(t_cost):.1f} Seconds")
        print("评价指标\t\tMAE\t\tRMSE\t\tMAPE")
        print("------------------------------------------------------")
        print("结果\t\t{:.3f}\t\t{:.3f}\t\t{:.3f}%".format(mae, rmse, mape * 100))
