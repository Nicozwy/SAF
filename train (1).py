import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, ViTImageProcessor, ViTModel
from PIL import Image
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import build_dataset

# ===================== 1. 基础配置（新增iet相关维度适配） =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{device}")
TEXT_HIDDEN = 768
IMAGE_HIDDEN = 768
FUSION_DIM = 512
SIM_DIM = 256
INIT_THRESHOLD = 0.7
EPOCHS = 30
BATCH_SIZE = 2
NUM_CLASSES = 4
EPS = 1e-6  # 数值稳定性常数
LR = 5e-5

# 四任务损失权重
ALPHA1 = 0.25
ALPHA2 = 0.25
ALPHA3 = 0.25
ALPHA4 = 0.25

# 标签定义（新增image-image_e_fused对应标签）
LABEL_MAP = {
    "真实": 0,
    "图文不一致": 1,
    "文本证据不一致": 2,
    "图像证据不一致": 3
}
LABEL_TO_SUBTASKS = {
    0: {"l_t": 0, "l_i": 0, "l_ti": 0, "l_f": 0},
    1: {"l_t": 0, "l_i": 0, "l_ti": 1, "l_f": 1},
    2: {"l_t": 1, "l_i": 0, "l_ti": 0, "l_f": 1},
    3: {"l_t": 0, "l_i": 1, "l_ti": 0, "l_f": 1}
}
# 新增image-image_e_fused（融合后图像证据）的映射
SIM_TO_LABEL = {
    "image-text": 1,
    "text-text_e": 2,
    "image-image_e": 3,
    "image-image_e_fused": 3  # 融合后仍归为图像证据不一致
}
LABEL_REVERSE = {v: k for k, v in LABEL_MAP.items()}

# 预处理工具
tokenizer = BertTokenizer.from_pretrained(r"E:\model\bert-base-uncased")
image_processor = ViTImageProcessor.from_pretrained(r"E:\model\vit-base-patch16-224")

# ===================== 2. 数据集构建（新增image_et字段） =====================
class SimDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        text_e_inputs = self._preprocess_text(item["text_e"])
        text_inputs = self._preprocess_text(item["text"])
        image_inputs = self._preprocess_image(item["image_path"])
        image_e_inputs = self._preprocess_image(item["image_e_path"])
        # 新增：预处理图片证据的文本（iet）
        image_et_inputs = self._preprocess_text(item["image_et"])
        
        label = torch.tensor(LABEL_MAP[item["label"]], dtype=torch.long)
        subtask_labels = LABEL_TO_SUBTASKS[label.item()]
        l_t = torch.tensor(subtask_labels["l_t"], dtype=torch.float32)
        l_i = torch.tensor(subtask_labels["l_i"], dtype=torch.float32)
        l_ti = torch.tensor(subtask_labels["l_ti"], dtype=torch.float32)
        l_f = torch.tensor(subtask_labels["l_f"], dtype=torch.float32)
        
        return {
            "text_e": text_e_inputs,
            "text": text_inputs,
            "image": image_inputs,
            "image_e": image_e_inputs,
            "image_et": image_et_inputs,  # 新增iet输入
            "label": label,
            "l_t": l_t,
            "l_i": l_i,
            "l_ti": l_ti,
            "l_f": l_f
        }

    def _preprocess_text(self, text: str) -> dict:
        inputs = tokenizer(
            text, truncation=True, padding="max_length", max_length=32, return_tensors="pt"
        )
        return {k: v.squeeze(0) for k, v in inputs.items()}

    def _preprocess_image(self, img_path: str) -> dict:
        img = Image.open(img_path).convert("RGB")
        inputs = image_processor(images=img, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in inputs.items()}

# ===================== 3. CrossTransformer（核心新增iet融合逻辑） =====================
class CrossTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = BertModel.from_pretrained(r"E:\model\bert-base-uncased").to(device)
        self.image_encoder = ViTModel.from_pretrained(r"E:\model\vit-base-patch16-224").to(device)
        
        self.text_proj = nn.Linear(TEXT_HIDDEN, FUSION_DIM).to(device)
        self.image_proj = nn.Linear(IMAGE_HIDDEN, FUSION_DIM).to(device)
        
        # 原有Transformer层
        self.trans_layer_img_text = nn.TransformerEncoderLayer(
            d_model=FUSION_DIM, nhead=8, batch_first=True, dropout=0.1
        ).to(device)
        self.trans_layer_img_ime = nn.TransformerEncoderLayer(
            d_model=FUSION_DIM, nhead=8, batch_first=True, dropout=0.1
        ).to(device)
        self.trans_layer_text_te = nn.TransformerEncoderLayer(
            d_model=FUSION_DIM, nhead=8, batch_first=True, dropout=0.1
        ).to(device)
        # 新增：处理iei_patch和iet_proj融合的Transformer层
        self.trans_layer_img_ie_fusion = nn.TransformerEncoderLayer(
            d_model=FUSION_DIM, nhead=8, batch_first=True, dropout=0.1
        ).to(device)
        # 新增：处理i和i_e_fused的Transformer层
        self.trans_layer_img_ie_final = nn.TransformerEncoderLayer(
            d_model=FUSION_DIM, nhead=8, batch_first=True, dropout=0.1
        ).to(device)

    def get_text_token_feat(self, text_inputs: dict) -> tuple:
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        outputs = self.text_encoder(**text_inputs)
        token_feat = outputs.last_hidden_state
        cls_feat = token_feat[:, 0, :]
        
        token_feat = self.text_proj(token_feat)
        cls_feat = self.text_proj(cls_feat)
        return token_feat, cls_feat

    def get_image_patch_feat(self, img_inputs: dict) -> tuple:
        img_inputs = {k: v.to(device) for k, v in img_inputs.items()}
        outputs = self.image_encoder(**img_inputs)
        patch_feat = outputs.last_hidden_state
        cls_feat = patch_feat[:, 0, :]
        
        patch_feat = self.image_proj(patch_feat)
        cls_feat = self.image_proj(cls_feat)
        return patch_feat, cls_feat

    def calculate_pair_sim(self, feat1: torch.Tensor, feat2: torch.Tensor, trans_layer: nn.TransformerEncoderLayer) -> torch.Tensor:
        concat_feat = torch.cat([feat1, feat2], dim=1)
        trans_feat = trans_layer(concat_feat)
        
        feat1_trans = trans_feat[:, :feat1.shape[1], :]
        feat2_trans = trans_feat[:, feat1.shape[1]:, :]
        
        feat1_norm = F.normalize(feat1_trans, p=2, dim=-1)
        feat2_norm = F.normalize(feat2_trans, p=2, dim=-1)
        sim_matrix = torch.bmm(feat1_norm, feat2_norm.transpose(1, 2))
        
        max_sim = torch.max(sim_matrix, dim=2)[0].mean(dim=1)
        mean_sim = torch.mean(sim_matrix, dim=(1, 2))
        final_sim = (max_sim + mean_sim) / 2
        final_sim = torch.clamp(final_sim, 0.0 + EPS, 1.0 - EPS)
        return final_sim

    def forward(self, batch: dict) -> dict:
        # 原有特征提取
        text_token, text_cls = self.get_text_token_feat(batch["text"])
        text_e_token, text_e_cls = self.get_text_token_feat(batch["text_e"])
        img_patch, img_cls = self.get_image_patch_feat(batch["image"])
        img_e_patch, img_e_cls = self.get_image_patch_feat(batch["image_e"])
        
        # 新增：提取图片证据文本（iet）的特征
        image_et_token, image_et_cls = self.get_text_token_feat(batch["image_et"])
        
        # 核心：计算i_e,fused = CrossTransformer(i_ei_patch, i_et_proj)
        # 步骤1：融合iei_patch（图片证据图片）和iet_proj（图片证据文本）得到i_e_fused
        i_e_fused_patch = self.trans_layer_img_ie_fusion(
            torch.cat([img_e_patch, image_et_token], dim=1)
        )
        
        # 步骤2：计算原图片i与i_e_fused的相似度（s_i_ie）
        trans_sim_img_ie_fused = self.calculate_pair_sim(
            img_patch, i_e_fused_patch, self.trans_layer_img_ie_final
        )
        
        # 原有相似度计算
        trans_sim_img_text = self.calculate_pair_sim(img_patch, text_token, self.trans_layer_img_text)
        trans_sim_img_ime = self.calculate_pair_sim(img_patch, img_e_patch, self.trans_layer_img_ime)
        trans_sim_text_te = self.calculate_pair_sim(text_token, text_e_token, self.trans_layer_text_te)
        
        return {
            # 原有返回值
            "trans_sim_img_text": trans_sim_img_text,
            "trans_sim_img_ime": trans_sim_img_ime,
            "trans_sim_text_te": trans_sim_text_te,
            "img_cls": img_cls,
            "text_cls": text_cls,
            "img_e_cls": img_e_cls,
            "text_e_cls": text_e_cls,
            # 新增返回值：融合后的图像证据相似度
            "trans_sim_img_ie_fused": trans_sim_img_ie_fused,
            "i_e_fused_patch": i_e_fused_patch,  # 可选：返回融合特征
            "image_et_cls": image_et_cls         # 可选：返回iet的cls特征
        }

# ===================== 4. SimEnhancer（适配新增的融合相似度） =====================
class SimEnhancer(nn.Module):
    def __init__(self):
        super().__init__()
        self.threshold_param = nn.Parameter(torch.tensor(INIT_THRESHOLD, dtype=torch.float32, device=device))
        
        # 得分头维度保持不变（如需更精细可调整，此处兼容原有逻辑）
        self.score_head_t = nn.Sequential(
            nn.Linear(FUSION_DIM, 1).to(device),
            nn.Sigmoid().to(device),
            nn.Hardtanh(min_val=EPS, max_val=1-EPS).to(device)
        )
        self.score_head_i = nn.Sequential(
            nn.Linear(FUSION_DIM, 1).to(device),
            nn.Sigmoid().to(device),
            nn.Hardtanh(min_val=EPS, max_val=1-EPS).to(device)
        )
        self.score_head_ti = nn.Sequential(
            nn.Linear(FUSION_DIM*2, 1).to(device),
            nn.Sigmoid().to(device),
            nn.Hardtanh(min_val=EPS, max_val=1-EPS).to(device)
        )
        self.score_head_f = nn.Sequential(
            nn.Linear(FUSION_DIM*3, 1).to(device),
            nn.Sigmoid().to(device),
            nn.Hardtanh(min_val=EPS, max_val=1-EPS).to(device)
        )
        
        self.bce_loss = nn.BCELoss(reduction='mean').to(device)

    def _get_min_sim_info(self, sim_dict: dict) -> tuple:
        # 新增：纳入融合后的图像证据相似度
        sim_img_text = sim_dict["image-text"]
        sim_text_te = sim_dict["text-text_e"]
        sim_img_ime = sim_dict["image-image_e"]
        sim_img_ie_fused = sim_dict["image-image_e_fused"]  # 融合后的相似度
        
        # 堆叠所有相似度计算最小值
        sim_stack = torch.stack([sim_img_text, sim_text_te, sim_img_ime, sim_img_ie_fused], dim=1)
        min_sim, min_sim_idx = torch.min(sim_stack, dim=1)
        
        # 适配新增的索引（3对应融合后的图像证据）
        min_sim_label = torch.where(
            min_sim_idx == 0, torch.tensor(1, device=device),
            torch.where(
                min_sim_idx == 1, torch.tensor(2, device=device),
                torch.where(
                    min_sim_idx == 2, torch.tensor(3, device=device),
                    torch.tensor(3, device=device)  # 索引3也归为图像证据不一致
                )
            )
        )
        
        return min_sim, min_sim_label

    def _calc_four_task_loss(self, trans_output: dict, subtask_labels: dict, min_sim: torch.Tensor) -> tuple:
        # 原有得分计算逻辑（如需优化可加入iet_cls，此处保持兼容）
        s_t = self.score_head_t(torch.abs(trans_output["text_cls"] - trans_output["text_e_cls"])).squeeze(-1)
        s_i = self.score_head_i(torch.abs(trans_output["img_cls"] - trans_output["img_e_cls"])).squeeze(-1)
        s_ti = self.score_head_ti(torch.cat([trans_output["img_cls"], trans_output["text_cls"]], dim=-1)).squeeze(-1)
        # 可选：score_head_f纳入iet_cls（trans_output["image_et_cls"]）
        s_f = self.score_head_f(torch.cat([trans_output["img_cls"], trans_output["text_cls"], trans_output["img_e_cls"]], dim=-1)).squeeze(-1)
        
        # 真实标签移到device
        y_t = subtask_labels["l_t"].to(device)
        y_i = subtask_labels["l_i"].to(device)
        y_ti = subtask_labels["l_ti"].to(device)
        y_f = subtask_labels["l_f"].to(device)
        
        # 四任务BCE损失
        l_ti = -torch.mean(y_ti * torch.log(s_ti) + (1 - y_ti) * torch.log(1 - s_ti))
        l_t = -torch.mean(y_t * torch.log(s_t) + (1 - y_t) * torch.log(1 - s_t))
        l_i = -torch.mean(y_i * torch.log(s_i) + (1 - y_i) * torch.log(1 - s_i))
        l_f = -torch.mean(y_f * torch.log(s_f) + (1 - y_f) * torch.log(1 - s_f))
        
        # 阈值约束损失（保留，基于融合后的最小相似度）
        true_mask = (y_f == 0)
        fake_mask = (y_f == 1)
        
        loss_true = torch.tensor(0.0, device=device)
        if true_mask.any():
            loss_true = torch.mean(torch.clamp(self.threshold_param - min_sim[true_mask], min=0.0))
        
        loss_fake = torch.tensor(0.0, device=device)
        if fake_mask.any():
            loss_fake = torch.mean(torch.clamp(min_sim[fake_mask] - self.threshold_param, min=0.0))
        
        loss_threshold = 0.5 * (loss_true + loss_fake)
        
        # 总损失
        total_loss = ALPHA1 * l_ti + ALPHA2 * l_t + ALPHA3 * l_i + ALPHA4 * l_f + loss_threshold
        
        # 兜底：防止NaN
        if torch.isnan(total_loss):
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        return total_loss, l_ti, l_t, l_i, l_f, loss_threshold

    def forward(self, trans_output: dict, true_labels: torch.Tensor, subtask_labels: dict) -> tuple:
        # 最终相似度字典：纳入融合后的图像证据相似度
        final_sim_dict = {
            "image-text": trans_output["trans_sim_img_text"],
            "image-image_e": trans_output["trans_sim_img_ime"],
            "text-text_e": trans_output["trans_sim_text_te"],
            "image-image_e_fused": trans_output["trans_sim_img_ie_fused"]  # 新增融合相似度
        }
        
        # 最小相似度计算（包含融合后的相似度）
        min_sim, min_sim_label = self._get_min_sim_info(final_sim_dict)
        
        # 计算损失
        total_loss, l_ti, l_t, l_i, l_f, loss_threshold = self._calc_four_task_loss(trans_output, subtask_labels, min_sim)
        
        # 可训练阈值约束
        self.threshold = torch.clamp(self.threshold_param, 0.0 + EPS, 1.0 - EPS)
        
        return total_loss, final_sim_dict, self.threshold, min_sim, min_sim_label, loss_threshold

# ===================== 5. 训练流程（新增Transformer层优化） =====================
def train_model(train_loader):
    cross_trans = CrossTransformer()
    sim_enhancer = SimEnhancer()
    
    # 固定主干
    for param in cross_trans.text_encoder.parameters():
        param.requires_grad = False
    for param in cross_trans.image_encoder.parameters():
        param.requires_grad = False
    
    # 优化器：新增融合层参数，阈值参数单独设学习率
    optimizer = torch.optim.AdamW([
        {'params': cross_trans.trans_layer_img_text.parameters()},
        {'params': cross_trans.trans_layer_img_ime.parameters()},
        {'params': cross_trans.trans_layer_text_te.parameters()},
        # 新增：融合层和最终层参数
        {'params': cross_trans.trans_layer_img_ie_fusion.parameters()},
        {'params': cross_trans.trans_layer_img_ie_final.parameters()},
        {'params': cross_trans.text_proj.parameters()},
        {'params': cross_trans.image_proj.parameters()},
        {'params': sim_enhancer.score_head_t.parameters()},
        {'params': sim_enhancer.score_head_i.parameters()},
        {'params': sim_enhancer.score_head_ti.parameters()},
        {'params': sim_enhancer.score_head_f.parameters()},
        {'params': sim_enhancer.threshold_param, 'lr': 1e-4}
    ], lr=LR, weight_decay=0.001)
    
    # 梯度裁剪
    grad_clip = torch.nn.utils.clip_grad_norm_
    
    # 训练循环
    cross_trans.train()
    sim_enhancer.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        total_loss_threshold = 0.0
        
        for batch in train_loader:
            optimizer.zero_grad()
            trans_output = cross_trans(batch)
            true_labels = batch["label"].to(device)
            
            subtask_labels = {
                "l_t": batch["l_t"],
                "l_i": batch["l_i"],
                "l_ti": batch["l_ti"],
                "l_f": batch["l_f"]
            }
            
            # 前向传播
            loss, final_sim_dict, threshold, min_sim, min_sim_label, loss_threshold = sim_enhancer(trans_output, true_labels, subtask_labels)
            
            loss.backward()
            # 梯度裁剪
            grad_clip(cross_trans.parameters(), max_norm=0.1)
            grad_clip(sim_enhancer.parameters(), max_norm=0.1)
            optimizer.step()
            
            total_loss += loss.item()
            total_loss_threshold += loss_threshold.item()
        
        # 计算平均损失
        avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        avg_loss_threshold = total_loss_threshold / len(train_loader) if len(train_loader) > 0 else 0.0
        current_threshold = sim_enhancer.threshold.item()
        
        # 打印时处理NaN
        avg_loss = avg_loss if not np.isnan(avg_loss) else 0.0
        avg_loss_threshold = avg_loss_threshold if not np.isnan(avg_loss_threshold) else 0.0
        print(f"Epoch [{epoch+1}/{EPOCHS}], Avg Loss: {avg_loss:.4f}, Threshold Loss: {avg_loss_threshold:.4f}, Current Threshold: {current_threshold:.4f}")
    
    final_threshold = sim_enhancer.threshold.item()
    print(f"\n训练完成！最终适配任务的阈值：{final_threshold:.4f}")
    
    return cross_trans, sim_enhancer, final_threshold

# ===================== 6. 测试流程（适配新增的融合相似度） =====================
def test_model(test_loader, cross_trans, sim_enhancer, final_threshold):
    cross_trans.eval()
    sim_enhancer.eval()
    correct = 0
    total = 0
    
    print("\n" + "="*80)
    print(f"测试结果（训练阈值{final_threshold:.4f}+最小相似度规则+四任务损失）：")
    print("="*80)
    
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            trans_output = cross_trans(batch)
            true_labels = batch["label"].to(device)
            
            subtask_labels = {
                "l_t": batch["l_t"],
                "l_i": batch["l_i"],
                "l_ti": batch["l_ti"],
                "l_f": batch["l_f"]
            }
            
            _, final_sim_dict, _, min_sim, min_sim_label, _ = sim_enhancer(trans_output, true_labels, subtask_labels)
            
            # 提取所有相似度（含融合后的）
            sim_img_text = final_sim_dict["image-text"].item()
            sim_img_ime = final_sim_dict["image-image_e"].item()
            sim_text_te = final_sim_dict["text-text_e"].item()
            sim_img_ie_fused = final_sim_dict["image-image_e_fused"].item()  # 新增融合相似度
            true_label = true_labels.item()
            
            sim_dict = {
                "image-text": sim_img_text,
                "text-text_e": sim_text_te,
                "image-image_e": sim_img_ime,
                "image-image_e_fused": sim_img_ie_fused
            }
            min_sim = min(sim_dict.values())
            min_sim_type = min(sim_dict, key=sim_dict.get)
            min_sim_label = SIM_TO_LABEL[min_sim_type]
            
            if min_sim > final_threshold:
                final_pred = 0
            else:
                final_pred = min_sim_label
            
            print(f"\n测试样本 {idx+1}：")
            print(f"  图文相似度：{sim_img_text:.4f}")
            print(f"  文本证据相似度：{sim_text_te:.4f}")
            print(f"  图像证据相似度（原始）：{sim_img_ime:.4f}")
            print(f"  图像证据相似度（融合iet）：{sim_img_ie_fused:.4f}")  # 新增打印
            print(f"  最小相似度：{min_sim:.4f} (训练阈值{final_threshold:.4f})，对应类型：{min_sim_type}")
            print(f"  规则预测标签：{LABEL_REVERSE[final_pred]}")
            print(f"  真实标签：{LABEL_REVERSE[true_label]}")
            print(f"  预测结果：{'正确' if final_pred == true_label else '错误'}")
            
            total += 1
            correct += (final_pred == true_label)
    
    accuracy = correct / total if total > 0 else 0.0
    print("\n" + "="*80)
    print(f"测试集准确率：{accuracy:.2%}")
    print(f"最终适配任务的阈值：{final_threshold:.4f}")
    print("="*80)

# ===================== 7. 主函数 =====================
if __name__ == "__main__":
    train_loader, test_loader = build_dataset()
    print("开始训练（新增iet融合逻辑）...")
    cross_trans, sim_enhancer, final_threshold = train_model(train_loader)
    print("\n开始测试...")
    test_model(test_loader, cross_trans, sim_enhancer, final_threshold)