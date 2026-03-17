#!/usr/bin/env python3
"""
LoRA微调脚本 - 基于NotaGen官方微调脚本改造
支持对NotaGen模型进行LoRA高效微调
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
import argparse
from typing import List, Dict

# LoRA相关依赖
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    print("警告: 未安装peft库，请先安装: pip install peft")
    PEFT_AVAILABLE = False

# ==================== 配置类 ====================
class LoRAFineTuneConfig:
    """LoRA微调配置"""
    
    # 数据配置
    DATA_TRAIN_INDEX_PATH = "../data/schubert_augmented_train.jsonl"
    DATA_EVAL_INDEX_PATH = "../data/schubert_augmented_eval.jsonl"
    
    # 模型架构配置 (对应NotaGen-large)
    PATCH_STREAM = True
    PATCH_SIZE = 16
    PATCH_LENGTH = 1024
    CHAR_NUM_LAYERS = 6
    PATCH_NUM_LAYERS = 20
    HIDDEN_SIZE = 1280
    
    # LoRA配置
    LORA_RANK = 16              # LoRA秩，控制低秩分解的维度
    LORA_ALPHA = 32             # LoRA缩放因子，通常为rank的2倍
    LORA_DROPOUT = 0.05          # LoRA层的dropout
    LORA_TARGET_MODULES = [      # 需要应用LoRA的目标模块
        "patch_attn.q_proj",     # patch-level decoder的query投影
        "patch_attn.k_proj",     # patch-level decoder的key投影
        "patch_attn.v_proj",     # patch-level decoder的value投影
        "char_attn.q_proj",      # character-level decoder的query投影
        "char_attn.k_proj",      # character-level decoder的key投影
        "char_attn.v_proj",      # character-level decoder的value投影
    ]
    
    # 训练配置
    BATCH_SIZE = 1               # 根据显存调整，LoRA可以用更大的batch
    LEARNING_RATE = 1e-4         # LoRA训练通常需要更高的学习率
    NUM_EPOCHS = 10              # 小数据集建议减少epoch数
    ACCUMULATION_STEPS = 1        # 梯度累积步数
    WARMUP_STEPS = 100            # 学习率warmup步数
    WEIGHT_DECAY = 0.01           # 权重衰减
    
    # 保存配置
    PRETRAINED_PATH = "../pretrain/weights_notagen_pretrain_p_size_16_p_length_1024_p_layers_20_c_layers_6_h_size_1280_lr_0.0001_batch_4.pth"
    EXP_TAG = "lora_schubert"
    SAVE_DIR = "./lora_checkpoints"
    
    # 系统配置
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MIXED_PRECISION = True        # 是否使用混合精度训练
    LOG_STEPS = 10                # 日志打印间隔
    EVAL_STEPS = 100              # 评估间隔
    
    @property
    def model_name(self):
        return f"weights_notagen_{self.EXP_TAG}_lora_r{self.LORA_RANK}_lr{self.LEARNING_RATE}_batch{self.BATCH_SIZE}"

# ==================== 数据集类 ====================
class NotagenDataset(Dataset):
    """NotaGen数据集加载器"""
    
    def __init__(self, data_path: str, config: LoRAFineTuneConfig):
        self.config = config
        self.data = self._load_data(data_path)
    
    def _load_data(self, data_path: str) -> List[Dict]:
        """加载数据"""
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        # 这里需要根据实际的NotaGen数据格式进行处理
        # 实际使用时需要根据数据集格式调整
        return {
            "abc_text": item.get("abc_text", ""),
            "prompt": item.get("prompt", ""),
            # 添加其他需要的字段
        }

# ==================== 模型定义 ====================
class NotagenDualDecoder(nn.Module):
    """
    简化的NotaGen模型结构 (基于Tunesformer架构)
    实际使用时需要替换为真实的NotaGen模型定义
    """
    
    def __init__(self, config: LoRAFineTuneConfig):
        super().__init__()
        self.config = config
        
        # Patch-level decoder
        self.patch_embedding = nn.Linear(config.PATCH_SIZE * 128, config.HIDDEN_SIZE)
        self.patch_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=config.HIDDEN_SIZE,
                nhead=config.HIDDEN_SIZE // 64,  # 计算head数量
                dim_feedforward=config.HIDDEN_SIZE * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=config.PATCH_NUM_LAYERS
        )
        
        # Character-level decoder
        self.char_embedding = nn.Embedding(128, config.HIDDEN_SIZE)  # ASCII字符
        self.char_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=config.HIDDEN_SIZE,
                nhead=config.HIDDEN_SIZE // 64,
                dim_feedforward=config.HIDDEN_SIZE * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=config.CHAR_NUM_LAYERS
        )
        
        self.output_proj = nn.Linear(config.HIDDEN_SIZE, 128)
        
        # 为LoRA识别标记各注意力层
        self._mark_attention_layers()
    
    def _mark_attention_layers(self):
        """为注意力层添加标记，方便LoRA配置"""
        # Patch-level attention
        for i, layer in enumerate(self.patch_decoder.layers):
            layer.patch_attn = layer.multihead_attn
            layer.multihead_attn.q_proj = layer.multihead_attn.in_proj_weight
            layer.multihead_attn.k_proj = layer.multihead_attn.in_proj_weight
            layer.multihead_attn.v_proj = layer.multihead_attn.in_proj_weight
        
        # Character-level attention  
        for i, layer in enumerate(self.char_decoder.layers):
            layer.char_attn = layer.multihead_attn
            layer.multihead_attn.q_proj = layer.multihead_attn.in_proj_weight
            layer.multihead_attn.k_proj = layer.multihead_attn.in_proj_weight
            layer.multihead_attn.v_proj = layer.multihead_attn.in_proj_weight
    
    def forward(self, patch_input, char_input):
        """
        前向传播
        patch_input: patch级别输入 [batch, patch_len, patch_dim]
        char_input: 字符级别输入 [batch, char_len]
        """
        # Patch-level processing
        patch_emb = self.patch_embedding(patch_input)
        patch_output = self.patch_decoder(patch_emb, patch_emb)
        
        # Character-level processing
        char_emb = self.char_embedding(char_input)
        char_output = self.char_decoder(char_emb, patch_output)
        
        # Output projection
        logits = self.output_proj(char_output)
        
        return logits

# ==================== 训练器类 ====================
class LoRATrainer:
    """LoRA微调训练器"""
    
    def __init__(self, config: LoRAFineTuneConfig):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # 创建保存目录
        os.makedirs(config.SAVE_DIR, exist_ok=True)
        
        # 初始化模型
        self.model = self._load_or_create_model()
        self.model = self._apply_lora()
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # 学习率调度器
        self.scheduler = self._create_scheduler()
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler() if config.MIXED_PRECISION else None
        
        # 数据加载器
        self.train_loader = self._create_dataloader(config.DATA_TRAIN_INDEX_PATH, shuffle=True)
        self.eval_loader = self._create_dataloader(config.DATA_EVAL_INDEX_PATH, shuffle=False)
        
        # 训练状态
        self.global_step = 0
        self.best_loss = float('inf')
    
    def _load_or_create_model(self) -> nn.Module:
        """加载预训练模型或创建新模型"""
        model = NotagenDualDecoder(self.config)
        model.to(self.device)
        
        # 加载预训练权重
        if os.path.exists(self.config.PRETRAINED_PATH):
            print(f"加载预训练权重: {self.config.PRETRAINED_PATH}")
            state_dict = torch.load(self.config.PRETRAINED_PATH, map_location=self.device)
            
            # 处理可能的key不匹配问题
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print(f"成功加载 {len(pretrained_dict)}/{len(model_dict)} 个参数")
        else:
            print("未找到预训练权重，使用随机初始化")
        
        return model
    
    def _apply_lora(self) -> nn.Module:
        """应用LoRA到模型"""
        if not PEFT_AVAILABLE:
            print("警告: PEFT库不可用，跳过LoRA应用")
            return self.model
        
        print("应用LoRA配置...")
        
        # 创建LoRA配置
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,  # 因果语言建模
            r=self.config.LORA_RANK,
            lora_alpha=self.config.LORA_ALPHA,
            lora_dropout=self.config.LORA_DROPOUT,
            target_modules=self.config.LORA_TARGET_MODULES,
            inference_mode=False,
        )
        
        # 应用LoRA
        self.model = get_peft_model(self.model, lora_config)
        
        # 打印可训练参数信息
        self.model.print_trainable_parameters()
        
        return self.model
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
        
        # Warmup + Cosine Annealing
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=self.config.WARMUP_STEPS
        )
        main_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.NUM_EPOCHS * len(self.train_loader)
        )
        
        from torch.optim.lr_scheduler import SequentialLR
        scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[self.config.WARMUP_STEPS]
        )
        
        return scheduler
    
    def _create_dataloader(self, data_path: str, shuffle: bool) -> DataLoader:
        """创建数据加载器"""
        dataset = NotagenDataset(data_path, self.config)
        return DataLoader(
            dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True
        )
    
    def train_epoch(self, epoch: int):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.NUM_EPOCHS}")
        
        for batch_idx, batch in enumerate(pbar):
            # 数据预处理
            batch = self._preprocess_batch(batch)
            
            # 前向传播
            if self.config.MIXED_PRECISION:
                with torch.cuda.amp.autocast():
                    logits = self.model(batch['patch_input'], batch['char_input'])
                    loss = self.criterion(
                        logits.view(-1, logits.size(-1)),
                        batch['target'].view(-1)
                    )
            else:
                logits = self.model(batch['patch_input'], batch['char_input'])
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    batch['target'].view(-1)
                )
            
            # 反向传播
            self.optimizer.zero_grad()
            
            if self.config.MIXED_PRECISION:
                self.scaler.scale(loss).backward()
                if (batch_idx + 1) % self.config.ACCUMULATION_STEPS == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                loss.backward()
                if (batch_idx + 1) % self.config.ACCUMULATION_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
            
            self.scheduler.step()
            
            # 记录损失
            total_loss += loss.item()
            self.global_step += 1
            
            # 更新进度条
            if batch_idx % self.config.LOG_STEPS == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })
            
            # 定期评估
            if self.global_step % self.config.EVAL_STEPS == 0:
                eval_loss = self.evaluate()
                print(f"\nStep {self.global_step}: Eval Loss = {eval_loss:.4f}")
                
                # 保存最佳模型
                if eval_loss < self.best_loss:
                    self.best_loss = eval_loss
                    self._save_checkpoint(f"best_model_step{self.global_step}")
                
                self.model.train()
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def _preprocess_batch(self, batch: Dict) -> Dict:
        """数据预处理"""
        # 这里需要根据实际的NotaGen数据处理流程进行调整
        batch_size = len(batch['abc_text'])
        
        # 示例：创建dummy数据 (实际使用时替换为真实的数据处理逻辑)
        patch_input = torch.randn(batch_size, self.config.PATCH_LENGTH, self.config.PATCH_SIZE * 128).to(self.device)
        char_input = torch.randint(0, 128, (batch_size, 512)).to(self.device)
        target = torch.randint(0, 128, (batch_size, 512)).to(self.device)
        
        return {
            'patch_input': patch_input,
            'char_input': char_input,
            'target': target
        }
    
    @torch.no_grad()
    def evaluate(self):
        """评估模型"""
        self.model.eval()
        total_loss = 0.0
        
        for batch in tqdm(self.eval_loader, desc="Evaluating", leave=False):
            batch = self._preprocess_batch(batch)
            
            logits = self.model(batch['patch_input'], batch['char_input'])
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                batch['target'].view(-1)
            )
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.eval_loader)
        return avg_loss
    
    def _save_checkpoint(self, filename: str):
        """保存检查点"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'config': self.config.__dict__
        }
        
        save_path = os.path.join(self.config.SAVE_DIR, filename)
        torch.save(checkpoint, save_path)
        print(f"模型已保存到: {save_path}")
    
    def train(self):
        """完整训练流程"""
        print(f"开始LoRA微调，配置:")
        print(f"  - 模型: NotaGen-large (516M)")
        print(f"  - LoRA rank: {self.config.LORA_RANK}")
        print(f"  - LoRA alpha: {self.config.LORA_ALPHA}")
        print(f"  - 学习率: {self.config.LEARNING_RATE}")
        print(f"  - Batch size: {self.config.BATCH_SIZE}")
        print(f"  - Epochs: {self.config.NUM_EPOCHS}")
        print(f"  - 设备: {self.device}")
        
        for epoch in range(self.config.NUM_EPOCHS):
            train_loss = self.train_epoch(epoch)
            print(f"\nEpoch {epoch+1} 完成. 平均损失: {train_loss:.4f}")
            
            # 每个epoch结束后评估
            eval_loss = self.evaluate()
            print(f"验证损失: {eval_loss:.4f}")
            
            # 保存epoch检查点
            self._save_checkpoint(f"checkpoint_epoch{epoch+1}")
        
        print(f"\n训练完成! 最佳验证损失: {self.best_loss:.4f}")

# ==================== 主函数 ====================
def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="NotaGen LoRA微调脚本")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--rank", type=int, default=16, help="LoRA秩")
    parser.add_argument("--alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--pretrained_path", type=str, default=None, help="预训练权重路径")
    parser.add_argument("--data_path", type=str, default=None, help="训练数据路径")
    
    args = parser.parse_args()
    
    # 创建配置
    config = LoRAFineTuneConfig()
    
    # 覆盖命令行参数
    if args.rank:
        config.LORA_RANK = args.rank
    if args.alpha:
        config.LORA_ALPHA = args.alpha
    if args.lr:
        config.LEARNING_RATE = args.lr
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.epochs:
        config.NUM_EPOCHS = args.epochs
    if args.pretrained_path:
        config.PRETRAINED_PATH = args.pretrained_path
    if args.data_path:
        config.DATA_TRAIN_INDEX_PATH = args.data_path
    
    # 检查依赖
    if not PEFT_AVAILABLE:
        print("错误: 请先安装PEFT库: pip install peft")
        return
    
    # 创建训练器并开始训练
    trainer = LoRATrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
