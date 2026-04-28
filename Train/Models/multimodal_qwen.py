import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

class MultimodalQwen(nn.Module):
    def __init__(
        self,
        encoder,
        qwen_model_path,
        llm_dim=None,
        llm_dtype=torch.float16,
        soft_prompt_len=4,
        use_gating=True,
        use_bridge_norm=True,
    ):
        super().__init__()
        self.encoder = encoder
        self.llm_dtype = llm_dtype
        self.soft_prompt_len = int(soft_prompt_len)
        self.use_gating = bool(use_gating)
        self.use_bridge_norm = bool(use_bridge_norm)

        print(f"正在加载基座模型: {qwen_model_path} (dtype={self.llm_dtype})...")
        self.llm = AutoModelForCausalLM.from_pretrained(
            qwen_model_path,
            dtype=self.llm_dtype,
            low_cpu_mem_usage=True,
        )

        try:
            actual_llm_dim = self.llm.get_input_embeddings().weight.shape[1]
            print(f"[INFO] 已加载基座模型: {qwen_model_path}, hidden_size={actual_llm_dim}")
        except Exception:
            actual_llm_dim = None
            print(f"[INFO] 已加载基座模型: {qwen_model_path} (hidden_size 读取失败)")
        if llm_dim is None:
            llm_dim = actual_llm_dim
        if llm_dim != actual_llm_dim:
            print(
                f"[WARN] 检测到 llm_dim={llm_dim} 与模型 hidden_size={actual_llm_dim} 不一致，"
                f"自动使用 {actual_llm_dim}。"
            )
            llm_dim = actual_llm_dim
        self.llm_dim = llm_dim

        self.pre_norm = nn.LayerNorm(self.encoder.output_dim) if self.use_bridge_norm else nn.Identity()

        self.projection = nn.Sequential(
            nn.Linear(self.encoder.output_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim * self.soft_prompt_len)
        )

        self.post_norm = nn.LayerNorm(llm_dim) if self.use_bridge_norm else nn.Identity()

        self.gate_proj = (
            nn.Linear(self.encoder.output_dim, llm_dim * self.soft_prompt_len)
            if self.use_gating
            else None
        )

        self.freeze_llm_parameters()

    def freeze_llm_parameters(self):
        for param in self.llm.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.projection.parameters():
            param.requires_grad = True
        for param in self.pre_norm.parameters():
            param.requires_grad = True
        for param in self.post_norm.parameters():
            param.requires_grad = True
        if self.gate_proj is not None:
            for param in self.gate_proj.parameters():
                param.requires_grad = True
        print("参数冻结完毕：只训练时序编码器与线性投影层。")

    def _encode_ts_soft_prompt(self, ts_data):
        ts_data = torch.nan_to_num(ts_data, nan=0.0, posinf=1e4, neginf=-1e4)
        ts_data = torch.clamp(ts_data, min=-1e4, max=1e4)

        ts_features = self.encoder(ts_data)
        ts_features_norm = self.pre_norm(ts_features)

        ts_proj = self.projection(ts_features_norm).view(-1, self.soft_prompt_len, self.llm_dim)
        ts_proj = self.post_norm(ts_proj)

        if self.use_gating:
            gate = torch.sigmoid(
                self.gate_proj(ts_features_norm).view(-1, self.soft_prompt_len, self.llm_dim)
            )
            return ts_proj * gate
        return ts_proj

    def _build_multimodal_inputs(self, ts_data, input_ids, attention_mask=None):
        ts_embeds = self._encode_ts_soft_prompt(ts_data)
        text_embeds = self.llm.get_input_embeddings()(input_ids)

        ts_embeds = ts_embeds.to(dtype=text_embeds.dtype)
        inputs_embeds = torch.cat([ts_embeds, text_embeds], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        prefix_mask = torch.ones(
            (attention_mask.size(0), self.soft_prompt_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        full_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        return inputs_embeds, full_attention_mask

    def forward(self, ts_data, input_ids, attention_mask=None, labels=None):
        inputs_embeds, full_attention_mask = self._build_multimodal_inputs(
            ts_data=ts_data,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        full_labels = None
        if labels is not None:
            prefix_labels = torch.full(
                (labels.size(0), self.soft_prompt_len),
                fill_value=-100,
                dtype=labels.dtype,
                device=labels.device,
            )
            full_labels = torch.cat([prefix_labels, labels], dim=1)

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels,
            return_dict=True,
        )
        return outputs

    @torch.no_grad()
    def generate(self, ts_data, input_ids, attention_mask=None, **generate_kwargs):
        inputs_embeds, full_attention_mask = self._build_multimodal_inputs(
            ts_data=ts_data,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            **generate_kwargs,
        )
