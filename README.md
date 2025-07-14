# On-the-Fly Adaptive Distillation of Transformer to Dual-State Linear Attention for Long-Context LLM Serving

[ICML'25](https://icml.cc/virtual/2025/poster/43982)

## Abstract

Large language models (LLMs) excel at capturing global token dependencies via self-attention but face prohibitive compute and memory costs on lengthy inputs. While sub-quadratic methods (e.g., linear attention) can reduce these costs, they often degrade accuracy due to overemphasizing recent tokens. In this work, we first propose dual-state linear attention (DSLA), a novel design that maintains two specialized hidden states—one for preserving historical context and one for tracking recency—thereby mitigating the short-range bias typical of linear-attention architectures. To further balance efficiency and accuracy under dynamic workload conditions, we introduce DSLA-Serve, an online adaptive distillation framework that progressively replaces Transformer layers with DSLA layers at inference time, guided by a sensitivity-based layer ordering. DSLA-Serve uses a chained fine-tuning strategy to ensure that each newly converted DSLA layer remains consistent with previously replaced layers, preserving the overall quality. Extensive evaluations on commonsense reasoning, long-context QA, and text summarization demonstrate that DSLA-Serve yields 2.3× faster inference than Llama2-7B and 3.0× faster than the hybrid Zamba-7B, while retaining comparable performance across downstream tasks. Our ablation studies show that DSLA’s dual states capture both global and local dependencies, addressing the historical-token underrepresentation seen in prior linear attentions.

![Fig1](/assets/distill.drawio.svg)

### Reference

The distillation trainer is built on top of an early version of the [flame](https://github.com/fla-org/flame) framework.

### Citation

```
@inproceedings{rofly,
  title={On-the-Fly Adaptive Distillation of Transformer to Dual-State Linear Attention for Long-Context LLM Serving},
  author={Ro, Yeonju and Zhang, Zhenyu and Kundu, Souvik and Wang, Zhangyang and Akella, Aditya},
  booktitle={Forty-second International Conference on Machine Learning}
}
```
