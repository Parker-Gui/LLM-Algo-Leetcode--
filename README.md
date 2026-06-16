# LLM-Algo-Leetcode--
# LLM Algo LeetCode 学习笔记

这是我学习 [datawhalechina/llm-algo-leetcode](https://github.com/datawhalechina/llm-algo-leetcode) 项目时整理的个人学习笔记与代码练习记录。

该项目是一个面向大模型入门到进阶的算法刷题教程，覆盖 Python、PyTorch、Transformer、推理优化、显存管理与 CUDA/Triton 实战。它不是只讲概念，而是把每个知识点设计成可运行、可验证、可回顾的练习，帮助学习者从“会看”逐步过渡到“会写、会调、会优化”。

本仓库主要用于记录我的学习过程、关键知识点理解、代码实现思路以及训练过程中的疑问和总结。

---

## 学习目标

通过本项目的学习，我希望系统掌握 LLM 算法工程中常见的底层基础能力，包括：

- 熟悉 PyTorch 中常用张量操作
- 理解 LLM 中常见模块的数学原理与代码实现
- 掌握 Transformer 相关组件的实现细节
- 提升阅读和复现大模型源码的能力
- 建立“概念理解 -> 手写实现 -> 测试验证 -> 笔记复盘”的学习闭环

---

## 当前进度

### Task 1: Chapter 2

当前已完成：

- `00 PyTorch Warmup`
- `01 RMSNorm`

本阶段主要学习内容包括：

- PyTorch 张量维度变换
- `view`、`reshape`、`permute`、`transpose` 等基础操作
- `einops` 的基本使用
- Embedding Layer 的本质
- 自定义 `torch.autograd.Function`
- Linear + ReLU 的前向与反向传播推导
- RMSNorm 的数学公式与 PyTorch 实现

---
