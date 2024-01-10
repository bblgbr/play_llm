
### 一、基础理论

LLM的局限性：
- 知识时效受限
- 专业能力有限
- 定制化成本高

解决方法
- RAG：低成本、可以实时更新、受基座模型影响大、单次回答知识有限，RAG的可落地性更高
- Finetune：可个性化微调、知识覆盖面广、成本高昂、无法实时更新

检索增强生成（Retrieval Augmented Generation），简称 RAG，已经成为当前最火热的LLM应用方案。通过检索获取相关的知识并将其融入Prompt，让大模型能够参考相应的知识从而给出合理回答。因此，可以将RAG的核心理解为“检索+生成”，前者主要是利用向量数据库的高效存储和检索能力，召回目标知识；后者则是利用大模型和Prompt工程，将召回的知识合理利用，生成目标答案。

大模型的开发流程：
在大模型开发中，主要尝试用 Prompt Engineering 来替代子模型的训练调优，通过 Prompt 链路组合来实现业务逻辑，用一个通用大模型 + 若干业务 Prompt 来解决任务，从而将传统的模型训练调优转变成了更简单、轻松、低成本的 Prompt 设计调优。大模型的开发流程如下：

- 确定目标并设计功能
- 确定技术框架并搭建整体架构
- 搭建数据库
- Prompt Engineering
- 验证迭代
- 前后端搭建
- 体验优化，部署上线

数据库搭建：
1. 收集和整理领域相关的文档：常用文档格式有 pdf、txt、doc 等，首先使用工具读取文本，通常使用 langchain 的文档加载器模块可以方便的将用户提供的文档加载进来，也可以使用一些 python 比较成熟的包进行读取。
2. 将文档词向量化：使用文本嵌入(Embeddings)对分割后的文档进行向量化，使语义相似的文本片段具有接近的向量表示。
3. 建立知识库索引：Langchain 集成了超过 30 个不同的向量存储库。可以选择 Chroma 向量库是因为它轻量级且数据存储在内存中，这使得它非常容易启动和开始使用。知识库内容经过 embedding 存入向量知识库，然后用户每一次提问也会经过 embedding，利用向量相关性算法（例如余弦算法）找到最匹配的几个知识库片段，将这些知识库片段作为上下文，与用户问题一起作为 prompt 提交给 LLM 回答。

### 二、实验流程

整个实验按照教程可以直接完成复现，相对简单

- 配置环境：复制已经提供的conda环境、配置、
- 下载模型：直接复制提供的模型权重
- 下载代码：下载InterLM的代码`git clone https://gitee.com/internlm/InternLM.git`
- 终端运行：在终端和LLM交互
- web运行：通过可视化网站和LLM交互，相对于终端更加方便直观。

### 三、相关技术

#### gradio


Gradio是一个简单直观的交互界面的SDK组件，支持多种输入输出格式。只需要在原有程序中新增几行代码，便可快速构建交互式的机器学习模型，将已有的算法模型以 Web 服务形式呈现给用户。

```python
import gradio as gr
import torch
import requests
from torchvision import transforms

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
# 加载预训练模型
model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True).eval()
# 获取标签
response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")

# 预测函数，输入图片，返回置信概率
def predict(inp):
  inp = transforms.ToTensor()(inp).unsqueeze(0)
  with torch.no_grad():
    prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
    confidences = {labels[i]: float(prediction[i]) for i in range(1000)}    
  return confidences
# 利用gradio制作前端web，输入为图片，调用预测函数，返回前三的预测结果
demo = gr.Interface(fn=predict, 
             inputs=gr.inputs.Image(type="pil"),
             outputs=gr.outputs.Label(num_top_classes=3),
             examples=[["cheetah.jpg"]],
             )
# 启动网页 
demo.launch()

```