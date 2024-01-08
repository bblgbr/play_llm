本次课程主要是跑通三个Demo。由于课程的markdown说明已经非常清晰，并且提供了已经创建好的conda环境和模型权重，直接复制粘贴就可以跑，因此整体难度较小。本次课程总共完成了三个Demo的测试，详细过程参考下面教程，由于课程相对简单，于是简单总结了一下遇到的问题以及解决方法，并简单阅读了一下代码，学习了一些相关知识。.

- InternLM-Chat-7B 智能对话 Demo
- Lagent 智能体工具调用 Demo
- 浦语·灵笔图文理解创作 Demo

> [tutorial/helloworld/hello_world.md at main · InternLM/tutorial (github.com)](https://github.com/InternLM/tutorial/blob/main/helloworld/hello_world.md)

### 一、实验流程

#### 1. InternLM-Chat-7B 智能对话 Demo

- 配置环境：复制已经提供的conda环境
- 下载模型：直接复制提供的模型权重
- 下载代码：下载InterLM的代码`git clone https://gitee.com/internlm/InternLM.git`
- 终端运行：在终端和LLM交互
- web运行：通过可视化网站和LLM交互，相对于终端更加方便直观。

huggingface无法直接下载问题，可以通过换源解决。在终端中直接加入`export HF_ENDPOINT=https://hf-mirror.com`就可以使用了。更多的可以参考下面两个链接

```python
import os

# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 下载模型
os.system('huggingface-cli download --resume-download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --local-dir /root/data/model/sentence-transformer')
```

>[如何快速下载huggingface模型——全方法总结 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/663712983)
>[hf-mirror.com - Huggingface 镜像站](https://hf-mirror.com/)

#### 2. Lagent 智能体工具调用 Demo

Lagent是一个是一个轻量级、开源的基于大语言模型的智能体（agent）框架，支持用户快速地将一个大语言模型转变为多种类型的智能体，并提供了一些典型工具为大语言模型赋能。通过 Lagent 框架可以更好的发挥 InternLM 的全部性能。目前支持gpt3.5和InternLM的调用。

- 配置环境：复制已经提供的conda环境
- 下载模型：直接复制提供的模型权重
- 下载代码：下载lagent的代码`git clone https://gitee.com/internlm/lagent.git`
- web运行：通过可视化网站和LLM交互，相对于终端更加方便直观。

#### 3. 浦语·灵笔图文理解创作 Demo

浦语灵笔支持多模态和联网，可以给定要给标题让其创作图文并茂的文章，也可以输入图片后让其理解。

- 配置环境：复制已经提供的conda环境
- 下载模型：直接复制提供的模型权重
- 下载代码：下载XComposer的代码 `git clone https://gitee.com/internlm/InternLM-XComposer.git`
- web运行：通过可视化网站和LLM交互，相对于终端更加方便直观。
### 二、相关技术

#### streamlit

InternLM的web demo是使用streamlit去实现的，streamlit 是一个用于机器学习、数据可视化的 Python 框架，它能几行代码就构建出一个精美的在线 app 应用。他是纯python写的，不需要什么前端的知识，可以完美支持图表、图片等形式，值得学习。

streamlit可以轻松部署，通过官方提供的云台，连接github，非常适合小代码量的显示性工作部署。

```python
import streamlit as st

st.header('st.button')

if st.button('Say hello'):
     st.write('Why hello there')
else:
     st.write('Goodbye')
```

> [streamlit/streamlit: Streamlit — A faster way to build and share data apps. (github.com)](https://github.com/streamlit/streamlit)
> [Streamlit (30days.streamlit.app)](https://30days.streamlit.app/?challenge=Day1)