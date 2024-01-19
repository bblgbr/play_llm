## 一、服务部署

大模型服务可以分为以下三个阶段

- 模型推理/服务。主要提供模型本身的推理，一般来说可以和具体业务解耦，专注模型推理本身性能的优化。可以以模块、API等多种方式提供。
- Client：可以理解为前端，与用户交互的地方。
- API Server。一般作为前端的后端，提供与产品和服务相关的数据和功能支持。

![](assets/05%20LMDeploy大模型部署/image-20240119164643389.png)
## 二、LMDeploy

### 1. 模型转换

使用 TurboMind 推理模型需要先将模型转化为 TurboMind 的格式，目前支持在线转换和离线转换两种形式。在线转换可以直接加载 Huggingface 模型，离线转换需需要先保存模型再加载。TurboMind 是一款关于 LLM 推理的高效推理引擎，基于英伟达的 FasterTransformer 研发而成。它的主要功能包括：LLaMa 结构模型的支持，persistent batch 推理模式和可扩展的 KV 缓存管理器。

#### 1.1 在线转换

- 在 huggingface.co 上面通过 lmdeploy 量化的模型
- huggingface.co 上面其他 LM 模型

```bash
# 需要能访问 Huggingface 的网络环境
# 使用量化版本
lmdeploy chat turbomind internlm/internlm-chat-20b-4bit --model-name internlm-chat-20b
# 使用huggface其他模型
lmdeploy chat turbomind Qwen/Qwen-7B-Chat --model-name qwen-7b
# 使用本地模型
lmdeploy chat turbomind /share/temp/model_repos/internlm-chat-7b/  --model-name internlm-chat-7b
```


#### 1.2 离线转换

离线转换需要在启动服务之前，将模型转为 lmdeploy TurboMind 的格式（FastTransformer）`weights` 和 `tokenizer` 目录分别放的是拆分后的参数和 Tokenizer。

```bash
lmdeploy convert internlm-chat-7b  /root/share/temp/model_repos/internlm-chat-7b/
```

### 2. TurboMind推理

#### 2.1 命令行本地对话

```bash
# Turbomind + Bash Local Chat
lmdeploy chat turbomind ./workspace
```
#### 2.2 API服务

”模型推理/服务“目前提供了 Turbomind 和 TritonServer 两种服务化方式。此时，Server 是 TurboMind 或 TritonServer，API Server 可以提供对外的 API 服务。我们推荐使用 TurboMind，TritonServer 使用方式详见《附录1》。

```bash
# 启动服务
# ApiServer+Turbomind   api_server => AsyncEngine => TurboMind
lmdeploy serve api_server ./workspace \
	--server_name 0.0.0.0 \
	--server_port 23333 \
	--instance_num 64 \
	--tp 1

# 新开一个窗口，执行下面的 Client 命令。
# ChatApiClient+ApiServer（注意是http协议，需要加http）
lmdeploy serve api_client http://localhost:23333
```


#### 2.3 网页Demo

```bash
# Gradio+ApiServer。必须先开启 Server，此时 Gradio 为 Client
lmdeploy serve gradio http://0.0.0.0:23333 \
	--server_name 0.0.0.0 \
	--server_port 6006 \
	--restful_api True

# Gradio+Turbomind(local)
lmdeploy serve gradio ./workspace
```

#### 2.4 python代码集成

```python
from lmdeploy import turbomind as tm

# load model
model_path = "/root/share/temp/model_repos/internlm-chat-7b/"
tm_model = tm.TurboMind.from_pretrained(model_path, model_name='internlm-chat-20b')
generator = tm_model.create_instance()

# process query
query = "你好啊兄嘚"
prompt = tm_model.model.get_prompt(query)
input_ids = tm_model.tokenizer.encode(prompt)

# inference
for outputs in generator.stream_infer(
        session_id=0,
        input_ids=[input_ids]):
    res, tokens = outputs[0]

response = tm_model.tokenizer.decode(res.tolist())
print(response)
```


## 三、模型量化

服务部署和量化是没有直接关联的，量化的最主要目的是降低显存占用，主要包括两方面的显存：模型参数和中间过程计算结果。前者对应W4A16 量化，后者对应KV Cache 量化。

KV Cache 可以节约大约 20% 的显存，精度不仅没有明显下降，相反在不少任务上还有一定的提升。

### 1. KV Cache量化

KV Cache 量化是将已经生成序列的 KV 变成 Int8，首先计算最大最小值，然后将其归一化后放缩到量化区间。可能得原因是，量化会导致一定的误差，有时候这种误差可能会减少模型对训练数据的拟合，从而提高泛化性能。量化可以被视为引入轻微噪声的正则化方法。或者，也有可能量化后的模型正好对某些数据集具有更好的性能。

### 2. W4A16量化

W4A16中的A是指Activation，保持FP16，只对参数进行 4bit 量化。

首先还是计算最大最小值，将其放缩到对应的区间，然后转成TurboMind格式

官方在 NVIDIA GeForce RTX 4090 上测试了 4-bit 的 Llama-2-7B-chat 和 Llama-2-13B-chat 模型的 token 生成速度。测试配置为 BatchSize = 1，prompt_tokens=1，completion_tokens=512。TurboMind 相比其他框架速度优势非常显著，比 mlc-llm 快了将近 30%