使用APIserver+gradio demo部署的大模型，完成了主人公为阿夫的三百字小故事。

- 首先启动APIServer服务
- 然后使用Gradio为Client作为前端得到交互的大模型界面

```bash
# 启动服务
# ApiServer+Turbomind   api_server => AsyncEngine => TurboMind
lmdeploy serve api_server ./workspace \
	--server_name 0.0.0.0 \
	--server_port 23333 \
	--instance_num 64 \
	--tp 1


# Gradio+ApiServer。必须先开启 Server，此时 Gradio 为 Client
lmdeploy serve gradio http://0.0.0.0:23333 \
	--server_name 0.0.0.0 \
	--server_port 6006 \
	--restful_api True

# Gradio+Turbomind(local)
lmdeploy serve gradio ./workspace
```

![](assets/05%20LMDeploy大模型部署-作业/image-20240121160831943.png)

