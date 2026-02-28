# Render 免费 H5 快速上线

目标：先让别人能在手机上直接用（不用安装，不用你电脑一直开着）。

## 1. 准备代码

1. 把项目推到 GitHub（私有仓库也可以）。
2. 确认 `.env` 没提交（当前已忽略）。

## 2. 在 Render 创建服务

1. 打开 Render 控制台，点 `New` -> `Blueprint`。
2. 选择你的 GitHub 仓库。
3. Render 会自动读取仓库里的 `render.yaml`。
4. 创建服务并等待部署。

## 3. 配置环境变量（必须）

在 Render 服务页 `Environment` 里填：

1. `LLM_PROVIDER=zhipuai`
2. `ZHIPU_API_KEY=你的密钥`

如果你用 OpenAI：

1. `LLM_PROVIDER=openai`
2. `OPENAI_API_KEY=你的密钥`

## 4. 验证上线

1. 打开 `https://<你的服务名>.onrender.com/health`
2. 打开 `https://<你的服务名>.onrender.com/`

两个都能打开就算上线成功。

## 5. 重要：首次部署知识库是空的

仓库默认忽略了：

1. `data/vector_store/`
2. `data/rules/*.pdf`

所以 Render 不会自动带上你本地规则。  
上线后请通过上传接口导入规则文件：

1. 打开 `https://<你的服务名>.onrender.com/docs`
2. 找到 `POST /api/documents/upload`
3. 逐个上传 PDF/TXT/MD，等待状态变成 `ready`
4. 注意：免费实例重启/重部署后，已上传文件可能丢失，需要重新导入

## 6. 给手机用户使用

把这个地址发给用户即可：

`https://<你的服务名>.onrender.com/`

微信内置浏览器可直接打开。

## 7. 你最关心的问题

Render 部署后不依赖你的本机在线，电脑可以关机。  
免费实例会休眠，首次访问可能会慢一些（冷启动）。
