# MAIB7030｜老师 Windows 电脑直播演示部署说明

## 部署目的

- 在老师 Windows 电脑上**只用浏览器打开网址**；不需要安装 Python，不需要把你的 Mac 带到教室。
- 真实运行 Jan–Apr 训练的两个**冻结**模型（HGB baseline 和 HGB + transition）；可以互动选择五月区域和小时，点击预测后再揭示实际值。
- 在线包只含 **Jan–May 的 34,550 条数据**。June / July 14,640 条保留测试记录**没有打包上传**。此数据裁剪不影响模型、预测和五月验证结果。
- 本包没有部署到互联网上，也没有创建真实 URL；你仍需通过自己的 GitHub + Streamlit Community Cloud 发布。

## 一｜先进行本地核验（推荐）

在终端进入解压后的文件夹，激活 Python 环境并安装 requirements，然后执行：

```
python -m pip install -r requirements.txt
python validate_demo.py
```

预期输出：`ACCEPTANCE CHECK: PASS`。核心验收：May HGB MAE≈22.7828、HGB+transition MAE≈22.6590；Zone 142 / 2025-05-01 22:00 预测 ≈472.11 / 478.34；实际 742。

## 二｜通过 GitHub 发布代码

1. 在 GitHub 创建一个新 repository，建议名称 `maib7030-taxi-demo`。
2. **建议先选 Private**，和队友确认哪些内容允许公开。部署后你可以决定是否使 App 对老师免登录可见。
3. 将本文件夹**内部所有文件和子目录**（`app.py`、`forecast_engine.py`、`validate_demo.py`、`requirements.txt`、`data/`、`models/`、`README_Deploy_CN.md` 等）上传至 repository 根目录；不要只上传 ZIP，也不要上传整个 .venv 文件夹。
4. 在 GitHub 网页确认 `models/manifest.json` 和 `data/taxi_demand_features.csv` 在相应文件夹里。
5. 不要在 GitHub 提交任何凭证、个人密钥、未授权私有资料或 June/July 数据集。

## 三｜通过 Streamlit Community Cloud 部署

1. 打开 https://share.streamlit.io/ 并使用 GitHub 登录。
2. 点击 Create app / Deploy app，选择自己的 repository、main branch、entrypoint `app.py`。
3. 打开 Advanced settings，**选择 Python 3.13**，以尽量匹配冻结模型的训练环境（Python 3.13.5、scikit-learn 1.8.0）。首次部署请不要使用 3.14。
4. 点击 Deploy，等待安装依赖、启动服务。
5. 获得形如 `https://<your-subdomain>.streamlit.app` 的真实网址（网址由平台创建，不能提前假定）。
6. 打开 App，运行 Zone 142 / May 1 / 22:00；确认 472.11/478.34，点击 Reveal Actual 确认 742，之后再选第二个区域测试。
7. 如果 App 来自私有 repo，可能默认要求 GitHub/邮箱登录。根据 App settings > Sharing 决定公开访问，或者邀请老师作为 viewer。**老师电脑只能浏览网站时，公开可访问的 App 更方便；公开前应获得团队同意。**

## 四｜11 月 26 日现场演示

- 尽可能**提前在老师的 Windows 浏览器中打开真实 URL，进行完整测试**。
- 现场可提前在演讲前打开网页并加载完模型，展示时刷新再操作。
- 不要直接打开 `localhost:8501`：它只指向运行应用的本机。老师 Windows 没有启动服务，就打不开你的 Mac localhost。
- 不要现场修改模型或数据；如果线上服务出现故障，使用事先录制的真实 Demo 流程作为**明确标注的备用材料**，不能当作现场正常运行。

## 五｜课堂讲解用例

- Zone 142 · 1 May 2025 · 22:00。模型先根据已完成历史小时数据作出两个预测；随后揭示 742 实际 recorded pickups。
- 对比模型输出及绝对误差，但不要把个别案例当作模型总体性能的证明。
- 说明一个关键局限：这是 historical replay，不是实时读取 NYC TLC API。

## 六｜遇到部署错误

- 提供 Streamlit 页面报错文字或 Manage app > Logs 内容。
- 如果报 pandas/NumPy/scikit-learn 加载模型的兼容性错误，先检查 cloud Python 3.13 与 requirements.txt 是否严格匹配。
- 不要通过放宽 `validate_demo.py` 断言来掩盖预测不一致。
