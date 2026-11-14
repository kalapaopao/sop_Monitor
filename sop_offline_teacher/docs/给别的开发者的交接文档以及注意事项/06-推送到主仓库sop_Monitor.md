# 06 · 推送到主仓库 `sop_Monitor`（含"别覆盖别人"的注意事项）

> 读者：接手代码的开发者。**这一步做错会覆盖整个 NX 推理工程，请先读完再操作。**

## 一、两个仓库的关系

| 位置 | 内容 |
|---|---|
| `https://github.com/kalapaopao/sop_Monitor` **根目录** | C++/TensorRT 的 NX 推理工程（`src/core/main.cpp`、`cmake/`、`models/`、`docs/design/` 等 95 个文件） |
| 同一仓库的 **`sop_offline_teacher/` 子目录** | 本项目（Python 教师链路 + Vue 演示页） |

本项目**独立的本地仓库**在 `E:\study\SOP_train_model_resource\sop_offline_teacher`，
它的提交历史被并入主仓库 `feat/roi-action` 分支（作为一条父提交保留）。

## 二、⛔ 绝对不要做的事

```powershell
# ❌ 会把远端 95 个文件的 NX 工程全部删掉
git push --force origin master:feat/roi-action

# ❌ 本地 master 与远端历史无关，直接 push 会被拒；用 force 就是上面那种后果
git push origin master
```

原因：本地仓库是 `git init` 出来的独立历史，与远端**完全无关**（unrelated histories）。
两者没有共同祖先，任何"直接推"都只能靠 force 覆盖。

## 三、正确做法：以子目录方式并入（保留双方历史）

在**临时克隆**里操作，不要动你自己的本地仓库目录：

```powershell
$env:GIT_TERMINAL_PROMPT='0'
$tmp = Join-Path $env:TEMP 'sopmon_push'
git clone --branch feat/roi-action --single-branch https://github.com/kalapaopao/sop_Monitor.git $tmp
cd $tmp

# 把本地项目挂成 remote，取它的历史
git remote add local 'E:/study/SOP_train_model_resource/sop_offline_teacher'
git fetch local master

# 1) 记录父提交（这样双方历史都保留），但先不改树
git merge -s ours --no-commit --allow-unrelated-histories local/master

# 2) 把对方整棵树放到子目录下
git read-tree --prefix=sop_offline_teacher/ -u local/master

# 3) 提交（消息写文件，避免 PowerShell 把 > 当重定向）
git commit -F merge_msg.txt

# 4) 推送到 feat/roi-action（fast-forward，不需要 force）
git push origin feat/roi-action
```

结果：`170 = 95（原有）+ 75（本项目）` 个文件，远端 `main` 不动。

## 四、必须核对的三个数字

```powershell
git ls-remote --heads https://github.com/kalapaopao/sop_Monitor.git
#   refs/heads/main          → 必须还是 b0f9696，没变
#   refs/heads/feat/roi-action → 变成新的提交

git ls-tree -r --name-only origin/feat/roi-action | Measure-Object   # 170
git ls-tree -r --name-only origin/feat/roi-action -- sop_offline_teacher | Measure-Object  # 75
```

> 注意：中文路径会被 git 转成八进制转义（`"\347\273..."`），
> 用 `^sop_offline_teacher/` 这种正则统计会把带引号的路径漏掉（实测少 14 个）。
> 统计请直接用 `-- <path>` 限定，或先 `git config core.quotepath false`。

## 五、已经推上去的内容（2026-11-15）

| 项 | 值 |
|---|---|
| 分支 | `feat/roi-action` |
| 合并提交 | `f298923`（父：`b0f9696` 远端 + `0e78b8c` 本地） |
| 本项目文件 | 75 个 / 约 5.7 MB |
| 远端 `main` | `b0f9696`，**未改动** |

## 六、入库前做过哪些精简

| 项 | 处理 |
|---|---|
| `web/public/data/local_pose_30s.json` | 删（未被任何代码引用） |
| 根 `src/style.css` | 删（孤儿文件，Vite 从不加载） |
| 30 张兜底抽帧 | 1920 宽压到 960 宽/q82：18.9 MB → 3.5 MB |
| 视频 / onnx 权重 / `.venv` / `node_modules` / `api_runs` / `.backup` | 均在 `.gitignore`，从未入库 |
| 凭据扫描 | 无硬编码；`README` 里只有占位符 `"你的 Key"` |
