# GitHub 新账号迁移记录 · 2026-11-03

**当前状态：原项目主线与本地保留分支已推送新仓库，提交编号均已核对一致。新仓库设为默认同步目标。**

## 起因与目标

- 用户要求尝试拉取最新，并将项目迁至本人新账号。
- 旧地址：https://github.com/1521183894w-ux/mythic-planet-lab
- 新地址：https://github.com/1521183894w-bit/mythic-planet-lab
- 用户本人创建了新的公开空仓库，并本人完成 Git Credential Manager 浏览器授权。公开属性沿用 2026-10-21 的用户决定。

## 已实际核对

- 匿名 fetch 旧仓库成功；本地 main 与旧 origin/main 均为 54f468bacb81b3d3346405d5ba8b10958b1c2414，左右差异 0/0。
- 旧远程只列出 main，无标签；主线包含 123 个提交。
- 本地另外保留 codex/u36-s10-local-4aa83e3，含主线之外的 1 个提交 4aa83e315fdc4c5c07f8a3e2f6d66d230b12d3b4，已原样推送新仓库同名分支，未合并到 main。
- 原已跟踪文件在开工时无改动；.codex/agents/world-reviewer.toml 与 scripts/design_template.py 两项原有未跟踪内容保留，未加入公开提交。
- git fsck --connectivity-only --no-dangling 退出码 0；git bundle verify 确认备份包含完整历史。

## D 盘本地恢复材料

目录：D:\mythic-planet-lab\.cache\repository-migration\2026-11-03\（已被 .gitignore 忽略，不上传）。

- before-fetch.bundle：全部本地引用的 Git 历史，478710716 字节（约 456.5 MiB）。
- before-fetch.bundle.sha256：SHA-256 校验记录。
- refs-before.txt、status-before.txt：拉取前引用和工作区状态。
- untracked/：两项未跟踪内容；已实际核对副本与原件字节相同。
- git-config-before.txt：变更前的本地 Git 配置；仅供本机恢复，不公开。
- tracked-before/：本次修改的原文件。

恢复时应先在 D 盘另建目录，从 bundle 克隆并核对分支；不要覆盖现有工作区。生成图、环境、大型科学数据仍留在原来的 D 盘位置，没有加入此次上传。

## 同步配置

- origin 指向新账号仓库，main 跟踪新的 origin/main。
- 旧地址保留为 legacy-origin，其 push URL 为 disabled://legacy-origin；不会向旧账号推送。
- 凭据用户名固定为 1521183894w-bit，仅修改本仓库配置。
- 今后提交使用 351335107+1521183894w-bit@users.noreply.github.com；账号 ID 351335107 来自 GitHub 公开用户 API。原有提交作者、邮箱及哈希不改写。
- AGENTS.md、CLAUDE.md、使用说明.md 及两个研究下载脚本的现行项目 URL 已更新；历史环境审计记录保留原样。
- 另一台已有电脑的切换命令见 使用说明.md §2.2；先切换地址、本人登录，再检查差异并快进拉取，不强制覆盖。

## 传输与核验记录

- 完整主线大包约 456.47 MiB；大包及 HTTP 代理突发上传曾连接重置，未把失败命令末尾的 Everything up-to-date 误判为成功。
- 最终按原提交顺序分批快进上传，以远程实际 SHA 核对；临时在本机将既有代理入口限速至约 1 MiB/s，直接转发原加密连接，不读取 TLS 正文，不改写原提交或文件。上传日志只保存在本机忽略目录，任务结束停止临时转发器。
- 本仓库设置既有 127.0.0.1:7890 代理用于日常拉取和小提交；未修改系统代理。大上传若重置，可参照本机交接记录由开发者重建临时限速连接。
- 原主线 123 个提交末端仍为 54f468b；本次另加迁移说明提交。保留分支末端为 4aa83e3。最终最新提交以 git log 和新 origin/main 为准。

## 范围与限制

这是向独立新仓库复制 Git 内容。GitHub 站点上的 Issue、PR、星标和仓库权限等元数据不随普通 Git 推送迁移；未执行旧账号的所有权转移，也未向旧账号上传。本次不运行世界生成、地学指标或标准出图，阶段 1 既有未通过结论不变。

参考：[GitHub 仓库复制说明](https://docs.github.com/en/repositories/creating-and-managing-repositories/duplicating-a-repository)、[GitHub 所有权转移说明](https://docs.github.com/en/repositories/creating-and-managing-repositories/transferring-a-repository)。
