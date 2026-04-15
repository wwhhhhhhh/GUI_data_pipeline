# 260414_统一IP蓝区claude code配置

## 内容包含敏感信息，切勿外传

---

## 一、VPN 软件下载与配置

### 1. 下载 v2rayN

从 GitHub 下载 v2rayN 客户端：

```
https://github.com/2dust/v2rayn
```

### 2. 导入节点

打开 v2rayN 后，选择「从剪贴板导入分享链接」，将以下链接复制到剪贴板后导入：

```
vmess://eyJwcyI6IkpNUy0yMjc0OTdAYzM4czQucG9ydGFibGVzdWJtYXJpbmVzLmNvbToyMDMyNiIsInBvcnQiOiIyMDMyNiIsImlkIjoiOWNlNjNhYTQtOTVhOC00ZTFhLThlYmYtZmE5NGQzYWI4MDQwIiwiYWlkIjowLCJuZXQiOiJ0Y3AiLCJ0eXBlIjoibm9uZSIsInRscyI6Im5vbmUiLCJhZGQiOiJjMzhzNC5wb3J0YWJsZXN1Ym1hcmluZXMuY29tIn0
```

### 3. 开启全局模式

导入成功后，将路由模式切换为「V4-全局(Global)」。
访问以下网址确认国内/国外访问的所有 IP 均已变更为 `23.106.133.209`：

```
https://ip111.cn/
```

# 务必保证IP已改变后使用，使用过程中不要开关VPN

---

## 二、v2rayN 详细设置

右键点击系统托盘中的 v2rayN 图标，确认以下设置：

- 系统代理：选择「自动配置系统代理」
- 路由：V4-全局(Global)
- 配置项：[VMess] JMS-227497@c38s4.portablesubmarines.com

---

## 三、Windows 代理端口配置（Mac应该可以跳过这步）

在 v2rayN 中进入「设置 → 参数设置 → Core: 基础设置」，将「本地混合监听端口」设置为 `1080`（或者其他数字，下面的命令对应修改）。

### 配置 PowerShell / 终端代理

在 PowerShell 终端中输入以下命令，使终端流量也走代理，或者直接在系统设置-环境变量设置中配置 HTTP_PROXY 与 HTTPS_PROXY 两个环境变量：

```powershell
$env:HTTP_PROXY="http://127.0.0.1:1080/"
$env:HTTPS_PROXY="http://127.0.0.1:1080/"
```

---

## 四、Claude 账号信息与登录

### 账号信息

账号格式：Claude账号/邮箱 ---- 邮箱密码 ---- 接码令牌 ---- Claude SK

```
JmdavycliffhZ@birdlover.com
邮箱密码：65Z7ZK3jcO
接码令牌：a8734102d69c1cd723da4fec98cfcd89
Claude SK：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
```

### 登录步骤

1. 打开 Claude 官网：Claude.ai
2. 输入账号邮箱登录
3. 需要验证码时，使用免登录邮箱方式获取：
   - 复制上方的「接码令牌」
   - 在浏览器打开以下地址并粘贴令牌：
   ```
   https://b.171mail.com/#/home/code?type=claude&token=
   ```
4. 点击「获取」即可拿到验证链接，点击后完成登录。

然后即可在终端中运行 claude 命令开始使用。
