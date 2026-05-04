# PnP 外参矫正方案 — 架构设计

## 背景

现有 LiDAR-Camera 标定的 extrinsic 为默认值（单位矩阵），导致 3D box 在 2D 图像上的投影错位（"点云倒置"）。本方案通过**在 2D 图像上拖拽 3D box 角点 → solvePnP 反算外参**来实现手动精标定。

## 核心流程

```
用户选中3D box → 提取正面4角点的3D坐标 (P0~P3)
              → 用当前外参投影到2D
              → SVG 绘制4个带编号的拖拽 handle (标记 P0~P3)
              → 用户按编号依次拖拽到图像上实际位置
              → POST /api/solve_pnp  (points_3d[i] ↔ points_2d[i] 按序对应)
              → IPPE 闭式求解 → 更新 extrinsic
              → 重绘所有 box 投影
```

## 技术选型

| 项目 | 选择 | 理由 |
|------|------|------|
| PnP 求解器 | `cv2.SOLVEPNP_IPPE` | 正面4点共面，闭式求解，无需初值 |
| 求解位置 | 后端 Python OpenCV | 复用现成 cv2，代码量小 |
| 交互模式 | 先拖拽4点 → 点击求解 | 初版简单可靠 |
| 前端渲染 | SVG (复用现有 image.js) | 项目已有完整 SVG 投影管线 |

## 架构图

```
┌──────────────────────────────────────────────┐
│  前端 (Browser)                                │
│                                                │
│  calib_pnp.js  ← 面板控制器 + 角点状态管理     │
│  image.js      ← SVG 渲染 + 拖拽事件 (修改)    │
│  util.js       ← psr_to_xyz (不改)            │
│                                                │
│  用户拖拽 → fetch POST /api/solve_pnp          │
└──────────────────────┬───────────────────────┘
                       │ HTTP JSON
┌──────────────────────┴───────────────────────┐
│  后端 (CherryPy)                               │
│                                                │
│  main.py  → 新增 @expose solve_pnp()          │
│  calibpy/calib_pnp.py → IPPE 求解             │
│  data/*/calib/camera/*.json  ← extrinsic 更新 │
└──────────────────────────────────────────────┘
```

## 接口设计

### POST /api/solve_pnp

**请求：**
```json
{
  "scene": "scene-xxx",
  "camera": "front",
  "points_3d": [[x,y,z], [x,y,z], [x,y,z], [x,y,z]],
  "points_2d": [[u,v], [u,v], [u,v], [u,v]]
}
```

**响应：**
```json
{
  "extrinsic": [4x4 矩阵 flatten],
  "rvec": [3x1],
  "tvec": [3x1],
  "reprojection_error": 1.23,
  "success": true
}
```

## 角点序号对应规则

solvePnP 的正确性完全依赖于 **points_3d[i] ↔ points_2d[i] 一一对应**，必须在交互和 API 层面保证序号一致。

### 3D 角点编号 (固定)

基于 `util.js` 中 `psr_to_xyz()` 的顶点顺序：

```
正面 4 个角点 (x = +width/2, 共面 → IPPE)：

P0: ( x,  y, -z)  前左下  ← local_coord[0:4]
P1: ( x, -y, -z)  前右下  ← local_coord[4:8]
P2: ( x, -y,  z)  前右上  ← local_coord[8:12]
P3: ( x,  y,  z)  前左上  ← local_coord[12:16]

背面 4 个角点 (x = -width/2)：
P4: (-x,  y, -z)  后左下
P5: (-x, -y, -z)  后右下
P6: (-x, -y,  z)  后右上
P7: (-x,  y,  z)  后左上
```

### 2D 投影编号 (与 3D 一一对应)

| 序号 | 3D 坐标 | 2D 投影 handle | 说明 |
|------|---------|---------------|------|
| P0 | ( x,  y, -z) | handle_0 (u₀, v₀) | 前左下 → 图像左下方 |
| P1 | ( x, -y, -z) | handle_1 (u₁, v₁) | 前右下 → 图像右下方 |
| P2 | ( x, -y,  z) | handle_2 (u₂, v₂) | 前右上 → 图像右上方 |
| P3 | ( x,  y,  z) | handle_3 (u₃, v₃) | 前左上 → 图像左上方 |

### 交互约束

- **SVG 上必须始终显示 P0~P3 编号标签**（不可省略），以便用户区分
- **推荐按 P0→P1→P2→P3 顺序拖拽**（左下→右下→右上→左上），方便记忆
- API 请求中 `points_3d` 和 `points_2d` 的数组顺序严格按 P0~P3 排列
- 后端不做重排序，前端保证传入顺序正确

## 交互设计

### 模式 A（初版）：先拖拽 → 再求解

```
1. 点击「外参矫正」进入模式
2. SVG 上显示 4 个带编号的 ○ handle (P0~P3)
3. 用户按编号顺序逐个拖拽 ○ 到图像上实际位置
4. 全部放置后点击「计算 PnP」
5. 显示结果 + 重投影误差
6. 满意 → 保存；不满意 → 微调重算
```

### 模式 B（优化）：实时求解

```
1. 拖拽过程中实时 PnP（每次 mouse move 节流 100ms）
2. 所有角点跟随更新
3. 松手后最终求解
```

### UI 布局

```
┌─────────────────────────────────────────┐
│ [PnP 外参矫正]                          │
├────────────────┬────────────────────────┤
│                │ 角点 (已放置2/4)       │
│  ┌──○───●──┐  │ P0 ● (235, 412)       │
│  │P₃    P₂ │  │ P1 ● (356, 418)       │
│  │         │  │ P2 ○ (---, ---) 未放置 │
│  │P₀    P₁ │  │ P3 ○ (---, ---) 未放置 │
│  └──●───○──┘  │                        │
│                │ [📐 计算PnP] [⟳ 重置]│
│  SVG 上编号标记 │ [💾 保存]             │
│  P₀~P₃ 与右侧  │ 误差: -- px           │
│  列表一一对应   │                        │
├────────────────┴────────────────────────┤
│ ⚠ 必须按 P0→P1→P2→P3 顺序拖拽/放置，    │
│   确保 points_3d[i] ↔ points_2d[i] 对应 │
└─────────────────────────────────────────┘
```

## 约束条件

| 约束 | 说明 |
|------|------|
| box 不变形 | PnP 的 3D 点固定，只改变外参，box 刚性不变 |
| 共面求解 | IPPE 专为共面点设计，正面4角点满足条件 |
| 无需初值 | IPPE 闭式求解，不依赖初始外参 |
| 最小输入 | 4 个非共线共面点即可唯一求解 |

## 涉及的文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `public/js/calib_pnp.js` | 新建 | 面板控制器、角点状态、拖拽逻辑、PnP 请求 |
| `calibpy/calib_pnp.py` | 新建 | IPPE 求解封装 |
| `main.py` | 修改 | 添加 `/api/solve_pnp` POST 端点 |
| `public/index.html` | 修改 | 添加 PnP 面板 HTML 模板 |
| `public/js/image.js` | 修改 | SVG 叠加 draggable corner handles |
| `public/css/main.css` | 修改 | PnP 面板样式 |

## 实现优先级

```
P0: calibpy/calib_pnp.py + main.py 端点     ← 后端 PnP 求解
P1: index.html 面板模板 + calib_pnp.js 骨架   ← 前端框架
P2: image.js SVG handles + 拖拽事件           ← 交互核心
P3: main.css 面板样式                         ← 界面美化
P4: 实时求解模式                              ← 优化
```
