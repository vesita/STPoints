# PnP 外参矫正功能开发笔记

## 概述

实现了一个基于 OpenCV IPPE 算法的 PnP 外参标定工具，用户在 3D 视图中选择 box 角点，在 2D 图像上拖拽对应位置，求解新的外参矩阵。

## 架构

```
3D 视图 (Three.js)          2D 图像 (SVG)
  box 角点选择  ──────→  角点拖拽定位
       │                      │
       └──── solvePnP ────────┘
              (IPPE)
                │
         新外参矩阵 → 重投影验证 → 保存到 JSON
```

## 踩坑记录

### 1. SVG viewBox 与 preserveAspectRatio 的坐标映射陷阱

**问题**：SVG 使用 `viewBox="0 0 2048 1536"` + `preserveAspectRatio="xMidYMid meet"` 后，图片保持宽高比居中显示，但 SVG 坐标系仍然是 0-2048 × 0-1536 的全区域。点击黑框区域（图片外的空白）会返回 SVG 坐标，但这些坐标映射到图片上是错误的。

**尝试过的方案**：
- `_getImageBounds()` 计算图片在 SVG 中的实际显示区域，然后做 CSS 像素 → SVG viewBox → 图片像素的多级转换
- `_imageToSvgCoords()` / `_svgToImageCoords()` 双向转换
- 每次渲染都要调用 `getBoundingClientRect()` + 缩放计算

**结果**：多次修补仍有边界 case，坐标链路过长容易出错。

**最终方案**：彻底放弃固定 viewBox。改为：
- SVG `viewBox` 动态设为图片 `naturalWidth × naturalHeight`
- 弹簧布局（flex spacer）居中图片
- 坐标 1:1 映射：SVG 坐标 = 图片像素坐标
- 删除所有 `_getImageBounds` / `_imageToSvgCoords` 等中间转换层

**教训**：不要试图在两个不匹配的坐标系之间做转换，直接统一坐标系更可靠。

### 2. Box 角点拾取精度

**问题**：Three.js 的 `Raycaster.intersectObjects` 对 `THREE.Points` 的拾取精度依赖 `threshold` 参数。默认值太大（1.0）会选到不相关的点，太小（0.1）则很难选中。

**解决方案**：
- 阈值设为 0.3（米），在精度和易用性之间平衡
- Box 角点使用射线距离计算（`_pointToRayDistance`），阈值 2.0 米
- 优先拾取 box 角点，其次点云，最后 z=0 平面兜底

### 3. Box 角点高亮大小

**问题**：角点高亮球体在 3D 视图中大小是世界坐标单位，不同缩放下视觉效果差异大。0.15 太大，0.05 仍然偏大。

**最终值**：`SphereGeometry(0.03, 8, 8)` — 在常见缩放下刚好能看到。

### 4. IPPE 的点序敏感性

**问题**：OpenCV 的 `SOLVEPNP_IPPE` 对 4 个 2D-3D 点的对应关系非常敏感。错误的点序会导致完全错误的外参，但重投影误差可能看起来不太大。

**解决方案**：穷举测试功能 — 遍历 24 种点序排列（4! = 24），每种都计算重投影误差，按误差排序展示。用户可以直接点击结果预览并保存最优解。

### 5. 3D 角点选取与 2D 点不对应

**问题**：用户在 3D 视图中选取 box 角点时，容易选错角点（比如选了背面的角点而非正面的），或者选取的 3D 角点和后续在 2D 图像上标注的角点并非同一物理位置。这导致 PnP 求解的 3D-2D 对应关系错误，计算出的外参完全不对，但表面看操作流程是正确的，很难排查。

**根因**：box 有 8 个角点，视觉上容易混淆（尤其是遮挡面和旋转后的 box）。手动选取 4 个角点时，用户需要在 3D 和 2D 之间保持一致的对应关系，这在没有辅助标记的情况下很容易出错。

**解决方案**：
- **自动选取**：默认从 box 正面 4 个角点自动获取 3D 坐标，避免手动选错
- **手动选取辅助**：手动模式下，选中的 3D 角点会变色（红/绿/蓝/黄对应 P0-P3），并在 2D 图像上同步显示标记，方便用户确认对应关系
- **穷举兜底**：穷举 24 种点序排列，即使用户选对了 4 个 3D 点和 4 个 2D 点但顺序不一致，也能通过最小重投影误差找到正确配对

### 6. 图像角点不遵从图片坐标系

**问题**：PnP 面板中，`box_to_2d_points()` 返回的是**图片像素坐标**（如 1920×1080），但 SVG 使用固定 `viewBox="0 0 2048 1536"`。两者之间存在缩放映射，所有渲染和交互都必须做坐标转换。更糟糕的是，`preserveAspectRatio="xMidYMid meet"` 导致图片实际显示区域不等于 SVG 元素区域，点击黑框区域会被错误映射到图片上。

**表现**：
- 手动拖拽的 2D 角点坐标与 `points3d_homo_to_image2d` 投影出的坐标对不上
- 重投影标记偏移，尤其是在图像边缘
- 点击图片外的黑框区域也能放置角点，但坐标是错的

**最终方案**：SVG `viewBox` 动态设为图片 `naturalWidth × naturalHeight`，实现坐标 1:1 映射。详见第 1 条。

### 7. 重投影误差计算的差异

**问题**：前端 JS 的重投影（`points3d_homo_to_image2d`）和后端 Python 的 `cv2.projectPoints` 结果不一致。

**原因**：
- JS 端未考虑畸变系数（`dist_coeffs`）
- JS 端可能有 `rect` 矩阵（KITTI 风格的校正矩阵）但未被应用
- 浮点精度差异

**解决方案**：重投影误差统一在 Python 端计算（使用 `cv2.projectPoints` + `dist_coeffs`），前端只做可视化展示。

### 8. 图像去畸变的实现

**问题**：镜头畸变导致 3D 投影在图像边缘对不齐。

**方案选择**：
- 前端 Canvas 去畸变：需要手写畸变模型或引入 OpenCV.js（太大）
- 后端 OpenCV 去畸变：简单可靠，`cv2.undistort` 一行搞定

**最终方案**：后端 `/undistort` 端点 + 前端 toggle。无 `dist_coeffs` 时重定向到原图，零开销。

**注意**：去畸变后的图像尺寸与原图相同，投影坐标无需修改（等价于针孔模型）。

### 9. image.js 中 BoxImageContext 的去畸变

**问题**：`BoxImageContext` 使用 Canvas `drawImage()` 直接绘制原始 `Image` 对象，不经过 `ImageContext.show_image()`，所以去畸变 toggle 对它无效。

**解决方案**：通过 `_imageSourceProvider` 模式，让 `BoxImageContext` 从 `ImageContextManager` 获取去畸变缓存图像。如果是 dataURL 字符串，先 `new Image()` 加载再绘制。

### 10. world.js 中的 `this` 上下文

**问题**：`World.new_line()` 中 `this.world.data.dbg.alloc()` 报错 — `World` 是构造函数，`this` 已经是 world 实例，不需要再 `.world`。

**修复**：改为 `this.data.dbg.alloc()`。

## 文件结构

```
main.py                    — /solve_pnp, /brute_force_pnp, /calib_save, /undistort 端点
calibpy/calib_pnp.py       — OpenCV IPPE 求解器
public/js/calib_pnp.js     — PnP 面板控制器（角点选择、拖拽、穷举、保存）
public/js/measure.js       — 3D 测量工具（box 角点拾取、距离测量）
public/js/image.js         — 图像显示（ImageContext + BoxImageContext + 去畸变）
public/js/editor.js        — 主编辑器（按钮绑定、状态管理）
public/js/world.js         — 世界/场景管理
```

## 标定文件格式

```json
{
    "extrinsic": [16 floats],    // 4x4 行优先 LiDAR→Camera 变换矩阵
    "intrinsic": [9 floats],     // 3x3 行优先相机内参 [fx,0,cx,0,fy,cy,0,0,1]
    "dist_coeffs": [5 floats]    // 可选，畸变系数 [k1,k2,p1,p2,k3]
}
```
