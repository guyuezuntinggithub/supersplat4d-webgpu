# twgsl（WebGPU GLSL→WGSL 转译）

PlayCanvas WebGPU 在将 GLSL 转成 WGSL 时需要 **twgsl**。npm 上的 `twgsl` 包已不可用（404），因此需要本地提供。

## 需要放置的文件

请将以下两个文件放到本目录（与 README.md 同级）：

- **twgsl.js** — 主模块
- **twgsl.wasm** — WASM 二进制（twgsl.js 会按同路径加载）

## 获取方式

1. **从预构建源下载（若可用）**  
   运行项目根目录下的下载脚本（若已提供）：
   ```bash
   node scripts/download-twgsl.js
   ```

2. **自行从 BabylonJS/twgsl 构建**  
   仓库：<https://github.com/BabylonJS/twgsl>  
   按仓库说明用 Emscripten 构建，将生成的 `twgsl.js` 和 `twgsl.wasm` 复制到本目录。

3. **从已有工程或备份复制**  
   若你在其他项目或备份里已有 `twgsl@0.0.10` 的 `dist/twgsl.js` 和 `dist/twgsl.wasm`，可直接复制到本目录并重命名为上述文件名。

未放置这两个文件时，使用 WebGPU 且涉及 GLSL 转 WGSL 的功能会报错；仅使用 WGSL 或 WebGL2 时不受影响。
