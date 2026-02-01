/**
 * 占位：未安装真实 twgsl 时使用，避免 404 并给出明确错误。
 * 若需 WebGPU + GLSL，请将 BabylonJS/twgsl 构建得到的 twgsl.js 与 twgsl.wasm 替换本文件。
 * @see static/lib/twgsl/README.md
 */
export default function (/* wasmUrl */) {
    return Promise.resolve({
        convertSpirV2WGSL() {
            throw new Error(
                'twgsl 未安装：需要将 twgsl.js 与 twgsl.wasm 放入 static/lib/twgsl/（参见该目录 README），或使用 WGSL/WebGL2。'
            );
        }
    });
}
