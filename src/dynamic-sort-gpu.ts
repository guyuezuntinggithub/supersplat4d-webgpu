/**
 * 动态高斯 WebGPU 深度排序：按 dot(center, viewDir) 升序（远到近），与 sort worker 深度排序原则一致。
 * 仅当 device 为 WebGPU 且可用时使用；否则返回 null，由调用方回退到 CPU 排序。
 */
/// <reference types="@webgpu/types" />

const KEY_INDEX_STRIDE = 8; // f32 key (4) + u32 index (4)

function nextPowerOf2(n: number): number {
    let p = 1;
    while (p < n) p *= 2;
    return p;
}

const computeKeysWGSL = /* wgsl */ `
struct Params {
    numElements: u32,
    dirX: f32,
    dirY: f32,
    dirZ: f32,
    padding: f32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> centers: array<vec3f>;
@group(0) @binding(2) var<storage, read_write> keysIndices: array<vec2f>; // x = key, y = bitcast u32 index

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.numElements) {
        keysIndices[i].x = 1e30;
        keysIndices[i].y = 0.0;
        return;
    }
    let c = centers[i];
    let key = c.x * params.dirX + c.y * params.dirY + c.z * params.dirZ;
    keysIndices[i].x = key;
    keysIndices[i].y = bitcast<f32>(i);
}
`;

const bitonicStepWGSL = /* wgsl */ `
struct Params {
    numElements: u32,
    stage: u32,
    step: u32,
    _pad: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> keysIndices: array<vec2f>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    let j = i ^ (1u << params.step);
    if (j <= i) { return; }
    if (max(i, j) >= params.numElements) { return; }

    let keyI = keysIndices[i].x;
    let keyJ = keysIndices[j].x;
    let idxI = keysIndices[i].y;
    let idxJ = keysIndices[j].y;

    let ascending = (i & (1u << (params.step + 1u))) == 0u;
    let swap = ascending && keyI > keyJ || !ascending && keyI < keyJ;

    if (swap) {
        keysIndices[i].x = keyJ;
        keysIndices[i].y = idxJ;
        keysIndices[j].x = keyI;
        keysIndices[j].y = idxI;
    }
}
`;

export type CameraParams = { x: number; y: number; z: number };

/**
 * 从 PlayCanvas GraphicsDevice 获取 WebGPU GPUDevice（若为 WebGPU）。
 */
export function getWebGPUDevice(graphicsDevice: unknown): GPUDevice | null {
    if (!graphicsDevice || typeof (graphicsDevice as { isWebGPU?: boolean }).isWebGPU !== 'boolean') return null;
    if (!(graphicsDevice as { isWebGPU: boolean }).isWebGPU) return null;
    const d = (graphicsDevice as { device?: GPUDevice }).device;
    return d ?? null;
}

let keysModule: GPUShaderModule | null = null;
let bitonicModule: GPUShaderModule | null = null;

/**
 * 使用 WebGPU 按深度排序：输入 centers (N*3)，输出 N 个本地索引（0..N-1）按 depth 升序。
 * centers 为当前帧的 active centers（与 activeIndices 一一对应），相机方向为 viewDir。
 * 返回 Promise<Uint32Array> 或 null（不可用时）。
 */
export async function sortByDepthGPU(
    device: GPUDevice,
    centers: Float32Array,
    cameraDir: CameraParams
): Promise<Uint32Array | null> {
    const N = centers.length / 3;
    if (N === 0) return new Uint32Array(0);

    const M = nextPowerOf2(N);
    const numElements = M;

    const centersBuffer = device.createBuffer({
        size: Math.max(16, numElements * 12),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(centersBuffer, 0, centers as unknown as GPUAllowSharedBufferSource, 0, N * 3);

    const keysIndicesBuffer = device.createBuffer({
        size: numElements * KEY_INDEX_STRIDE,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    const paramsBuffer = device.createBuffer({
        size: 32,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    const paramsData = new ArrayBuffer(32);
    const paramsView = new DataView(paramsData);
    paramsView.setUint32(0, N, true);
    paramsView.setFloat32(4, cameraDir.x, true);
    paramsView.setFloat32(8, cameraDir.y, true);
    paramsView.setFloat32(12, cameraDir.z, true);
    device.queue.writeBuffer(paramsBuffer, 0, paramsData);

    if (!keysModule) {
        keysModule = device.createShaderModule({ code: computeKeysWGSL });
    }
    if (!bitonicModule) {
        bitonicModule = device.createShaderModule({ code: bitonicStepWGSL });
    }

    const keysPipeline = device.createComputePipeline({
        layout: 'auto',
        compute: {
            module: keysModule,
            entryPoint: 'main'
        }
    });

    const keysBindGroup = device.createBindGroup({
        layout: keysPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: paramsBuffer } },
            { binding: 1, resource: { buffer: centersBuffer } },
            { binding: 2, resource: { buffer: keysIndicesBuffer } }
        ]
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(keysPipeline);
    pass.setBindGroup(0, keysBindGroup);
    pass.dispatchWorkgroups(Math.ceil(numElements / 256));

    const stages = Math.ceil(Math.log2(numElements));
    const bitonicPipeline = device.createComputePipeline({
        layout: 'auto',
        compute: {
            module: bitonicModule!,
            entryPoint: 'main'
        }
    });

    const stepParamsBuffer = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    const stepParamsData = new ArrayBuffer(16);
    const stepParamsView = new DataView(stepParamsData);

    for (let stage = 0; stage < stages; stage++) {
        for (let step = stage; step >= 0; step--) {
            stepParamsView.setUint32(0, numElements, true);
            stepParamsView.setUint32(4, stage, true);
            stepParamsView.setUint32(8, step, true);
            device.queue.writeBuffer(stepParamsBuffer, 0, stepParamsData);

            const stepBindGroup = device.createBindGroup({
                layout: bitonicPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: stepParamsBuffer } },
                    { binding: 1, resource: { buffer: keysIndicesBuffer } }
                ]
            });

            pass.setPipeline(bitonicPipeline);
            pass.setBindGroup(0, stepBindGroup);
            pass.dispatchWorkgroups(Math.ceil(numElements / 2 / 256));
        }
    }
    pass.end();

    const readbackBuffer = device.createBuffer({
        size: numElements * KEY_INDEX_STRIDE,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    encoder.copyBufferToBuffer(keysIndicesBuffer, 0, readbackBuffer, 0, numElements * KEY_INDEX_STRIDE);

    device.queue.submit([encoder.finish()]);

    await readbackBuffer.mapAsync(GPUMapMode.READ);
    const mappedRange = readbackBuffer.getMappedRange();
    const sortedLocal = new Uint32Array(N);
    const view = new DataView(mappedRange);
    for (let i = 0; i < N; i++) {
        sortedLocal[i] = view.getUint32(i * KEY_INDEX_STRIDE + 4, true);
    }
    readbackBuffer.unmap();

    centersBuffer.destroy();
    keysIndicesBuffer.destroy();
    paramsBuffer.destroy();
    stepParamsBuffer.destroy();
    readbackBuffer.destroy();

    return sortedLocal;
}
