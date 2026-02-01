查看 `supersplat4d` 和 `engine` 中与 sort 相关的代码，梳理高斯排序的实现逻辑。


[4 tools called]


读取关键文件，梳理排序逻辑：


[5 tools called]


查看 `supersplat4d` 中如何使用排序：


[3 tools called]


下面是 engine 中高斯排序的实现逻辑：

## 高斯排序的整体架构

engine 提供两种排序路径：

### 1. GPU 排序路径 (`sortGpu`)
使用 GPU compute shader 进行排序，适合支持 WebGPU 的环境。

### 2. CPU 排序路径 (`sortCpu`)
使用 Web Worker 在后台线程进行排序，作为 GPU 的降级方案。

## 排序流程详解

### 阶段 1: 计算距离范围

```936:1036:engine/src/scene/gsplat-unified/gsplat-manager.js
    sortGpu(worldState) {
        const keyGenerator = this.keyGenerator;
        const gpuSorter = this.gpuSorter;
        Debug.assert(keyGenerator && gpuSorter, 'GPU sorter not initialized');
        if (!keyGenerator || !gpuSorter) return;

        const elementCount = worldState.totalUsedPixels;
        if (elementCount === 0) return;

        // Handle first-time setup for GPU path
        if (!worldState.sortedBefore) {
            worldState.sortedBefore = true;
            this.rebuildWorkBuffer(worldState, elementCount);

            // clean up old world states
            this.cleanupOldWorldStates(worldState.version);
            this.sortedVersion = worldState.version;
        }

        // number of bits used for sorting to match CPU sorter
        const numBits = Math.max(10, Math.min(20, Math.round(Math.log2(elementCount / 4))));
        // Round up to multiple of 4 for radix sort
        const roundedNumBits = Math.ceil(numBits / 4) * 4;

        // Compute min/max distances for key normalization
        const { minDist, maxDist } = this.computeDistanceRange(worldState);

        // Generate sort keys from work buffer world-space positions
        const keysBuffer = keyGenerator.generate(
            this.workBuffer,
            this.cameraNode,
            this.scene.gsplat.radialSorting,
            elementCount,
            roundedNumBits,
            minDist,
            maxDist
        );

        // Run GPU radix sort
        const sortedIndices = gpuSorter.sort(keysBuffer, elementCount, roundedNumBits);

        // Set sorted indices directly on renderer (no CPU upload needed)
        this.renderer.setOrderBuffer(sortedIndices);

        // Update renderer with count
        this.renderer.update(elementCount, worldState.textureSize);
    }

    /**
     * Computes the min/max effective distances for the current world state.
     *
     * @param {GSplatWorldState} worldState - The world state.
     * @returns {{minDist: number, maxDist: number}} The distance range.
     */
    computeDistanceRange(worldState) {
        const cameraNode = this.cameraNode;
        const cameraMat = cameraNode.getWorldTransform();
        cameraMat.getTranslation(cameraPosition);
        cameraMat.getZ(cameraDirection).normalize();

        const radialSort = this.scene.gsplat.radialSorting;

        // For radial: minDist is always 0, only track maxDist
        // For linear: track both min and max along camera direction
        let minDist = radialSort ? 0 : Infinity;
        let maxDist = radialSort ? 0 : -Infinity;

        for (const splat of worldState.splats) {
            const modelMat = splat.node.getWorldTransform();
            const aabbMin = splat.aabb.getMin();
            const aabbMax = splat.aabb.getMax();

            // Check all 8 corners of local-space AABB
            for (let i = 0; i < 8; i++) {
                _cornerPoint.x = (i & 1) ? aabbMax.x : aabbMin.x;
                _cornerPoint.y = (i & 2) ? aabbMax.y : aabbMin.y;
                _cornerPoint.z = (i & 4) ? aabbMax.z : aabbMin.z;

                // Transform to world space
                modelMat.transformPoint(_cornerPoint, _cornerPoint);

                if (radialSort) {
                    // Radial: distance from camera
                    const dist = _cornerPoint.distance(cameraPosition);
                    if (dist > maxDist) maxDist = dist;
                } else {
                    // Linear: distance along camera direction
                    const dist = _cornerPoint.sub(cameraPosition).dot(cameraDirection);
                    if (dist < minDist) minDist = dist;
                    if (dist > maxDist) maxDist = dist;
                }
            }
        }

        // Handle empty state
        if (maxDist === 0 || maxDist === -Infinity) {
            return { minDist: 0, maxDist: 1 };
        }

        return { minDist, maxDist };
    }
```

要点：
- 遍历所有 splat 的 AABB 的 8 个角点
- 计算每个角点到相机的距离（线性或径向）
- 确定最小/最大距离范围用于归一化

### 阶段 2: 生成排序键

排序键生成支持两种模式：

#### 线性模式（Linear）
- 计算每个高斯点沿相机前向向量的投影距离
- 公式：`dist = dot(splatPosition - cameraPosition, cameraDirection)`

#### 径向模式（Radial）
- 计算每个高斯点到相机的欧氏距离
- 公式：`dist = length(splatPosition - cameraPosition)`

排序键生成还使用相机相对 bin 权重优化：

```1:85:engine/src/scene/shader-lib/wgsl/chunks/gsplat/compute-gsplat-sort-key.js
// Compute shader for generating GSplat sort keys from world-space positions in work buffer
// Uses camera-relative bin weighting for precision optimization near the camera
// Supports both linear (forward vector) and radial (distance) sorting modes

export const computeGsplatSortKeySource = /* wgsl */`

// Work buffer texture containing world-space centers (RGBA32U: xyz as floatBitsToUint)
@group(0) @binding(0) var splatTexture0: texture_2d<u32>;

// Output sort keys (one u32 per splat)
@group(0) @binding(1) var<storage, read_write> sortKeys: array<u32>;

// Uniforms
struct SortKeyUniforms {
    cameraPosition: vec3f,
    elementCount: u32,
    cameraDirection: vec3f,
    numBits: u32,
    textureSize: u32,
    minDist: f32,
    invRange: f32,
    numWorkgroupsX: u32,
    numBins: u32
};
@group(0) @binding(2) var<uniform> uniforms: SortKeyUniforms;

// Camera-relative bin weighting (entries with base and divider)
struct BinWeight {
    base: f32,
    divider: f32
};
@group(0) @binding(3) var<storage, read> binWeights: array<BinWeight>;

@compute @workgroup_size({WORKGROUP_SIZE_X}, {WORKGROUP_SIZE_Y}, 1)
fn computeSortKey(@builtin(global_invocation_id) global_id: vec3u) {
    let gid = global_id.x + global_id.y * ({WORKGROUP_SIZE_X} * uniforms.numWorkgroupsX);
    
    // Early exit for out-of-bounds threads
    if (gid >= uniforms.elementCount) {
        return;
    }
    
    // Calculate texture UV from linear index
    let textureSize = uniforms.textureSize;
    let uv = vec2i(i32(gid % textureSize), i32(gid / textureSize));
    
    // Load world-space center from work buffer (stored as floatBitsToUint)
    let packed = textureLoad(splatTexture0, uv, 0);
    let worldCenter = vec3f(
        bitcast<f32>(packed.r),
        bitcast<f32>(packed.g),
        bitcast<f32>(packed.b)
    );
    
    // Calculate distance based on sort mode
    var dist: f32;
    
    #ifdef RADIAL_SORT
        // Radial mode: distance from camera (inverted so far objects get small keys)
        let delta = worldCenter - uniforms.cameraPosition;
        let radialDist = length(delta);
        // Invert distance so far objects get small keys (rendered first, back-to-front)
        dist = (1.0 / uniforms.invRange) - radialDist - uniforms.minDist;
    #else
        // Linear mode: distance along camera forward vector
        let toSplat = worldCenter - uniforms.cameraPosition;
        dist = dot(toSplat, uniforms.cameraDirection) - uniforms.minDist;
    #endif
    
    // Apply bin-based mapping for camera-relative precision weighting
    let numBins = uniforms.numBins;
    let d = dist * uniforms.invRange * f32(numBins);
    let binFloat = clamp(d, 0.0, f32(numBins) - 0.001);
    let bin = u32(binFloat);
    let binFrac = binFloat - f32(bin);
    
    // Calculate final sort key using pre-computed bin weighting
    let sortKey = u32(binWeights[bin].base + binWeights[bin].divider * binFrac);
    
    // Write sort key
    sortKeys[gid] = sortKey;
}
`;
```

相机相对 bin 权重：
- 将距离范围分成 32 个 bin
- 相机附近的 bin 分配更多精度（权重更高）
- 远离相机的 bin 分配较少精度

```27:35:engine/src/scene/gsplat-unified/gsplat-sort-bin-weights.js
    static get WEIGHT_TIERS() {
        return [
            { maxDistance: 0, weight: 40.0 },   // Camera bin
            { maxDistance: 2, weight: 20.0 },   // Adjacent bins
            { maxDistance: 5, weight: 8.0 },    // Nearby bins
            { maxDistance: 10, weight: 3.0 },   // Medium distance
            { maxDistance: Infinity, weight: 1.0 }  // Far bins
        ];
    }
```

### 阶段 3: 执行排序

#### GPU 路径：4-bit Radix Sort
```463:530:engine/src/scene/graphics/compute-radix-sort.js
    sort(keysBuffer, elementCount, numBits = 16) {
        Debug.assert(keysBuffer, 'ComputeRadixSort.sort: keysBuffer is required');
        Debug.assert(elementCount > 0, 'ComputeRadixSort.sort: elementCount must be > 0');
        Debug.assert(numBits % BITS_PER_PASS === 0, `ComputeRadixSort.sort: numBits must be a multiple of ${BITS_PER_PASS}`);

        this._elementCount = elementCount;

        // Allocate buffers and create passes if needed
        this._allocateBuffers(elementCount, numBits);

        const device = this.device;
        const numPasses = numBits / BITS_PER_PASS;

        // First pass reads directly from input buffer (no copy needed)
        // Subsequent passes ping-pong between internal buffers
        let currentKeys = keysBuffer;
        let currentValues = this._values0;
        let nextKeys = this._keys0;
        let nextValues = this._values1;

        // Execute radix passes using cached compute instances
        for (let pass = 0; pass < numPasses; pass++) {
            const { blockSumCompute, reorderCompute } = this._passes[pass];
            const isLastPass = (pass === numPasses - 1);

            // Phase 1: Compute local prefix sums and block sums
            blockSumCompute.setParameter('input', currentKeys);
            blockSumCompute.setParameter('local_prefix_sums', this._localPrefixSums);
            blockSumCompute.setParameter('block_sums', this._blockSums);

            // Set uniform values
            blockSumCompute.setParameter('workgroupCount', this._workgroupCount);
            blockSumCompute.setParameter('elementCount', elementCount);
            blockSumCompute.setupDispatch(this._dispatchSize.x, this._dispatchSize.y, 1);
            device.computeDispatch([blockSumCompute], 'RadixSort-BlockSum');

            // Phase 2: Prefix sum on block sums
            this._prefixSumKernel.dispatch(device);

            // Phase 3: Reorder elements to sorted positions
            // On last pass, write directly to sortedIndices to avoid final copy
            const outputValues = isLastPass ? this._sortedIndices : nextValues;

            reorderCompute.setParameter('inputKeys', currentKeys);
            reorderCompute.setParameter('outputKeys', nextKeys);
            reorderCompute.setParameter('local_prefix_sum', this._localPrefixSums);
            reorderCompute.setParameter('prefix_block_sum', this._blockSums);
            reorderCompute.setParameter('inputValues', currentValues);
            reorderCompute.setParameter('outputValues', outputValues);

            // Set uniform values
            reorderCompute.setParameter('workgroupCount', this._workgroupCount);
            reorderCompute.setParameter('elementCount', elementCount);
            reorderCompute.setupDispatch(this._dispatchSize.x, this._dispatchSize.y, 1);
            device.computeDispatch([reorderCompute], 'RadixSort-Reorder');

            // Swap buffers for next pass (skip on last pass - not needed)
            if (!isLastPass) {
                currentKeys = nextKeys;
                nextKeys = (currentKeys === this._keys0) ? this._keys1 : this._keys0;
                const tempValues = currentValues;
                currentValues = nextValues;
                nextValues = tempValues;
            }
        }

        return this._sortedIndices;
    }
```

- 每次处理 4 位（16 个桶）
- 16 位键需要 4 次 pass，32 位需要 8 次
- 每 pass 三步：计算局部前缀和、全局前缀和、重排

#### CPU 路径：Counting Sort
```160:173:engine/src/scene/gsplat-unified/gsplat-unified-sort-worker.js
    const countingSort = (bucketCount, countBuffer, numVertices, distances, order) => {

        // accumulate counts
        for (let i = 1; i < bucketCount; i++) {
            countBuffer[i] += countBuffer[i - 1];
        }

        // build output array
        for (let i = 0; i < numVertices; i++) {
            const distance = distances[i];
            const destIndex = --countBuffer[distance];
            order[destIndex] = i;
        }
    };
```

- 计数排序，时间复杂度 O(n)
- 在 Web Worker 中执行，不阻塞主线程

## 性能瓶颈分析

可能的瓶颈点：
1. 距离范围计算：遍历所有 splat 的 AABB 角点
2. 排序键生成：每个高斯点都要计算距离和 bin 映射
3. 排序本身：Radix Sort 需要多次 pass

优化建议：
- 减少每帧排序次数（仅在相机移动/旋转超过阈值时排序）
- 使用更粗粒度的距离范围计算
- 考虑增量排序（只对变化的部分重新排序）

需要我进一步分析某个具体部分吗？