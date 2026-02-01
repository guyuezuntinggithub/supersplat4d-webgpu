/**
 * 自定义高斯 z 向排序器：仅对高斯单元做 z（深度）排序，从远到近渲染，配合 alpha blending。
 * 不依赖 PlayCanvas 自带的 Worker 排序，在主线程同步完成排序并写入 orderTexture。
 */

import { EventHandler, TEXTURELOCK_WRITE } from 'playcanvas';

type TextureWithLevels = import('playcanvas').Texture & { _levels?: (Uint32Array | null)[] };

/** PlayCanvas GSplat 每个实例的顶点数（用于 instancingCount） */
export const GSPLAT_INSTANCE_SIZE = 128;

export class GSplatZSorter extends EventHandler {
    orderTexture: TextureWithLevels | null = null;
    centers: Float32Array | null = null;
    mapping: Uint32Array | null = null;
    /** 原始完整 centers，setMapping(null) 时恢复用 */
    private fullCenters: Float32Array | null = null;

    private cameraPos = { x: 0, y: 0, z: 0 };
    private cameraDir = { x: 0, y: 0, z: 0 };
    private lastCameraPos = { x: 0, y: 0, z: 0 };
    private lastCameraDir = { x: 0, y: 0, z: 0 };
    private sortIndices: Uint32Array | null = null;
    private depthScratch: Float32Array | null = null;

    destroy(): void {
        this.orderTexture = null;
        this.centers = null;
        this.fullCenters = null;
        this.mapping = null;
        this.sortIndices = null;
        this.depthScratch = null;
    }

    init(orderTexture: TextureWithLevels, centers: Float32Array, _chunks?: Float32Array | null): void {
        this.orderTexture = orderTexture;
        this.centers = centers.slice();
        this.fullCenters = this.centers.slice();
        const numSplats = this.centers.length / 3;
        this.sortIndices = new Uint32Array(numSplats);
        this.depthScratch = new Float32Array(numSplats);
        for (let i = 0; i < numSplats; i++) this.sortIndices![i] = i;
        this.uploadOrder(this.sortIndices!, numSplats);
        this.fire('updated', numSplats);
    }

    setMapping(mapping: Uint32Array | null): void {
        if (mapping) {
            const centers = new Float32Array(mapping.length * 3);
            for (let i = 0; i < mapping.length; i++) {
                const src = mapping[i] * 3;
                const dst = i * 3;
                centers[dst + 0] = this.fullCenters![src + 0];
                centers[dst + 1] = this.fullCenters![src + 1];
                centers[dst + 2] = this.fullCenters![src + 2];
            }
            this.centers = centers;
            this.mapping = mapping;
            const n = mapping.length;
            if (!this.sortIndices || this.sortIndices.length !== n) {
                this.sortIndices = new Uint32Array(n);
                this.depthScratch = new Float32Array(n);
            }
            for (let i = 0; i < n; i++) this.sortIndices![i] = i;
        } else {
            this.centers = this.fullCenters ? this.fullCenters.slice() : null;
            this.mapping = null;
            const n = this.centers ? this.centers.length / 3 : 0;
            if (n) {
                if (!this.sortIndices || this.sortIndices.length !== n) {
                    this.sortIndices = new Uint32Array(n);
                    this.depthScratch = new Float32Array(n);
                }
                for (let i = 0; i < n; i++) this.sortIndices![i] = i;
            }
        }
        const numSplats = this.centers ? this.centers.length / 3 : 0;
        if (numSplats && this.sortIndices) {
            this.sortByDepth();
            this.uploadOrder(this.sortIndices!, numSplats);
            this.fire('updated', numSplats);
        }
    }

    setCamera(pos: { x: number; y: number; z: number }, dir: { x: number; y: number; z: number }): void {
        const eps = 1e-6;
        if (
            Math.abs(pos.x - this.lastCameraPos.x) < eps &&
            Math.abs(pos.y - this.lastCameraPos.y) < eps &&
            Math.abs(pos.z - this.lastCameraPos.z) < eps &&
            Math.abs(dir.x - this.lastCameraDir.x) < eps &&
            Math.abs(dir.y - this.lastCameraDir.y) < eps &&
            Math.abs(dir.z - this.lastCameraDir.z) < eps
        ) {
            return;
        }
        this.lastCameraPos.x = this.cameraPos.x = pos.x;
        this.lastCameraPos.y = this.cameraPos.y = pos.y;
        this.lastCameraPos.z = this.cameraPos.z = pos.z;
        this.lastCameraDir.x = this.cameraDir.x = dir.x;
        this.lastCameraDir.y = this.cameraDir.y = dir.y;
        this.lastCameraDir.z = this.cameraDir.z = dir.z;
        this.sortByDepth();
        const numSplats = this.centers ? this.centers.length / 3 : 0;
        if (numSplats && this.sortIndices) {
            this.uploadOrder(this.sortIndices!, numSplats);
            this.fire('updated', numSplats);
        }
    }

    /** 按视线方向深度排序：深度 = dot(center, cameraDir)，升序 = 远到近，用于 alpha blending */
    private sortByDepth(): void {
        const centers = this.centers!;
        const dx = this.cameraDir.x;
        const dy = this.cameraDir.y;
        const dz = this.cameraDir.z;
        const n = centers.length / 3;
        const depths = this.depthScratch!;
        const order = this.sortIndices!;
        for (let i = 0; i < n; i++) {
            const o = i * 3;
            depths[i] = centers[o] * dx + centers[o + 1] * dy + centers[o + 2] * dz;
        }
        // 升序：深度小的（更远）在前 -> 从远到近绘制
        order.sort((a, b) => depths[a] - depths[b]);
        const mapping = this.mapping;
        if (mapping) {
            for (let i = 0; i < n; i++) order[i] = mapping[order[i]];
        }
    }

    private uploadOrder(order: Uint32Array, count: number): void {
        const tex = this.orderTexture;
        if (!tex) return;
        const buf = tex.lock({ mode: TEXTURELOCK_WRITE }) as Uint32Array;
        const len = Math.min(buf.length, count);
        for (let i = 0; i < len; i++) buf[i] = order[i];
        for (let i = len; i < buf.length; i++) buf[i] = 0;
        tex.unlock();
    }
}
