import {
    EVENT_POSTRENDER_LAYER,
    EVENT_PRERENDER_LAYER,
    LAYERID_DEPTH,
    SORTMODE_NONE,
    BoundingBox,
    CameraComponent,
    Color,
    Entity,
    Layer,
    MiniStats,
    GraphicsDevice
} from 'playcanvas';

import { AssetLoader } from './asset-loader';
import { Camera } from './camera';
import { DataProcessor } from './data-processor';
import { Element, ElementType, ElementTypeList } from './element';
import { Events } from './events';
import { InfiniteGrid as Grid } from './infinite-grid';
import { Outline } from './outline';
import { PCApp } from './pc-app';
import { SceneConfig } from './scene-config';
import { SceneState } from './scene-state';
import { Splat } from './splat';
import { SplatOverlay } from './splat-overlay';
import { Underlay } from './underlay';

class Scene {
    events: Events;
    config: SceneConfig;
    canvas: HTMLCanvasElement;
    app: PCApp;
    backgroundLayer: Layer;
    shadowLayer: Layer;
    debugLayer: Layer;
    overlayLayer: Layer;
    gizmoLayer: Layer;
    sceneState = [new SceneState(), new SceneState()];
    elements: Element[] = [];
    boundStorage = new BoundingBox();
    boundDirty = true;
    forceRender = false;

    lockedRenderMode = false;
    lockedRender = false;

    canvasResize: {width: number; height: number} | null = null;
    targetSize = {
        width: 0,
        height: 0
    };

    dataProcessor: DataProcessor;
    assetLoader: AssetLoader;
    camera: Camera;
    miniStats: MiniStats | null = null;
    splatOverlay: SplatOverlay;
    grid: Grid;
    outline: Outline;
    underlay: Underlay;

    contentRoot: Entity;
    cameraRoot: Entity;

    constructor(
        events: Events,
        config: SceneConfig,
        canvas: HTMLCanvasElement,
        graphicsDevice: GraphicsDevice
    ) {
        this.events = events;
        this.config = config;
        this.canvas = canvas;

        // configure the playcanvas application. we render to an offscreen buffer so require
        // only the simplest of backbuffers.
        this.app = new PCApp(canvas, { graphicsDevice });

        // Enable United pipeline GPU sorting (WebGPU only, uses sortGpu in gsplat-manager)
        if (this.app.graphicsDevice.isWebGPU) {
            this.app.scene.gsplat.gpuSorting = true;
        }

        // only render the scene when instructed
        this.app.autoRender = false;
        // @ts-ignore
        this.app._allowResize = false;
        this.app.scene.clusteredLightingEnabled = false;

        // hack: disable lightmapper first bake until we expose option for this
        // @ts-ignore
        this.app.off('prerender', this.app._firstBake, this.app);

        // @ts-ignore
        this.app.loader.getHandler('texture').imgParser.crossOrigin = 'anonymous';

        // this is required to get full res AR mode backbuffer
        this.app.graphicsDevice.maxPixelRatio = window.devicePixelRatio;

        // configure application canvas
        const observer = new ResizeObserver((entries: ResizeObserverEntry[]) => {
            if (entries.length > 0) {
                const entry = entries[0];
                if (entry) {
                    if (entry.devicePixelContentBoxSize) {
                        // on non-safari browsers, we are given the pixel-perfect canvas size
                        this.canvasResize = {
                            width: entry.devicePixelContentBoxSize[0].inlineSize,
                            height: entry.devicePixelContentBoxSize[0].blockSize
                        };
                    } else if (entry.contentBoxSize.length > 0) {
                        // on safari browsers we must calculate pixel size from CSS size ourselves
                        // and hope the browser performs the same calculation.
                        const pixelRatio = window.devicePixelRatio;
                        this.canvasResize = {
                            width: Math.ceil(entry.contentBoxSize[0].inlineSize * pixelRatio),
                            height: Math.ceil(entry.contentBoxSize[0].blockSize * pixelRatio)
                        };
                    }
                }
                this.forceRender = true;
            }
        });

        observer.observe(window.document.getElementById('canvas-container'));

        // configure depth layers to handle dynamic refraction
        const depthLayer = this.app.scene.layers.getLayerById(LAYERID_DEPTH);
        this.app.scene.layers.remove(depthLayer);
        this.app.scene.layers.insertOpaque(depthLayer, 2);

        // register application callbacks
        this.app.on('update', (deltaTime: number) => this.onUpdate(deltaTime));
        this.app.on('prerender', () => this.onPreRender());
        this.app.on('postrender', () => this.onPostRender());

        // 性能统计叠加层（DrawCalls / Frame / GPU / CPU）
        //this.miniStats = new MiniStats(this.app);
        const miniStatsOptions = MiniStats.getDefaultOptions() as any;
        miniStatsOptions.startSizeIndex = 1;
        // 增加第四档：更宽更高，方便看所有 GPU/CPU 子项
        miniStatsOptions.sizes = [
            { width: 100, height: 16, spacing: 0, graphs: false },
            { width: 128, height: 32, spacing: 2, graphs: true },
            { width: 256, height: 64, spacing: 2, graphs: true },
            { width: 320, height: 96, spacing: 2, graphs: true }
        ];

        // display additional counters
        // Note: for most of these to report values, either debug or profiling engine build needs to be used.
        miniStatsOptions.stats = [
            // frame update time in ms
            {
                name: 'Update',
                stats: ['frame.updateTime'],
                decimalPlaces: 1,
                unitsName: 'ms',
                watermark: 33
            },

            // total number of draw calls
            {
                name: 'DrawCalls',
                stats: ['drawCalls.total'],
                watermark: 2000
            },

            // total number of triangles, in 1000s
            // {
            //     name: 'triCount',
            //     stats: ['frame.triangles'],
            //     decimalPlaces: 1,
            //     multiplier: 1 / 1000,
            //     unitsName: 'k',
            //     watermark: 500
            // },

            // // number of materials used in a frame
            // {
            //     name: 'materials',
            //     stats: ['frame.materials'],
            //     watermark: 2000
            // },

            // frame time it took to do frustum culling
            {
                name: 'cull',
                stats: ['frame.cullTime'],
                decimalPlaces: 1,
                watermark: 1,
                unitsName: 'ms'
            },

            // used VRAM, displayed using 2 colors - red for textures, green for geometry
            {
                name: 'VRAM',
                stats: ['vram.tex', 'vram.geom'],
                decimalPlaces: 1,
                multiplier: 1 / (1024 * 1024),
                unitsName: 'MB',
                watermark: 100
            },

            // frames per second
            {
                name: 'FPS',
                stats: ['frame.fps'],
                watermark: 60
            },

            // delta time
            {
                name: 'Frame',
                stats: ['frame.ms'],
                decimalPlaces: 1,           
                unitsName: 'ms',
                watermark: 33
            }
        ];



        this.miniStats = new MiniStats(this.app, miniStatsOptions);
        
        // set up options for mini-stats, start with the default options
        // const miniStatsOptions = MiniStats.getDefaultOptions();

        // // configure sizes
        // miniStatsOptions.sizes = [
        //     { width: 128, height: 16, spacing: 0, graphs: false },
        //     { width: 256, height: 32, spacing: 2, graphs: true },
        //     { width: 500, height: 64, spacing: 2, graphs: true }
        // ];

        // // when the application starts, use the largest size
        // miniStatsOptions.startSizeIndex = 2;

        // // display additional counters
        // // Note: for most of these to report values, either debug or profiling engine build needs to be used.
        // miniStatsOptions.stats = [
        //     // frame update time in ms
        //     {
        //         name: 'Update',
        //         stats: ['frame.updateTime'],
        //         decimalPlaces: 1,
        //         unitsName: 'ms',
        //         watermark: 33
        //     },

        //     // total number of draw calls
        //     {
        //         name: 'DrawCalls',
        //         stats: ['drawCalls.total'],
        //         watermark: 2000
        //     },

        //     // total number of triangles, in 1000s
        //     {
        //         name: 'triCount',
        //         stats: ['frame.triangles'],
        //         decimalPlaces: 1,
        //         multiplier: 1 / 1000,
        //         unitsName: 'k',
        //         watermark: 500
        //     },

        //     // number of materials used in a frame
        //     {
        //         name: 'materials',
        //         stats: ['frame.materials'],
        //         watermark: 2000
        //     },

        //     // frame time it took to do frustum culling
        //     {
        //         name: 'cull',
        //         stats: ['frame.cullTime'],
        //         decimalPlaces: 1,
        //         watermark: 1,
        //         unitsName: 'ms'
        //     },

        //     // used VRAM, displayed using 2 colors - red for textures, green for geometry
        //     {
        //         name: 'VRAM',
        //         stats: ['vram.tex', 'vram.geom'],
        //         decimalPlaces: 1,
        //         multiplier: 1 / (1024 * 1024),
        //         unitsName: 'MB',
        //         watermark: 100
        //     },

        //     // frames per second
        //     {
        //         name: 'FPS',
        //         stats: ['frame.fps'],
        //         watermark: 60
        //     },

        //     // delta time
        //     {
        //         name: 'Frame',
        //         stats: ['frame.ms'],
        //         decimalPlaces: 1,           
        //         unitsName: 'ms',
        //         watermark: 33
        //     }
        // ];

        // // create mini-stats system
        // this.miniStats = new MiniStats(this.app, miniStatsOptions); // eslint-disable-line no-unused-vars

        // force render on device restored
        this.app.graphicsDevice.on('devicerestored', () => {
            this.forceRender = true;
        });

        // fire pre and post render events on the camera
        this.app.scene.on(EVENT_PRERENDER_LAYER, (camera: CameraComponent, layer: Layer, transparent: boolean) => {
            camera.fire('preRenderLayer', layer, transparent);
        });

        this.app.scene.on(EVENT_POSTRENDER_LAYER, (camera: CameraComponent, layer: Layer, transparent: boolean) => {
            camera.fire('postRenderLayer', layer, transparent);
        });

        // background layer
        // CRITICAL: Background layer should render first and not clear depth buffer
        this.backgroundLayer = new Layer({
            enabled: true,
            name: 'Background Layer',
            opaqueSortMode: SORTMODE_NONE,
            transparentSortMode: SORTMODE_NONE,
            clearDepthBuffer: false // Don't clear depth, so it renders behind everything
        });

        // shadow layer
        // this layer contains shadow caster scene mesh instances, shadow-casting
        // virtual light, shadow catching plane geometry and the main camera.
        this.shadowLayer = new Layer({
            name: 'Shadow Layer'
        });

        // debug layer
        this.debugLayer = new Layer({
            enabled: true,
            name: 'Debug Layer',
            opaqueSortMode: SORTMODE_NONE,
            transparentSortMode: SORTMODE_NONE
        });

        // overlay layer
        this.overlayLayer = new Layer({
            name: 'Overlay',
            clearDepthBuffer: false,
            opaqueSortMode: SORTMODE_NONE,
            transparentSortMode: SORTMODE_NONE
        });

        // gizmo layer
        this.gizmoLayer = new Layer({
            name: 'Gizmo',
            clearDepthBuffer: true,
            opaqueSortMode: SORTMODE_NONE,
            transparentSortMode: SORTMODE_NONE
        });

        const layers = this.app.scene.layers;
        const worldLayer = layers.getLayerByName('World');
        const idx = layers.getOpaqueIndex(worldLayer);
        layers.insert(this.backgroundLayer, idx);
        layers.insert(this.shadowLayer, idx + 1);
        layers.insert(this.debugLayer, idx + 1);
        layers.push(this.overlayLayer);
        layers.push(this.gizmoLayer);

        this.dataProcessor = new DataProcessor(this.app.graphicsDevice);
        this.assetLoader = new AssetLoader(this.app, events, this.app.graphicsDevice.maxAnisotropy);

        // create root entities
        this.contentRoot = new Entity('contentRoot');
        this.app.root.addChild(this.contentRoot);

        this.cameraRoot = new Entity('cameraRoot');
        this.app.root.addChild(this.cameraRoot);

        // create elements
        this.camera = new Camera();
        this.add(this.camera);

        this.splatOverlay = new SplatOverlay();
        this.add(this.splatOverlay);

        this.grid = new Grid();
        this.add(this.grid);

        this.outline = new Outline();
        this.add(this.outline);
        this.underlay = new Underlay();
        this.add(this.underlay);
    }

    start() {
        // start the app
        this.app.start();
    }

    clear() {
        const splats = this.getElementsByType(ElementType.splat);
        splats.forEach((splat) => {
            this.remove(splat);
            (splat as Splat).destroy();
        });
    }

    // add a scene element
    add(element: Element) {
        if (!element.scene) {
            // add the new element
            element.scene = this;
            element.add();
            this.elements.push(element);

            // notify all elements of scene addition
            this.forEachElement(e => e !== element && e.onAdded(element));

            // notify listeners
            this.events.fire('scene.elementAdded', element);
        }
    }

    // remove an element from the scene
    remove(element: Element) {
        if (element.scene === this) {
            // remove from list
            this.elements.splice(this.elements.indexOf(element), 1);

            // notify listeners
            this.events.fire('scene.elementRemoved', element);

            // notify all elements of scene removal
            this.forEachElement(e => e.onRemoved(element));

            element.remove();
            element.scene = null;
        }
    }

    // get the scene bound
    get bound() {
        if (this.boundDirty) {
            let valid = false;
            this.forEachElement((e) => {
                const bound = e.worldBound;
                if (bound) {
                    if (!valid) {
                        valid = true;
                        this.boundStorage.copy(bound);
                    } else {
                        this.boundStorage.add(bound);
                    }
                }
            });

            this.boundDirty = false;
            this.events.fire('scene.boundChanged', this.boundStorage);
        }

        return this.boundStorage;
    }

    getElementsByType(elementType: ElementType) {
        return this.elements.filter(e => e.type === elementType);
    }

    get graphicsDevice() {
        return this.app.graphicsDevice;
    }

    private forEachElement(action: (e: Element) => void) {
        this.elements.forEach(action);
    }

    private onUpdate(deltaTime: number) {
        // allow elements to update
        this.forEachElement(e => e.onUpdate(deltaTime));

        // fire global update
        this.events.fire('update', deltaTime);

        // fire a 'serialize' event which listers will use to store their state. we'll use
        // this to decide if the view has changed and so requires rendering.
        const i = this.app.frame % 2;
        const state = this.sceneState[i];
        state.reset();
        this.forEachElement(e => state.pack(e));

        // diff with previous state
        const result = state.compare(this.sceneState[1 - i]);

        // generate the set of all element types that changed
        const all = new Set([...result.added, ...result.removed, ...result.moved, ...result.changed]);

        // compare with previously serialized
        if (this.lockedRenderMode) {
            this.app.renderNextFrame = this.lockedRender;
            this.lockedRender = false;
        } else if (!this.app.renderNextFrame) {
            this.app.renderNextFrame = this.forceRender || all.size > 0;
        }
        this.forceRender = false;

        // raise per-type update events
        ElementTypeList.forEach((type) => {
            if (all.has(type)) {
                this.events.fire(`updated:${type}`);
            }
        });

        // allow elements to postupdate
        this.forEachElement(e => e.onPostUpdate());
    }

    private onPreRender() {
        if (this.canvasResize) {
            this.canvas.width = this.canvasResize.width;
            this.canvas.height = this.canvasResize.height;
            this.canvasResize = null;
        }

        // update render target size
        this.targetSize.width = Math.ceil(this.app.graphicsDevice.width / this.config.camera.pixelScale);
        this.targetSize.height = Math.ceil(this.app.graphicsDevice.height / this.config.camera.pixelScale);

        this.forEachElement(e => e.onPreRender());

        this.events.fire('prerender', this.camera.entity.getWorldTransform());

        // debug - display scene bound
        if (this.config.debug.showBound) {
            // draw element bounds
            this.forEachElement((e: Element) => {
                if (e.type === ElementType.splat) {
                    const splat = e as Splat;

                    const local = splat.localBound;
                    this.app.drawWireAlignedBox(
                        local.getMin(),
                        local.getMax(),
                        Color.RED,
                        true,
                        undefined,
                        splat.entity.getWorldTransform());

                    const world = splat.worldBound;
                    this.app.drawWireAlignedBox(
                        world.getMin(),
                        world.getMax(),
                        Color.GREEN);
                }
            });

            // draw scene bound
            this.app.drawWireAlignedBox(this.bound.getMin(), this.bound.getMax(), Color.BLUE);
        }
    }

    private onPostRender() {
        this.forEachElement(e => e.onPostRender());

        this.events.fire('postrender');
    }
}

export { SceneConfig, Scene };
