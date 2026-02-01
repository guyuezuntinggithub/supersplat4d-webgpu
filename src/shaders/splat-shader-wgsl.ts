/**
 * WebGPU (WGSL) 版本的 splat shader，与 splat-shader.ts 的 GLSL 逻辑对应。
 * 供 PlayCanvas 在 WebGPU 设备上使用；需在 rebuildMaterial 中同时设置 WGSL chunks。
 */

const vertexShaderWGSL = /* wgsl */ `
#include "gsplatCommonVS"

var splatState: texture_2d<f32>;
var selectedClr: vec4f;
var lockedClr: vec4f;
var clrOffset: vec3f;
var clrScale: vec4f;
var saturation: f32;

#ifdef PICK_PASS
  var pickMode: u32;
  var uFrameOnlyMode: bool;
#endif

const discardVec: vec4f = vec4f(0.0, 0.0, 2.0, 1.0);

struct VertexOutputCustom {
  @location(0) position: vec4f,
  @location(1) texCoordIsLocked: vec3f,
  @location(2) color: vec4f,
}

fn applySaturation(color: vec3f) -> vec3f {
  let grey = vec3f(dot(color, vec3f(0.299, 0.587, 0.114)));
  return grey + (color - grey) * saturation;
}

@vertex
fn vertexMain(input: VertexInput) -> VertexOutputCustom {
  var output: VertexOutputCustom;
  var source: SplatSource;
  if (!initSource(&source)) {
    output.position = discardVec;
    return output;
  }

  let vertexState = u32(textureLoad(splatState, source.uv, 0).r * 255.0 + 0.5) & 15u;

  #if OUTLINE_PASS
    if (vertexState != 1u) {
      output.position = discardVec;
      return output;
    }
  #elif UNDERLAY_PASS
    if (vertexState != 1u) {
      output.position = discardVec;
      return output;
    }
  #elif PICK_PASS
    if (pickMode == 0u) {
      if (vertexState != 0u) {
        output.position = discardVec;
        return output;
      }
    } else if (pickMode == 1u) {
      if (vertexState != 1u) {
        output.position = discardVec;
        return output;
      }
    } else {
      if ((vertexState & 6u) != 0u) {
        output.position = discardVec;
        return output;
      }
    }
    #ifdef DYNAMIC_MODE
    if (uFrameOnlyMode && uIsDynamic) {
      let trbfData = textureLoad(splatTrbf, source.uv, 0).rg;
      let trbfCenter = trbfData.r;
      let trbfScale = trbfData.g;
      let dt = (uCurrentTime - trbfCenter) / max(trbfScale, 1e-6);
      let gaussian = exp(-dt * dt);
      if (gaussian < 0.05) {
        output.position = discardVec;
        return output;
      }
    }
    #endif
  #else
    if ((vertexState & 4u) != 0u) {
      output.position = discardVec;
      return output;
    }
    #ifdef DYNAMIC_MODE
    if (uIsDynamic && (vertexState & 8u) != 0u) {
      output.position = discardVec;
      return output;
    }
    #endif
  #endif

  let modelCenter = readCenter(&source);
  var center: SplatCenter;
  if (!initCenter(&source, modelCenter, &center)) {
    output.position = discardVec;
    return output;
  }

  var corner: SplatCorner;
  if (!initCorner(&source, &center, &corner)) {
    output.position = discardVec;
    return output;
  }

  output.position = center.proj + vec4f(corner.offset, 0.0, 0.0);
  output.texCoordIsLocked = vec3f(corner.uv, select(0.0, 1.0, (vertexState & 2u) != 0u));

  #if UNDERLAY_PASS
    var clr = readColor(&source);
    clr.xyz = mix(clr.xyz, selectedClr.xyz * 0.2, selectedClr.a) * selectedClr.a;
    output.color = clr;
  #elif PICK_PASS
    output.color = vec4f(f32(source.id & 255u), f32((source.id >> 8u) & 255u), f32((source.id >> 16u) & 255u), f32((source.id >> 24u) & 255u)) / 255.0;
  #elif FORWARD_PASS
    var clr = readColor(&source);
    #ifdef DYNAMIC_MODE
    if (uIsDynamic) {
      let trbfData = textureLoad(splatTrbf, source.uv, 0).rg;
      let trbfCenter = trbfData.r;
      let trbfScale = trbfData.g;
      let dt = (uCurrentTime - trbfCenter) / max(trbfScale, 1e-6);
      let gaussian = exp(-dt * dt);
      clr.a *= gaussian;
      if (clr.a < 0.005) {
        output.position = discardVec;
        return output;
      }
    }
    #endif
    #if SH_BANDS > 0
    let modelView3x3 = mat3x3f(center.modelView[0].xyz, center.modelView[1].xyz, center.modelView[2].xyz);
    let dir = normalize(center.view * modelView3x3);
    var sh: array<vec3f, SH_COEFFS>;
    var scale: f32;
    readSHData(&source, &sh, &scale);
    clr.xyz += evalSH(&sh, dir) * scale;
    #endif
    clr = clr * clrScale + vec4f(clrOffset, 0.0);
    clr.xyz = applySaturation(clr.xyz);
    clr.a = clamp(clr.a, 0.0, 1.0);
    clr = vec4f(prepareOutputFromGamma(max(clr.xyz, vec3f(0.0))), clr.w);
    if ((vertexState & 2u) != 0u) {
      clr *= lockedClr;
    } else if ((vertexState & 1u) != 0u) {
      clr.xyz = mix(clr.xyz, selectedClr.xyz * 0.8, selectedClr.a);
    }
    output.color = clr;
  #endif

  return output;
}
`;

const fragmentShaderWGSL = /* wgsl */ `
struct FragmentInputCustom {
  @location(0) position: vec4f,
  @location(1) texCoordIsLocked: vec3f,
  @location(2) color: vec4f,
  @builtin(position) fragCoord: vec4f,
}

struct FragmentOutputCustom {
  @location(0) color: vec4f,
}

var<uniform> mode: i32;
var<uniform> pickerAlpha: f32;
var<uniform> ringSize: f32;

const EXP4 = exp(-4.0);
const INV_EXP4 = 1.0 / (1.0 - EXP4);

fn normExp(x: f32) -> f32 {
  return (exp(x * -4.0) - EXP4) * INV_EXP4;
}

@fragment
fn fragmentMain(input: FragmentInputCustom) -> FragmentOutputCustom {
  var output: FragmentOutputCustom;
  let A = dot(input.texCoordIsLocked.xy, input.texCoordIsLocked.xy);

  if (A > 1.0) {
    discard;
    return output;
  }

  #if OUTLINE_PASS
    output.color = vec4f(1.0, 1.0, 1.0, select(1.0, exp(-A * 4.0) * input.color.a, mode == 0));
  #else
    #ifdef PICK_PASS
      output.color = input.color;
    #else
      var alpha = normExp(A) * input.color.a;
      if (input.texCoordIsLocked.z == 0.0 && ringSize > 0.0) {
        if (A < 1.0 - ringSize) {
          alpha = max(0.05, alpha);
        } else {
          alpha = 0.6;
        }
      }
      output.color = vec4f(input.color.xyz * alpha, alpha);
    #endif
  #endif

  return output;
}
`;

const gsplatCenterWGSL = /* wgsl */ `
var<uniform> matrix_model: mat4x4f;
var<uniform> matrix_view: mat4x4f;
var<uniform> matrix_projection: mat4x4f;
var<uniform> camera_params: vec4f;              // 1/far, far, near, isOrtho (for engine gsplatCorner compatibility)
var splatTransform: texture_2d<u32>;
var transformPalette: texture_2d<f32>;
var<uniform> uCurrentTime: f32;
var<uniform> uIsDynamic: bool;

#ifdef DYNAMIC_MODE
  var splatMotion: texture_2d<f32>;
  var splatTrbf: texture_2d<f32>;
#endif

fn applyPaletteTransform(source: ptr<function, SplatSource>, model: mat4x4f) -> mat4x4f {
  let transformIndex = textureLoad(splatTransform, (*source).uv, 0).r;
  if (transformIndex == 0u) {
    return model;
  }
  let u = i32(transformIndex % 512u) * 3;
  let v = i32(transformIndex / 512u);
  let r0 = textureLoad(transformPalette, vec2i(u, v), 0);
  let r1 = textureLoad(transformPalette, vec2i(u + 1, v), 0);
  let r2 = textureLoad(transformPalette, vec2i(u + 2, v), 0);
  var t: mat4x4f;
  t[0] = r0;
  t[1] = r1;
  t[2] = r2;
  t[3] = vec4f(0.0, 0.0, 0.0, 1.0);
  return model * transpose(t);
}

fn computeDynamicPosition(source: ptr<function, SplatSource>, basePos: vec3f) -> vec3f {
  #ifndef DYNAMIC_MODE
  return basePos;
  #else
  if (!uIsDynamic) {
    return basePos;
  }
  let motionData = textureLoad(splatMotion, (*source).uv, 0);
  let trbfData = textureLoad(splatTrbf, (*source).uv, 0).rg;
  let motion = motionData.rgb;
  let trbfCenter = trbfData.r;
  let dt = uCurrentTime - trbfCenter;
  return basePos + motion * dt;
  #endif
}

fn initCenter(source: ptr<function, SplatSource>, modelCenter: vec3f, center: ptr<function, SplatCenter>) -> bool {
  let dynamicCenter = select(modelCenter, computeDynamicPosition(source, modelCenter), uIsDynamic);
  let modelView = matrix_view * applyPaletteTransform(source, matrix_model);
  var centerView = modelView * vec4f(dynamicCenter, 1.0);
  if (centerView.z > 0.0) {
    return false;
  }
  var centerProj = matrix_projection * centerView;
  centerProj.z = clamp(centerProj.z, -abs(centerProj.w), abs(centerProj.w));
  (*center).view = centerView.xyz / centerView.w;
  (*center).proj = centerProj;
  (*center).projMat00 = matrix_projection[0][0];
  (*center).modelView = modelView;
  return true;
}
`;

export { vertexShaderWGSL, fragmentShaderWGSL, gsplatCenterWGSL };
