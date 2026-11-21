// 渲染着色器：顶点变换 + 物理感光照（含 AO、半透明 PETG、色谱、双面）
struct Camera {
  viewProj: mat4x4<f32>,
  eye: vec4<f32>,
};

struct Material {
  baseColor: vec4<f32>,   // rgb + opacity
  mode: f32,              // 0=实体材质 1=顶点色谱 2=扫描细节
  aoStrength: f32,
  specular: f32,
  _pad: f32,
};

@group(0) @binding(0) var<uniform> cam: Camera;
@group(1) @binding(0) var<uniform> mat: Material;
@group(1) @binding(1) var<storage, read> positions: array<f32>;
@group(1) @binding(2) var<storage, read> normals: array<f32>;
@group(1) @binding(3) var<storage, read> colors: array<f32>;
@group(1) @binding(4) var<storage, read> aoVals: array<f32>;
@group(1) @binding(6) var<storage, read> indexBuf: array<u32>;

struct VSOut {
  @builtin(position) clip: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) vColor: vec3<f32>,
  @location(3) ao: f32,
};

@vertex
fn vsMain(@builtin(vertex_index) vi: u32) -> VSOut {
  let realIndex = indexBuf[vi];
  let p = vec3<f32>(positions[realIndex * 3u], positions[realIndex * 3u + 1u], positions[realIndex * 3u + 2u]);
  let n = vec3<f32>(normals[realIndex * 3u], normals[realIndex * 3u + 1u], normals[realIndex * 3u + 2u]);
  var out: VSOut;
  out.clip = cam.viewProj * vec4<f32>(p, 1.0);
  out.worldPos = p;
  out.normal = n;
  out.vColor = vec3<f32>(colors[realIndex * 3u], colors[realIndex * 3u + 1u], colors[realIndex * 3u + 2u]);
  out.ao = aoVals[realIndex];
  return out;
}

// 三点光（主光 + 补光 + 轮廓光），背面两面对称照
@fragment
fn fsMain(in: VSOut) -> @location(0) vec4<f32> {
  let n = normalize(in.normal);
  let viewDir = normalize(cam.eye.xyz - in.worldPos);
  let key = normalize(vec3<f32>(0.5, 0.9, 0.6));
  let fill = normalize(vec3<f32>(-0.6, 0.25, -0.4));
  let rim = normalize(vec3<f32>(-0.2, 0.4, 0.9));
  let diff = max(dot(n, key), 0.0) * 0.72
           + max(dot(n, fill), 0.0) * 0.22
           + max(dot(n, rim), 0.0) * 0.18;
  let halfV = normalize(key + viewDir);
  let spec = pow(max(dot(n, halfV), 0.0), 48.0) * mat.specular;
  var albedo = mat.baseColor.rgb;
  if (mat.mode > 0.5) {
    albedo = in.vColor;
  }
  let aoFactor = mix(1.0, in.ao, mat.aoStrength);
  let shaded = albedo * (0.22 + diff * 0.78) * aoFactor + vec3<f32>(spec);
  return vec4<f32>(shaded, mat.baseColor.a);
}

// 线框/截面轮廓/标记点渲染（纯色无光照）
struct LineMaterial {
  color: vec4<f32>,
  _a: f32,
  _b: f32,
  _c: f32,
};
@group(1) @binding(5) var<uniform> lineMat: LineMaterial;

@fragment
fn fsLine(in: VSOut) -> @location(0) vec4<f32> {
  return lineMat.color;
}
