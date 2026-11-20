/**
 * 轻量 WebGL2 网格渲染器（零三方依赖）。
 * 支持：轨道相机、顶点色/材质色、半透明 PETG 与不透明打印料切换、
 * 屏幕空间环境光遮蔽（SSAO）+ 半球/方向光。
 * 渲染走 WebGL2（普及度高），通用计算走 WebGPU（见 gpu/GpuCompute）。
 */

export interface RenderMesh {
  id: string;
  positions: Float32Array;
  indices: Uint32Array;
  normals?: Float32Array;
  colors?: Float32Array;
  material: {
    /** 不透明度 1=不透明打印料，~0.35=半透明 PETG */
    alpha: number;
    baseColor: [number, number, number];
    useVertexColor: boolean;
    visible: boolean;
  };
}

const DEPTH_NORMALS_VS = `#version 300 es
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
uniform mat4 uVP;
uniform mat4 uModel;
out vec3 vNormal;
out vec3 vWorldPos;
void main(){
  vec4 wp = uModel * vec4(aPos,1.0);
  vWorldPos = wp.xyz;
  vNormal = mat3(uModel) * aNormal;
  gl_Position = uVP * wp;
}`;

const DEPTH_NORMALS_FS = `#version 300 es
precision highp float;
in vec3 vNormal;
in vec3 vWorldPos;
uniform float uDebugGBuffer;
layout(location=0) out vec4 outPos;
layout(location=1) out vec4 outNormal;
void main(){
  if (uDebugGBuffer > 0.5) {
    outPos = vec4(1.0, 0.5, 0.25, 1.0);
    outNormal = vec4(1.0);
    return;
  }
  outPos = vec4(vWorldPos, gl_FragCoord.z);
  outNormal = vec4(normalize(vNormal) * 0.5 + 0.5, 1.0);
}`;

const SSAO_FS = `#version 300 es
precision highp float;
uniform sampler2D tPos;
uniform sampler2D tNormal;
uniform vec2 uTexel;
uniform mat4 uProj;
uniform float uRadius;
out float outOcclusion;
void main(){
  vec2 uv = gl_FragCoord.xy * uTexel;
  vec3 pos = texture(tPos, uv).xyz;
  vec3 n = normalize(texture(tNormal, uv).xyz * 2.0 - 1.0);
  float occ = 0.0;
  // 固定 16 核螺旋采样
  for(int i=0;i<16;i++){
    float fi = float(i);
    vec3 sample3 = normalize(vec3(
      fract(sin(fi*12.9898)*43758.5453),
      fract(sin(fi*78.233)*12543.134),
      fract(sin(fi*37.719)*23421.631)
    ) * 2.0 - 1.0);
    sample3 = mix(sample3, n, 0.6);
    vec3 sp = pos + sample3 * uRadius;
    vec4 clip = uProj * vec4(sp,1.0);
    vec2 suv = (clip.xy/clip.w)*0.5+0.5;
    vec3 occPos = texture(tPos, suv).xyz;
    float rangeCheck = smoothstep(1.0, 0.0, abs(length(pos-occPos))/uRadius);
    occ += (occPos.z >= sp.z ? 1.0 : 0.0) * rangeCheck;
  }
  outOcclusion = 1.0 - occ/16.0;
}`;

const SCENE_VS = `#version 300 es
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec3 aColor;
uniform mat4 uVP;
uniform mat4 uModel;
out vec3 vNormal;
out vec3 vColor;
out vec3 vWorldPos;
void main(){
  vec4 wp = uModel * vec4(aPos,1.0);
  vWorldPos = wp.xyz;
  vNormal = mat3(uModel) * aNormal;
  vColor = aColor;
  gl_Position = uVP * wp;
}`;

const SCENE_FS = `#version 300 es
precision highp float;
in vec3 vNormal;
in vec3 vColor;
in vec3 vWorldPos;
uniform vec3 uCamPos;
uniform vec3 uBaseColor;
uniform float uUseVertexColor;
uniform float uAlpha;
uniform float uDebug;
uniform sampler2D tSSAO;
uniform vec2 uTexel;
out vec4 frag;
void main(){
  vec3 n = normalize(vNormal);
  vec3 V = normalize(uCamPos - vWorldPos);
  // 半球环境光 + 主方向光
  vec3 sky = vec3(0.62,0.70,0.82);
  vec3 ground = vec3(0.30,0.27,0.25);
  float hemi = n.y*0.5+0.5;
  vec3 ambient = mix(ground, sky, hemi);
  vec3 L = normalize(vec3(0.45,0.8,0.35));
  float diff = max(dot(n,L),0.0);
  vec3 H = normalize(L+V);
  float spec = pow(max(dot(n,H),0.0), 48.0) * 0.35;
  vec3 base = mix(uBaseColor, vColor, uUseVertexColor);
  // 调试模式：输出未照亮的 base 色，排查光照/法线链路
  if (uDebug > 0.5) { frag = vec4(base, uAlpha); return; }
  float ao = texture(tSSAO, gl_FragCoord.xy*uTexel).r;
  // 调试：ao 异常（全黑）时退化为 1，避免整体黑场
  ao = ao < 0.02 ? 1.0 : ao;
  vec3 lit = base * (ambient*0.65 + vec3(1.0)*diff*0.7) * ao + vec3(spec);
  frag = vec4(lit, uAlpha);
}`;

function createColorTexture(
  gl: WebGL2RenderingContext, internal: number, format: number, type: number,
  w: number, h: number, filter: number = gl.NEAREST,
): WebGLTexture {
  const tex = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, internal, w, h, 0, format, type, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
  return tex;
}

function compile(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
  const sh = gl.createShader(type)!;
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    throw new Error('着色器编译失败: ' + gl.getShaderInfoLog(sh));
  }
  return sh;
}

function program(gl: WebGL2RenderingContext, vs: string, fs: string): WebGLProgram {
  const p = gl.createProgram()!;
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vs));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    throw new Error('着色器链接失败: ' + gl.getProgramInfoLog(p));
  }
  return p;
}

/** 透视投影（列主序，直接可用于 uniformMatrix4fv transpose=false） */
function perspective(fovy: number, aspect: number, near: number, far: number): Float32Array {
  const f = 1 / Math.tan(fovy / 2);
  const nf = 1 / (near - far);
  // 列主序：元素 [col*4 + row]
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, 2 * far * near * nf, 0,
  ]);
}

/** 视图矩阵 lookAt（列主序） */
function lookAt(eye: number[], center: number[], up: number[]): Float32Array {
  const z = norm3(sub3(eye, center));
  const x = norm3(cross3(up, z));
  const y = cross3(z, x);
  // 行主序的旋转平移 R|t，转置为列主序：基础向量按列存放
  return new Float32Array([
    x[0], x[1], x[2], 0,
    y[0], y[1], y[2], 0,
    z[0], z[1], z[2], 0,
    -dot3(x, eye), -dot3(y, eye), -dot3(z, eye), 1,
  ]);
}

function sub3(a: number[], b: number[]): number[] { return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]; }
function dot3(a: number[], b: number[]): number { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
function cross3(a: number[], b: number[]): number[] {
  return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
}
function norm3(a: number[]): number[] { const l = Math.hypot(a[0],a[1],a[2])||1; return [a[0]/l,a[1]/l,a[2]/l]; }
/** 列主序矩阵乘法 out = a * b（a、b 均为列主序） */
function mul4(a: Float32Array, b: Float32Array): Float32Array {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      let sum = 0;
      for (let k = 0; k < 4; k++) sum += a[k * 4 + r] * b[c * 4 + k];
      o[c * 4 + r] = sum;
    }
  }
  return o;
}
const IDENTITY = new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);

export class MeshRenderer {
  private gl: WebGL2RenderingContext;
  private progGBuffer!: WebGLProgram;
  private progSSAO!: WebGLProgram;
  private progScene!: WebGLProgram;
  private meshes: RenderMesh[] = [];
  private vao = new Map<string, { vao: WebGLVertexArrayObject; count: number }>();
  // 相机
  private yaw = 0.6;
  private pitch = 0.25;
  private distance = 900;
  private center: number[] = [0, 0, 0];
  private dragging = false;
  private lastX = 0;
  private lastY = 0;
  // 离屏缓冲
  private gBuffer!: { fbo: WebGLFramebuffer; posTex: WebGLTexture; normalTex: WebGLTexture; depth: WebGLRenderbuffer };
  private ssaoFbo!: { fbo: WebGLFramebuffer; tex: WebGLTexture };
  private whiteTex: WebGLTexture | null = null;
  private aoRadius = 28;
  /** SSAO 开关。浮点 G-Buffer 在个别驱动异常时可关闭，场景光照不受影响 */
  ssaoEnabled = true;
  /** 调试：跳过光照直接输出 base 色 */
  debug = false;
  /** 调试：G-buffer 输出固定色，验证光栅化 */
  debugGBuffer = false;

  constructor(private canvas: HTMLCanvasElement) {
    const gl = canvas.getContext('webgl2', { antialias: true, alpha: false });
    if (!gl) throw new Error('当前环境不支持 WebGL2');
    this.gl = gl;
    // 浮点颜色纹理作为渲染目标需要显式启用该扩展
    const ext = gl.getExtension('EXT_color_buffer_float');
    if (!ext) console.warn('[Render] EXT_color_buffer_float 不可用，SSAO 浮点 G-Buffer 可能失败');
    this.initPrograms();
    this.initGBuffer();
    this.bindCamera();
  }

  private initPrograms(): void {
    const gl = this.gl;
    this.progGBuffer = program(gl, DEPTH_NORMALS_VS, DEPTH_NORMALS_FS);
    // SSAO 只有片元，配一个全屏三角顶点
    const fullVs = `#version 300 es
      void main(){ vec2 p = vec2((gl_VertexID<<1)&2, gl_VertexID&2); gl_Position = vec4(p*2.0-1.0,0.0,1.0); }`;
    this.progSSAO = program(gl, fullVs, SSAO_FS);
    this.progScene = program(gl, SCENE_VS, SCENE_FS);
  }

  private initGBuffer(): void {
    const gl = this.gl;
    const w = this.canvas.width, h = this.canvas.height;
    // 位置用单个浮点附件（单浮点 RT 普遍支持），法线用 RGBA8，规避双浮点 MRT 兼容问题
    const posTex = createColorTexture(gl, gl.RGBA16F, gl.RGBA, gl.FLOAT, w, h);
    const normalTex = createColorTexture(gl, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, w, h);
    const depth = gl.createRenderbuffer()!;
    gl.bindRenderbuffer(gl.RENDERBUFFER, depth);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, w, h);
    const fbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, posTex, 0);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, normalTex, 0);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, depth);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
    const gStatus = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (gStatus !== gl.FRAMEBUFFER_COMPLETE) {
      console.warn('[Render] G-Buffer 帧缓冲不完整，状态码 0x' + gStatus.toString(16));
    }
    this.gBuffer = { fbo, posTex, normalTex, depth };

    const aoTex = createColorTexture(gl, gl.R8, gl.RED, gl.UNSIGNED_BYTE, w, h, gl.LINEAR);
    const aoFbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, aoFbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, aoTex, 0);
    this.ssaoFbo = { fbo: aoFbo, tex: aoTex };
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  private bindCamera(): void {
    const canvas = this.canvas;
    canvas.onmousedown = (e) => { this.dragging = true; this.lastX = e.clientX; this.lastY = e.clientY; };
    window.addEventListener('mouseup', () => { this.dragging = false; });
    window.addEventListener('mousemove', (e) => {
      if (!this.dragging) return;
      this.yaw -= (e.clientX - this.lastX) * 0.01;
      this.pitch = Math.max(-1.4, Math.min(1.4, this.pitch + (e.clientY - this.lastY) * 0.01));
      this.lastX = e.clientX; this.lastY = e.clientY;
    });
    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.distance = Math.max(120, Math.min(2500, this.distance * (1 + Math.sign(e.deltaY) * 0.08)));
    }, { passive: false });
  }

  setMeshes(meshes: RenderMesh[]): void {
    this.meshes = meshes;
    const gl = this.gl;
    this.vao.forEach(({ vao }) => gl.deleteVertexArray(vao));
    this.vao.clear();
    for (const m of meshes) {
      const vao = gl.createVertexArray()!;
      gl.bindVertexArray(vao);
      const vbo = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
      // 交错 pos(3)+normal(3)+color(3)，缺省补默认
      const V = m.positions.length / 3;
      const interleaved = new Float32Array(V * 9);
      for (let v = 0; v < V; v++) {
        interleaved[v*9] = m.positions[v*3];
        interleaved[v*9+1] = m.positions[v*3+1];
        interleaved[v*9+2] = m.positions[v*3+2];
        interleaved[v*9+3] = m.normals ? m.normals[v*3] : 0;
        interleaved[v*9+4] = m.normals ? m.normals[v*3+1] : 1;
        interleaved[v*9+5] = m.normals ? m.normals[v*3+2] : 0;
        interleaved[v*9+6] = m.colors ? m.colors[v*3] : 0.85;
        interleaved[v*9+7] = m.colors ? m.colors[v*3+1] : 0.85;
        interleaved[v*9+8] = m.colors ? m.colors[v*3+2] : 0.85;
      }
      gl.bufferData(gl.ARRAY_BUFFER, interleaved, gl.STATIC_DRAW);
      const stride = 9 * 4;
      gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, stride, 0);
      gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.FLOAT, false, stride, 3*4);
      gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 3, gl.FLOAT, false, stride, 6*4);
      const ibo = gl.createBuffer();
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, m.indices, gl.STATIC_DRAW);
      gl.bindVertexArray(null);
      this.vao.set(m.id, { vao, count: m.indices.length });
    }
  }

  /** 调试：扫描 G-buffer 非零位置像素数量与一个样本 */
  debugScanGBuffer(): { nonzero: number; sample: number[] } {
    const gl = this.gl;
    const w = this.canvas.width, h = this.canvas.height;
    const buf = new Float32Array(w * h * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.gBuffer.fbo);
    gl.readBuffer(gl.COLOR_ATTACHMENT0);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.FLOAT, buf);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    let nonzero = 0;
    let sample: number[] = [];
    for (let i = 0; i < buf.length; i += 4) {
      if (buf[i] !== 0 || buf[i + 1] !== 0 || buf[i + 2] !== 0) {
        nonzero++;
        if (sample.length === 0) sample = [buf[i], buf[i + 1], buf[i + 2]];
      }
    }
    return { nonzero, sample };
  }

  /** 调试：读取 G-buffer 位置附件中心像素 */
  debugReadGBuffer(): number[] {
    const gl = this.gl;
    const buf = new Float32Array(4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.gBuffer.fbo);
    gl.readBuffer(gl.COLOR_ATTACHMENT0);
    gl.readPixels(this.canvas.width >> 1, this.canvas.height >> 1, 1, 1, gl.RGBA, gl.FLOAT, buf);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return Array.from(buf);
  }

  setCenter(c: number[]): void { this.center = c; }

  private getWhiteTexture(): WebGLTexture {
    if (this.whiteTex) return this.whiteTex;
    const gl = this.gl;
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, 1, 1, 0, gl.RED, gl.UNSIGNED_BYTE, new Uint8Array([255]));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    this.whiteTex = tex;
    return tex;
  }

  private viewProj(): { vp: Float32Array; camPos: number[]; proj: Float32Array } {
    const eye = [
      this.center[0] + this.distance * Math.cos(this.pitch) * Math.sin(this.yaw),
      this.center[1] + this.distance * Math.sin(this.pitch),
      this.center[2] + this.distance * Math.cos(this.pitch) * Math.cos(this.yaw),
    ];
    const proj = perspective(Math.PI / 4, this.canvas.width / this.canvas.height, 5, 6000);
    const view = lookAt(eye, this.center, [0, 1, 0]);
    return { vp: mul4(proj, view), camPos: eye, proj };
  }

  render(): void {
    const gl = this.gl;
    const { vp, camPos, proj } = this.viewProj();
    const texel = new Float32Array([1 / this.canvas.width, 1 / this.canvas.height]);
    const visible = this.meshes.filter((m) => m.material.visible);

    // Pass 1: G-Buffer（位置+法线+深度）
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.gBuffer.fbo);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
      // 实时诊断：打印十进制状态码与画布尺寸
      console.warn('[Render] 运行时 G-Buffer 不完整', gl.checkFramebufferStatus(gl.FRAMEBUFFER), this.canvas.width, this.canvas.height);
    }
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.useProgram(this.progGBuffer);
    const gbVP = gl.getUniformLocation(this.progGBuffer, 'uVP');
    const gbModel = gl.getUniformLocation(this.progGBuffer, 'uModel');
    gl.uniformMatrix4fv(gbVP, false, vp);
    gl.uniformMatrix4fv(gbModel, false, IDENTITY);
    gl.uniform1f(gl.getUniformLocation(this.progGBuffer, 'uDebugGBuffer'), this.debugGBuffer ? 1 : 0);
    for (const m of visible) {
      const vao = this.vao.get(m.id);
      if (!vao) continue;
      gl.bindVertexArray(vao.vao);
      gl.drawElements(gl.TRIANGLES, vao.count, gl.UNSIGNED_INT, 0);
    }

    // Pass 2: SSAO（可关闭；关闭时场景用全白遮蔽纹理，即不产生暗化）
    if (this.ssaoEnabled) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.ssaoFbo.fbo);
      gl.disable(gl.DEPTH_TEST);
      gl.useProgram(this.progSSAO);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.gBuffer.posTex);
      gl.uniform1i(gl.getUniformLocation(this.progSSAO, 'tPos'), 0);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, this.gBuffer.normalTex);
      gl.uniform1i(gl.getUniformLocation(this.progSSAO, 'tNormal'), 1);
      gl.uniform2fv(gl.getUniformLocation(this.progSSAO, 'uTexel'), texel);
      gl.uniformMatrix4fv(gl.getUniformLocation(this.progSSAO, 'uProj'), false, proj);
      gl.uniform1f(gl.getUniformLocation(this.progSSAO, 'uRadius'), this.aoRadius);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }

    // Pass 3: 场景着色（先不透明后透明）
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0.12, 0.13, 0.15, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.useProgram(this.progScene);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.progScene, 'uVP'), false, vp);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.progScene, 'uModel'), false, IDENTITY);
    gl.uniform3fv(gl.getUniformLocation(this.progScene, 'uCamPos'), new Float32Array(camPos));
    gl.uniform2fv(gl.getUniformLocation(this.progScene, 'uTexel'), texel);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.ssaoEnabled ? this.ssaoFbo.tex : this.getWhiteTexture());
    gl.uniform1i(gl.getUniformLocation(this.progScene, 'tSSAO'), 0);

    const opaque = visible.filter((m) => m.material.alpha >= 0.999);
    const transparent = visible.filter((m) => m.material.alpha < 0.999);
    const drawOne = (m: RenderMesh) => {
      const vao = this.vao.get(m.id);
      if (!vao) return;
      gl.uniform3fv(gl.getUniformLocation(this.progScene, 'uBaseColor'), new Float32Array(m.material.baseColor));
      gl.uniform1f(gl.getUniformLocation(this.progScene, 'uUseVertexColor'), m.material.useVertexColor ? 1 : 0);
      gl.uniform1f(gl.getUniformLocation(this.progScene, 'uAlpha'), m.material.alpha);
      gl.uniform1f(gl.getUniformLocation(this.progScene, 'uDebug'), this.debug ? 1 : 0);
      gl.bindVertexArray(vao.vao);
      gl.drawElements(gl.TRIANGLES, vao.count, gl.UNSIGNED_INT, 0);
    };
    gl.enable(gl.DEPTH_TEST);
    gl.depthMask(true);
    opaque.forEach(drawOne);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.depthMask(false);
    transparent.forEach(drawOne);
    gl.depthMask(true);
    gl.disable(gl.BLEND);
    gl.bindVertexArray(null);
  }

  /** 射线拾取：返回最近命中的网格 id 与世界坐标（标记点拖动用） */
  pick(clientX: number, clientY: number): { meshId: string; point: number[] } | null {
    const rect = this.canvas.getBoundingClientRect();
    const nx = ((clientX - rect.left) / rect.width) * 2 - 1;
    const ny = -(((clientY - rect.top) / rect.height) * 2 - 1);
    // 简化：在每个网格上做射线-三角形，取最近（足够标记点交互用）
    const { camPos } = this.viewProj();
    const dir = norm3([nx, ny, -1.2]);
    let best: { meshId: string; point: number[]; t: number } | null = null;
    for (const m of this.meshes) {
      if (!m.material.visible) continue;
      for (let f = 0; f < m.indices.length; f += 3) {
        const tri = [m.indices[f], m.indices[f+1], m.indices[f+2]];
        const a = tri.map((v) => [m.positions[v*3], m.positions[v*3+1], m.positions[v*3+2]]);
        const hit = rayTri(camPos, dir, a[0], a[1], a[2]);
        if (hit && (!best || hit.t < best.t)) best = { meshId: m.id, point: hit.point, t: hit.t };
      }
    }
    return best ? { meshId: best.meshId, point: best.point } : null;
  }
}

function rayTri(o: number[], d: number[], a: number[], b: number[], c: number[]): { t: number; point: number[] } | null {
  const e1 = sub3(b, a), e2 = sub3(c, a);
  const p = cross3(d, e2);
  const det = dot3(p, e1);
  if (Math.abs(det) < 1e-8) return null;
  const tvec = sub3(o, a);
  const u = dot3(p, tvec) / det;
  if (u < 0 || u > 1) return null;
  const q = cross3(tvec, e1);
  const v = dot3(q, d) / det;
  if (v < 0 || u + v > 1) return null;
  const t = dot3(e2, q) / det;
  if (t < 0) return null;
  const point = [o[0]+d[0]*t, o[1]+d[1]*t, o[2]+d[2]*t];
  return { t, point };
}
