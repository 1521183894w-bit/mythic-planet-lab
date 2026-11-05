"use strict";

const byId = id => document.getElementById(id);
const scene = byId("scene");
const manifest = JSON.parse(byId("world").textContent);
const radians = Math.PI / 180;
const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
const wrap = value => ((value + 180) % 360 + 360) % 360 - 180;
const views = [
  { name: "全球总览", lon: 35, lat: 23, zoom: 1 },
  { name: "主内海", lon: 6, lat: 36, zoom: 2.35 },
  { name: "西北海贸带", lon: -34, lat: 51, zoom: 2.75 },
  { name: "东方出海走廊", lon: 112, lat: 34, zoom: 3.45 },
  { name: "远方大陆", lon: -130, lat: -8, zoom: 1.62 },
  { name: "南极大陆", lon: 35, lat: -83, zoom: 1.55 },
];
const camera = { ...views[0] };
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
const pointers = new Map();
let layer = "relief";
let selected = 0;
let pin = null;
let flight = null;
let gestureMoved = false;
let renderer;
let frame = 0;

function basis() {
  const lon = camera.lon * radians;
  const lat = camera.lat * radians;
  const cl = Math.cos(lon), sl = Math.sin(lon), cp = Math.cos(lat), sp = Math.sin(lat);
  return [[-sl, 0, cl], [-sp * cl, cp, -sp * sl], [cp * cl, sp, cp * sl]];
}

function geometry() {
  const { clientWidth: width, clientHeight: height } = scene;
  const radius = Math.min(width * .41, height * .45) * camera.zoom;
  return { width, height, cx: width * .48, cy: height * .495, radius, axes: basis() };
}

function unit(lon, lat) {
  return [Math.cos(lat) * Math.cos(lon), Math.sin(lat), Math.cos(lat) * Math.sin(lon)];
}

function project(vector, geo) {
  const [x, y, front] = geo.axes.map(axis => axis.reduce((sum, v, i) => sum + v * vector[i], 0));
  return [geo.cx + geo.radius * x, geo.cy - geo.radius * y, front];
}

function pick(x, y, geo) {
  const u = (x - geo.cx) / geo.radius;
  const v = (geo.cy - y) / geo.radius;
  const depth = 1 - u * u - v * v;
  if (depth < 0) return null;
  const [right, up, front] = geo.axes;
  const vector = front.map((component, i) => u * right[i] + v * up[i] + Math.sqrt(depth) * component);
  return { lon: Math.atan2(vector[2], vector[0]) / radians, lat: Math.asin(clamp(vector[1], -1, 1)) / radians };
}

function webgl(canvas, images) {
  const gl = canvas.getContext("webgl", { alpha: true, premultipliedAlpha: false, antialias: false });
  if (!gl) throw new Error("WebGL 不可用");
  function compile(type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
    return shader;
  }
  const program = gl.createProgram();
  const vertex = compile(gl.VERTEX_SHADER, `
    attribute vec2 position;
    void main() { gl_Position = vec4(position, 0.0, 1.0); }
  `);
  const fragment = compile(gl.FRAGMENT_SHADER, `
    precision highp float;
    uniform sampler2D mapTexture;
    uniform mat3 axes;
    uniform vec2 center;
    uniform float radius;
    const float PI = 3.14159265359;
    void main() {
      vec2 xy = (gl_FragCoord.xy - center) / radius;
      float r = length(xy);
      if (r > 1.045) discard;
      if (r >= 1.0) {
        gl_FragColor = vec4(0.55, 0.71, 0.78, 0.36 * exp(-112.0 * (r - 1.0)));
        return;
      }
      float z = sqrt(max(0.0, 1.0 - dot(xy, xy)));
      vec3 p = axes * vec3(xy, z);
      vec2 uv = vec2(0.5 + atan(p.z, p.x) / (2.0 * PI), 0.5 - asin(clamp(p.y, -1.0, 1.0)) / PI);
      vec3 color = texture2D(mapTexture, uv).rgb;
      float light = 0.80 + 0.22 * max(0.0, dot(vec3(xy, z), vec3(-0.36, 0.46, 0.81)));
      float rim = 0.13 * pow(1.0 - z, 4.0);
      gl_FragColor = vec4(mix(color * light, vec3(0.51, 0.69, 0.76), rim), 1.0);
    }
  `);
  gl.attachShader(program, vertex);
  gl.attachShader(program, fragment);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
  gl.useProgram(program);
  gl.deleteShader(vertex);
  gl.deleteShader(fragment);

  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  const position = gl.getAttribLocation(program, "position");
  gl.enableVertexAttribArray(position);
  gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
  const uniforms = Object.fromEntries(["axes", "center", "radius"].map(name => [name, gl.getUniformLocation(program, name)]));
  const textures = {};
  for (const [name, image] of Object.entries(images)) {
    if (image.width > gl.getParameter(gl.MAX_TEXTURE_SIZE) || image.height > gl.getParameter(gl.MAX_TEXTURE_SIZE)) {
      throw new Error("球面贴图超出 WebGL 纹理限制");
    }
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
    const powerOfTwo = [image.width, image.height].every(size => (size & (size - 1)) === 0);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, powerOfTwo ? gl.REPEAT : gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    textures[name] = texture;
  }

  return (geo, name) => {
    const scale = Math.min(window.devicePixelRatio || 1, 1.5, 1920 / Math.max(geo.width, geo.height));
    const width = Math.round(geo.width * scale), height = Math.round(geo.height * scale);
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
      gl.viewport(0, 0, width, height);
    }
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.uniformMatrix3fv(uniforms.axes, false, geo.axes.flat());
    gl.uniform2f(uniforms.center, geo.cx * scale, (geo.height - geo.cy) * scale);
    gl.uniform1f(uniforms.radius, geo.radius * scale);
    gl.bindTexture(gl.TEXTURE_2D, textures[name]);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  };
}

function software(canvas, images) {
  const context = canvas.getContext("2d");
  const scratch = document.createElement("canvas");
  const reader = scratch.getContext("2d", { willReadFrequently: true });
  const rasters = {};
  for (const [name, image] of Object.entries(images)) {
    scratch.width = image.width;
    scratch.height = image.height;
    reader.drawImage(image, 0, 0);
    rasters[name] = reader.getImageData(0, 0, image.width, image.height);
  }
  let target;
  return (geo, name) => {
    const scale = Math.min(1, 620 / Math.max(geo.width, geo.height));
    const width = Math.round(geo.width * scale), height = Math.round(geo.height * scale);
    if (!target || target.width !== width || target.height !== height) {
      canvas.width = width;
      canvas.height = height;
      target = context.createImageData(width, height);
    }
    const raster = rasters[name];
    const [right, up, front] = geo.axes;
    const radius = geo.radius * scale;
    const cx = geo.cx * scale, cy = geo.cy * scale;
    const dst = target.data;
    dst.fill(0);
    for (let row = 0; row < height; row++) {
      const y = (cy - row - .5) / radius;
      if (Math.abs(y) >= 1) continue;
      const reach = radius * Math.sqrt(1 - y * y);
      const first = Math.max(0, Math.ceil(cx - reach - .5));
      const last = Math.min(width - 1, Math.floor(cx + reach - .5));
      for (let col = first; col <= last; col++) {
        const x = (col + .5 - cx) / radius;
        const z = Math.sqrt(Math.max(0, 1 - x * x - y * y));
        const wx = x * right[0] + y * up[0] + z * front[0];
        const wy = y * up[1] + z * front[1];
        const wz = x * right[2] + y * up[2] + z * front[2];
        const u = .5 + Math.atan2(wz, wx) / (2 * Math.PI);
        const v = .5 - Math.asin(clamp(wy, -1, 1)) / Math.PI;
        const ix = Math.floor(u * raster.width) % raster.width;
        const iy = Math.min(Math.floor(v * raster.height), raster.height - 1);
        const src = 4 * (iy * raster.width + ix);
        const off = 4 * (row * width + col);
        const light = .80 + .22 * Math.max(0, -.36 * x + .46 * y + .81 * z);
        for (let channel = 0; channel < 3; channel++) dst[off + channel] = raster.data[src + channel] * light;
        dst[off + 3] = 255;
      }
    }
    context.putImageData(target, 0, 0);
  };
}

function drawGrid(geo) {
  const svg = byId("graticule");
  if (svg.toggleAttribute("hidden", !byId("grid").checked)) return;
  const parts = [];
  function trace(from, to, vectorAt) {
    let visible = false;
    for (let angle = from; angle <= to; angle++) {
      const [x, y, front] = project(vectorAt(angle * radians), geo);
      if (front >= 0) parts.push(`${visible ? "L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`);
      visible = front >= 0;
    }
  }
  for (let lon = -180; lon < 180; lon += 30) trace(-90, 90, lat => unit(lon * radians, lat));
  for (let lat = -60; lat <= 60; lat += 30) trace(-180, 180, lon => unit(lon, lat * radians));
  byId("grid-lines").setAttribute("d", parts.join(" "));
}

function formatAngle(value, positive, negative) {
  return `${Math.abs(value).toFixed(1)}°${Math.abs(value) < .05 ? "" : value > 0 ? positive : negative}`;
}

function drawOverlays(geo) {
  drawGrid(geo);
  const marker = byId("pin");
  marker.hidden = true;
  if (pin) {
    const [x, y, front] = project(unit(pin.lon * radians, pin.lat * radians), geo);
    marker.hidden = front < 0 || x < 0 || x > geo.width || y < 0 || y > geo.height;
    marker.style.left = `${x}px`;
    marker.style.top = `${y}px`;
  }
  const point = pin || camera;
  byId("coordinates").textContent = `${pin ? "选点" : "中心"}  ${formatAngle(point.lat, "N", "S")}  ${formatAngle(point.lon, "E", "W")}`;
  const maxKm = 2 * manifest.radius_km * Math.asin(Math.min(110 / (2 * geo.radius), 1));
  const magnitude = 10 ** Math.floor(Math.log10(maxKm));
  const km = [5, 2, 1].find(value => value * magnitude <= maxKm) * magnitude;
  const scale = byId("scale");
  scale.style.width = `${2 * geo.radius * Math.sin(km / (2 * manifest.radius_km))}px`;
  scale.textContent = `${km.toLocaleString("en")} km`;
}

function draw(time) {
  frame = 0;
  if (flight) {
    const t = clamp((time - flight.start) / 540, 0, 1);
    const ease = t * t * (3 - 2 * t);
    camera.lon = wrap(flight.from.lon + flight.delta * ease);
    camera.lat = flight.from.lat + (flight.to.lat - flight.from.lat) * ease;
    camera.zoom = flight.from.zoom * (flight.to.zoom / flight.from.zoom) ** ease;
    if (t < 1) refresh();
    else flight = null;
  }
  const geo = geometry();
  if (geo.radius <= 0) return;
  renderer?.(geo, layer);
  drawOverlays(geo);
}

function refresh() {
  if (!frame) frame = requestAnimationFrame(draw);
}

function selectView(index) {
  selected = index;
  for (const [i, button] of Array.from(byId("views").children).entries()) button.setAttribute("aria-pressed", i === selected);
}

function go(index) {
  const target = views[index];
  selectView(index);
  pin = null;
  flight = null;
  if (reducedMotion.matches) Object.assign(camera, target);
  else flight = { from: { ...camera }, to: target, delta: wrap(target.lon - camera.lon), start: performance.now() };
  refresh();
}

function navigate(dx, dy, zoom = 1) {
  flight = null;
  if (selected !== -1) selectView(-1);
  const degreesPerPixel = 58 / Math.max(1, geometry().radius);
  camera.lon = wrap(camera.lon - dx * degreesPerPixel);
  camera.lat = clamp(camera.lat + dy * degreesPerPixel, -89.8, 89.8);
  camera.zoom = clamp(camera.zoom * zoom, .82, 3.8);
  refresh();
}

const layerToggle = byId("layer-toggle");
function showLayers(show) {
  byId("layers").hidden = !show;
  layerToggle.setAttribute("aria-expanded", show);
}

function setLayer(name) {
  layer = name;
  const chips = name === "relief"
    ? [["#2a4c63", "海洋"], ["#809266", "低地"], ["#b48b67", "山地"], ["#d4e1e5", "冰盖"]]
    : [["#1e3b4d", "海洋"], ["#bdb089", "陆地"], ["#497b96", "湖泊"], ["#629796", "咸湖"], ["#b4cbd7", "冰架"]];
  byId("legend").replaceChildren(...chips.map(([color, label]) => {
    const chip = document.createElement("span");
    const swatch = document.createElement("i");
    swatch.style.backgroundColor = color;
    chip.append(swatch, label);
    return chip;
  }));
  const method = manifest.layers[name];
  byId("layer-name").textContent = method.name;
  byId("method").textContent = name === "relief"
    ? `终图设计高程与设色 · ${method.method_class}`
    : `终图海陆设计 · ${method.method_class}`;
  refresh();
}

for (const [index, view] of views.entries()) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "view";
  button.innerHTML = `<span class="view-index">0${index + 1}</span>${view.name}<svg class="icon" aria-hidden="true"><use href="#arrow-icon"/></svg>`;
  button.addEventListener("click", () => go(index));
  byId("views").append(button);
}
selectView(0);
setLayer("relief");
byId("resolution").textContent = manifest.grid.join(" × ");
layerToggle.addEventListener("click", () => showLayers(byId("layers").hidden));
byId("layers").addEventListener("change", event => {
  if (event.target.name === "layer") setLayer(event.target.value);
  else refresh();
});
document.addEventListener("keydown", event => {
  if (event.key === "Escape" && !byId("layers").hidden) {
    showLayers(false);
    layerToggle.focus();
  }
});
byId("zoom-in").addEventListener("click", () => navigate(0, 0, 1.26));
byId("zoom-out").addEventListener("click", () => navigate(0, 0, 1 / 1.26));
byId("reset").addEventListener("click", () => go(0));

scene.addEventListener("wheel", event => {
  event.preventDefault();
  const multiplier = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? scene.clientHeight : 1;
  navigate(0, 0, Math.exp(clamp(-event.deltaY * multiplier * .0015, -.5, .5)));
}, { passive: false });
scene.addEventListener("pointerdown", event => {
  if (event.button !== 0 || pointers.size === 2) return;
  if (!pointers.size) gestureMoved = false;
  else gestureMoved = true;
  flight = null;
  scene.setPointerCapture(event.pointerId);
  pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
});
scene.addEventListener("pointermove", event => {
  const old = pointers.get(event.pointerId);
  if (!old) return;
  const x = event.clientX, y = event.clientY;
  if (!gestureMoved && Math.hypot(x - old.x, y - old.y) < 4) return;
  gestureMoved = true;
  pointers.set(event.pointerId, { x, y });
  if (pointers.size === 1) {
    navigate(x - old.x, y - old.y);
  } else {
    const other = [...pointers].find(([id]) => id !== event.pointerId)[1];
    const previous = Math.hypot(old.x - other.x, old.y - other.y);
    navigate((x - old.x) / 2, (y - old.y) / 2, Math.hypot(x - other.x, y - other.y) / Math.max(1, previous));
  }
});
scene.addEventListener("pointerup", event => {
  if (!pointers.delete(event.pointerId) || gestureMoved) return;
  const bounds = scene.getBoundingClientRect();
  pin = pick(event.clientX - bounds.left, event.clientY - bounds.top, geometry());
  refresh();
});
for (const event of ["pointercancel", "lostpointercapture"]) {
  scene.addEventListener(event, pointer => pointers.delete(pointer.pointerId));
}
scene.addEventListener("keydown", event => {
  const moves = { ArrowUp: [0, 38], ArrowDown: [0, -38], ArrowLeft: [38, 0], ArrowRight: [-38, 0],
    "+": [0, 0, 1.26], "=": [0, 0, 1.26], "-": [0, 0, 1 / 1.26] };
  if (event.key === "Home") go(0);
  else if (Object.hasOwn(moves, event.key)) navigate(...moves[event.key]);
  else return;
  event.preventDefault();
});
new ResizeObserver(refresh).observe(scene);

async function initialize() {
  const images = Object.fromEntries(await Promise.all(Object.entries(manifest.textures).map(async ([name, source]) => {
    const image = new Image();
    image.src = source;
    await image.decode();
    if (image.width !== manifest.grid[0] || image.height !== manifest.grid[1]) throw new Error("格网尺寸与清单不一致");
    return [name, image];
  })));
  const canvas = byId("surface");
  function useSoftware() {
    const fallback = canvas.cloneNode();
    canvas.replaceWith(fallback);
    renderer = software(fallback, images);
    fallback.dataset.renderer = "canvas2d";
    refresh();
  }
  try {
    renderer = webgl(canvas, images);
    canvas.dataset.renderer = "webgl";
    canvas.addEventListener("webglcontextlost", useSoftware, { once: true });
  } catch (error) {
    console.info("启用 CPU 球面绘制", error.message);
    useSoftware();
  }
  byId("status").hidden = true;
  refresh();
}

initialize().catch(error => {
  byId("status").textContent = "球面数据加载失败。请重新生成终图地球仪。";
  console.error(error);
});
