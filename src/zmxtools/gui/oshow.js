import * as THREE from 'three';

let SCENE_DATA = __SCENE__;
const SERVED = __SERVED__;
const root = document.documentElement;
const view = document.getElementById('view');
const panel = document.getElementById('panel');

function css(name, fallback = ''){
  return getComputedStyle(root).getPropertyValue(name).trim() || fallback;
}
function cssNumber(name, fallback){
  const value = Number.parseFloat(css(name));
  return Number.isFinite(value) ? value : fallback;
}

const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(view.clientWidth || innerWidth, view.clientHeight || innerHeight, false);
view.prepend(renderer.domElement);
const canvas = renderer.domElement;
canvas.style.touchAction = 'none';

const camera = new THREE.PerspectiveCamera(cssNumber('--view-fov', 50), 1, 1e-5, 100);
const target = new THREE.Vector3();

const scene = new THREE.Scene();
const THEME_LIGHT = window.matchMedia && matchMedia('(prefers-color-scheme: light)').matches;
let theme = THEME_LIGHT ? 'light' : 'dark';
function applyTheme(){
  root.dataset.theme = theme;
  scene.background = new THREE.Color(css('--scene-bg', '#101018'));
}
document.getElementById('theme').addEventListener('click', () => {
  theme = (theme === 'dark') ? 'light' : 'dark';
  applyTheme();
});
applyTheme();
scene.add(new THREE.AmbientLight(css('--light-color', '#ffffff'), 0.7));
const key = new THREE.DirectionalLight(css('--light-color', '#ffffff'), 0.9);
key.position.set(0.3, -0.4, 0.6).multiplyScalar(0.1);
scene.add(key);

let presetAxis = 'x';
let autoFit = true;
let state = {surfaces:new Map(), rims:new Map(), rays:[], image:null};

function addLine(a, b, color, linewidth = 2){
  const g = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(...a), new THREE.Vector3(...b)]);
  const line = new THREE.Line(g, new THREE.LineBasicMaterial({color:color, linewidth:linewidth}));
  line.userData.dyn = true;
  scene.add(line);
}
function addArrow(origin, direction, length, color){
  const dir = new THREE.Vector3(...direction);
  if(length <= 0 || dir.lengthSq() === 0){
    return;
  }
  dir.normalize();
  const width = cssNumber('--axis-linewidth', 2);
  const arrow = new THREE.ArrowHelper(
    dir,
    new THREE.Vector3(...origin),
    length,
    color,
    Math.max(length * 0.12, 1e-12),
    Math.max(length * 0.055, 1e-12),
  );
  arrow.line.material.linewidth = width;
  arrow.line.material.color.set(color);
  arrow.cone.material.color.set(color);
  arrow.userData.dyn = true;
  scene.add(arrow);
}
function addTo(map, key, object){
  if(!map.has(key)){
    map.set(key, []);
  }
  map.get(key).push(object);
}
function disposeObject(object){
  object.traverse(child => {
    if(child.geometry){
      child.geometry.dispose();
    }
    if(Array.isArray(child.material)){
      child.material.forEach(material => material.dispose());
    }else if(child.material){
      child.material.dispose();
    }
  });
}
function disposeDynamic(){
  for(const child of scene.children.filter(c => c.userData.dyn)){
    scene.remove(child);
    disposeObject(child);
  }
}
function resetHighlight(){
  for(const list of state.surfaces.values()){
    for(const mesh of list){
      mesh.material.color.setHex(mesh.userData.baseColor);
      mesh.material.emissive.setHex(0x000000);
      mesh.material.emissiveIntensity = 0;
      if(mesh.userData.baseSide !== undefined){
        mesh.material.side = mesh.userData.baseSide;
      }
    }
  }
  for(const list of state.rims.values()){
    for(const item of list){
      if(item.userData.baseColor !== undefined){
        item.material.color.setHex(item.userData.baseColor);
      }
      if(item.material.emissive){
        item.material.emissive.setHex(0x000000);
        item.material.emissiveIntensity = 0;
      }
      if(item.userData.baseSide !== undefined && item.isMesh){
        item.material.side = item.userData.baseSide;
      }
    }
  }
  for(const line of state.rays){
    line.material.color.setHex(line.userData.baseColor);
  }
  if(state.image){
    state.image.material.color.setHex(state.image.userData.baseColor);
  }
}
function applyHighlight(target){
  if(!target){
    return;
  }
  const color = css('--highlight-color', '#ff0000');
  if(target.kind === 'rays'){
    state.rays.forEach(line => line.material.color.set(color));
  }else if(target.kind === 'surface'){
    (state.surfaces.get(target.id) || []).forEach(mesh => {
      mesh.material.color.set(color);
      mesh.material.emissive.set(color);
      mesh.material.emissiveIntensity = 0.4;
      mesh.material.side = THREE.DoubleSide;
    });
  }else if(target.kind === 'rim'){
    const highlight = new THREE.Color(color);
    for(const list of state.surfaces.values()){
      for(const mesh of list){
        if(mesh.userData.lensId === target.id){
          const base = new THREE.Color(mesh.userData.baseColor);
          mesh.material.color.copy(base.clone().lerp(highlight, 0.5));
          mesh.material.emissive.set(color);
          mesh.material.emissiveIntensity = 0.25;
          if(mesh.material.side !== undefined){
            mesh.material.side = THREE.DoubleSide;
          }
        }
      }
    }
    for(const item of (state.rims.get(target.id) || [])){
      const base = new THREE.Color(item.userData.baseColor);
      item.material.color.copy(base.clone().lerp(highlight, 0.5));
      if(item.material.emissive){
        item.material.emissive.set(color);
        item.material.emissiveIntensity = 0.25;
      }
      if(item.material.side !== undefined){
        item.material.side = THREE.DoubleSide;
      }
    }
  }else if(target.kind === 'image' && state.image){
    state.image.material.color.set(color);
  }
}
function selectBlock(element, target){
  panel.querySelectorAll('.block.selected').forEach(selected => selected.classList.remove('selected'));
  element.classList.add('selected');
  resetHighlight();
  applyHighlight(target);
}
function addDetailRow(parent, label, value){
  const row = document.createElement('div');
  row.className = 'detail';
  const name = document.createElement('span');
  name.className = 'detail-label';
  name.textContent = label;
  const content = document.createElement('span');
  content.className = 'detail-value';
  content.textContent = value;
  row.append(name, content);
  parent.appendChild(row);
}
function buildPanel(blocks){
  panel.replaceChildren();
  for(const block of blocks){
    const element = document.createElement('div');
    element.className = 'block ' + block.type;
    if(block.stop){
      element.classList.add('stop');
    }
    if(block.color){
      element.classList.add('colored');
      element.style.setProperty('--block-bg', block.color);
    }
    if(block.target){
      element.dataset.kind = block.target.kind;
      element.dataset.id = block.target.id || '';
      element.addEventListener('click', () => selectBlock(element, block.target));
    }
    const details = block.details || [];
    if(block.type === 'surface' || block.type === 'image'){
      const line = document.createElement('div');
      line.className = 'surface-line';
      if(block.title && block.title !== ''){
        const title = document.createElement('span');
        title.className = 'block-title';
        title.textContent = block.title;
        line.appendChild(title);
      }
      if(block.summary){
        const summary = document.createElement('span');
        summary.className = 'surface-detail';
        summary.innerHTML = block.summary;
        line.appendChild(summary);
      }else{
        for(const [label, value] of details){
          const detail = document.createElement('span');
          detail.className = 'surface-detail';
          detail.textContent = label + ': ' + value;
          line.appendChild(detail);
        }
      }
      element.appendChild(line);
    }else{
      if(block.title && block.title !== ''){
        const title = document.createElement('div');
        title.className = 'block-title';
        title.textContent = block.title;
        element.appendChild(title);
      }
      if(block.summary){
        const summary = document.createElement('div');
        summary.className = 'summary';
        summary.innerHTML = block.summary;
        element.appendChild(summary);
      }
      for(const [label, value] of details){
        addDetailRow(element, label, value);
      }
    }
    panel.appendChild(element);
  }
}
function fitDistance(height){
  const fill = cssNumber('--view-fill', 0.9);
  const fov = camera.fov * Math.PI / 180;
  return (height / 2) / (Math.tan(fov / 2) * fill);
}
function setView(axis){
  const b = SCENE_DATA.bounds;
  const height = Math.max(b.zmax - b.zmin, 1e-9);
  const distance = fitDistance(height);
  const zc = (b.zmin + b.zmax) / 2;
  target.set(0, 0, zc);
  camera.up.set(0, 0, -1);
  if(axis === 'y'){
    camera.position.set(0, distance, zc);
  }else{
    camera.position.set(distance, 0, zc);
  }
  camera.near = Math.max(1e-7, distance / 1000);
  camera.far = Math.max(100, 20 * distance + 10 * height);
  camera.updateProjectionMatrix();
  camera.lookAt(target);
  presetAxis = axis;
  autoFit = true;
}
function viewForward(){
  return new THREE.Vector3().subVectors(target, camera.position).normalize();
}
function viewRight(){
  return new THREE.Vector3().crossVectors(viewForward(), camera.up).normalize();
}
function viewUp(){
  return new THREE.Vector3().crossVectors(viewRight(), viewForward()).normalize();
}
function orbit(dx, dy){
  const speed = cssNumber('--orbit-speed', 0.008);
  const right = viewRight();
  const up = viewUp();
  if(right.lengthSq() === 0 || up.lengthSq() === 0){
    return;
  }
  const offset = new THREE.Vector3().subVectors(camera.position, target);
  const rotation = new THREE.Quaternion().setFromAxisAngle(right, -dy * speed);
  rotation.multiply(new THREE.Quaternion().setFromAxisAngle(up, -dx * speed));
  offset.applyQuaternion(rotation);
  camera.up.applyQuaternion(rotation).normalize();
  camera.position.copy(target).add(offset);
  camera.lookAt(target);
  autoFit = false;
}
function zoomBy(factor){
  const offset = new THREE.Vector3().subVectors(camera.position, target);
  offset.multiplyScalar(factor);
  camera.position.copy(target).add(offset);
  camera.lookAt(target);
  autoFit = false;
}
function moveViewer(direction, distance){
  camera.position.addScaledVector(direction, distance);
  camera.lookAt(target);
  autoFit = false;
}
function roll(degrees){
  const axis = viewForward();
  camera.up.applyAxisAngle(axis, THREE.MathUtils.degToRad(degrees)).normalize();
  camera.lookAt(target);
  autoFit = false;
}
function resize(){
  const width = view.clientWidth || innerWidth;
  const height = view.clientHeight || innerHeight;
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  if(autoFit){
    setView(presetAxis);
  }
}
function build(data){
  disposeDynamic();
  state = {surfaces:new Map(), rims:new Map(), rays:[], image:null};
  const L = data.axes.length;
  addArrow([0, 0, 0], [1, 0, 0], L, css('--axis-x-color', '#ff5555'));
  addArrow([0, 0, 0], [0, 1, 0], L, css('--axis-y-color', '#55ff55'));
  addArrow([0, 0, 0], [0, 0, 1], L, css('--axis-z-color', '#5555ff'));

  for(const lens of data.lenses){
    const opacity = lens.opacity != null ? lens.opacity : cssNumber('--lens-opacity', 0.5);
    for(const m of lens.meshes){
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.Float32BufferAttribute(m.vertices.flat(), 3));
      g.setIndex(m.triangles.flat());
      g.computeVertexNormals();
      const mesh = new THREE.Mesh(g, new THREE.MeshPhongMaterial(
        {color:lens.color, transparent:true, opacity:opacity, side:THREE.FrontSide, depthWrite:false}));
      mesh.userData.dyn = true;
      mesh.userData.surfaceId = m.surface_id || null;
      mesh.userData.lensId = lens.id || null;
      mesh.userData.baseColor = mesh.material.color.getHex();
      mesh.userData.baseEmissive = 0x000000;
      mesh.userData.baseSide = mesh.material.side;
      if(m.surface_id){
        addTo(state.surfaces, m.surface_id, mesh);
        if(String(m.surface_id).startsWith("rim-")){
          addTo(state.rims, lens.id, mesh);
        }
      }
      scene.add(mesh);
    }
    const rimPolylines = lens.rims || (lens.rim_lines ? [lens.rim_lines] : []);
    for(const rim of rimPolylines){
      const line = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(rim.map(p => new THREE.Vector3(...p))),
        new THREE.LineBasicMaterial({color:lens.color}));
      line.userData.dyn = true;
      if(line.userData.baseColor === undefined) line.userData.baseColor = line.material.color.getHex();
      line.userData.lensId = lens.id;
      addTo(state.rims, lens.id, line);
      scene.add(line);
    }
  }
  for(const ray of data.rays){
    const g = new THREE.BufferGeometry().setFromPoints(ray.points.map(p => new THREE.Vector3(...p)));
    const line = new THREE.Line(g, new THREE.LineBasicMaterial({color:ray.color}));
    line.userData.dyn = true;
    line.userData.baseColor = line.material.color.getHex();
    state.rays.push(line);
    scene.add(line);
  }
  const ip = data.image_plane;
  const disk = new THREE.Mesh(
    new THREE.CircleGeometry(ip.radius, 48),
    new THREE.MeshBasicMaterial({color:css('--image-color', '#ffffff'), side:THREE.DoubleSide}));
  disk.position.set(ip.center[0], ip.center[1], ip.center[2]);
  disk.userData.dyn = true;
  disk.userData.baseColor = disk.material.color.getHex();
  state.image = disk;
  scene.add(disk);
  const ap = data.aperture_stop;
  const stop = new THREE.Mesh(
    new THREE.RingGeometry(ap.inner_radius, ap.outer_radius, 48),
    new THREE.MeshBasicMaterial({color:ap.color || css('--aperture-color', '0x808088'),
      transparent:true, opacity:ap.opacity != null ? ap.opacity : cssNumber('--aperture-opacity', 0.5),
      side:THREE.DoubleSide, depthWrite:false}));
  stop.position.set(ap.center[0], ap.center[1], ap.center[2]);
  stop.userData.dyn = true;
  scene.add(stop);
  buildPanel(data.table || []);
  setView(presetAxis);
}

let dragging = false;
let lastX = 0;
let lastY = 0;
canvas.addEventListener('pointerdown', ev => {
  if(ev.button !== 0 && ev.button !== 2){
    return;
  }
  dragging = true;
  lastX = ev.clientX;
  lastY = ev.clientY;
  canvas.setPointerCapture?.(ev.pointerId);
});
canvas.addEventListener('pointermove', ev => {
  if(!dragging){
    return;
  }
  const dx = ev.clientX - lastX;
  const dy = ev.clientY - lastY;
  lastX = ev.clientX;
  lastY = ev.clientY;
  orbit(dx, dy);
});
function endDrag(ev){
  if(!dragging){
    return;
  }
  dragging = false;
  canvas.releasePointerCapture?.(ev.pointerId);
}
canvas.addEventListener('pointerup', endDrag);
canvas.addEventListener('pointercancel', endDrag);
canvas.addEventListener('contextmenu', ev => ev.preventDefault());
canvas.addEventListener('wheel', ev => {
  ev.preventDefault();
  zoomBy(Math.exp(ev.deltaY * cssNumber('--zoom-speed', 0.001)));
}, {passive:false});

new ResizeObserver(resize).observe(view);
addEventListener('resize', resize);
addEventListener('keydown', ev => {
  const b = SCENE_DATA.bounds;
  const step = Math.max(0.05 * (b.zmax - b.zmin), 0.2 * b.radius, 1e-9);
  switch(ev.key){
    case 'Home': setView('y'); break;
    case 'End': setView('x'); break;
    case 'PageUp': roll(90); break;
    case 'PageDown': roll(-90); break;
    case 'ArrowUp': moveViewer(viewForward(), step); break;
    case 'ArrowDown': moveViewer(viewForward(), -step); break;
    case 'ArrowLeft': moveViewer(viewRight(), -step); break;
    case 'ArrowRight': moveViewer(viewRight(), step); break;
    default: return;
  }
  ev.preventDefault();
});
build(SCENE_DATA);
if (SERVED){
  let version = null;
  setInterval(async () => {
    try {
      const v = await (await fetch('/version')).text();
      if (version !== null && v !== version){
        SCENE_DATA = await (await fetch('/scene.json')).json();
        build(SCENE_DATA);
      }
      version = v;
    } catch (err) {
      /* Server exited; the embedded scene keeps rendering. */
    }
  }, 1000);
}
renderer.setAnimationLoop(() => renderer.render(scene, camera));
