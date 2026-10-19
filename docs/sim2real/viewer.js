'use strict';
const canvas=document.getElementById('model'),ctx=canvas.getContext('2d'),el=id=>document.getElementById(id);
let kind='cube',model=null,yaw=.65,pitch=-.5,drag=null,last=0,tagPixels={};
async function select(k){
  kind=k;model=await(await fetch(k+'.json')).json();tagPixels=await(await fetch('tag-pixels.json')).json();
  el('cubeBtn').classList.toggle('active',k==='cube');el('penBtn').classList.toggle('active',k==='pen');el('objectName').textContent=k.toUpperCase();
  el('description').textContent=k==='cube'?'放大后的 55 mm 方块。六面都有唯一标识，平整表面居中贴码；按掌部网格尺度核对，仍需真机抓持验证。':'圆柱中杆 + 方形端头。默认沿轴线分为两半，平面朝下打印后定位粘合；组装外形与仿真一致。';
  const mm=x=>Math.round(x*1000),tag=model.tags[0];
  const rows=k==='cube'?[['外形',model.dimensions_m.map(mm).join(' × ')+' mm'],['黑框 / 白纸',mm(tag.black_size_m)+' / '+mm(tag.paper_size_m)+' mm'],['标签 ID','0 — 5'],['名义质量（待测）',mm(model.nominal_mass_kg)+' g']]:[['总长 / 圆杆直径','170 / 16 mm'],['方头 / 黑框 / 白纸','24 / 16 / 20 mm'],['标签 ID / 两端基线','10 — 19 / 148 mm'],['默认分体打印高度','12 mm（组装长170）']];
  el('facts').replaceChildren(...rows.map(([a,b])=>{let p=document.createElement('p'),s=document.createElement('span'),v=document.createElement('strong');s.textContent=a;v.textContent=b;p.append(s,v);return p}));
  const printName=k==='pen'?'pen-split':k;
  el('print3mf').href=printName+'-a1mini-pla.3mf';el('petg3mf').href=printName+'-a1mini-petg.3mf';el('stl').href=k==='pen'?'pen-half.stl':k+'-a1mini.stl';el('stl').textContent=k==='pen'?'半壳 STL（需2片+销）':'已定向 STL';el('cad').href=k+'.scad';el('readout').textContent='1 : 1 文件尺寸 · 视窗按显示区域缩放';
}
const add=(a,b)=>a.map((x,i)=>x+b[i]),mul=(a,s)=>a.map(x=>x*s);
function camera(p){let [x,y,z]=p;[x,y]=[Math.cos(yaw)*x-Math.sin(yaw)*y,Math.sin(yaw)*x+Math.cos(yaw)*y];return [x,Math.cos(pitch)*y-Math.sin(pitch)*z,Math.sin(pitch)*y+Math.cos(pitch)*z]}
function project(p,w,h){const [x,y,z]=camera(p),scale=Math.min(w,h)*(kind==='cube'?4.8:2.5)/(.65-z);return [w/2+x*scale,h/2-y*scale]}
function poly(points,fill,stroke,w,h){ctx.beginPath();points.forEach((p,i)=>{const [x,y]=project(p,w,h);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.closePath();ctx.fillStyle=fill;ctx.fill();if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=.7;ctx.stroke()}}
function draw(t){const dpr=window.devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr)}ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);if(model){
if(el('spin').checked&&!drag)yaw+=(t-last)*.00015;
ctx.strokeStyle='#c8d2c1';ctx.lineWidth=.5;for(let i=-5;i<=5;i++)for(const line of [[[-.12,i*.024,-.045],[.12,i*.024,-.045]],[[i*.024,-.12,-.045],[i*.024,.12,-.045]]]){const a=project(line[0],w,h),b=project(line[1],w,h);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke()}
const surfaces=model.triangles.map(tri=>{const p=tri.map(i=>model.vertices_m[i]);return {p,z:p.reduce((s,v)=>s+camera(v)[2],0)/3,tag:false}});
if(el('tags').checked)for(const tag of model.tags){const n=[tag.u[1]*tag.v[2]-tag.u[2]*tag.v[1],tag.u[2]*tag.v[0]-tag.u[0]*tag.v[2],tag.u[0]*tag.v[1]-tag.u[1]*tag.v[0]];if(camera(n)[2]<=0)continue;const c=add(tag.center_m,mul(n,.0001+Number(el('explode').value)/1000));surfaces.push({p:[c],z:camera(c)[2]+.0001,tag,c})}
surfaces.sort((a,b)=>(Number(!!a.tag)-Number(!!b.tag)) || a.z-b.z);for(const s of surfaces){if(!s.tag){const bright=Math.round(173+35*s.z/.08);poly(s.p,'rgb('+(bright-24)+','+(bright+4)+','+(bright-13)+')','#9ab09a',w,h);continue}
const tag=s.tag,point=(u,v)=>add(s.c,add(mul(tag.u,u),mul(tag.v,v))),size=tag.black_size_m,paper=tag.paper_size_m;
poly([point(-paper/2,paper/2),point(paper/2,paper/2),point(paper/2,-paper/2),point(-paper/2,-paper/2)],'#fcfcf1','#4f8260',w,h);
const pixels=tagPixels[String(tag.id)];if(pixels)for(let iy=0;iy<8;iy++)for(let ix=0;ix<8;ix++){if(pixels[iy][ix])continue;const x=(ix/8-.5)*size,y=(.5-iy/8)*size,d=size/8;poly([point(x,y),point(x+d,y),point(x+d,y-d),point(x,y-d)],'#20342f',null,w,h)}
if(Number(el('explode').value)>5){const p=project(add(s.c,mul(tag.v,paper*.8)),w,h);ctx.fillStyle='#335e46';ctx.font='11px monospace';ctx.fillText('ID '+tag.id,...p)}}}
last=t;requestAnimationFrame(draw)}
canvas.addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)});
canvas.addEventListener('pointermove',e=>{if(!drag)return;yaw+=(e.clientX-drag[0])*.008;pitch=Math.max(-1.4,Math.min(1.4,pitch+(e.clientY-drag[1])*.008));drag=[e.clientX,e.clientY]});
canvas.addEventListener('pointerup',()=>drag=null);canvas.addEventListener('pointercancel',()=>drag=null);el('cubeBtn').onclick=()=>select('cube');el('penBtn').onclick=()=>select('pen');
select('pen').catch(e=>el('boot').textContent='模型加载失败；请通过 HTTP 服务打开本页：'+e.message);requestAnimationFrame(draw);
