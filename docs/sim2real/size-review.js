'use strict';
(async()=>{
  const report=await(await fetch('pen-size-review.json')).json();
  const buttons=[...document.querySelectorAll('[data-pen-length]')];
  function show(length){
    const row=report.candidates.find(r=>r.length_mm===length);
    const probe=row.centers.find(r=>r.center_x_m===.19);
    const image=document.getElementById('sizeCandidate');
    image.src=`pen-size-${length}.png`;image.alt=`${length}毫米笔，相同手型和物体姿态的静态尺寸对照`;
    document.getElementById('sizeCandidateCaption').textContent=`${length} mm 总长 / ${row.shaft_mm} mm 纯圆杆 / ${row.tag_baseline_mm} mm 侧码基线`;
    const rows=[['总长 / 圆杆',`${length} / ${row.shaft_mm} mm`],
      ['方头相交的静态快照',`${probe.heads} / ${probe.snapshots}`],
      ['近似偏航惯量 / 170版',`${row.yaw_inertia_relative_to_170.toFixed(2)} ×`],
      ['示例相机两端几何可见快照',`${probe.both_ends_geometrically_visible} / ${probe.snapshots}`],
      ['对角半壳包络',row.half_plate_xy_mm.map(x=>x.toFixed(1)).join(' × ')+' mm'],
      ['含brim后的最小床边余量',row.edge_margin_with_3mm_brim_and_01gap_mm.toFixed(1)+' mm'],
      ['独立重力落稳检查',`${row.settle.supported} / ${row.settle.trials} 有手部支撑`]];
    document.getElementById('sizeFacts').replaceChildren(...rows.map(([a,b])=>{const p=document.createElement('p'),s=document.createElement('span'),v=document.createElement('strong');s.textContent=a;v.textContent=b;p.append(s,v);return p}));
    buttons.forEach(b=>b.classList.toggle('active',Number(b.dataset.penLength)===length));
  }
  buttons.forEach(b=>b.addEventListener('click',()=>show(Number(b.dataset.penLength))));
  show(210);
})().catch(()=>{document.getElementById('sizeCandidateCaption').textContent='尺寸比较数据未载入，请刷新或下载原始JSON。'});
