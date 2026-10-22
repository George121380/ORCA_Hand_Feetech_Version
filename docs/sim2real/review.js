/* Fixed checkpoints and all five predeclared cases; no outcome filtering. */
'use strict';
(() => {
  const el = id => document.getElementById(id);
  const videos = [el('reviewOld'), el('reviewPure')];
  let review, index = 0, playing = false, time = 0, started = 0;
  const speed = () => Number(el('reviewSpeed').value);
  const names = ['comparison', 'pure'];
  const cases = () => names.map(name => review.models[name].trials[index]);
  const duration = () => Math.max(...cases().map(c => c.duration_s));
  function pause() {
    playing = false;
    videos.forEach(v => v.pause());
    el('reviewPlay').textContent = '播放 RL 对照';
  }
  function frame(t) {
    time = Math.max(0, Math.min(duration(), t));
    cases().forEach((c, i) => {
      const target = c.offset_s + Math.min(time, Math.max(0, c.duration_s - .025));
      if (videos[i].readyState > 0 && (!playing || Math.abs(videos[i].currentTime - target) > .12)) {
        videos[i].currentTime = target;
      }
      if (time >= c.duration_s) videos[i].pause();
    });
    el('reviewSeek').value = time;
    el('reviewClock').textContent = `${time.toFixed(2)} / ${duration().toFixed(2)} s`;
  }
  function animate(now) {
    if (!playing) return;
    frame((now - started) / 1000 * speed());
    if (time >= duration()) pause();
    else requestAnimationFrame(animate);
  }
  function row(label, values) {
    const r = document.createElement('tr');
    [label, ...values].forEach((text, i) => {
      const cell = document.createElement(i ? 'td' : 'th');
      cell.textContent = text;
      if (!i) cell.scope = 'row';
      r.append(cell);
    });
    return r;
  }
  function select(i) {
    pause(); index = i; el('reviewEpisode').value = i;
    const c = cases();
    el('reviewSeek').max = duration();
    frame(0);
    el('reviewCaseTitle').textContent = `第 ${i + 1} 回合 · seed ${c[0].seed} · 负方向 15°`;
    const yes = x => x ? '满足' : '未满足';
    const measured = (x, unit, scale = 1) => Number.isFinite(x) ? `${(x * scale).toFixed(2)} ${unit}` : '无有效窗口数据';
    const endReason = x => ({horizon:'到达回合时限', displacement_limit:'物体位移超过限制', vision_timeout:'视觉超时', success:'成功'})[x.end_reason] ?? x.end_reason;
    const rows = [
      row('完整回合结果', c.map(x => x.is_success ? '成功' : '失败')),
      row('末帧完整姿态误差 / 要求 <8°', c.map(x => `${x.angle_error_deg.toFixed(2)}°`)),
      row('末段 1 秒旋转范围 / 要求 <2°', c.map(x => measured(x.final_1s_orientation_span_deg, '°'))),
      row('末段 1 秒位置范围 / 要求 <2 mm', c.map(x => measured(x.final_1s_position_span_m, 'mm', 1000))),
      row('末帧受载对握', c.map(x => yes(x.loaded_opposition))),
      row('末帧静止条件', c.map(x => yes(x.resting))),
      row('抓持前缀', c.map(x => x.prefix_grasp_success ? '成功' : '失败（计入结果）')),
      row('掉落', c.map(x => x.dropped ? '是' : '否')),
      row('最大物体位移 / 限制 40 mm', c.map(x => measured(x.maximum_displacement_m, 'mm', 1000))),
      row('环境结束原因', c.map(endReason)),
      row('执行长度', c.map(x => `${x.steps} 步 / ${(x.duration_s).toFixed(2)} s`))
    ];
    el('reviewRows').replaceChildren(...rows);
    ['reviewOldResult', 'reviewPureResult'].forEach((id, k) => {
      el(id).textContent = `${c[k].is_success ? '成功' : '失败'} · 末态误差 ${c[k].angle_error_deg.toFixed(2)}°`;
    });
  }
  function chart() {
    const ns = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(ns, 'svg');
    svg.setAttribute('viewBox', '0 0 640 240'); svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', '旧 PPO 内部成功率在5万、10万、15万、20万步分别为30%、40%、40%、25%；纯 PPO在54096步为0%。各点20回合，不是正式成绩。');
    const add = (tag, attrs, text) => {
      const n = document.createElementNS(ns, tag);
      Object.entries(attrs).forEach(([k, v]) => n.setAttribute(k, v));
      if (text !== undefined) n.textContent = text;
      svg.append(n); return n;
    };
    const x = steps => 48 + steps / 200000 * 550;
    const y = rate => 194 - rate / .5 * 155;
    [0, .1, .2, .3, .4, .5].forEach(rate => {
      add('line', {x1:48, x2:600, y1:y(rate), y2:y(rate), stroke:'#cdd3c5'});
      add('text', {x:10, y:y(rate)+4, fill:'#68746d', 'font-size':12}, `${rate*100}%`);
    });
    [50000, 100000, 150000, 200000].forEach(steps => add('text', {x:x(steps), y:220, 'text-anchor':'middle', 'font-size':12, fill:'#68746d'}, `${steps/10000}万步`));
    names.forEach((name, k) => {
      const points = review.curves[name]; const color = k ? '#ba5937' : '#335e46';
      add('polyline', {points:points.map(p => `${x(p.steps)},${y(p.successes/p.episodes)}`).join(' '), fill:'none', stroke:color, 'stroke-width':3});
      points.forEach(p => {
        const dot = add('circle', {cx:x(p.steps), cy:y(p.successes/p.episodes), r:5, fill:color});
        const title = document.createElementNS(ns, 'title'); title.textContent = `${name} ${p.steps}步：${p.successes}/${p.episodes}`; dot.append(title);
      });
    });
    el('reviewChart').replaceChildren(svg);
  }
  el('reviewPlay').addEventListener('click', async () => {
    if (!review) return;
    if (playing) { pause(); return; }
    if (time >= duration() - .025) frame(0);
    videos.forEach(v => v.playbackRate = speed());
    try {
      await Promise.all(videos.map((v, i) => time < cases()[i].duration_s ? v.play() : Promise.resolve()));
      playing = true; started = performance.now() - time / speed() * 1000;
      el('reviewPlay').textContent = '暂停'; requestAnimationFrame(animate);
    } catch { pause(); el('reviewError').textContent = '录像暂不能播放，请使用下方 MP4 下载链接。'; }
  });
  el('reviewEpisode').addEventListener('change', e => select(Number(e.target.value)));
  el('reviewRestart').addEventListener('click', () => { pause(); frame(0); });
  el('reviewFrame').addEventListener('click', () => { pause(); frame(Math.min(duration(), time + .05)); });
  el('reviewSeek').addEventListener('input', e => { pause(); frame(Number(e.target.value)); });
  el('reviewSpeed').addEventListener('change', () => { pause(); frame(time); });
  videos.forEach(v => v.addEventListener('loadedmetadata', () => { if (review) frame(time); }));
  fetch('intermediate-policy-review.json', {cache:'no-store'}).then(r => {
    if (!r.ok) throw Error('中间结果数据加载失败'); return r.json();
  }).then(d => {
    review = d;
    d.models.pure.trials.forEach((c, i) => {
      const o = document.createElement('option'); o.value = i; o.textContent = `${i+1}/5 · seed ${c.seed}`; el('reviewEpisode').append(o);
    });
    el('reviewSummary').textContent = `这 5 个固定初态：旧 PPO ${d.models.comparison.successes}/5，新纯 PPO ${d.models.pure.successes}/5。平均末态误差 ${d.models.comparison.mean_error_deg.toFixed(2)}° → ${d.models.pure.mean_error_deg.toFixed(2)}°；当前纯 RL 中期表现更差。`;
    document.querySelectorAll('#review button,#review select,#review input').forEach(n => n.disabled = false);
    select(0); chart();
  }).catch(e => { el('reviewError').textContent = e.message; });
})();
