/* Fixed checkpoints and all five predeclared cases; no outcome filtering. */
'use strict';
(() => {
  const el = id => document.getElementById(id);
  const videos = [el('reviewOld'), el('reviewPure')];
  let review, index = 0, playing = false, time = 0, started = 0;
  let snapshots, request = 0;
  const fmt = x => Number(x).toLocaleString('en-US');
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
    svg.setAttribute('aria-label', names.map(name => `${name}：${review.curves[name].map(p => `${p.steps}步${p.successes}/${p.episodes}`).join('；')}`).join('。') + '。内部检查不是正式成绩。');
    const add = (tag, attrs, text) => {
      const n = document.createElementNS(ns, tag);
      Object.entries(attrs).forEach(([k, v]) => n.setAttribute(k, v));
      if (text !== undefined) n.textContent = text;
      svg.append(n); return n;
    };
    const points = names.flatMap(name => review.curves[name]);
    const xmax = Math.max(200000, ...points.map(p => p.steps));
    const ymax = Math.max(.5, Math.ceil(Math.max(...points.map(p => p.successes/p.episodes))*10)/10);
    const x = steps => 48 + steps / xmax * 550;
    const y = rate => 194 - rate / ymax * 155;
    Array.from({length:6}, (_, i) => ymax * i / 5).forEach(rate => {
      add('line', {x1:48, x2:600, y1:y(rate), y2:y(rate), stroke:'#cdd3c5'});
      add('text', {x:10, y:y(rate)+4, fill:'#68746d', 'font-size':12}, `${Math.round(rate*100)}%`);
    });
    [1, 2, 3, 4].map(i => xmax*i/4).forEach(steps => add('text', {x:x(steps), y:220, 'text-anchor':'middle', 'font-size':12, fill:'#68746d'}, `${(steps/10000).toFixed(1)}万步`));
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
  async function loadSnapshot(id) {
    pause(); const serial = ++request;
    document.querySelectorAll('#review button,#review input,#reviewEpisode,#reviewSpeed').forEach(n => n.disabled = true);
    try {
      const snapshot = snapshots.versions.find(v => String(v.steps) === String(id));
      if (!snapshot) throw Error('检查点不存在');
      const response = await fetch(snapshot.data, {cache:'no-store'});
      if (!response.ok) throw Error('中间结果数据加载失败');
      const d = await response.json(); if (serial !== request) return;
      review = d; time = 0;
      videos[1].poster = snapshot.poster; videos[1].src = snapshot.video; videos[1].load();
      videos[1].setAttribute('aria-label', `纯PPO${fmt(snapshot.steps)}步固定五回合录像`);
      el('reviewPureLabel').textContent = `新路线 / 纯 PPO · ${fmt(snapshot.steps)} 步`;
      el('reviewPureHeading').textContent = `新纯 PPO · ${fmt(snapshot.steps)} 步`;
      el('reviewPureCount').textContent = `固定诊断回合 ${d.models.pure.successes} / 5`;
      const check = d.curves.pure.filter(p => p.steps <= snapshot.steps).at(-1);
      el('reviewPureCheck').textContent = check ? `${fmt(check.steps)}步内部检查 ${check.successes}/${check.episodes}` : '尚无内部检查';
      el('reviewDownload').href = snapshot.video; el('reviewRaw').href = snapshot.data;
      el('reviewRecorded').textContent = `录像记录：${new Date(d.recorded_at).toISOString().replace('T',' ').slice(0,19)} UTC · 两边均为固定诊断五回合`;
      const selected = Math.min(index, d.models.pure.trials.length - 1);
      el('reviewEpisode').replaceChildren(...d.models.pure.trials.map((c, i) => {
        const o = document.createElement('option'); o.value = i; o.textContent = `${i+1}/5 · seed ${c.seed}`; return o;
      }));
      const old = d.models.comparison, fresh = d.models.pure;
      const relation = fresh.mean_error_deg > old.mean_error_deg ? '更高' : fresh.mean_error_deg < old.mean_error_deg ? '更低' : '相同';
      el('reviewSummary').textContent = `这 5 个固定初态：旧 PPO ${old.successes}/5，纯 PPO ${fmt(snapshot.steps)}步 ${fresh.successes}/5。平均末态误差 ${old.mean_error_deg.toFixed(2)}° → ${fresh.mean_error_deg.toFixed(2)}°；纯 RL 在这五例的误差${relation}。`;
      document.querySelectorAll('#review button,#review select,#review input').forEach(n => n.disabled = false);
      el('reviewError').textContent = ''; select(selected); chart();
    } catch (e) {
      if (serial === request) el('reviewError').textContent = e.message;
    }
  }
  el('reviewSnapshot').addEventListener('change', e => loadSnapshot(e.target.value));
  fetch('review-snapshots.json', {cache:'no-store'}).then(r => {
    if (!r.ok) throw Error('检查点列表加载失败'); return r.json();
  }).then(d => {
    snapshots = d;
    el('reviewSnapshot').replaceChildren(...d.versions.map(v => {
      const o = document.createElement('option'); o.value = v.steps; o.textContent = `纯 PPO · ${fmt(v.steps)} 步`; return o;
    }));
    el('reviewSnapshot').value = d.latest_steps;
    el('reviewNext').textContent = d.status === 'paused_by_user' ? '已按用户要求暂停训练与自动录像，所有既有版本保留。当前策略尚未通过验收。' : `旧路线未通过，纯 RL 尚未取得可用结果。下一录像点为 ${fmt(d.next_review_steps)} 步，继续使用相同五回合，保留所有旧版本。`;
    return loadSnapshot(d.latest_steps);
  }).catch(e => { el('reviewError').textContent = e.message; });
})();
