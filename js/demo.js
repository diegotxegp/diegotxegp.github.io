/* AutoML mini-demo: UI layer. Models and metrics live in ml.js. */
(function () {
  'use strict';
  var $ = function (id) { return document.getElementById(id); };
  var el = {
    form: $('demo-form'), dataset: $('d-dataset'), target: $('d-target'), model: $('d-model'),
    scale: $('d-scale'), test: $('d-test'), seed: $('d-seed'), feats: $('d-feats'),
    info: $('d-info'), run: $('d-run'), out: $('d-out'), hist: $('d-hist'), histBody: $('d-hist-body'),
    state: $('d-state')
  };
  var datasets = [], current = null, runs = [], busy = false;

  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }
  function pct(v) { return (v * 100).toFixed(1) + '%'; }
  function num(v, d) { return (Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(d || 3)); }
  function setState(html, cls) { el.out.innerHTML = '<p class="d-state ' + (cls || '') + '">' + html + '</p>'; }

  /* ---------- loading ---------- */
  function load() {
    el.run.disabled = true;
    setState('Loading datasets…', 'loading');
    fetch('data/datasets.json').then(function (r) {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.json();
    }).then(function (d) {
      datasets = d;
      el.dataset.innerHTML = d.map(function (x, i) { return '<option value="' + i + '">' + esc(x.name) + '</option>'; }).join('');
      selectDataset(0);
      el.run.disabled = false;
      setState('Pick a setup above and press <strong>Run experiment</strong>. Nothing has been trained yet.');
    }).catch(function () {
      el.out.innerHTML = '<p class="d-state err">The demo data could not be loaded. <button type="button" class="linklike" id="d-retry">Try again</button></p>';
      $('d-retry').addEventListener('click', load);
    });
  }

  /* ---------- controls ---------- */
  function taskOf(col) { return col.type === 'cat' ? 'classification' : 'regression'; }

  function selectDataset(i) {
    current = datasets[i];
    el.info.textContent = current.desc + ' Source: ' + current.source + '.';
    el.target.innerHTML = current.columns.map(function (c) { return '<option>' + esc(c.name) + '</option>'; }).join('');
    el.target.value = current.default_target;
    onTarget();
  }

  function targetCol() { return current.columns.filter(function (c) { return c.name === el.target.value; })[0]; }

  function onTarget() {
    var t = targetCol(), task = taskOf(t);
    el.feats.innerHTML = current.columns.filter(function (c) { return c.name !== t.name; }).map(function (c, i) {
      return '<label class="check"><input type="checkbox" name="feat" value="' + esc(c.name) + '" checked> ' + esc(c.name) + '</label>';
    }).join('');
    var opts = ['<option value="auto">Auto: try them all and keep the best</option>'].concat(
      ML.CATALOG[task].map(function (m) { return '<option value="' + m.id + '">' + esc(m.label) + '</option>'; }));
    el.model.innerHTML = opts.join('');
    $('d-task').textContent = task === 'classification'
      ? 'Predicting a category (' + uniqueLabels(t).length + ' classes).'
      : 'Predicting a number (regression).';
  }

  function uniqueLabels(col) {
    var ci = current.columns.indexOf(col), seen = {}, out = [];
    current.rows.forEach(function (r) { if (!seen[r[ci]]) { seen[r[ci]] = 1; out.push(r[ci]); } });
    return out.sort();
  }

  el.dataset.addEventListener('change', function () { selectDataset(+el.dataset.value); });
  el.target.addEventListener('change', onTarget);

  /* ---------- experiment ---------- */
  function readConfig() {
    var t = targetCol(), task = taskOf(t);
    var feats = Array.prototype.map.call(el.feats.querySelectorAll('input:checked'), function (i) { return i.value; });
    return {
      task: task, target: t, features: feats, model: el.model.value,
      standardize: el.scale.checked, testFrac: +el.test.value / 100,
      seed: Math.max(1, Math.floor(+el.seed.value) || 1)
    };
  }

  function prepare(cfg) {
    var cols = current.columns, ti = cols.indexOf(cfg.target);
    var fi = cfg.features.map(function (n) { return cols.map(function (c) { return c.name; }).indexOf(n); });
    var labels = cfg.task === 'classification' ? uniqueLabels(cfg.target) : null;
    var X = current.rows.map(function (r) { return fi.map(function (j) { return r[j]; }); });
    var y = current.rows.map(function (r) { return labels ? labels.indexOf(r[ti]) : r[ti]; });
    return { X: X, y: y, labels: labels, K: labels ? labels.length : 0 };
  }

  function experiment(cfg) {
    var d = prepare(cfg), task = cfg.task, K = d.K;
    var sp = ML.splitIndices(d.y, task, cfg.testFrac, cfg.seed);
    var Xtr = ML.pick(d.X, sp.train), ytr = ML.pick(d.y, sp.train);
    var Xte = ML.pick(d.X, sp.test), yte = ML.pick(d.y, sp.test);
    var cat = ML.CATALOG[task], board = null, spec;

    if (cfg.model === 'auto') {
      board = cat.map(function (m) {
        return { spec: m, cv: ML.crossValidate(m, cfg.standardize, Xtr, ytr, task, K, 5, cfg.seed) };
      });
      board.sort(function (a, b) { return b.cv - a.cv; });
      spec = board[0].spec;
    } else {
      spec = cat.filter(function (m) { return m.id === cfg.model; })[0];
    }

    var m = ML.pipeline(spec, cfg.standardize, K); m.fit(Xtr, ytr);
    var pred = m.predict(Xte);
    var b = ML.pipeline(ML.BASELINE, false, K); b.fit(Xtr, ytr);
    var bpred = b.predict(Xte);

    var res = { spec: spec, board: board, nTrain: Xtr.length, nTest: Xte.length, yte: yte, pred: pred, Xte: Xte, d: d, cfg: cfg };
    if (task === 'classification') {
      res.cm = ML.confusion(yte, pred, K);
      res.main = ML.accuracy(yte, pred); res.f1 = ML.macroF1(res.cm); res.base = ML.accuracy(yte, bpred);
    } else {
      res.reg = ML.regMetrics(yte, pred); res.main = res.reg.r2; res.base = ML.regMetrics(yte, bpred).r2;
    }
    return res;
  }

  el.form.addEventListener('submit', function (e) {
    e.preventDefault();
    if (busy || !current) return;
    var cfg = readConfig();
    if (!cfg.features.length) { setState('Select at least one input column.', 'err'); return; }
    busy = true; el.run.disabled = true; el.run.textContent = 'Training…';
    setState('Training in your browser…', 'loading');
    setTimeout(function () {               // let the loading message paint first
      try {
        var res = experiment(cfg);
        render(res);
        addRun(res);
      } catch (err) {
        setState('Something went wrong while training: ' + esc(err.message), 'err');
      }
      busy = false; el.run.disabled = false; el.run.textContent = 'Run experiment';
    }, 30);
  });

  /* ---------- rendering ---------- */
  function render(r) {
    var cls = r.cfg.task === 'classification', h = '';
    h += '<div class="d-sum"><div><span class="d-k">Model</span><span class="d-v small">' + esc(r.spec.label) + '</span></div>';
    if (cls) {
      h += stat('Test accuracy', pct(r.main)) + stat('Macro F1', r.f1.toFixed(3)) + stat('Baseline accuracy', pct(r.base));
    } else {
      h += stat('Test R²', num(r.main)) + stat('RMSE', num(r.reg.rmse, 2)) + stat('MAE', num(r.reg.mae, 2));
    }
    h += '</div>';
    h += '<p class="d-note">Evaluated on <strong>' + r.nTest + '</strong> held-out cases never used for training (' + r.nTrain + ' used to train). ' +
      'Baseline = ' + (cls ? 'always guessing the most common class' : 'always predicting the average (R² ≈ 0)') + '.</p>';

    if (r.board) {
      var worst = Math.min.apply(null, r.board.map(function (b) { return b.cv; }).concat([0]));
      h += '<h3>Model comparison</h3><p class="d-note">5-fold cross-validation on the training data (' + (cls ? 'accuracy' : 'R²') + ').</p><ul class="d-bars">' +
        r.board.map(function (b, i) {
          var w = Math.max(2, (Math.max(b.cv, 0) - worst) / ((1 - worst) || 1) * 100);
          return '<li' + (i === 0 ? ' class="best"' : '') + '><span class="n">' + esc(b.spec.label) + (i === 0 ? ' · chosen' : '') + '</span>' +
            '<span class="bar" aria-hidden="true"><i style="width:' + w.toFixed(1) + '%"></i></span><span class="s">' + (cls ? pct(b.cv) : num(b.cv)) + '</span></li>';
        }).join('') + '</ul>';
    }

    h += cls ? confusionHtml(r) : scatterSvg(r);
    h += samplesHtml(r);
    el.out.innerHTML = h;
  }

  function stat(k, v) { return '<div><span class="d-k">' + k + '</span><span class="d-v">' + v + '</span></div>'; }

  function confusionHtml(r) {
    var L = r.d.labels, max = Math.max.apply(null, [].concat.apply([], r.cm)) || 1;
    var h = '<h3>Confusion matrix</h3><p class="d-note">Rows: true class. Columns: predicted class. The diagonal is correct.</p>' +
      '<div class="d-scroll"><table class="d-cm"><thead><tr><th></th>' + L.map(function (l) { return '<th scope="col">' + esc(l) + '</th>'; }).join('') + '</tr></thead><tbody>';
    r.cm.forEach(function (row, i) {
      h += '<tr><th scope="row">' + esc(L[i]) + '</th>' + row.map(function (v, j) {
        return '<td class="' + (i === j ? 'ok' : 'bad') + '" style="--a:' + (v / max).toFixed(2) + '">' + v + '</td>';
      }).join('') + '</tr>';
    });
    return h + '</tbody></table></div>';
  }

  function scatterSvg(r) {
    var all = r.yte.concat(r.pred), lo = Math.min.apply(null, all), hi = Math.max.apply(null, all), pad = (hi - lo) * 0.05 || 1;
    lo -= pad; hi += pad;
    var S = 300, m = 34, sc = function (v) { return m + (v - lo) / (hi - lo) * (S - m - 8); }, sy = function (v) { return S - m - (v - lo) / (hi - lo) * (S - m - 8) + 8; };
    var h = '<h3>Predicted vs. true</h3><p class="d-note">Each dot is a test case; the closer to the diagonal, the better.</p>' +
      '<svg class="d-plot" viewBox="0 0 ' + S + ' ' + S + '" role="img" aria-label="Scatter plot of predicted against true values for the test cases">' +
      '<line x1="' + sc(lo) + '" y1="' + sy(lo) + '" x2="' + sc(hi) + '" y2="' + sy(hi) + '" class="diag"/>' +
      '<line x1="' + m + '" y1="' + (S - m + 8) + '" x2="' + (S - 8) + '" y2="' + (S - m + 8) + '" class="axis"/><line x1="' + m + '" y1="8" x2="' + m + '" y2="' + (S - m + 8) + '" class="axis"/>';
    r.yte.forEach(function (v, i) { h += '<circle cx="' + sc(v).toFixed(1) + '" cy="' + sy(r.pred[i]).toFixed(1) + '" r="3.2"/>'; });
    h += '<text x="' + (S / 2 + 12) + '" y="' + (S - 8) + '" text-anchor="middle">true value</text>' +
      '<text transform="translate(11 ' + (S / 2 - 10) + ') rotate(-90)" text-anchor="middle">predicted</text></svg>';
    return h;
  }

  function samplesHtml(r) {
    var cls = r.cfg.task === 'classification', n = Math.min(8, r.yte.length), rows = '';
    for (var i = 0; i < n; i++) {
      var t = r.yte[i], p = r.pred[i];
      if (cls) rows += '<tr><td>' + (i + 1) + '</td><td>' + esc(r.d.labels[t]) + '</td><td>' + esc(r.d.labels[p]) + '</td><td class="' + (t === p ? 'good' : 'wrong') + '">' + (t === p ? '✓ correct' : '✗ wrong') + '</td></tr>';
      else rows += '<tr><td>' + (i + 1) + '</td><td>' + num(t, 1) + '</td><td>' + num(p, 1) + '</td><td>' + (p - t >= 0 ? '+' : '−') + num(Math.abs(p - t), 1) + '</td></tr>';
    }
    return '<h3>Example predictions</h3><p class="d-note">The first ' + n + ' test cases.</p><div class="d-scroll"><table class="d-tab"><thead><tr><th scope="col">#</th><th scope="col">True</th><th scope="col">Predicted</th><th scope="col">' + (cls ? 'Result' : 'Error') + '</th></tr></thead><tbody>' + rows + '</tbody></table></div>';
  }

  function addRun(r) {
    var c = r.cfg;
    runs.unshift({
      n: (runs.length ? runs[0].n : 0) + 1,
      what: esc(current.name) + ' → ' + esc(c.target.name),
      cfg: esc(r.spec.label) + (c.model === 'auto' ? ' (auto)' : '') + ' · ' + (c.standardize ? 'scaled' : 'unscaled') + ' · ' + c.features.length + ' inputs · ' + Math.round(c.testFrac * 100) + '% test',
      score: c.task === 'classification' ? pct(r.main) : 'R² ' + num(r.main)
    });
    runs = runs.slice(0, 6);
    el.hist.hidden = false;
    el.histBody.innerHTML = runs.map(function (x) {
      return '<tr><td>' + x.n + '</td><td>' + x.what + '</td><td>' + x.cfg + '</td><td>' + x.score + '</td></tr>';
    }).join('');
  }

  load();
})();
