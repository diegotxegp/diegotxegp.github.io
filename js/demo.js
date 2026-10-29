/* AutoML mini-demo: UI layer. Models and metrics live in ml.js.
   Result views follow the ones in the Hamelin desktop app: metric cards with a
   plain-language summary, a clickable confusion matrix, an ROC curve and a
   decision threshold for binary tasks. */
(function () {
  'use strict';
  var $ = function (id) { return document.getElementById(id); };
  var el = {
    form: $('demo-form'), dataset: $('d-dataset'), target: $('d-target'), model: $('d-model'),
    scale: $('d-scale'), missing: $('d-missing'), test: $('d-test'), seed: $('d-seed'), feats: $('d-feats'),
    info: $('d-info'), chips: $('d-chips'), preview: $('d-preview'), run: $('d-run'), out: $('d-out'),
    hist: $('d-hist'), histBody: $('d-hist-body')
  };
  var datasets = [], current = null, runs = [], busy = false, last = null;
  var view = { thr: 0.5, pct: false, cell: null };

  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }
  function pct(v, d) { return (v * 100).toFixed(d === undefined ? 1 : d) + '%'; }
  function num(v, d) { return (Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(d || 3)); }
  function lab(s) { return esc(s).replace(/_/g, '_<wbr>'); }   // lets long snake_case labels wrap
  function cell(v) { return v === null ? '—' : esc(v); }
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
  function colIndex(name) { return current.columns.map(function (c) { return c.name; }).indexOf(name); }
  function targetCol() { return current.columns.filter(function (c) { return c.name === el.target.value; })[0]; }

  function selectDataset(i) {
    current = datasets[i];
    var cells = current.rows.length * current.columns.length;
    el.info.textContent = current.desc + ' Source: ' + current.source + '.';
    el.chips.innerHTML = [current.rows.length.toLocaleString('en') + ' rows', current.columns.length + ' columns',
      (current.missing / cells * 100).toFixed(1) + '% missing'].map(function (t) { return '<span>' + t + '</span>'; }).join('');
    el.preview.innerHTML = '<thead><tr>' + current.columns.map(function (c) { return '<th scope="col">' + esc(c.name) + '</th>'; }).join('') + '</tr></thead><tbody>' +
      current.rows.slice(0, 6).map(function (r) { return '<tr>' + r.map(function (v) { return '<td>' + cell(v) + '</td>'; }).join('') + '</tr>'; }).join('') + '</tbody>';
    el.missing.disabled = !current.missing;
    el.target.innerHTML = current.columns.map(function (c) { return '<option>' + esc(c.name) + '</option>'; }).join('');
    el.target.value = current.default_target;
    onTarget();
  }

  function uniqueLabels(col) {
    var ci = current.columns.indexOf(col), seen = {}, out = [];
    current.rows.forEach(function (r) { if (r[ci] !== null && !seen[r[ci]]) { seen[r[ci]] = 1; out.push(r[ci]); } });
    out.sort();
    if (out.length === 2 && current.positive && col.name === current.default_target) {   // negative first, positive last
      out = out.filter(function (l) { return l !== current.positive; }).concat([current.positive]);
    }
    return out;
  }

  function onTarget() {
    var t = targetCol(), task = taskOf(t);
    el.feats.innerHTML = current.columns.filter(function (c) { return c.name !== t.name && c.type === 'num'; }).map(function (c) {
      return '<label class="check"><input type="checkbox" name="feat" value="' + esc(c.name) + '" checked> ' + esc(c.name) + '</label>';
    }).join('');
    el.model.innerHTML = ['<option value="auto">Auto: try them all and keep the best</option>'].concat(
      ML.CATALOG[task].map(function (m) { return '<option value="' + m.id + '">' + esc(m.label) + '</option>'; })).join('');
    var n = task === 'classification' ? uniqueLabels(t).length : 0;
    $('d-task').textContent = task === 'classification'
      ? 'Predicting a category (' + n + ' classes' + (n === 2 ? ', binary' : '') + ').'
      : 'Predicting a number (regression).';
  }

  el.dataset.addEventListener('change', function () { selectDataset(+el.dataset.value); });
  el.target.addEventListener('change', onTarget);

  /* ---------- experiment ---------- */
  function readConfig() {
    var t = targetCol();
    return {
      task: taskOf(t), target: t, model: el.model.value,
      features: Array.prototype.map.call(el.feats.querySelectorAll('input:checked'), function (i) { return i.value; }),
      standardize: el.scale.checked, impute: el.missing.value === 'fill', dropMissing: el.missing.value === 'drop' && !el.missing.disabled,
      testFrac: +el.test.value / 100, seed: Math.max(1, Math.floor(+el.seed.value) || 1)
    };
  }

  function prepare(cfg) {
    var ti = colIndex(cfg.target.name), fi = cfg.features.map(colIndex);
    var labels = cfg.task === 'classification' ? uniqueLabels(cfg.target) : null;
    var rows = current.rows.filter(function (r) {
      return r[ti] !== null && !(cfg.dropMissing && fi.some(function (j) { return r[j] === null; }));
    });
    return {
      X: rows.map(function (r) { return fi.map(function (j) { return r[j]; }); }),
      y: rows.map(function (r) { return labels ? labels.indexOf(r[ti]) : r[ti]; }),
      labels: labels, K: labels ? labels.length : 0, nRows: rows.length
    };
  }

  function experiment(cfg) {
    var d = prepare(cfg), task = cfg.task, K = d.K, opts = { standardize: cfg.standardize, impute: cfg.impute };
    var sp = ML.splitIndices(d.y, task, cfg.testFrac, cfg.seed);
    var Xtr = ML.pick(d.X, sp.train), ytr = ML.pick(d.y, sp.train);
    var Xte = ML.pick(d.X, sp.test), yte = ML.pick(d.y, sp.test);
    var cat = ML.CATALOG[task], board = null, spec;

    if (!cfg.impute && Xtr.concat(Xte).some(function (r) { return r.indexOf(null) >= 0; })) throw new Error('missing values need to be filled or dropped.');

    if (cfg.model === 'auto') {
      board = cat.map(function (m) { return { spec: m, cv: ML.crossValidate(m, opts, Xtr, ytr, task, K, 5, cfg.seed) }; });
      board.sort(function (a, b) { return b.cv - a.cv; });
      spec = board[0].spec;
    } else {
      spec = cat.filter(function (m) { return m.id === cfg.model; })[0];
    }

    var m = ML.pipeline(spec, opts, K); m.fit(Xtr, ytr);
    var pred = m.predict(Xte);
    var b = ML.pipeline(ML.BASELINE, {}, K); b.fit(Xtr, ytr);
    var bpred = b.predict(Xte);

    var res = {
      spec: spec, board: board, nTrain: Xtr.length, nTest: Xte.length, yte: yte, pred: pred, Xte: Xte, d: d, cfg: cfg,
      imp: ML.permutationImportance(m, Xte, yte, task, cfg.seed, 5)
    };
    if (task === 'classification') {
      res.base = ML.accuracy(yte, bpred); res.baseClass = bpred[0];
      if (K === 2) {
        res.prob = m.proba(Xte).map(function (p) { return p[1]; });
        res.pos = yte.map(function (v) { return v === 1; });
        res.roc = ML.roc(res.pos, res.prob);
      } else {
        res.cm = ML.confusion(yte, pred, K);
        res.main = ML.accuracy(yte, pred); res.f1 = ML.macroF1(res.cm);
      }
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
        last = experiment(cfg);
        view = { thr: 0.5, pct: false, cell: null };
        render(last);
        addRun(last);
      } catch (err) {
        setState('Something went wrong while training: ' + esc(err.message), 'err');
      }
      busy = false; el.run.disabled = false; el.run.textContent = 'Run experiment';
    }, 30);
  });

  /* ---------- rendering ---------- */
  function binary(r) { return r.d.K === 2 && r.cfg.task === 'classification'; }

  /* Everything that depends on the decision threshold, for binary tasks. */
  function binStats(r) {
    var s = ML.atThreshold(r.pos, r.prob, view.thr);
    s.cm = [[s.tn, s.fp], [s.fn, s.tp]];
    s.pred = r.prob.map(function (p) { return p >= view.thr ? 1 : 0; });
    return s;
  }

  function kpi(label, value, caption, tone) {
    return '<div class="kpi ' + tone + '"><span class="d-k">' + label + '</span><span class="d-v">' + value + '</span><span class="cap">' + caption + '</span></div>';
  }
  function tone(ok) { return ok ? 'good' : 'warn'; }

  function cardsAndSummary(r) {
    var cfg = r.cfg, L = r.d.labels, h, text;
    if (binary(r)) {
      var s = binStats(r), auc = r.roc.auc, q = auc >= .9 ? 'excellent' : auc >= .8 ? 'good' : auc >= .7 ? 'acceptable' : auc >= .6 ? 'limited' : 'almost no';
      h = kpi('AUC-ROC', auc.toFixed(3), 'ranking quality (0.5–1)', tone(auc >= .7)) +
        kpi('Accuracy', pct(s.acc), 'correct predictions', tone(s.acc >= r.base + .05)) +
        kpi('Sensitivity', pct(s.sens), 'positive cases detected', tone(s.sens >= .7)) +
        kpi('Specificity', pct(s.spec), 'negative cases correct', tone(s.spec >= .7));
      text = 'The model correctly classifies <strong>' + pct(s.acc) + '</strong> of the test cases (always guessing “' + esc(L[r.baseClass]) + '” would get ' + pct(r.base) + '). ' +
        'AUC-ROC = ' + auc.toFixed(3) + ': ' + q + ' discriminative ability. It detects ' + pct(s.sens) + ' of the “' + esc(L[1]) + '” cases (sensitivity) and correctly identifies ' + pct(s.spec) + ' of the “' + esc(L[0]) + '” ones (specificity).';
    } else if (cfg.task === 'classification') {
      h = kpi('Accuracy', pct(r.main), 'correct predictions', tone(r.main >= r.base + .05)) +
        kpi('Macro F1', r.f1.toFixed(3), 'balance across classes', tone(r.f1 >= r.base + .05)) +
        kpi('Baseline', pct(r.base), 'always the most common class', 'neu');
      text = 'The model correctly classifies <strong>' + pct(r.main) + '</strong> of the test cases; always guessing the most common class gets ' + pct(r.base) + ', so it is ' +
        (r.main >= r.base + .05 ? 'clearly better than guessing.' : 'not clearly better than guessing.');
    } else {
      var rg = r.reg;
      h = kpi('R²', num(rg.r2), 'share of variation explained', tone(rg.r2 >= .3)) +
        kpi('RMSE', num(rg.rmse, 2), 'typical error, big misses weigh more', 'neu') +
        kpi('MAE', num(rg.mae, 2), 'average absolute error', 'neu');
      text = 'On unseen cases the model explains <strong>' + pct(Math.max(rg.r2, 0), 0) + '</strong> of the variation in “' + esc(cfg.target.name) + '” (R² = ' + num(rg.r2) + '). ' +
        'Its predictions are off by ' + num(rg.mae, 2) + ' on average; always predicting the mean gives R² ≈ ' + num(r.base) + '.';
    }
    return { cards: h, text: text };
  }

  function render(r) {
    var cls = r.cfg.task === 'classification', h = '';
    h += '<p class="d-model"><span class="d-k">Model</span> <strong>' + esc(r.spec.label) + '</strong> · trained on ' + r.nTrain + ' cases · evaluated on <strong>' + r.nTest + '</strong> held-out cases</p>';
    h += '<div class="d-banner" id="d-banner"></div><div class="kpis" id="d-kpis"></div>';
    h += '<p class="d-note">Green: AUC, sensitivity or specificity ≥ 0.7, accuracy or F1 at least 5 points above the baseline, or R² ≥ 0.3. Orange: below that.</p>';

    if (binary(r)) {
      h += '<div class="thr"><label for="d-thr">Decision threshold: <output id="d-thr-out">50%</output></label>' +
        '<input type="range" id="d-thr" min="5" max="95" step="5" value="50" aria-describedby="d-thr-help">' +
        '<button type="button" class="btn small" id="d-pct" aria-pressed="false">Show %</button></div>' +
        '<p class="d-note" id="d-thr-help">A case counts as “' + esc(r.d.labels[1]) + '” when the model’s probability is at least the threshold. Lower it to catch more positives, at the price of more false alarms.</p>';
    }
    if (cls) {
      h += '<div class="pair"><div><h3>Confusion matrix</h3><p class="d-note">Rows: true class. Columns: predicted. Click a cell to see the cases behind it.</p><div class="d-scroll" id="d-cm"></div></div>';
      if (binary(r)) h += '<div><h3>ROC curve</h3><p class="d-note">How well the model separates the two classes at every threshold.</p><div id="d-roc"></div></div>';
      h += '</div><div id="d-cases" class="cases" aria-live="polite"></div>';
    } else {
      h += scatterSvg(r);
    }

    if (r.board) {
      var worst = Math.min.apply(null, r.board.map(function (b) { return b.cv; }).concat([0]));
      h += '<h3>Model comparison</h3><p class="d-note">5-fold cross-validation on the training data (' + (cls ? 'accuracy' : 'R²') + ').</p>' +
        bars(r.board.map(function (b, i) {
          return { n: b.spec.label + (i === 0 ? ' · chosen' : ''), v: b.cv, s: cls ? pct(b.cv) : num(b.cv), best: i === 0 };
        }), worst, 1);
    }

    var order = r.imp.map(function (v, i) { return { n: r.cfg.features[i], v: v }; }).sort(function (a, b) { return b.v - a.v; });
    var top = Math.max(order[0].v, 1e-9);
    h += '<h3>What mattered most</h3><p class="d-note">Score lost on the test cases when one input is shuffled (permutation importance). Bigger means the model relied on it more.</p>' +
      bars(order.map(function (o, i) { return { n: o.n, v: o.v, s: (o.v >= 0 ? '−' : '+') + Math.abs(o.v).toFixed(3), best: i === 0 && o.v > 0 }; }), 0, top);

    h += samplesHtml(r);
    el.out.innerHTML = h;
    update();
  }

  function bars(items, lo, hi) {
    return '<ul class="d-bars">' + items.map(function (b) {
      var w = Math.max(1.5, (Math.max(b.v, lo) - lo) / ((hi - lo) || 1) * 100);
      return '<li' + (b.best ? ' class="best"' : '') + '><span class="n">' + esc(b.n) + '</span><span class="bar" aria-hidden="true"><i style="width:' + w.toFixed(1) + '%"></i></span><span class="s">' + b.s + '</span></li>';
    }).join('') + '</ul>';
  }

  /* Re-draws the parts that depend on the threshold or the % toggle. */
  function update() {
    var r = last, cs = cardsAndSummary(r);
    $('d-banner').innerHTML = cs.text; $('d-kpis').innerHTML = cs.cards;
    if (r.cfg.task !== 'classification') return;
    var bs = binary(r) ? binStats(r) : null;
    $('d-cm').innerHTML = confusionHtml(r, bs ? bs.cm : r.cm);
    if (bs) $('d-roc').innerHTML = rocSvg(r, bs);
    showCases(r);
  }

  function confusionHtml(r, cm) {
    var L = r.d.labels, max = Math.max.apply(null, [].concat.apply([], cm)) || 1;
    var rowTot = cm.map(function (row) { return row.reduce(function (a, b) { return a + b; }, 0) || 1; });
    var h = '<table class="d-cm"><thead><tr><th></th>' + L.map(function (l) { return '<th scope="col">' + lab(l) + '</th>'; }).join('') + '</tr></thead><tbody>';
    cm.forEach(function (row, i) {
      h += '<tr><th scope="row">' + lab(L[i]) + '</th>' + row.map(function (v, j) {
        var on = view.cell && view.cell[0] === i && view.cell[1] === j;
        return '<td class="' + (i === j ? 'ok' : 'bad') + '" style="--a:' + (v / max).toFixed(2) + '"><button type="button" data-r="' + i + '" data-c="' + j + '" aria-pressed="' + !!on + '" aria-label="' +
          v + ' cases: true ' + esc(L[i]) + ', predicted ' + esc(L[j]) + '">' + (view.pct ? pct(v / rowTot[i], 0) : v) + '</button></td>';
      }).join('') + '</tr>';
    });
    return h + '</tbody></table>';
  }

  function showCases(r) {
    var box = $('d-cases'), c = view.cell;
    if (!c) { box.innerHTML = ''; return; }
    var pred = binary(r) ? binStats(r).pred : r.pred, L = r.d.labels, rows = [];
    r.yte.forEach(function (t, i) { if (t === c[0] && pred[i] === c[1]) rows.push(i); });
    var shown = rows.slice(0, 8);
    box.innerHTML = '<h3>' + rows.length + ' case' + (rows.length === 1 ? '' : 's') + ': true “' + esc(L[c[0]]) + '”, predicted “' + esc(L[c[1]]) + '”</h3>' +
      (rows.length ? '<div class="d-scroll"><table class="d-tab"><thead><tr>' + r.cfg.features.map(function (f) { return '<th scope="col">' + esc(f) + '</th>'; }).join('') + '</tr></thead><tbody>' +
        shown.map(function (i) { return '<tr>' + r.Xte[i].map(function (v) { return '<td>' + cell(v) + '</td>'; }).join('') + '</tr>'; }).join('') + '</tbody></table></div>' +
        (rows.length > shown.length ? '<p class="d-note">…and ' + (rows.length - shown.length) + ' more.</p>' : '') : '<p class="d-note">No test case falls in this cell.</p>');
  }

  el.out.addEventListener('click', function (e) {
    var b = e.target.closest('.d-cm button');
    if (b && last) {
      var c = [+b.dataset.r, +b.dataset.c];
      view.cell = view.cell && view.cell[0] === c[0] && view.cell[1] === c[1] ? null : c;
      update(); return;
    }
    if (e.target.id === 'd-pct') {
      view.pct = !view.pct; e.target.textContent = view.pct ? 'Show counts' : 'Show %'; e.target.setAttribute('aria-pressed', view.pct); update();
    }
  });
  el.out.addEventListener('input', function (e) {
    if (e.target.id !== 'd-thr' || !last) return;
    view.thr = e.target.value / 100; $('d-thr-out').textContent = e.target.value + '%'; update();
  });

  function rocSvg(r, s) {
    var W = 320, H = 300, l = 46, rt = 12, t = 12, bt = 42, pw = W - l - rt, ph = H - t - bt;
    var X = function (v) { return (l + v * pw).toFixed(1); }, Y = function (v) { return (t + (1 - v) * ph).toFixed(1); };
    var d = r.roc.fpr.map(function (f, i) { return (i ? 'L' : 'M') + X(f) + ' ' + Y(r.roc.tpr[i]); }).join('');
    var h = '<svg class="d-plot" viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="ROC curve, AUC ' + r.roc.auc.toFixed(3) + '. At the current threshold sensitivity is ' + pct(s.sens, 0) + ' and the false positive rate is ' + pct(1 - s.spec, 0) + '.">' +
      '<rect x="' + l + '" y="' + t + '" width="' + pw + '" height="' + ph + '" class="frame"/>';
    for (var k = 0; k <= 5; k++) {
      var v = k / 5;
      h += '<line x1="' + X(v) + '" y1="' + t + '" x2="' + X(v) + '" y2="' + (t + ph) + '" class="grid"/><line x1="' + l + '" y1="' + Y(v) + '" x2="' + (l + pw) + '" y2="' + Y(v) + '" class="grid"/>' +
        '<text x="' + X(v) + '" y="' + (t + ph + 15) + '" text-anchor="middle">' + v.toFixed(1) + '</text><text x="' + (l - 6) + '" y="' + (+Y(v) + 4) + '" text-anchor="end">' + v.toFixed(1) + '</text>';
    }
    h += '<line x1="' + X(0) + '" y1="' + Y(0) + '" x2="' + X(1) + '" y2="' + Y(1) + '" class="diag"/><path d="' + d + '" class="roc"/>' +
      '<circle cx="' + X(1 - s.spec) + '" cy="' + Y(s.sens) + '" r="5" class="op"/>' +
      '<text x="' + (l + pw / 2) + '" y="' + (H - 6) + '" text-anchor="middle">False positive rate</text>' +
      '<text transform="translate(12 ' + (t + ph / 2) + ') rotate(-90)" text-anchor="middle">True positive rate</text>' +
      '<line x1="' + (l + pw - 118) + '" y1="' + (t + ph - 16) + '" x2="' + (l + pw - 98) + '" y2="' + (t + ph - 16) + '" class="roc"/><text x="' + (l + pw - 92) + '" y="' + (t + ph - 12) + '" class="legend">AUC = ' + r.roc.auc.toFixed(3) + '</text></svg>';
    return h;
  }

  function scatterSvg(r) {
    var all = r.yte.concat(r.pred), lo = Math.min.apply(null, all), hi = Math.max.apply(null, all), pad = (hi - lo) * 0.05 || 1;
    lo -= pad; hi += pad;
    var S = 300, m = 34, sc = function (v) { return m + (v - lo) / (hi - lo) * (S - m - 8); }, sy = function (v) { return S - m - (v - lo) / (hi - lo) * (S - m - 8) + 8; };
    var h = '<h3>Predicted vs. true</h3><p class="d-note">Each dot is a test case; the closer to the diagonal, the better.</p>' +
      '<svg class="d-plot" viewBox="0 0 ' + S + ' ' + S + '" role="img" aria-label="Scatter plot of predicted against true values for the test cases">' +
      '<line x1="' + sc(lo) + '" y1="' + sy(lo) + '" x2="' + sc(hi) + '" y2="' + sy(hi) + '" class="diag"/>' +
      '<line x1="' + m + '" y1="' + (S - m + 8) + '" x2="' + (S - 8) + '" y2="' + (S - m + 8) + '" class="axis"/><line x1="' + m + '" y1="8" x2="' + m + '" y2="' + (S - m + 8) + '" class="axis"/>';
    r.yte.forEach(function (v, i) { h += '<circle cx="' + sc(v).toFixed(1) + '" cy="' + sy(r.pred[i]).toFixed(1) + '" r="3.2" class="dot"/>'; });
    h += '<text x="' + (S / 2 + 12) + '" y="' + (S - 8) + '" text-anchor="middle">true value</text>' +
      '<text transform="translate(11 ' + (S / 2 - 10) + ') rotate(-90)" text-anchor="middle">predicted</text></svg>';
    return h;
  }

  function samplesHtml(r) {
    var cls = r.cfg.task === 'classification', n = Math.min(8, r.yte.length), rows = '';
    for (var i = 0; i < n; i++) {
      var t = r.yte[i], p = r.pred[i];
      if (cls) rows += '<tr><td>' + (i + 1) + '</td><td>' + lab(r.d.labels[t]) + '</td><td>' + lab(r.d.labels[p]) + '</td><td class="' + (t === p ? 'good' : 'wrong') + '">' + (t === p ? '✓ correct' : '✗ wrong') + '</td></tr>';
      else rows += '<tr><td>' + (i + 1) + '</td><td>' + num(t, 1) + '</td><td>' + num(p, 1) + '</td><td>' + (p - t >= 0 ? '+' : '−') + num(Math.abs(p - t), 1) + '</td></tr>';
    }
    return '<h3>Example predictions</h3><p class="d-note">The first ' + n + ' test cases' + (binary(r) ? ' (at the default 50% threshold)' : '') + '.</p><div class="d-scroll"><table class="d-tab pred"><thead><tr><th scope="col">#</th><th scope="col">True</th><th scope="col">Predicted</th><th scope="col">' + (cls ? 'Result' : 'Error') + '</th></tr></thead><tbody>' + rows + '</tbody></table></div>';
  }

  function addRun(r) {
    var c = r.cfg, s0 = binary(r) ? ML.atThreshold(r.pos, r.prob, 0.5) : null;
    var score = binary(r) ? 'AUC ' + r.roc.auc.toFixed(3) + ' · acc ' + pct(s0.acc, 0)
      : c.task === 'classification' ? pct(r.main) : 'R² ' + num(r.main);
    runs.unshift({
      n: (runs.length ? runs[0].n : 0) + 1,
      what: esc(current.name) + ' → ' + esc(c.target.name),
      cfg: esc(r.spec.label) + (c.model === 'auto' ? ' (auto)' : '') + ' · ' + (c.standardize ? 'scaled' : 'unscaled') + ' · ' + c.features.length + ' inputs · ' + Math.round(c.testFrac * 100) + '% test',
      score: score
    });
    runs = runs.slice(0, 6);
    el.hist.hidden = false;
    el.histBody.innerHTML = runs.map(function (x) {
      return '<tr><td>' + x.n + '</td><td>' + x.what + '</td><td>' + x.cfg + '</td><td>' + x.score + '</td></tr>';
    }).join('');
  }

  load();
})();
