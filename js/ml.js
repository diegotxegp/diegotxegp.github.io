/* Tiny machine-learning toolkit for the in-browser demo.
   Real implementations (no simulated numbers) of a few classic models, a
   preprocessing step, cross-validation and metrics. Pure JS, no dependencies.
   Classification targets are integers 0..K-1; regression targets are floats. */
(function (root) {
  'use strict';

  /* ---------- utilities ---------- */
  function rng(seed) {                       // mulberry32: small, deterministic
    var a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      var t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function shuffled(n, rand) {
    var idx = []; for (var i = 0; i < n; i++) idx.push(i);
    for (var j = n - 1; j > 0; j--) {
      var k = Math.floor(rand() * (j + 1)), t = idx[j]; idx[j] = idx[k]; idx[k] = t;
    }
    return idx;
  }
  function pick(arr, idx) { return idx.map(function (i) { return arr[i]; }); }
  function mean(a) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i]; return a.length ? s / a.length : 0; }
  function argmax(a) { var b = 0; for (var i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; }

  /* Stratified for classification so small classes survive the split. */
  function splitIndices(y, task, testFrac, seed) {
    var rand = rng(seed), test = [], train = [];
    if (task === 'classification') {
      var groups = {};
      y.forEach(function (c, i) { (groups[c] = groups[c] || []).push(i); });
      Object.keys(groups).forEach(function (c) {
        var g = groups[c], order = shuffled(g.length, rand);
        var nt = Math.max(1, Math.round(g.length * testFrac));
        order.forEach(function (o, p) { (p < nt ? test : train).push(g[o]); });
      });
    } else {
      var order = shuffled(y.length, rand), nt = Math.max(1, Math.round(y.length * testFrac));
      order.forEach(function (o, p) { (p < nt ? test : train).push(o); });
    }
    return { train: train, test: test };
  }

  /* ---------- preprocessing ---------- */
  function fitScaler(X) {
    var d = X[0].length, m = new Array(d).fill(0), s = new Array(d).fill(0);
    X.forEach(function (r) { for (var j = 0; j < d; j++) m[j] += r[j]; });
    for (var j = 0; j < d; j++) m[j] /= X.length;
    X.forEach(function (r) { for (var j = 0; j < d; j++) s[j] += (r[j] - m[j]) * (r[j] - m[j]); });
    for (j = 0; j < d; j++) { s[j] = Math.sqrt(s[j] / X.length) || 1; }
    return function (A) { return A.map(function (r) { return r.map(function (v, j) { return (v - m[j]) / s[j]; }); }); };
  }

  /* ---------- models: each returns {fit(X,y,K), predict(X)} ---------- */
  function baseline() {                       // majority class / mean: the bar to beat
    var out, task;
    return {
      fit: function (X, y, K) {
        task = K ? 'c' : 'r';
        if (task === 'c') { var c = new Array(K).fill(0); y.forEach(function (v) { c[v]++; }); out = argmax(c); }
        else out = mean(y);
      },
      predict: function (X) { return X.map(function () { return out; }); }
    };
  }

  function knn(k) {
    var Xt, yt, K;
    return {
      fit: function (X, y, nc) { Xt = X; yt = y; K = nc; },
      predict: function (X) {
        var kk = Math.min(k, Xt.length);
        return X.map(function (q) {
          var d = Xt.map(function (r, i) {
            var s = 0; for (var j = 0; j < r.length; j++) { var e = r[j] - q[j]; s += e * e; }
            return [s, i];
          }).sort(function (a, b) { return a[0] - b[0]; }).slice(0, kk);
          if (K) { var v = new Array(K).fill(0); d.forEach(function (p) { v[yt[p[1]]]++; }); return argmax(v); }
          return mean(d.map(function (p) { return yt[p[1]]; }));
        });
      }
    };
  }

  function tree(maxDepth, minLeaf) {
    var root, K;
    function leafVal(idx, y) {
      if (K) { var c = new Array(K).fill(0); idx.forEach(function (i) { c[y[i]]++; }); return argmax(c); }
      return mean(idx.map(function (i) { return y[i]; }));
    }
    function build(X, y, idx, depth) {
      var n = idx.length, node = { v: leafVal(idx, y) };
      if (depth >= maxDepth || n < 2 * minLeaf) return node;
      var best = null, d = X[0].length, total, ssTot, sTot;
      if (K) { total = new Array(K).fill(0); idx.forEach(function (i) { total[y[i]]++; }); if (Math.max.apply(null, total) === n) return node; }
      else { sTot = 0; ssTot = 0; idx.forEach(function (i) { sTot += y[i]; ssTot += y[i] * y[i]; }); }
      for (var f = 0; f < d; f++) {
        var o = idx.slice().sort(function (a, b) { return X[a][f] - X[b][f]; });
        var left = K ? new Array(K).fill(0) : null, sL = 0, ssL = 0;
        for (var p = 0; p < n - 1; p++) {
          var i = o[p];
          if (K) left[y[i]]++; else { sL += y[i]; ssL += y[i] * y[i]; }
          var nl = p + 1, nr = n - nl;
          if (nl < minLeaf || nr < minLeaf || X[o[p]][f] === X[o[p + 1]][f]) continue;
          var cost;
          if (K) {
            var gl = 1, gr = 1;
            for (var c = 0; c < K; c++) { var pl = left[c] / nl, pr = (total[c] - left[c]) / nr; gl -= pl * pl; gr -= pr * pr; }
            cost = nl * gl + nr * gr;
          } else {
            cost = (ssL - sL * sL / nl) + ((ssTot - ssL) - (sTot - sL) * (sTot - sL) / nr);
          }
          if (!best || cost < best.cost) best = { cost: cost, f: f, t: (X[o[p]][f] + X[o[p + 1]][f]) / 2 };
        }
      }
      if (!best) return node;
      var L = [], R = [];
      idx.forEach(function (i) { (X[i][best.f] <= best.t ? L : R).push(i); });
      node.f = best.f; node.t = best.t;
      node.l = build(X, y, L, depth + 1); node.r = build(X, y, R, depth + 1);
      return node;
    }
    return {
      fit: function (X, y, nc) { K = nc; var idx = []; for (var i = 0; i < X.length; i++) idx.push(i); root = build(X, y, idx, 0); },
      predict: function (X) {
        return X.map(function (r) { var n = root; while (n.l) n = r[n.f] <= n.t ? n.l : n.r; return n.v; });
      }
    };
  }

  function logistic(iters, lr, l2) {          // multinomial, batch gradient descent
    var W, K;
    function probs(r) {
      var z = W.map(function (w) { var s = w[w.length - 1]; for (var j = 0; j < r.length; j++) s += w[j] * r[j]; return s; });
      var m = Math.max.apply(null, z), e = z.map(function (v) { return Math.exp(v - m); }), s = 0;
      e.forEach(function (v) { s += v; });
      return e.map(function (v) { return v / s; });
    }
    return {
      fit: function (X, y, nc) {
        K = nc; var d = X[0].length, n = X.length;
        W = []; for (var c = 0; c < K; c++) W.push(new Array(d + 1).fill(0));
        for (var it = 0; it < iters; it++) {
          var G = W.map(function (w) { return new Array(w.length).fill(0); });
          for (var i = 0; i < n; i++) {
            var p = probs(X[i]);
            for (c = 0; c < K; c++) {
              var e = p[c] - (y[i] === c ? 1 : 0);
              for (var j = 0; j < d; j++) G[c][j] += e * X[i][j];
              G[c][d] += e;
            }
          }
          for (c = 0; c < K; c++) for (j = 0; j <= d; j++) W[c][j] -= lr * (G[c][j] / n + (j < d ? l2 * W[c][j] : 0));
        }
      },
      predict: function (X) { return X.map(function (r) { return argmax(probs(r)); }); }
    };
  }

  function gaussianNB() {
    var st, prior, K;
    return {
      fit: function (X, y, nc) {
        K = nc; var d = X[0].length; st = []; prior = [];
        for (var c = 0; c < K; c++) {
          var rows = X.filter(function (_, i) { return y[i] === c; });
          prior.push(Math.log((rows.length + 1e-9) / X.length));
          var m = [], v = [];
          for (var j = 0; j < d; j++) {
            var col = rows.map(function (r) { return r[j]; }), mu = mean(col);
            m.push(mu); v.push(mean(col.map(function (a) { return (a - mu) * (a - mu); })) + 1e-6);
          }
          st.push({ m: m, v: v });
        }
      },
      predict: function (X) {
        return X.map(function (r) {
          var s = st.map(function (c, k) {
            var ll = prior[k];
            for (var j = 0; j < r.length; j++) ll += -0.5 * Math.log(2 * Math.PI * c.v[j]) - (r[j] - c.m[j]) * (r[j] - c.m[j]) / (2 * c.v[j]);
            return ll;
          });
          return argmax(s);
        });
      }
    };
  }

  function ridge(lambda) {                    // closed form, intercept not penalised
    var w;
    return {
      fit: function (X, y) {
        var d = X[0].length + 1, A = [], b = new Array(d).fill(0), i, j, k;
        for (i = 0; i < d; i++) A.push(new Array(d).fill(0));
        X.forEach(function (r, n) {
          var z = r.concat([1]);
          for (i = 0; i < d; i++) { b[i] += z[i] * y[n]; for (j = 0; j < d; j++) A[i][j] += z[i] * z[j]; }
        });
        for (i = 0; i < d - 1; i++) A[i][i] += lambda;
        for (i = 0; i < d; i++) A[i].push(b[i]);          // Gauss-Jordan with pivoting
        for (i = 0; i < d; i++) {
          var p = i; for (k = i + 1; k < d; k++) if (Math.abs(A[k][i]) > Math.abs(A[p][i])) p = k;
          var t = A[i]; A[i] = A[p]; A[p] = t;
          var piv = A[i][i] || 1e-12;
          for (j = i; j <= d; j++) A[i][j] /= piv;
          for (k = 0; k < d; k++) if (k !== i) { var f = A[k][i]; for (j = i; j <= d; j++) A[k][j] -= f * A[i][j]; }
        }
        w = A.map(function (r) { return r[d]; });
      },
      predict: function (X) { return X.map(function (r) { var s = w[w.length - 1]; for (var j = 0; j < r.length; j++) s += w[j] * r[j]; return s; }); }
    };
  }

  /* Model catalogue per task. `make` builds a fresh, untrained model. */
  var CATALOG = {
    classification: [
      { id: 'logistic', label: 'Logistic regression', make: function () { return logistic(300, 0.1, 1e-3); } },
      { id: 'knn', label: 'k-nearest neighbours', make: function () { return knn(5); } },
      { id: 'tree', label: 'Decision tree', make: function () { return tree(4, 3); } },
      { id: 'nb', label: 'Naive Bayes', make: gaussianNB }
    ],
    regression: [
      { id: 'ridge', label: 'Linear regression (ridge)', make: function () { return ridge(1); } },
      { id: 'knn', label: 'k-nearest neighbours', make: function () { return knn(7); } },
      { id: 'tree', label: 'Decision tree', make: function () { return tree(4, 8); } }
    ]
  };
  var BASELINE = { id: 'baseline', label: 'Baseline (no learning)', make: baseline };

  /* ---------- pipeline = optional scaling + model ---------- */
  function pipeline(spec, standardize, K) {
    var model = spec.make(), tf = null;
    return {
      fit: function (X, y) { tf = standardize ? fitScaler(X) : null; model.fit(tf ? tf(X) : X, y, K); },
      predict: function (X) { return model.predict(tf ? tf(X) : X); }
    };
  }

  /* ---------- metrics ---------- */
  function accuracy(t, p) { var h = 0; t.forEach(function (v, i) { if (v === p[i]) h++; }); return h / t.length; }
  function confusion(t, p, K) {
    var m = []; for (var i = 0; i < K; i++) m.push(new Array(K).fill(0));
    t.forEach(function (v, i) { m[v][p[i]]++; });
    return m;
  }
  function macroF1(cm) {
    var f = cm.map(function (_, c) {
      var tp = cm[c][c], fp = 0, fn = 0;
      cm.forEach(function (row, r) { if (r !== c) { fp += row[c]; fn += cm[c][r]; } });
      return tp === 0 ? 0 : 2 * tp / (2 * tp + fp + fn);
    });
    return mean(f);
  }
  function regMetrics(t, p) {
    var m = mean(t), ssr = 0, sst = 0, sae = 0;
    t.forEach(function (v, i) { ssr += (v - p[i]) * (v - p[i]); sst += (v - m) * (v - m); sae += Math.abs(v - p[i]); });
    return { r2: sst ? 1 - ssr / sst : 0, rmse: Math.sqrt(ssr / t.length), mae: sae / t.length };
  }
  function score(task, t, p) { return task === 'classification' ? accuracy(t, p) : regMetrics(t, p).r2; }

  function crossValidate(spec, standardize, X, y, task, K, folds, seed) {
    var order = shuffled(X.length, rng(seed)), scores = [];
    for (var f = 0; f < folds; f++) {
      var va = order.filter(function (_, i) { return i % folds === f; });
      var tr = order.filter(function (_, i) { return i % folds !== f; });
      var m = pipeline(spec, standardize, K);
      m.fit(pick(X, tr), pick(y, tr));
      scores.push(score(task, pick(y, va), m.predict(pick(X, va))));
    }
    return mean(scores);
  }

  root.ML = {
    CATALOG: CATALOG, BASELINE: BASELINE, splitIndices: splitIndices, pick: pick,
    pipeline: pipeline, crossValidate: crossValidate, score: score,
    accuracy: accuracy, confusion: confusion, macroF1: macroF1, regMetrics: regMetrics
  };
})(typeof window !== 'undefined' ? window : globalThis);
