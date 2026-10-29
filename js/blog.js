/* Blog: the list comes from posts/posts.json and each entry is an HTML fragment in posts/<slug>.html */
(function () {
  var fmt = function (iso) {
    var d = new Date(iso + 'T12:00:00');
    return isNaN(d) ? iso : d.toLocaleDateString('es-ES', { year: 'numeric', month: 'long', day: 'numeric' });
  };
  function get(url, json) {
    return fetch(url).then(function (r) {
      if (!r.ok) throw new Error(r.status);
      return json ? r.json() : r.text();
    });
  }
  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text) e.textContent = text;
    return e;
  }

  var list = document.getElementById('posts');
  if (list) {
    get('posts/posts.json', true).then(function (posts) {
      list.textContent = '';
      posts.sort(function (a, b) { return a.date < b.date ? 1 : -1; });
      if (!posts.length) { list.appendChild(el('li', 'd-state', 'Todavía no hay entradas.')); return; }
      posts.forEach(function (p) {
        var li = el('li'), a = el('a', 'post-card');
        a.href = 'post.html?p=' + encodeURIComponent(p.slug);
        a.appendChild(el('span', 'meta', fmt(p.date)));
        a.appendChild(el('strong', 'post-title', p.title));
        if (p.summary) a.appendChild(el('span', 'post-sum', p.summary));
        li.appendChild(a);
        list.appendChild(li);
      });
    }).catch(function () {
      list.textContent = '';
      list.appendChild(el('li', 'd-state err', 'No se pudieron cargar las entradas. Recarga la página.'));
    });
  }

  var body = document.getElementById('p-body');
  if (body) {
    var slug = new URLSearchParams(location.search).get('p') || '';
    var title = document.getElementById('p-title'), date = document.getElementById('p-date');
    var fail = function () { title.textContent = 'Entrada no encontrada'; body.appendChild(el('p', 'hint', 'Esa entrada no existe. Vuelve al blog para ver las disponibles.')); };
    if (!/^[a-z0-9-]+$/.test(slug)) fail();
    else Promise.all([get('posts/posts.json', true), get('posts/' + slug + '.html')]).then(function (r) {
      var meta = r[0].filter(function (p) { return p.slug === slug; })[0];
      if (!meta) return fail();
      title.textContent = meta.title;
      date.textContent = fmt(meta.date);
      document.title = meta.title + ' · Diego García-Prieto';
      body.innerHTML = r[1]; /* the fragments are written by the site owner */
    }).catch(fail);
  }
})();
