/* Reveals the email address on demand (keeps it out of the HTML source for scrapers). */
(function () {
  var b = document.getElementById('mail');
  if (!b) return;
  b.addEventListener('click', function () {
    var a = document.createElement('a');
    a.className = 'btn';
    a.textContent = atob(b.dataset.e).split('').reverse().join('');
    a.href = 'mailto:' + a.textContent;
    b.replaceWith(a);
    a.focus();
  });
})();

/* Typewriter: cycles the last word of the lead (static "people" without JS or with reduced motion) */
(function () {
  var rot = document.querySelector('.rot');
  if (!rot || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  var words = rot.dataset.words.split('|'), wi = 0, ci = words[0].length, del = true;
  (function tick() {
    var delay = 90;
    if (del) { ci--; delay = 50; if (ci <= 0) { del = false; wi = (wi + 1) % words.length; } }
    else { ci++; if (ci >= words[wi].length) { del = true; delay = 1700; } }
    rot.textContent = words[wi].slice(0, Math.max(ci, 0));
    setTimeout(tick, delay);
  })();
})();

/* Highlight the index link of the section being read */
(function () {
  var links = document.querySelectorAll('.top nav a'), map = {};
  links.forEach(function (a) { map[a.getAttribute('href').slice(1)] = a; });
  var secs = Object.keys(map).map(function (id) { return document.getElementById(id); }).filter(Boolean);
  function mark() {
    var cur = secs[0], y = window.scrollY + window.innerHeight * 0.3;
    secs.forEach(function (el) { if (el.offsetTop <= y) cur = el; });
    if (window.innerHeight + window.scrollY >= document.body.scrollHeight - 4) cur = secs[secs.length - 1];
    links.forEach(function (a) { a.classList.toggle('on', a === map[cur.id]); });
  }
  window.addEventListener('scroll', mark, { passive: true });
  window.addEventListener('resize', mark);
  mark();
})();
