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
