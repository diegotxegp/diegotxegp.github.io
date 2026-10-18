document.addEventListener('click', function (ev) {
  var b = ev.target.closest && ev.target.closest('#mail');
  if (!b) return;
  var addr = atob(b.getAttribute('data-e')).split('').reverse().join('');
  var a = document.createElement('a');
  a.className = 'btn primary';
  a.href = 'mailto:' + addr;
  a.textContent = addr;
  b.parentNode.replaceChild(a, b);
});
