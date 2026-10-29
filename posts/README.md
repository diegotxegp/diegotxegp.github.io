# Cómo añadir una entrada

1. Crea `posts/mi-entrada.html` con solo el texto de la entrada (párrafos `<p>`, `<h2>`, `<ul>`, `<a>`, `<img>`...). El nombre solo puede llevar minúsculas, números y guiones.
2. Añade un bloque en `posts/posts.json` con el mismo `slug` (sin `.html`), la `date` (AAAA-MM-DD), el `title` y un `summary` breve.
3. Haz commit y push: la entrada aparece sola en `blog.html` y se abre en `post.html?p=mi-entrada`.

Para esconder una entrada, quítala de `posts.json`.
