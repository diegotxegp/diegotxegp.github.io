# How to add a post (posts are written in English)

1. Create `posts/my-post.html` with only the text of the post (`<p>`, `<h2>`, `<ul>`, `<a>`, `<img>`...). The name may only contain lowercase letters, numbers and hyphens.
2. Add an entry to `posts/posts.json` with the same `slug` (no `.html`), the `date` (YYYY-MM-DD), the `title` and, optionally, a short `summary`.
3. Commit and push: the post appears in `blog.html` and opens at `post.html?p=my-post`.

To hide a post, remove it from `posts.json`.
