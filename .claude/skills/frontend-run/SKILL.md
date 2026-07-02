---
name: frontend-run
description: Start the Vue 3 + Vite frontend dev server. Use when the user wants to launch the frontend UI.
user-invocable: true
---

# Frontend Run

Start the Vue 3 + Element Plus + Vite frontend dev server.

## Steps

1. First run or missing `node_modules`:

```bash
cd frontend && npm install
```

2. Start dev server:

```bash
cd frontend && npm run dev
```

Frontend will be available at `http://127.0.0.1:5173`.

## Other commands

```bash
cd frontend && npm run build     # Production build with type-check
cd frontend && npm run preview   # Preview production build
```
