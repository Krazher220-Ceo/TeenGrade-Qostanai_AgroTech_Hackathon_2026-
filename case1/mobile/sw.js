// Service Worker: АгроСкаут Олжа Агро (PWA Offline First)
const CACHE_NAME = 'agrovision-field-v12';
const STATIC_ASSETS = [
  '/mobile/',
  '/mobile/index.html',
  '/mobile/styles.css?v=12',
  '/mobile/app.js?v=12',
  '/mobile/api.js?v=12',
  '/mobile/db.js?v=12',
  '/mobile/store.js?v=12',
  '/mobile/types.js?v=12',
  '/mobile/catalog.js?v=12',
  '/mobile/manifest.json',
  '/mobile/icon.svg'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => Promise.all(
      STATIC_ASSETS.map(async (url) => {
        const response = await fetch(url, { cache: 'reload' });
        if (!response.ok) throw new Error(`Не удалось кэшировать ${url}: HTTP ${response.status}`);
        await cache.put(url, response);
      })
    ))
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(
        keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key))
      );
    })
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  if (event.request.method !== 'GET') {
    return;
  }

  // Стратегия для вырезок сорняков (/crops/...): Cache First с сохранением в кэш
  if (url.pathname.startsWith('/crops/')) {
    event.respondWith(
      caches.open(CACHE_NAME).then((cache) => {
        return cache.match(event.request).then((response) => {
          if (response) {
            return response;
          }
          return fetch(event.request).then((networkResponse) => {
            if (networkResponse && networkResponse.status === 200) {
              cache.put(event.request, networkResponse.clone());
            }
            return networkResponse;
          }).catch(() => {
            // В случае полного оффлайна и отсутствия в кэше
            return new Response('Offline crop unavailable', { status: 503 });
          });
        });
      })
    );
    return;
  }

  // API остаётся под контролем app.js: AbortController и retry должны
  // отменять исходный запрос, а не копию Request внутри Service Worker.
  if (url.pathname.startsWith('/api/')) {
    return;
  }

  // Навигация: сеть сначала, затем гарантированный офлайн-shell.
  if (event.request.mode === 'navigate') {
    event.respondWith(
      fetch(event.request)
        .then((response) => {
          const copy = response.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put('/mobile/index.html', copy));
          return response;
        })
        .catch(() => caches.match('/mobile/index.html'))
    );
    return;
  }

  // Статика: cache-first; версии кэша меняются вместе с релизом shell.
  event.respondWith(
    caches.match(event.request).then((response) => {
      return response || fetch(event.request);
    })
  );
});
