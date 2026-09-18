# AgroVision AI — дизайн-система PWA

Все стили лежат в `styles.css` и разбиты на три слоя:

1. **Примитивы** (`--green-900`, `--lime-300`, …): сырая палитра. В компонентах их не используем.
2. **Семантические токены** (`--color-*`, `--space-*`, `--text-*`, `--radius-*`, `--shadow-*`, `--duration-*`). Компоненты ссылаются только на них. Тёмная тема переопределяет только этот слой.
3. **Компоненты**: `.button`, `.icon-button`, `.filter-tabs`, `.review-warning`, `.dialog-sheet`, `.toast` и т. д.

## Токены

| Группа | Токены | Правило |
|---|---|---|
| Поверхности | `--color-bg`, `--color-surface`, `--color-surface-muted`, `--color-overlay`, `--color-nav-bg` | Карточки ставим на `surface`, подложки — на `surface-muted` |
| Текст | `--color-text`, `--color-text-muted`, `--color-text-strong`, `--color-text-inverse` | `muted` только для вторичных подписей |
| Действия | `--color-primary` / `--color-on-primary`, `--color-accent` / `--color-on-accent`, `--color-selected-bg` / `-fg` | Пары фон/текст используем вместе |
| Статусы | `--color-success`, `--color-warning(-bg)`, `--color-danger(-bg)` | Статус всегда дублируется иконкой или текстом |
| Типографика | `--text-xs` 12 · `sm` 13 · `md` 14 · `lg` 17 · `xl` 22 · `2xl` 21–28 · `display` 30 | **Минимум 12px**: интерфейсом пользуются на ярком солнце |
| Отступы | `--space-1…14` (сетка 4px) | Произвольные значения в px не используем |
| Скругления | `--radius-xs` 6 · `sm` 10 · `md` 14 · `lg` 18 · `xl` 22 · `full` | Кнопки — `md`, оболочки и диалоги — `xl` |
| Тени | `--shadow-sm/md/lg` | `lg` оставляем для плавающих элементов (нижняя навигация, диалоги, offline-bar) |
| Анимация | `--duration-fast/base/slow`, `--ease-spring`, `--ease-press` | `prefers-reduced-motion` отключает анимацию глобально |
| Касание | `--touch` 54 (52 на телефоне), `--touch-min` 44 | Интерактивная зона не меньше 44×44 |

## Темы

- По умолчанию тема повторяет системную (`prefers-color-scheme`).
- Кнопка ☀ записывает `data-theme="light|dark"` на `<html>` и сохраняет выбор в `localStorage`.
- `<meta name="theme-color">` синхронизируется с темой в `app.js` (`applyTheme` / `syncThemeColor`).
- Поддерживается `prefers-contrast: more`: границы и вторичный текст становятся контрастнее.

## Кнопка

| Вариант | Когда использовать |
|---|---|
| `button--primary` | Одно главное действие на экране (Подтвердить, Следующий) |
| `button--secondary` | Альтернативное действие (Исправить вид) |
| `button--danger` | Отказ от обработки (Не опрыскивать) |
| `button--ghost` | Навигация и закрытие |
| `button--action` | Модификатор для крупных кнопок решения с подсказкой `<kbd>` |

Состояния: hover (подъём на 2px), active (scale .975), disabled (opacity .48), `:focus-visible` (кольцо `--color-focus`).

## Как добавлять стили

1. Нужен новый цвет — сначала добавьте примитив, затем семантический токен в обе темы.
2. Hex и rgba внутри правил компонентов не пишем (исключение — градиент `.image-shade` поверх фото).
3. После правки CSS поднимите `?v=N` в `index.html`, `sw.js` и в импортах JS, а также `CACHE_NAME` в `sw.js`.
