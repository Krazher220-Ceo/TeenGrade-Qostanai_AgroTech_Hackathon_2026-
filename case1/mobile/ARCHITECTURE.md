# AgroVision AI — frontend architecture

Production-oriented PWA for a field agronomist who validates model detections in bright sunlight, often without a stable connection. The UI does not claim local model accuracy: it renders the server's hypothesis and keeps the agronomist as the final decision-maker.

## 1. Data and state model

The executable JSDoc contracts live in `types.js`; the authoritative backend contract remains in `case1/server/schemas.py`.

```ts
type ViewMode = 'booting' | 'loading' | 'ready' | 'empty' | 'error' | 'mutating';
type NetworkMode = 'online' | 'offline';
type SyncMode = 'idle' | 'syncing' | 'conflict' | 'failed';
type QueueFilter = 'all' | 'low_conf' | 'very_low' | 'wheat';

interface ReviewItem {
  image_id: string;
  object_id: number;
  crop_url: string | null;
  crop_base64: string | null;
  detector_conf: number;       // 0..1
  predicted_species: string;
  predicted_species_ru: string;
  species_conf: number;        // 0..1
  stage_ru: string;
  review_required: boolean;
  bbox_xyxy: number[] | null;
  drone_lat: number | null;    // coordinate of the frame, not of an individual weed
  drone_lon: number | null;
}

interface VerificationAction {
  localId: string;             // client idempotency / undo key
  itemKey: string;             // image_id:object_id, deduplicates local decisions
  createdAt: number;
  image_id: string;
  object_id: number;
  verified_species: string;
  verified_species_ru: string;
  verified_stage_ru: string;
  is_crop: boolean;
  action: 'spray_weed' | 'do_not_spray' | 'manual_review';
  device_id: string;
  verified_by: string;
  timestamp: string;
}
```

### State machine

| Current state | Trigger | Guard / lock | Next state | Side effect |
|---|---|---|---|---|
| `booting` | IndexedDB opened | storage available | `loading` | read pending actions, request queue |
| `loading` | API success | payload passes normalization | `ready` / `empty` | cache normalized queue |
| `loading` | network/API failure | cached queue exists | `ready` | render cached data, announce fallback |
| `loading` | network/API failure | no cache | `error` | show retry without inventing sample data |
| `ready` | confirm / reject / reclassify | one action at a time | `mutating` | persist action before changing UI |
| `mutating` | IndexedDB commit | commit succeeded | `ready` / `empty` | optimistic removal, 4 s undo window |
| `mutating` | storage failure | commit failed | `error` | queue remains unchanged |
| any | `offline` event | — | same view + `network=offline` | sync is blocked, data stays local |
| any | manual/automatic sync | online and queue non-empty | `syncing` | send an immutable snapshot |
| `syncing` | full success | `failed_count == 0` | `idle` | delete only the snapshot IDs; preserve newer actions |
| `syncing` | partial response / HTTP 409 | — | `conflict` | retain every local action for human resolution/retry |
| `syncing` | transient failure | — | `failed` | retain queue and expose retry |

Race-condition controls:

- A new queue request aborts the previous request.
- A decision is written to IndexedDB before it disappears from the screen.
- Actions use a stable `itemKey`; a second local decision replaces the older unsynced decision for the same object.
- Sync deletes only IDs from its own snapshot, so a decision created while sync is running cannot be lost.
- Partial server success keeps the full snapshot locally. Server storage is keyed by `image_id:object_id`, so retry is idempotent for the current backend.

## 2. Design system

The visual direction is an outdoor-first operational surface: true white / cool green surfaces, dark forest navigation, high-contrast text, large controls, and restrained depth. Decorative dashboards, accuracy claims, and card grids are intentionally absent.

Core tokens:

- Color: `forest #063B2B`, `leaf #2E7D32`, `lime #B8E26D`, `background #F2F6F3`, `ink #10251D`, `warning #9B4600`, `danger #A83A18`.
- Typography: `Avenir Next` → `Segoe UI Variable` → system sans; display 20–30 px, body 12–14 px, utility 10–12 px.
- Touch: 54 px default, never below 48 px; mobile action controls remain 52–56 px.
- Radius: 10 / 16 / 22 px. Major surfaces use a 6 px outer shell plus an inner surface (double bezel).
- Elevation: one soft ambient shadow; focus is represented by a visible 3 px outline rather than shadow alone.
- Motion: only `transform` and `opacity`, spring-like cubic Bézier curves; `prefers-reduced-motion` disables non-essential motion.
- Layers: navigation 20, modal 40, toast 50. No arbitrary z-index values.

Themes:

- Light is the default outdoor theme.
- Dark uses the same semantic tokens and persists locally.
- `prefers-contrast: more` strengthens boundaries and muted text without altering information hierarchy.

Responsive rules:

- `> 980 px`: fixed navigation rail, image workspace, decision rail.
- `681–980 px`: compact navigation rail; decision panel moves above the canvas for tablet reach.
- `≤ 680 px`: bottom navigation, single-column layout, fixed mobile-safe controls, reduced secondary metadata.

## 3. Components and user flows

```text
AppShell
├── Sidebar / BottomNavigation
├── Topbar
│   ├── FieldSelector
│   ├── NetworkStatus
│   ├── SyncSummary
│   └── ThemeToggle
├── QueueFilters + SortControl
├── ReviewWorkspace
│   ├── LoadingSkeleton | ErrorState | EmptyState
│   └── ReviewLayout
│       ├── MediaReview (pointer gestures + queue navigation)
│       └── DecisionPanel (confirm / reclassify / do-not-spray)
├── SpeciesDialog (search + growth stage)
├── SyncQueueDialog
├── OfflineBar
└── ToastRegion + ARIA live region
```

Happy path:

1. The app normalizes the API queue and caches it.
2. The agronomist inspects the crop, confidence and growth stage.
3. Confirm, reclassify, or mark as crop/no spray via 54 px controls, keyboard, or swipe.
4. The decision is committed locally, the next item appears immediately, and undo remains available for four seconds.
5. After the undo window the client syncs the immutable action snapshot.

Offline path:

1. A network event changes only the network substate; the current screen remains usable.
2. The cached queue and crop cache continue to work.
3. Each decision is persisted and counted in the offline bar.
4. Reconnection starts background sync. A failed or partial batch stays on-device.

Human-in-the-loop path:

1. Low confidence or `review_required` is announced visually and through semantic text.
2. “Исправить вид” opens a keyboard-accessible native dialog with search and stage selection.
3. The agronomist's explicit species/stage is stored as the final local decision.
4. No spray prescription is treated as final until the local action has been accepted by the server.

## 4. Runtime modules

- `index.html`: semantic component skeleton, inline SVG symbol set, dialogs and live regions.
- `styles.css`: tokens, themes, responsive layout, focus/contrast/reduced-motion rules.
- `types.js`: contracts, enums and API boundary normalization.
- `store.js`: reducer/FSM and derived selectors.
- `db.js`: IndexedDB cache and durable sync queue.
- `api.js`: abortable, timeout-bound requests with retry for transient server failures.
- `catalog.js`: domain catalog and growth stages.
- `app.js`: controller, rendering, gestures, keyboard controls and orchestration.
- `sw.js`: versioned app-shell cache, crop cache, network-only API behavior and navigation fallback.
