/** @typedef {'booting'|'loading'|'ready'|'empty'|'error'|'mutating'} ViewMode */
/** @typedef {'online'|'offline'} NetworkMode */
/** @typedef {'idle'|'syncing'|'conflict'|'failed'} SyncMode */
/** @typedef {'all'|'low_conf'|'very_low'|'wheat'} QueueFilter */

/**
 * @typedef {Object} ReviewItem
 * @property {string} image_id
 * @property {number} object_id
 * @property {string|null} crop_url
 * @property {string|null} crop_base64
 * @property {number} detector_conf
 * @property {string} predicted_species
 * @property {string} predicted_species_ru
 * @property {number} species_conf
 * @property {string} stage_ru
 * @property {boolean} review_required
 * @property {number[]|null} bbox_xyxy
 * @property {number|null} drone_lat
 * @property {number|null} drone_lon
 */

/**
 * @typedef {Object} VerificationAction
 * @property {string} localId
 * @property {string} itemKey
 * @property {number} createdAt
 * @property {string} image_id
 * @property {number} object_id
 * @property {string} verified_species
 * @property {string} verified_species_ru
 * @property {string} verified_stage_ru
 * @property {boolean} is_crop
 * @property {'spray_weed'|'do_not_spray'|'manual_review'} action
 * @property {string} device_id
 * @property {string} verified_by
 * @property {string} timestamp
 */

export const VIEW_MODE = Object.freeze({BOOTING:'booting',LOADING:'loading',READY:'ready',EMPTY:'empty',ERROR:'error',MUTATING:'mutating'});
export const NETWORK_MODE = Object.freeze({ONLINE:'online',OFFLINE:'offline'});
export const SYNC_MODE = Object.freeze({IDLE:'idle',SYNCING:'syncing',CONFLICT:'conflict',FAILED:'failed'});
export const FILTER = Object.freeze({ALL:'all',LOW:'low_conf',CRITICAL:'very_low',WHEAT:'wheat'});

const finite01 = (value) => Math.max(0, Math.min(1, Number.isFinite(Number(value)) ? Number(value) : 0));

/** @param {Partial<ReviewItem>} raw @returns {ReviewItem} */
export function normalizeReviewItem(raw) {
  const objectId = Number(raw.object_id);
  if (!raw.image_id || !Number.isFinite(objectId)) throw new TypeError('Некорректный объект очереди');
  return {
    image_id:String(raw.image_id), object_id:objectId,
    crop_url:raw.crop_url ? String(raw.crop_url) : null,
    crop_base64:raw.crop_base64 ? String(raw.crop_base64) : null,
    detector_conf:finite01(raw.detector_conf), predicted_species:String(raw.predicted_species || 'unknown'),
    predicted_species_ru:String(raw.predicted_species_ru || 'Не определено'), species_conf:finite01(raw.species_conf),
    stage_ru:String(raw.stage_ru || 'Не определено'), review_required:raw.review_required !== false,
    bbox_xyxy:Array.isArray(raw.bbox_xyxy) ? raw.bbox_xyxy.map(Number) : null,
    drone_lat:Number.isFinite(Number(raw.drone_lat)) ? Number(raw.drone_lat) : null,
    drone_lon:Number.isFinite(Number(raw.drone_lon)) ? Number(raw.drone_lon) : null,
  };
}

/** @param {ReviewItem} item */
export const itemKey = (item) => `${item.image_id}:${item.object_id}`;
