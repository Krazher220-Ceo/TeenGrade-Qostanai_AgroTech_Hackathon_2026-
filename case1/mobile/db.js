const DB_NAME='AgroVision_Field_DB';
const DB_VERSION=1;
let connection;

export function openDatabase(){
  if(connection)return connection;
  connection=new Promise((resolve,reject)=>{
    const request=indexedDB.open(DB_NAME,DB_VERSION);
    request.onupgradeneeded=()=>{
      const db=request.result;
      if(!db.objectStoreNames.contains('actions'))db.createObjectStore('actions',{keyPath:'localId'});
      if(!db.objectStoreNames.contains('cache'))db.createObjectStore('cache',{keyPath:'key'});
    };
    request.onsuccess=()=>resolve(request.result);
    request.onerror=()=>reject(request.error || new Error('IndexedDB недоступна'));
  });
  return connection;
}

async function store(name,mode='readonly'){const db=await openDatabase();return db.transaction(name,mode).objectStore(name)}
function requestResult(request){return new Promise((resolve,reject)=>{request.onsuccess=()=>resolve(request.result);request.onerror=()=>reject(request.error)})}

export async function cacheQueue(items){const s=await store('cache','readwrite');return requestResult(s.put({key:'reviewQueue',items,savedAt:Date.now()}))}
export async function readCachedQueue(){const s=await store('cache');const row=await requestResult(s.get('reviewQueue'));return row?.items || []}
export async function listActions(){const s=await store('actions');return requestResult(s.getAll())}
export async function removeAction(localId){const s=await store('actions','readwrite');return requestResult(s.delete(localId))}
export async function clearActions(){const s=await store('actions','readwrite');return requestResult(s.clear())}

export async function upsertAction(action){
  const actions=await listActions();
  const duplicate=actions.find((row)=>row.itemKey===action.itemKey);
  if(duplicate)await removeAction(duplicate.localId);
  const s=await store('actions','readwrite');
  await requestResult(s.put(action));
  return action;
}
