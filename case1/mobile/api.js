import { normalizeReviewItem } from './types.js?v=12';

let activeLoadController=null;
const wait=(ms)=>new Promise((resolve)=>setTimeout(resolve,ms));

async function fetchJson(url,options={},retries=1){
  let lastError;
  for(let attempt=0;attempt<=retries;attempt+=1){
    const controller=options.controller || new AbortController();
    let timer;
    const timeout=new Promise((_,reject)=>{
      timer=setTimeout(()=>{
        controller.abort('timeout');
        const error=new Error('Сервер не ответил за 3 секунды');
        error.name='TimeoutError';
        reject(error);
      },3000);
    });
    try{
      const response=await Promise.race([
        fetch(url,{...options,signal:controller.signal,headers:{Accept:'application/json',...(options.headers||{})}}),
        timeout,
      ]);
      if(!response.ok){const error=new Error(`Сервер ответил HTTP ${response.status}`);error.status=response.status;throw error}
      return await response.json();
    }catch(error){lastError=error;if(error.name==='AbortError'||attempt===retries||error.status<500)break;await wait(350*(attempt+1))}
    finally{clearTimeout(timer)}
  }
  throw lastError;
}

export async function loadReviewQueue(sort){
  activeLoadController?.abort('superseded');
  activeLoadController=new AbortController();
  const data=await fetchJson(`/api/v1/review-queue?limit=150&sort_by=${encodeURIComponent(sort)}&include_base64=false`,{controller:activeLoadController},1);
  if(!Array.isArray(data.items))throw new TypeError('Сервер вернул очередь неизвестного формата');
  return data.items.map(normalizeReviewItem);
}

export async function syncActions(actions,deviceId){
  const payloadActions=actions.map(({localId,itemKey,createdAt,...payload})=>payload);
  return fetchJson('/api/v1/sync',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({device_id:deviceId,client_timestamp:new Date().toISOString(),actions:payloadActions})},1);
}
