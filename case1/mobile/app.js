import { loadReviewQueue, syncActions } from './api.js?v=12';
import { cacheQueue, listActions, openDatabase, readCachedQueue, removeAction, upsertAction } from './db.js?v=12';
import { SPECIES, STAGES } from './catalog.js?v=12';
import { createStore, filtered, selectCurrent } from './store.js?v=12';
import { FILTER, NETWORK_MODE, SYNC_MODE, VIEW_MODE, itemKey } from './types.js?v=12';
const darkQuery=window.matchMedia('(prefers-color-scheme: dark)');
function currentTheme(){return document.documentElement.dataset.theme || (darkQuery.matches?'dark':'light')}
function syncThemeColor(){const meta=document.querySelector('meta[name="theme-color"]');if(meta)meta.content=currentTheme()==='dark'?'#0a2219':'#063b2b'}
function applyTheme(theme){document.documentElement.dataset.theme=theme;syncThemeColor()}
darkQuery.addEventListener('change',syncThemeColor);

const $=(id)=>document.getElementById(id);
const ui={
  loading:$('loadingState'),review:$('reviewState'),empty:$('emptyState'),error:$('errorState'),errorMessage:$('errorMessage'),
  network:$('networkState'),pending:$('pendingCount'),offlineCount:$('offlineCount'),offlineBar:$('offlineBar'),syncHint:$('syncHint'),
  allCount:$('allCount'),image:$('cropImage'),media:document.querySelector('.media-core'),object:$('objectLabel'),gps:$('gpsLabel'),
  species:$('speciesName'),stage:$('stageName'),confidence:$('confidenceValue'),detector:$('detectorValue'),warning:$('reviewWarning'),
  progressText:$('progressText'),frame:$('fieldFrame'),progress:$('queueProgress'),previous:$('previousButton'),next:$('nextButton'),
  confirm:$('confirmButton'),edit:$('editButton'),quickEdit:$('quickEditButton'),reject:$('rejectButton'),sort:$('sortSelect'),
  speciesDialog:$('speciesDialog'),speciesSearch:$('speciesSearch'),speciesList:$('speciesList'),stageOptions:$('stageOptions'),
  syncDialog:$('syncDialog'),syncList:$('syncList'),toast:$('toastRegion'),status:$('srStatus'),
};
const store=createStore();
let syncTimer=0;
let dragStart=null;
let selectedStage=STAGES[0];

function getDeviceId(){
  let id=localStorage.getItem('agrovision_device_id');
  if(!id){id=`field_${crypto.randomUUID?.() || Math.random().toString(36).slice(2)}`;localStorage.setItem('agrovision_device_id',id)}
  return id;
}
const pct=(value)=>`${Math.round(Number(value||0)*100)}%`;
const currentImage=(item)=>item.crop_base64?`data:image/jpeg;base64,${item.crop_base64}`:(item.crop_url || '/mobile/icon.svg');
const localId=()=>crypto.randomUUID?.() || `${Date.now()}_${Math.random().toString(36).slice(2)}`;

function render(state){
  const items=filtered(state); const current=selectCurrent(state);
  ui.loading.hidden=state.mode!==VIEW_MODE.LOADING&&state.mode!==VIEW_MODE.BOOTING;
  ui.review.hidden=!current || state.mode===VIEW_MODE.LOADING || state.mode===VIEW_MODE.ERROR;
  ui.empty.hidden=state.mode!==VIEW_MODE.EMPTY;
  ui.error.hidden=state.mode!==VIEW_MODE.ERROR;
  ui.errorMessage.textContent=state.error?.message || 'Проверьте соединение или повторите загрузку.';
  ui.pending.textContent=String(state.pending.length);ui.offlineCount.textContent=String(state.pending.length);ui.offlineBar.hidden=state.pending.length===0;
  ui.syncHint.textContent=state.sync===SYNC_MODE.SYNCING?'Отправляем решения…':state.sync===SYNC_MODE.CONFLICT?'Нужна ручная проверка':'Очередь синхронизации';
  ui.allCount.textContent=String(state.queue.length);ui.sort.value=state.sort;
  document.querySelectorAll('[data-filter]').forEach((button)=>button.setAttribute('aria-selected',String(button.dataset.filter===state.filter)));
  renderNetwork(state); renderSyncList(state.pending);
  if(!current)return;
  ui.image.src=currentImage(current); ui.image.onerror=()=>{ui.image.onerror=null;ui.image.src='/mobile/icon.svg';showToast('Изображение недоступно офлайн')};
  ui.object.textContent=`Объект #${current.object_id}`;
  ui.gps.textContent=current.drone_lat!==null&&current.drone_lon!==null?`Координаты кадра ${current.drone_lat.toFixed(5)}, ${current.drone_lon.toFixed(5)}`:'Координаты кадра не указаны';
  ui.species.textContent=current.predicted_species_ru;ui.stage.textContent=current.stage_ru;ui.confidence.textContent=pct(current.species_conf);ui.detector.textContent=pct(current.detector_conf);
  const needsReview=current.review_required || current.species_conf<.65;
  ui.warning.classList.toggle('is-safe',!needsReview);
  ui.warning.querySelector('strong').textContent=needsReview?'Требуется подтверждение агронома':'Уверенное предположение модели';
  ui.warning.querySelector('small').textContent=needsReview?'Проверьте объект до включения в карту обработки.':'Финальное решение всё равно принимает агроном.';
  ui.progressText.textContent=`Объект ${state.index+1} из ${items.length}`;ui.frame.textContent=current.image_id;ui.progress.max=Math.max(1,items.length);ui.progress.value=state.index+1;
  ui.previous.disabled=state.index===0;ui.next.disabled=state.index>=items.length-1;
  const locked=state.mode===VIEW_MODE.MUTATING;[ui.confirm,ui.edit,ui.quickEdit,ui.reject].forEach((button)=>button.disabled=locked);
}

function renderNetwork(state){
  const online=state.network===NETWORK_MODE.ONLINE;
  ui.network.className=`status-block ${online?'is-online':'is-offline'}`;
  ui.network.setAttribute('aria-label',online?'Связь доступна, синхронизация разрешена':'Офлайн-режим, данные сохраняются на устройстве');
  ui.network.querySelector('strong').textContent=online?'Связь доступна':'Офлайн-режим';
  ui.network.querySelector('small').textContent=online?'Синхронизация разрешена':'Данные сохраняются на устройстве';
}

function renderSyncList(actions){
  ui.syncList.replaceChildren();
  if(!actions.length){const p=document.createElement('p');p.textContent='Локальная очередь пуста.';p.className='center-state-copy';ui.syncList.append(p);return}
  actions.forEach((action)=>{
    const row=document.createElement('div');row.className='sync-row';
    const copy=document.createElement('div');const title=document.createElement('strong');title.textContent=action.verified_species_ru;
    const meta=document.createElement('small');meta.textContent=`${action.image_id} · объект #${action.object_id}`;copy.append(title,meta);
    const state=document.createElement('span');state.textContent='Ожидает';row.append(copy,state);ui.syncList.append(row);
  });
}

async function refreshPending(){const actions=await listActions();store.dispatch({type:'PENDING',actions});return actions}

async function loadQueue(){
  store.dispatch({type:'LOAD_START'});
  try{
    if(!navigator.onLine)throw new Error('Нет сети');
    const items=await loadReviewQueue(store.getState().sort);await cacheQueue(items);store.dispatch({type:'NETWORK',online:true});store.dispatch({type:'LOAD_SUCCESS',items});
  }catch(error){
    store.dispatch({type:'NETWORK',online:false});
    const cached=await readCachedQueue();
    if(cached.length){store.dispatch({type:'LOAD_SUCCESS',items:cached});showToast('Открыта сохранённая очередь')}
    else store.dispatch({type:'LOAD_ERROR',error:new Error('На устройстве пока нет сохранённой очереди. Подключитесь к серверу и повторите.')});
  }
}

function createAction(item,details){
  return {localId:localId(),itemKey:itemKey(item),createdAt:Date.now(),image_id:item.image_id,object_id:item.object_id,
    verified_species:details.species,verified_species_ru:details.speciesRu,verified_stage_ru:details.stage || item.stage_ru,
    is_crop:Boolean(details.isCrop),action:details.action,device_id:getDeviceId(),verified_by:'Полевой агроном («Олжа Агро»)',timestamp:new Date().toISOString()};
}

async function decide(details){
  const item=selectCurrent(store.getState());if(!item || store.getState().mode===VIEW_MODE.MUTATING)return;
  store.dispatch({type:'MUTATE_START'});
  const action=createAction(item,details);
  try{
    await upsertAction(action);store.dispatch({type:'REMOVE_CURRENT'});await refreshPending();
    ui.status.textContent=`Решение по объекту ${item.object_id} сохранено на устройстве`;
    showToast('Решение сохранено',async()=>{await removeAction(action.localId);store.dispatch({type:'UNDO_REMOVE'});await refreshPending();ui.status.textContent='Последнее решение отменено'});
    clearTimeout(syncTimer);syncTimer=window.setTimeout(()=>syncPending(false),4800);
  }catch(error){store.dispatch({type:'LOAD_ERROR',error:new Error('Не удалось сохранить решение на устройстве. Очередь не изменена.')})}
}

async function syncPending(force=true){
  if(!navigator.onLine){showToast('Нет сети: решения остаются на устройстве');return}
  const all=await listActions();const actions=force?all:all.filter((action)=>Date.now()-action.createdAt>4300);
  if(!actions.length)return;
  store.dispatch({type:'SYNC_START'});
  try{
    const result=await syncActions(actions,getDeviceId());
    if(!result.success || result.failed_count>0)throw Object.assign(new Error('Сервер принял не все решения. Локальная копия сохранена.'),{conflict:true});
    await Promise.all(actions.map((action)=>removeAction(action.localId)));const remaining=await listActions();store.dispatch({type:'SYNC_SUCCESS',actions:remaining});showToast(`Синхронизировано: ${result.synced_count}`);
  }catch(error){if(!error.status)store.dispatch({type:'NETWORK',online:false});store.dispatch({type:'SYNC_FAILED',error,conflict:Boolean(error.conflict||error.status===409)});showToast(error.message || 'Синхронизация не удалась')}
}

function showToast(message,onUndo=null){
  const toast=document.createElement('div');toast.className='toast';const text=document.createElement('span');text.textContent=message;toast.append(text);
  let active=true;
  if(onUndo){const button=document.createElement('button');button.type='button';button.textContent='Отменить';button.onclick=async()=>{if(!active)return;active=false;await onUndo();dismiss()};toast.append(button)}
  const dismiss=()=>{toast.classList.add('is-leaving');setTimeout(()=>toast.remove(),310)};
  ui.toast.append(toast);setTimeout(()=>{active=false;dismiss()},4000);
}

function renderCatalog(query=''){
  const normalized=query.trim().toLocaleLowerCase('ru');ui.speciesList.replaceChildren();
  SPECIES.filter((species)=>!normalized || species.ru.toLocaleLowerCase('ru').includes(normalized)||species.id.includes(normalized)).forEach((species)=>{
    const button=document.createElement('button');button.type='button';button.className='species-option';button.setAttribute('role','option');
    const info=document.createElement('span');const name=document.createElement('strong');name.textContent=species.ru;const meta=document.createElement('small');meta.textContent=species.category;info.append(name,meta);
    const choose=document.createElement('span');choose.textContent='Выбрать';button.append(info,choose);
    button.onclick=()=>{ui.speciesDialog.close();decide({species:species.id,speciesRu:species.ru,stage:selectedStage,isCrop:species.isCrop,action:species.isCrop?'do_not_spray':'spray_weed'})};ui.speciesList.append(button);
  });
}

function openCatalog(){
  selectedStage=selectCurrent(store.getState())?.stage_ru || STAGES[0];ui.stageOptions.replaceChildren();
  STAGES.forEach((stage)=>{const button=document.createElement('button');button.type='button';button.textContent=stage;button.setAttribute('aria-pressed',String(stage===selectedStage));button.onclick=()=>{selectedStage=stage;[...ui.stageOptions.children].forEach((item)=>item.setAttribute('aria-pressed',String(item===button)))};ui.stageOptions.append(button)});
  ui.speciesSearch.value='';renderCatalog();ui.speciesDialog.showModal();setTimeout(()=>ui.speciesSearch.focus(),40);
}

function openSync(){renderSyncList(store.getState().pending);ui.syncDialog.showModal()}
function navigate(delta){store.dispatch({type:'INDEX',index:store.getState().index+delta})}

function bindEvents(){
  document.querySelectorAll('[data-filter]').forEach((button)=>button.addEventListener('click',()=>store.dispatch({type:'FILTER',filter:button.dataset.filter})));
  document.querySelectorAll('[data-toast]').forEach((button)=>button.addEventListener('click',()=>showToast(button.dataset.toast)));
  ui.sort.addEventListener('change',()=>{store.dispatch({type:'SORT',sort:ui.sort.value});loadQueue()});
  ui.previous.addEventListener('click',()=>navigate(-1));ui.next.addEventListener('click',()=>navigate(1));
  ui.confirm.addEventListener('click',()=>{const item=selectCurrent(store.getState());if(item)decide({species:item.predicted_species,speciesRu:item.predicted_species_ru,stage:item.stage_ru,isCrop:false,action:'spray_weed'})});
  ui.reject.addEventListener('click',()=>decide({species:'crop_wheat',speciesRu:'Пшеница (культура / фон)',stage:'Не определено',isCrop:true,action:'do_not_spray'}));
  ui.edit.addEventListener('click',openCatalog);ui.quickEdit.addEventListener('click',openCatalog);ui.speciesSearch.addEventListener('input',()=>renderCatalog(ui.speciesSearch.value));
  $('retryButton').addEventListener('click',loadQueue);$('resetFilterButton').addEventListener('click',()=>store.dispatch({type:'FILTER',filter:FILTER.ALL}));
  [$('syncButton'),$('offlineSyncButton')].forEach((button)=>button.addEventListener('click',openSync));$('dialogSyncButton').addEventListener('click',()=>syncPending(true));
  $('themeButton').addEventListener('click',()=>{const next=currentTheme()==='dark'?'light':'dark';applyTheme(next);try{localStorage.setItem('agrovision_theme',next)}catch{}});
  window.addEventListener('online',()=>{store.dispatch({type:'NETWORK',online:true});syncPending(false)});window.addEventListener('offline',()=>store.dispatch({type:'NETWORK',online:false}));
  document.addEventListener('keydown',(event)=>{if(ui.speciesDialog.open||ui.syncDialog.open||event.target.matches('input,select'))return;if(event.key==='ArrowLeft')navigate(-1);if(event.key==='ArrowRight')navigate(1);if(event.key.toLowerCase()==='d')ui.confirm.click();if(event.key.toLowerCase()==='e')openCatalog();if(event.key.toLowerCase()==='a')ui.reject.click()});
  bindGesture();
}

function bindGesture(){
  ui.media.addEventListener('pointerdown',(event)=>{dragStart={id:event.pointerId,x:event.clientX};ui.media.setPointerCapture(event.pointerId)});
  ui.media.addEventListener('pointermove',(event)=>{if(!dragStart||dragStart.id!==event.pointerId)return;const dx=event.clientX-dragStart.x;ui.image.style.transform=`translateX(${Math.max(-120,Math.min(120,dx))}px) rotate(${dx/35}deg)`;ui.media.dataset.drag=dx>35?'right':dx<-35?'left':''});
  const finish=(event)=>{if(!dragStart||dragStart.id!==event.pointerId)return;const dx=event.clientX-dragStart.x;dragStart=null;ui.image.style.transform='';delete ui.media.dataset.drag;if(dx>90)ui.confirm.click();else if(dx<-90)ui.reject.click()};
  ui.media.addEventListener('pointerup',finish);ui.media.addEventListener('pointercancel',()=>{dragStart=null;ui.image.style.transform='';delete ui.media.dataset.drag});
}

async function boot(){
  let savedTheme=null;try{savedTheme=localStorage.getItem('agrovision_theme')}catch{}if(savedTheme)applyTheme(savedTheme);else syncThemeColor();store.subscribe(render);render(store.getState());bindEvents();
  try{await openDatabase();await refreshPending()}catch(error){store.dispatch({type:'LOAD_ERROR',error:new Error('Локальное хранилище заблокировано браузером.')});return}
  if('serviceWorker' in navigator)navigator.serviceWorker.register('/mobile/sw.js').catch(()=>showToast('Офлайн-кэш недоступен в этом браузере'));
  await loadQueue();
}

boot();
