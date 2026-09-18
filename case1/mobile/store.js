import { FILTER, NETWORK_MODE, SYNC_MODE, VIEW_MODE } from './types.js?v=12';

const initialState = Object.freeze({
  mode:VIEW_MODE.BOOTING, network:navigator.onLine ? NETWORK_MODE.ONLINE : NETWORK_MODE.OFFLINE,
  sync:SYNC_MODE.IDLE, queue:[], filter:FILTER.ALL, sort:'confidence_asc', index:0,
  pending:[], error:null, selectedStage:'Розетка', lastRemoved:null,
});

export function createStore() {
  let state = {...initialState};
  const listeners = new Set();
  const emit = () => listeners.forEach((listener) => listener(state));
  return {
    getState:() => state,
    subscribe(listener){listeners.add(listener);return () => listeners.delete(listener)},
    dispatch(event){state = reduce(state,event);emit();return state;},
  };
}

function reduce(state,event) {
  switch(event.type) {
    case 'LOAD_START': return {...state,mode:VIEW_MODE.LOADING,error:null};
    case 'LOAD_SUCCESS': return {...state,queue:event.items,index:0,mode:event.items.length?VIEW_MODE.READY:VIEW_MODE.EMPTY,error:null};
    case 'LOAD_ERROR': return {...state,mode:state.queue.length?VIEW_MODE.READY:VIEW_MODE.ERROR,error:event.error};
    case 'NETWORK': return {...state,network:event.online?NETWORK_MODE.ONLINE:NETWORK_MODE.OFFLINE};
    case 'FILTER': return {...state,filter:event.filter,index:0,mode:filtered({...state,filter:event.filter}).length?VIEW_MODE.READY:VIEW_MODE.EMPTY};
    case 'SORT': return {...state,sort:event.sort,index:0};
    case 'INDEX': { const items=filtered(state); const index=Math.max(0,Math.min(items.length-1,event.index)); return {...state,index}; }
    case 'MUTATE_START': return {...state,mode:VIEW_MODE.MUTATING};
    case 'REMOVE_CURRENT': {
      const current=selectCurrent(state); if(!current)return state;
      const queueIndex=state.queue.findIndex((item)=>item.image_id===current.image_id&&item.object_id===current.object_id);
      const queue=state.queue.filter((item)=>!(item.image_id===current.image_id&&item.object_id===current.object_id));
      const next={...state,queue,index:Math.max(0,Math.min(state.index,filtered({...state,queue}).length-1)),lastRemoved:{item:current,viewIndex:state.index,queueIndex}};
      return {...next,mode:filtered(next).length?VIEW_MODE.READY:VIEW_MODE.EMPTY};
    }
    case 'UNDO_REMOVE': {
      if(!state.lastRemoved)return state;
      const queue=[...state.queue];queue.splice(Math.max(0,state.lastRemoved.queueIndex),0,state.lastRemoved.item);
      const next={...state,queue,index:state.lastRemoved.viewIndex,lastRemoved:null}; return {...next,mode:VIEW_MODE.READY};
    }
    case 'PENDING': return {...state,pending:event.actions};
    case 'SYNC_START': return {...state,sync:SYNC_MODE.SYNCING};
    case 'SYNC_SUCCESS': return {...state,sync:SYNC_MODE.IDLE,pending:event.actions};
    case 'SYNC_FAILED': return {...state,sync:event.conflict?SYNC_MODE.CONFLICT:SYNC_MODE.FAILED,error:event.error};
    case 'STAGE': return {...state,selectedStage:event.stage};
    default:return state;
  }
}

export function filtered(state) {
  return state.queue.filter((item)=>{
    if(state.filter===FILTER.LOW)return item.species_conf<.6;
    if(state.filter===FILTER.CRITICAL)return item.species_conf<.5;
    if(state.filter===FILTER.WHEAT)return item.predicted_species==='crop_wheat';
    return true;
  });
}
export function selectCurrent(state){return filtered(state)[state.index] || null}
