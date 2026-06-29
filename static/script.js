/* ═══════════════════════════════════════════════════════════
   script.js — Blood Vessel Blockage Detector
   Connected to Flask API: http://localhost:5000
═══════════════════════════════════════════════════════════ */

const API_BASE = 'http://localhost:5000';

const FEAT_CONFIG = [
  { id:'psv',   label:'Peak Systolic Velocity', unit:'cm/s', thresh:125,  dir:'above', norm:'70 – 125',    w:2 },
  { id:'ri',    label:'Resistive Index',         unit:'',     thresh:0.75, dir:'above', norm:'0.55 – 0.75', w:2 },
  { id:'bfv',   label:'Blood Flow Velocity',     unit:'cm/s', thresh:70,   dir:'below', norm:'> 70',        w:2 },
  { id:'csa',   label:'Cold Spot Area',          unit:'%',    thresh:12,   dir:'above', norm:'< 12%',       w:2 },
  { id:'temp',  label:'Avg Temperature',         unit:'°C',   thresh:null, dir:null,    norm:'36.1 – 37.1', w:0 },
  { id:'tdiff', label:'Temperature Difference',  unit:'°C',   thresh:3.5,  dir:'above', norm:'< 3.5',       w:1 },
  { id:'hr',    label:'Heart Rate',              unit:'bpm',  thresh:null, dir:null,    norm:'60 – 100',    w:0 },
  { id:'pa',    label:'Pulse Amplitude',         unit:'mmHg', thresh:1.2,  dir:'below', norm:'> 1.2',       w:1 },
  { id:'ptt',   label:'Pulse Transit Time',      unit:'sec',  thresh:0.28, dir:'above', norm:'< 0.28',      w:1 },
  { id:'hrv',   label:'HRV',                     unit:'ms',   thresh:40,   dir:'below', norm:'> 40',        w:1 },
];

const SAMPLES = {
  normal:   { psv:83.09, ri:0.70, bfv:96.25, temp:36.57, tdiff:2.41, csa:6.8,   hr:69,  pa:2.30, ptt:0.20, hrv:48 },
  blockage: { psv:129.25,ri:0.85, bfv:64.92, temp:35.73, tdiff:3.71, csa:20.32, hr:113, pa:0.84, ptt:0.29, hrv:24 }
};

let savedData = {};
function $(id) { return document.getElementById(id); }
function decimals(id) { return id==='ri'||id==='ptt'?3:id==='pa'?2:1; }

function liveCheck(id, thresh, dir, badgeId) {
  const v=parseFloat($(id).value), el=$(badgeId);
  if(isNaN(v)||!el) return;
  const bad=dir==='above'?v>thresh:v<thresh;
  el.className='thresh-tag '+(bad?'danger':'ok');
  el.textContent=bad?'⚠ Abnormal':'✓ Normal';
}

function loadSample(type) {
  const s=SAMPLES[type];
  for(const[k,v]of Object.entries(s)){const el=$(k);if(el)el.value=v;}
  [['psv',125,'above','v-psv'],['ri',0.75,'above','v-ri'],['bfv',70,'below','v-bfv'],
   ['csa',12,'above','v-csa'],['tdiff',3.5,'above','v-tdiff'],['pa',1.2,'below','v-pa'],
   ['ptt',0.28,'above','v-ptt'],['hrv',40,'below','v-hrv']].forEach(([id,t,d,b])=>liveCheck(id,t,d,b));
  if(type==='normal'){$('pat-name').value='Arjun Sharma';$('pat-age').value='42';$('pat-gender').value='Male';$('pat-doc').value='Dr. Priya Mehta';}
  else{$('pat-name').value='Sunita Verma';$('pat-age').value='58';$('pat-gender').value='Female';$('pat-doc').value='Dr. Ravi Kumar';}
  $('pat-date').value=new Date().toISOString().split('T')[0];
}

document.addEventListener('keydown',e=>{
  if(e.ctrlKey&&e.shiftKey&&e.key==='D'){const dt=$('dev-tools');dt.style.display=dt.style.display==='none'?'flex':'none';}
});

function getVals(){
  const r={};
  ['psv','ri','bfv','csa','temp','tdiff','hr','pa','ptt','hrv'].forEach(id=>{r[id]=parseFloat($(id).value)||0;});
  return r;
}

function getRiskClass(risk){
  const map={CRITICAL:'risk-critical',HIGH:'risk-high',MODERATE:'risk-moderate',LOW:'risk-low'};
  const dot={CRITICAL:'🔴',HIGH:'🟠',MODERATE:'🟡',LOW:'🟢'};
  return[map[risk]||'risk-low',dot[risk]||'🟢'];
}

function updateSteps(active){
  for(let i=1;i<=3;i++){const el=$('s'+i);if(i<active)el.className='step-circle done';else if(i===active)el.className='step-circle active';else el.className='step-circle';}
  for(let i=1;i<=2;i++)$('line'+i).className='step-line'+(i<active?' done':'');
}

function showPage(n){
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
  $('page'+n).classList.add('active');updateSteps(n);window.scrollTo({top:0,behavior:'smooth'});
}
function goToPage1(){showPage(1);}

function showLoading(msg='Analyzing...'){
  let o=$('loading-overlay');
  if(!o){o=document.createElement('div');o.id='loading-overlay';
    o.style.cssText='position:fixed;inset:0;background:rgba(15,42,61,0.7);display:flex;flex-direction:column;align-items:center;justify-content:center;z-index:9999;backdrop-filter:blur(4px);';
    o.innerHTML=`<div style="background:var(--white);border-radius:16px;padding:36px 48px;text-align:center;box-shadow:0 8px 32px rgba(0,0,0,0.2);"><div style="width:48px;height:48px;border:4px solid var(--accent-mid);border-top-color:var(--accent);border-radius:50%;animation:spin 0.8s linear infinite;margin:0 auto 16px;"></div><div id="loading-msg" style="font-size:15px;font-weight:600;color:var(--text);">${msg}</div><div style="font-size:12px;color:var(--muted);margin-top:6px;">Running MLP + VAE + Ensemble</div></div><style>@keyframes spin{to{transform:rotate(360deg)}}</style>`;
    document.body.appendChild(o);}
  o.style.display='flex';
}
function hideLoading(){const o=$('loading-overlay');if(o)o.style.display='none';}

async function callAPI(vals){
  const payload={peak_systolic_velocity:vals.psv,resistive_index:vals.ri,blood_flow_velocity:vals.bfv,avg_temperature:vals.temp,temperature_difference:vals.tdiff,cold_spot_area_percent:vals.csa,heart_rate:vals.hr,pulse_amplitude:vals.pa,pulse_transit_time:vals.ptt,hrv:vals.hrv};
  const response=await fetch(`${API_BASE}/predict`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  if(!response.ok){const err=await response.json();throw new Error(err.error||`Server error ${response.status}`);}
  return await response.json();
}

async function goToPage2(){
  const name=$('pat-name').value.trim(),age=$('pat-age').value,psv=$('psv').value;
  if(!name){alert('Please enter Patient Name');return;}
  if(!age){alert('Please enter Patient Age');return;}
  if(!psv){alert('Please enter Peak Systolic Velocity');return;}
  const vals=getVals();
  savedData={patient:{name,age,gender:$('pat-gender').value||'N/A',id:$('pat-id').value||('PT-'+Date.now().toString().slice(-6)),date:$('pat-date').value||new Date().toLocaleDateString(),doc:$('pat-doc').value||'N/A',notes:$('pat-notes').value||''},vals,result:null};
  showLoading('Running AI Analysis...');
  try{
    const result=await callAPI(vals);savedData.result=result;hideLoading();renderPage2(vals,result);showPage(2);
  }catch(err){
    hideLoading();console.warn('API fallback:',err.message);
    const fb=simulateFallback(vals);fb._fallback=true;savedData.result=fb;renderPage2(vals,fb);showPage(2);
    setTimeout(()=>{const w=document.createElement('div');w.style.cssText='position:fixed;bottom:20px;left:50%;transform:translateX(-50%);background:#fff0ee;border:1px solid #f5b8b3;color:#d93025;padding:10px 20px;border-radius:8px;font-size:13px;box-shadow:0 4px 12px rgba(0,0,0,0.1);z-index:999;';w.innerHTML='⚠️ <b>Server not running</b> — simulated results shown. Run: <code>python3 app.py</code>';document.body.appendChild(w);setTimeout(()=>w.remove(),6000);},300);
  }
}

function simulateFallback(vals){
  let s=0;
  if(vals.psv>125)s+=2;if(vals.ri>0.75)s+=2;if(vals.bfv<70)s+=2;if(vals.csa>12)s+=2;
  if(vals.tdiff>3.5)s+=1;if(vals.ptt>0.28)s+=1;if(vals.pa<1.2)s+=1;if(vals.hrv<40)s+=1;
  const sig=x=>1/(1+Math.exp(-x)),base=sig((s-4)*0.8);
  const mp=Math.min(Math.max(base+(Math.random()-.5)*0.03,0.01),0.99);
  const vp=Math.min(Math.max(s/12*1.05+(Math.random()-.5)*0.04,0.01),0.99);
  const mv=mp>0.5?1:0,vv=vp>0.45?1:0,votes=mv+vv,ep=mp*.55+vp*.45;
  const risk=s>=8||ep>=0.75?'CRITICAL':s>=6||ep>=0.55?'HIGH':s>=4||ep>=0.35?'MODERATE':'LOW';
  return{blockage:votes>=1?1:0,risk,clinical_score:s,ensemble_prob:ep,votes,mlp:{prediction:mv,probability:mp},vae:{prediction:vv,probability:vp,reconstruction_error:0,threshold:0}};
}

function goToPage3(){renderPage3();showPage(3);}

function renderPage2(vals,res){
  const isBlock=res.blockage===1,risk=res.risk||'LOW';
  const[riskClass,riskDot]=getRiskClass(risk);
  const score=res.clinical_score,ensProb=res.ensemble_prob,mlpProb=res.mlp.probability,vaeProb=res.vae.probability,mlpPred=res.mlp.prediction,vaePred=res.vae.prediction,votes=res.votes;
  $('r-hero').className='result-hero '+(isBlock?'blockage':'normal');
  $('r-icon').textContent=isBlock?'🚨':'✅';
  $('r-label').className='hero-label '+(isBlock?'red':'green');
  $('r-label').textContent=isBlock?'BLOCKAGE DETECTED':'NORMAL BLOOD FLOW';
  $('r-sub').textContent=isBlock?`Clinical score: ${score}/12 — Significant vascular obstruction markers found`:`Clinical score: ${score}/12 — Parameters within acceptable range`;
  const rb=$('r-risk-badge');rb.className='risk-badge '+riskClass;rb.textContent=riskDot+' '+risk+' RISK';
  $('r-score-val').textContent=score;$('r-score-val').style.color=isBlock?'var(--red)':'var(--green)';
  function setCard(cid,pred,det){$(cid).className='model-card '+(pred?'m-block':'m-norm');const r=$(cid.replace('-card','-res'));r.className='model-result '+(pred?'red':'green');r.textContent=pred?'BLOCKAGE':'NORMAL';$(cid.replace('-card','-det')).textContent=det;}
  setCard('mlp-card',mlpPred,`Prob: ${(mlpProb*100).toFixed(1)}%`);
  setCard('vae-card',vaePred,`Anomaly: ${(vaeProb*100).toFixed(1)}%`);
  setCard('ens-card',res.blockage,`${votes}/2 votes for blockage`);
  $('prob-bars').innerHTML=[{label:'MLP',prob:mlpProb,pred:mlpPred},{label:'VAE',prob:vaeProb,pred:vaePred},{label:'Ensemble',prob:ensProb,pred:res.blockage}].map(m=>`<div class="prob-row"><div class="prob-label">${m.label}</div><div class="prob-track"><div class="prob-fill ${m.pred?'fill-red':'fill-green'}" style="width:${(m.prob*100).toFixed(1)}%"><span>${(m.prob*100).toFixed(1)}%</span></div><div class="prob-midline"></div></div></div>`).join('');
  const criteria=[{label:'PSV > 125',ok:vals.psv>125,w:2},{label:'RI > 0.75',ok:vals.ri>0.75,w:2},{label:'BFV < 70',ok:vals.bfv<70,w:2},{label:'CSA > 12%',ok:vals.csa>12,w:2},{label:'TDiff > 3.5',ok:vals.tdiff>3.5,w:1},{label:'PTT > 0.28',ok:vals.ptt>0.28,w:1},{label:'PA < 1.2',ok:vals.pa<1.2,w:1},{label:'HRV < 40',ok:vals.hrv<40,w:1}];
  $('score-bars').innerHTML=criteria.map(c=>`<div class="score-bar-row"><div class="score-bar-label">${c.label}</div><div class="score-bar-track"><div class="score-bar-fill ${c.ok?'hit':'miss'}" style="width:${c.ok?100:0}%"></div></div><div class="score-bar-pts ${c.ok?'hit':'miss'}">${c.ok?'+'+(c.w*2)+' pts':'0 pts'}</div></div>`).join('');
  $('feat-tbody').innerHTML=FEAT_CONFIG.map(f=>{const v=vals[f.id],disp=v.toFixed(decimals(f.id));let isBad=false;if(f.dir==='above')isBad=v>f.thresh;if(f.dir==='below')isBad=v<f.thresh;const badge=f.dir===null?'<span class="badge badge-neutral">—</span>':isBad?'<span class="badge badge-bad">🔴 Abnormal</span>':'<span class="badge badge-ok">✅ Normal</span>';return`<tr><td style="font-size:12px;">${f.label}</td><td class="mono" style="color:${isBad?'var(--red)':'var(--text)'};font-size:13px;">${disp} ${f.unit}</td><td>${badge}</td></tr>`;}).join('');
}

function renderPage3(){
  const{patient,vals,result:res}=savedData;
  const isBlock=res.blockage===1,risk=res.risk||'LOW';
  const[riskClass,riskDot]=getRiskClass(risk);
  const repId='RPT-'+Date.now().toString().slice(-8).toUpperCase();
  const now=new Date(),dateStr=now.toLocaleDateString('en-IN',{day:'2-digit',month:'long',year:'numeric'}),timeStr=now.toLocaleTimeString('en-IN',{hour:'2-digit',minute:'2-digit'});
  $('rep-date').textContent='Date: '+dateStr;$('rep-id').textContent=repId;$('rep-footer-date').textContent='Generated: '+dateStr+' '+timeStr;
  $('rp-name').textContent=patient.name;$('rp-age').textContent=patient.age+' yrs / '+patient.gender;$('rp-pid').textContent=patient.id;$('rp-doc').textContent=patient.doc;
  $('rep-result-box').className='report-result-box '+(isBlock?'blockage':'normal');$('rep-icon').textContent=isBlock?'🚨':'✅';$('rep-label').className='rr-label '+(isBlock?'red':'green');$('rep-label').textContent=isBlock?'BLOCKAGE DETECTED':'NORMAL BLOOD FLOW';$('rep-sub').textContent=isBlock?'Significant vascular obstruction markers identified by AI ensemble':'No significant blockage detected — vascular parameters within normal range';
  const rr=$('rep-risk');rr.className='risk-badge '+riskClass;rr.textContent=riskDot+' '+risk+' RISK';
  $('rep-score').textContent=res.clinical_score+'/12';$('rep-score').style.color=isBlock?'var(--red)':'var(--green)';
  $('rep-models').innerHTML=[{name:'MLP',pred:res.mlp.prediction,prob:res.mlp.probability},{name:'VAE',pred:res.vae.prediction,prob:res.vae.probability},{name:'Ensemble',pred:res.blockage,prob:res.ensemble_prob}].map(m=>`<div class="ms-card ${m.pred?'b':'n'}"><div class="ms-name">${m.name}</div><div class="ms-res ${m.pred?'r':'g'}">${m.pred?'BLOCKAGE':'NORMAL'}</div><div class="ms-prob">${(m.prob*100).toFixed(1)}%</div></div>`).join('');
  const criteria=[{c:'PSV > 125 cm/s',v:vals.psv.toFixed(1)+' cm/s',ok:vals.psv>125,w:2},{c:'RI > 0.75',v:vals.ri.toFixed(3),ok:vals.ri>0.75,w:2},{c:'BFV < 70 cm/s',v:vals.bfv.toFixed(1)+' cm/s',ok:vals.bfv<70,w:2},{c:'CSA > 12%',v:vals.csa.toFixed(1)+'%',ok:vals.csa>12,w:2},{c:'TDiff > 3.5°C',v:vals.tdiff.toFixed(2)+'°C',ok:vals.tdiff>3.5,w:1},{c:'PTT > 0.28s',v:vals.ptt.toFixed(3)+'s',ok:vals.ptt>0.28,w:1},{c:'PA < 1.2 mmHg',v:vals.pa.toFixed(2)+' mmHg',ok:vals.pa<1.2,w:1},{c:'HRV < 40 ms',v:vals.hrv.toFixed(0)+' ms',ok:vals.hrv<40,w:1}];
  $('rep-score-tbody').innerHTML=criteria.map(c=>`<tr><td>${c.c}</td><td class="mono">${c.v}</td><td class="${c.ok?'abnormal':'normal-val'}">${c.ok?'+'+(c.w*2)+' pts':'0 pts'}</td></tr>`).join('');
  $('rep-feat-tbody').innerHTML=FEAT_CONFIG.map(f=>{const v=vals[f.id],disp=v.toFixed(decimals(f.id))+' '+(f.unit||'');let isBad=false;if(f.dir==='above')isBad=v>f.thresh;if(f.dir==='below')isBad=v<f.thresh;return`<tr><td>${f.label}</td><td class="mono ${isBad?'abnormal':''}">${disp}</td><td>${f.norm}</td><td>${f.dir===null?'—':isBad?'🔴 Abnormal':'✅ Normal'}</td></tr>`;}).join('');
  if(patient.notes){$('rep-notes-section').style.display='block';$('rep-notes-text').textContent=patient.notes;}
  else $('rep-notes-section').style.display='none';
}

document.addEventListener('DOMContentLoaded',()=>{
  $('pat-date').value=new Date().toISOString().split('T')[0];
  $('pat-id').placeholder='PT-'+Date.now().toString().slice(-6);
});
