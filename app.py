
import io, base64, numpy as np, nest_asyncio, uvicorn, asyncio
from PIL import Image
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pyngrok import ngrok
from tensorflow import keras

nest_asyncio.apply()
model = keras.models.load_model("/content/mnist_cnn.keras")
app = FastAPI(title="MNIST CNN")

class Img(BaseModel):
    image: str

def preprocess(data_url):
    raw = base64.b64decode(data_url.split(",")[1])
    arr = np.array(Image.open(io.BytesIO(raw)).convert("L"))
    ys, xs = np.where(arr > 30)
    if len(xs) == 0:
        return None
    arr = arr[ys.min():ys.max()+1, xs.min():xs.max()+1]
    h, w = arr.shape
    s = 20 / max(h, w)
    im = Image.fromarray(arr).resize((max(1, int(w*s)), max(1, int(h*s))), Image.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(im, ((28-im.width)//2, (28-im.height)//2))
    return np.array(canvas, dtype="float32") / 255.0

@app.post("/predict")
def predict(d: Img):
    x = preprocess(d.image)
    if x is None:
        return {"error": "Draw a digit first"}
    p = model.predict(x[None, ..., None], verbose=0)[0]
    return {"digit": int(p.argmax()), "confidence": float(p.max()), "probs": p.tolist()}

HTML = """
<!DOCTYPE html><html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Digit AI</title>
<style>
*{box-sizing:border-box;margin:0;padding:0;font-family:system-ui,sans-serif}
body{min-height:100vh;display:flex;align-items:center;justify-content:center;padding:16px;
background:radial-gradient(circle at 20% 10%,#2b1b5e,#0b0f1f 60%);color:#e8ecff}
.card{width:100%;max-width:380px;padding:22px;border-radius:24px;
background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.12);
backdrop-filter:blur(14px);box-shadow:0 20px 60px rgba(0,0,0,.5)}
h1{font-size:22px;text-align:center;
background:linear-gradient(90deg,#7c5cff,#22d3ee);-webkit-background-clip:text;color:transparent}
.sub{text-align:center;font-size:12px;opacity:.6;margin:4px 0 16px}
canvas{display:block;width:100%;aspect-ratio:1;border-radius:18px;background:#000;
border:2px solid #7c5cff;touch-action:none;box-shadow:0 0 24px rgba(124,92,255,.35)}
.row{display:flex;gap:10px;margin:14px 0}
button{flex:1;padding:12px;border:0;border-radius:14px;font-weight:600;font-size:15px;color:#fff;cursor:pointer}
#pred{background:linear-gradient(90deg,#7c5cff,#22d3ee)}
#clr{background:rgba(255,255,255,.12)}
.res{text-align:center;margin:6px 0 12px}
#digit{font-size:64px;font-weight:800;line-height:1;color:#22d3ee}
#conf{font-size:13px;opacity:.7}
.bar{display:flex;align-items:center;gap:8px;font-size:12px;margin:4px 0}
.bar span{width:12px;text-align:right;opacity:.8}
.track{flex:1;height:8px;border-radius:6px;background:rgba(255,255,255,.1);overflow:hidden}
.fill{height:100%;width:0;border-radius:6px;background:linear-gradient(90deg,#7c5cff,#22d3ee);transition:width .3s}
.bar em{width:44px;font-style:normal;opacity:.7;font-size:11px}
</style></head><body>
<div class="card">
<h1>Handwritten Digit AI</h1>
<div class="sub">CNN trained on MNIST &middot; 99.19% accuracy</div>
<canvas id="c" width="280" height="280"></canvas>
<div class="row"><button id="clr">Clear</button><button id="pred">Predict</button></div>
<div class="res"><div id="digit">?</div><div id="conf">Draw a digit above</div></div>
<div id="bars"></div>
</div>
<script>
const c=document.getElementById('c'),x=c.getContext('2d');
const bars=document.getElementById('bars');
for(let i=0;i<10;i++)bars.innerHTML+=`<div class="bar"><span>${i}</span><div class="track"><div class="fill" id="f${i}"></div></div><em id="p${i}">0%</em></div>`;
function reset(){x.fillStyle='#000';x.fillRect(0,0,280,280);x.strokeStyle='#fff';x.lineWidth=20;x.lineCap='round';x.lineJoin='round'}
reset();
let d=false,t;
function pos(e){const r=c.getBoundingClientRect(),p=e.touches?e.touches[0]:e;
return[(p.clientX-r.left)*280/r.width,(p.clientY-r.top)*280/r.height]}
function start(e){e.preventDefault();d=true;const[a,b]=pos(e);x.beginPath();x.moveTo(a,b);x.lineTo(a+.1,b+.1);x.stroke()}
function move(e){if(!d)return;e.preventDefault();const[a,b]=pos(e);x.lineTo(a,b);x.stroke()}
function end(){if(!d)return;d=false;clearTimeout(t);t=setTimeout(predict,300)}
c.addEventListener('mousedown',start);c.addEventListener('mousemove',move);window.addEventListener('mouseup',end);
c.addEventListener('touchstart',start,{passive:false});c.addEventListener('touchmove',move,{passive:false});c.addEventListener('touchend',end);
async function predict(){
const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image:c.toDataURL('image/png')})});
const j=await r.json();
if(j.error){document.getElementById('conf').innerText=j.error;return}
document.getElementById('digit').innerText=j.digit;
document.getElementById('conf').innerText='Confidence: '+(j.confidence*100).toFixed(2)+'%';
j.probs.forEach((p,i)=>{document.getElementById('f'+i).style.width=(p*100)+'%';document.getElementById('p'+i).innerText=(p*100).toFixed(1)+'%'})}
document.getElementById('pred').onclick=predict;
document.getElementById('clr').onclick=()=>{reset();document.getElementById('digit').innerText='?';
document.getElementById('conf').innerText='Draw a digit above';
for(let i=0;i<10;i++){document.getElementById('f'+i).style.width='0';document.getElementById('p'+i).innerText='0%'}}
</script></body></html>
"""

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML
