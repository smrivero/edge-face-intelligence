import cv2, time, json, threading
import numpy as np
import pyrealsense2 as rs
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from depth_liveness import get_metrics

BASE = Path(__file__).resolve().parent
DATA = BASE / 'data/face_template.npy'
MODEL = str(BASE / 'models/arcface.onnx')
W,H,PORT = 640,480,8765
SAMPLES,THRESHOLD = 8,0.45  # provisional: calibrar con impostores

face_detector = cv2.FaceDetectorYN_create(str(BASE/'models/yunet.onnx'), '', (W,H), 0.65,0.3,5000)
net = cv2.dnn.readNetFromONNX(MODEL)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# InsightFace ArcFace 112x112 reference landmarks
TARGET = np.array([[38.2946,51.6963],[73.5318,51.5014],[56.0252,71.7366],[41.5493,92.3655],[70.7299,92.2041]],dtype=np.float32)
lock=threading.Lock()
condition=threading.Condition()
state={'jpeg':None,'seq':0,'enrolling':False,'vectors':[],'last_sample':0,'message':'Ready','metrics':{'decision':'WAITING'}}
template=np.load(DATA).astype(np.float32) if DATA.exists() else None
if template is not None: template /= np.linalg.norm(template)+1e-12

def embedding(img,face):
    # YuNet: eye1, eye2, nose, mouth1, mouth2
    p = np.array(face[4:14],dtype=np.float32).reshape(5,2)
    eyes = sorted(p[:2],key=lambda x:x[0])
    mouth = sorted(p[3:],key=lambda x:x[0])
    landmarks=np.array([eyes[0],eyes[1],p[2],mouth[0],mouth[1]],dtype=np.float32)
    matrix,_=cv2.estimateAffinePartial2D(landmarks,TARGET,method=cv2.LMEDS)
    if matrix is None: return None
    aligned=cv2.warpAffine(img,matrix,(112,112),borderValue=0)
    # blobFromImage swaps BGR -> RGB with swapRB=True
    blob=cv2.dnn.blobFromImage(aligned,1/127.5,(112,112),(127.5,127.5,127.5),swapRB=True,crop=False)
    net.setInput(blob)
    v=net.forward().reshape(-1).astype(np.float32)
    v/=np.linalg.norm(v)+1e-12
    return v

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        global template
        if self.path=='/enroll':
            with lock:
                state['vectors']=[]
                state['enrolling']=True
                state['last_sample']=0
                state['message']='Enrollment started'
            return self.reply(200,{'status':'capturing','samples_required':SAMPLES})
        if self.path=='/delete':
            with lock:
                template=None
                state['enrolling']=False
                state['vectors']=[]
                DATA.unlink(missing_ok=True)
                state['message']='Enrollment deleted'
            return self.reply(200,{'status':'deleted'})
        self.reply(404,{'error':'not found'})
    def reply(self,status,obj):
        b=json.dumps(obj).encode()
        self.send_response(status)
        self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(b)))
        self.end_headers()
        self.wfile.write(b)
    def do_GET(self):
        if self.path=='/status':
            with lock:
                result={'enrolled':template is not None,'enrolling':state['enrolling'],'samples':len(state['vectors']),'message':state['message'],'metrics':state['metrics']}
            return self.reply(200,result)
        if self.path=='/':
            page=b'<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Edge Face Intelligence | Face Verification</title><style>body{font:16px Arial,sans-serif;background:#151515;color:#eee;text-align:center;margin:15px}img{max-width:1280px;width:100%}button{padding:12px 18px;margin:6px;border:0;border-radius:6px;cursor:pointer}#enroll{background:#76b900}#delete{background:#555;color:white}pre{background:#252525;padding:15px;white-space:pre-wrap;text-align:left;display:inline-block;border-radius:6px;min-width:300px;max-width:90%;overflow:auto}progress{width:250px;accent-color:#76b900}</style></head><body><h2>Edge Face Intelligence | ArcFace + 3D Depth</h2><p>Indicador experimental. NO es autenticacion biometrica segura.</p><button id="enroll" onclick="enroll()">Registrar cara (8 muestras)</button><button id="delete" onclick="removeTemplate()">Borrar registro</button><p id="info">Cargando...</p><progress id="progress" max="8" value="0"></progress><div><img src="/stream.mjpg"></div><pre id="metrics"></pre><script>async function act(path){let r=await fetch(path,{method:\'POST\'});if(!r.ok)throw Error(\'HTTP \'+r.status);return r.json()}async function enroll(){if(!confirm(\'Registrar cara nuevamente?\'))return;try{await act(\'/enroll\')}catch(e){alert(e)}}async function removeTemplate(){if(!confirm(\'Borrar plantilla facial?\'))return;try{await act(\'/delete\')}catch(e){alert(e)}}async function update(){try{const r=await fetch(\'/status\',{cache:\'no-store\'});const s=await r.json();document.getElementById(\'info\').textContent=(s.enrolling?\'Registrando... \':s.enrolled?\'Registro completo. \':\'Sin registro. \')+(s.samples||0)+\'/8 muestras | \'+(s.message||\'\');document.getElementById(\'progress\').value=s.samples||0;document.getElementById(\'enroll\').disabled=s.enrolling;document.getElementById(\'delete\').disabled=s.enrolling;document.getElementById(\'metrics\').textContent=JSON.stringify(s.metrics||{},null,2)}catch(e){document.getElementById(\'info\').textContent=\'Error: \'+e}}update();setInterval(update,600)</script></body></html>'
            self.send_response(200)
            self.send_header('Content-Type','text/html')
            self.send_header('Content-Length',str(len(page)))
            self.end_headers()
            return self.wfile.write(page)
        if self.path!='/stream.mjpg': return self.reply(404,{'error':'not found'})
        self.send_response(200)
        self.send_header('Content-Type','multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control','no-store')
        self.end_headers()
        seq=-1
        try:
            while True:
                with condition:
                    condition.wait_for(lambda: state['seq']!=seq,timeout=5)
                    jpg=state['jpeg'];seq=state['seq']
                if jpg:
                    self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(jpg)).encode()+b'\r\n\r\n'+jpg+b'\r\n')
        except (BrokenPipeError,ConnectionResetError,OSError): pass
    def log_message(self,*args): pass

camera=rs.pipeline();config=rs.config()
config.enable_stream(rs.stream.color,W,H,rs.format.bgr8,30)
config.enable_stream(rs.stream.depth,W,H,rs.format.z16,30)
align=rs.align(rs.stream.color)
camera.start(config)
server=ThreadingHTTPServer(('127.0.0.1',PORT),Handler)
threading.Thread(target=server.serve_forever,daemon=True).start()
print(f'Listening at http://127.0.0.1:{PORT}; template={DATA}',flush=True)
last=time.perf_counter();fps=0
try:
    while True:
        frames=align.process(camera.wait_for_frames(5000))
        cf,df=frames.get_color_frame(),frames.get_depth_frame()
        if not cf or not df: continue
        img=np.asarray(cf.get_data()).copy()
        raw=np.asarray(df.get_data())
        depth_mm=raw.astype(np.float32)*df.get_units()*1000
        _,faces=face_detector.detect(img)
        count=0 if faces is None else len(faces)
        elapsed_ms=0
        with lock:
            enrolling=state['enrolling']
        metrics={'decision':'INCONCLUSIVE','reason':'NO_FACE' if count==0 else 'MULTIPLE_FACES', 'faces':count, 'identity_threshold_experimental':THRESHOLD}
        if count==1:
            face=faces[0]
            x,y,w,h=face[:4].astype(int)
            depth_result, pts=get_metrics(depth_mm,face)
            metrics.update(depth_result)
            start=time.perf_counter()
            vec=embedding(img,face)
            elapsed_ms=(time.perf_counter()-start)*1000
            with lock:
                if vec is not None and state['enrolling'] and time.monotonic()-state['last_sample']>.3:
                    state['vectors'].append(vec)
                    state['last_sample']=time.monotonic()
                    state['message']=f"Capturing {len(state['vectors'])}/{SAMPLES}"
                    if len(state['vectors'])>=SAMPLES:
                        avg=np.mean(state['vectors'],axis=0)
                        avg/=np.linalg.norm(avg)+1e-12
                        DATA.parent.mkdir(parents=True,exist_ok=True)
                        np.save(DATA,avg)
                        DATA.chmod(0o600)
                        template=avg
                        state['enrolling']=False
                        state['message']='Enrollment complete'
                enrolled=template.copy() if template is not None else None
                ongoing=state['enrolling']
                samples=len(state['vectors'])
            similarity=float(np.dot(vec,enrolled)) if vec is not None and enrolled is not None else None
            identity_ok=similarity is not None and similarity>=THRESHOLD
            metrics.update({'similarity':round(similarity,4) if similarity is not None else None,'identity_match':identity_ok,'arcface_ms':round(elapsed_ms,1)})
            if ongoing:
                decision='INCONCLUSIVE'; reason=f'ENROLLING_{samples}_OF_{SAMPLES}'
            elif enrolled is None:
                decision='INCONCLUSIVE';reason='NOT_ENROLLED'
            elif similarity is None:
                decision='INCONCLUSIVE';reason='EMBEDDING_UNAVAILABLE'
            elif not identity_ok:
                decision='REJECTED';reason='IDENTITY_MISMATCH'
            elif depth_result['status']=='LOW_RELIEF':
                decision='REJECTED';reason='LOW_RELIEF'
            elif depth_result['status']=='3D_RELIEF_DETECTED':
                decision='PASS_EXPERIMENTAL';reason='IDENTITY_AND_RELIEF'
            else:
                decision='INCONCLUSIVE';reason='DEPTH_INCONCLUSIVE'
            metrics.update({'decision':decision,'reason':reason})
            boxcolor=(0,180,0) if decision=='PASS_EXPERIMENTAL' else (0,0,255) if decision=='REJECTED' else (0,200,255)
            cv2.rectangle(img,(x,y),(x+w,y+h),boxcolor,2)
            label=f'{decision} sim={similarity:.3f}' if similarity is not None else decision
            cv2.putText(img,label,(max(0,x),max(22,y-10)),cv2.FONT_HERSHEY_SIMPLEX,.48,boxcolor,2)
            cv2.putText(img,f"Depth: {depth_result['status']} relief={depth_result['relief_mm']}mm",(10,52),cv2.FONT_HERSHEY_SIMPLEX,.50,boxcolor,2)
            for pt in pts.values():
                cv2.circle(img,tuple(np.round(pt).astype(int)),3,(0,255,255),-1)
        elif enrolling:
            with lock: state['message']='Enrollment requires exactly one face'
        with lock:
            state['metrics']=metrics
        now=time.perf_counter();fps=.9*fps+.1/(max(now-last,.001));last=now
        cv2.putText(img,f'Faces {count} | FPS {fps:.1f} | ArcFace {elapsed_ms:.1f}ms',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.53,(0,255,0),2)
        depth_vis=cv2.applyColorMap(cv2.convertScaleAbs(depth_mm,alpha=255/2500),cv2.COLORMAP_JET)
        depth_vis[raw==0]=0
        ok,buffer=cv2.imencode('.jpg',np.hstack((img,depth_vis)),[cv2.IMWRITE_JPEG_QUALITY,80])
        if ok:
            with condition:
                state['jpeg']=buffer.tobytes();state['seq']+=1;condition.notify_all()
except KeyboardInterrupt: print('Stopping')
finally:
    server.shutdown();server.server_close();camera.stop()
