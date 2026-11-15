import pymupdf,re,json,os,glob,sys
from PIL import Image,ImageOps
D='/home/claude/greetjekoffie/hacktothefuture/ifc_models';OUT='/home/claude/blueline-stoa/db'
TYPES=[('bathroom',['bad','bath','wc','toilet','tualett','baño','aseo','łazienk','toalet','badkamer','sanit','dusch','shower','kylpy']),
 ('kitchen',['küche','kueche','kitchen','köök','køkken','keittiö','cocina','kuchn','keuken','pantry','teekeuken']),
 ('bedroom',['schlaf','bedroom','magamis','dormit','sypial','slaapk','soveværelse']),
 ('living',['wohn','living','elutuba','olohuone','salón','salon','stue','woonkamer','dzienny','estar']),
 ('office',['büro','buero','office','kontor','toimisto','oficina','biuro','kantoor','despacho','arbeit','werkraum']),
 ('meeting',['besprech','meeting','møde','konferenz','conference','reuni','neuvottelu','vergader']),
 ('corridor',['flur','korridor','koridor','corridor','gang','hall','esik','eteinen','pasillo','entré','vestib','foyer','diele','circul','lobby','windfang','verkehr']),
 ('stair',['trepp','stair','trepikoda','porras','escalera','klatka','trapp','stiege']),
 ('technical',['lager','storage','abstell','teknik','technik','tekn','skakt','shaft','varasto','almac','schacht','elektro','hvac','server','haustechn','heiz','installat','depot','archiv','garderob','vaatehuone','closet','trastero','berging']),
 ('sauna',['sauna']),('balcony',['balkon','balcony','terras','parveke','terraza','loggia']),
 ('teaching',['hörsaal','hoersaal','auditor','seminar','class','labor','lab ','unterricht','aula','gym','turnhalle','sport','zaal']),
 ('lounge',['lounge','café','cafe','kantine','canteen','aufenthalt','dining','comedor','ruokailu','spise'])]
PRE={'bathroom':['bañ','ban','douche','wc'],'bedroom':['hab','dorm'],'kitchen':['coc'],'living':['sal','estar'],'corridor':['dist','entrance','vorr'],'technical':['kleed','kast','utility','boiler','abiruum','cv'],'stair':['aufzug','lift','elevator','ascensor'],'parking':['stellplatz','garage','garaje','parking','autotalli']}
def rtype(s):
    l=' '+s.lower()+' '
    for t,ks in TYPES:
        if any(k in l for k in ks):return t
    toks=[t for t in re.split(r'[^a-zà-ž]+',s.lower()) if t]
    for t,ks in PRE.items():
        if any(tok.startswith(k) for tok in toks for k in ks):return t
    return 'other'
NUM=re.compile(r'^(?:(?:rum|room|raum|nr|no|n°)\.?\s*)*(?:nr\.?\s*)?([A-Z]?\d{1,4}(?:[.\-/]\d{1,3})*[A-Z]?)$',re.I)
SKIP=re.compile(r'(maßstab|massstab|scale|schnitth|©|http|licen|ifc|erzeugt|generated|grundriss|floor plan|plan:|storey|level|copyright|cc[- ]by|^0$|^\d+\s*m$)',re.I)
def thumb(pix,path,w):
    im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples).convert('L');im=ImageOps.invert(im)
    im=im.point(lambda v:min(255,int(28+v*0.78)));im.thumbnail((w,w*2));im.save(path,optimize=True)
plans=[]
for f in sorted(glob.glob(D+'/*.pdf')):
    n=os.path.basename(f);slug=n[:-4];doc=pymupdf.open(f);p=doc[0];W,H=p.rect.width,p.rect.height
    full=p.get_text();sc=re.search(r'1\s*:\s*(\d{2,4})',full);scale=int(sc.group(1)) if sc else None
    lines=[l.strip() for l in full.split('\n') if l.strip()]
    title=next((l for l in lines if '—' in l or ' - ' in l and len(l)>20),'')
    proj,_,fl=slug.partition('_plan-');src=proj.split('-')[0];pname=proj.split('_',1)[1] if '_' in proj else proj
    spans=[]
    for b in p.get_text('dict')['blocks']:
        for li in b.get('lines',[]):
            for s in li['spans']:
                t=s['text'].strip()
                if t:spans.append((t,(s['bbox'][0]+s['bbox'][2])/2,(s['bbox'][1]+s['bbox'][3])/2,s['size']))
    # footer: spans in the bottom band holding title/licence text
    body=[s for s in spans if not SKIP.search(s[0]) and s[0]!=title and len(s[0])<60]
    names=[s for s in body if not NUM.match(s[0]) and re.search(r'[A-Za-zÀ-ž]',s[0])]
    nums=[s for s in body if NUM.match(s[0])]
    rooms=[];used=set()
    isplan='_plan-' in n
    if not isplan:names=[]
    for nm in names:
        best=None
        for i,u in enumerate(nums):
            if i in used:continue
            d=((u[1]-nm[1])**2+(u[2]-nm[2])**2)**.5
            if d<nm[3]*4 and (best is None or d<best[0]):best=(d,i)
        num=None
        if best:used.add(best[1]);num=NUM.match(nums[best[1]][0]).group(1)
        rooms.append({'name':nm[0].title() if nm[0].isupper() and len(nm[0])>3 else nm[0],'num':num,'type':rtype(nm[0]),'x':round(nm[1]/W,4),'y':round(nm[2]/H,4)})
    m=0.0254/72*(scale or 100)
    dr=[g['rect'] for g in p.get_drawings() if g['rect'].width<W*.95]
    if dr:
        x0=min(r.x0 for r in dr);y0=min(r.y0 for r in dr);x1=max(r.x1 for r in dr);y1=max(r.y1 for r in dr)
    else:x0,y0,x1,y1=0,0,W,H
    tb=[b['bbox'] for b in p.get_text('dict')['blocks']]
    gx0,gy0,gx1,gy1=x0,y0,x1,y1
    for r in rooms:
        px,py=r['x']*W,r['y']*H;gx0=min(gx0,px-10);gy0=min(gy0,py-10);gx1=max(gx1,px+10);gy1=max(gy1,py+10)
    mg=max(gx1-gx0,gy1-gy0)*0.04;cx0,cy0,cx1,cy1=max(0,gx0-mg),max(0,gy0-mg),min(W,gx1+mg),min(H,gy1+mg)
    clip=pymupdf.Rect(cx0,cy0,cx1,cy1)
    for r in rooms:r['x']=round((r['x']*W-cx0)/(cx1-cx0),4);r['y']=round((r['y']*H-cy0)/(cy1-cy0),4)
    pix=p.get_pixmap(dpi=150,clip=clip);thumb(pix,f'{OUT}/thumbs/{slug}.png',560)
    aspect=round((cy1-cy0)/(cx1-cx0),3)
    cnt={}
    for r in rooms:cnt[r['type']]=cnt.get(r['type'],0)+1
    plans.append({'id':slug,'file':n,'kind':'cad' if isplan else ('section' if '_section' in n else 'elevation'),'source':src,'project':pname,'floor':fl,'title':title[:140],'scale':scale,'sizeM':[round((x1-x0)*m,1),round((y1-y0)*m,1)],'rooms':rooms,'types':cnt,'lang':None,'aspect':aspect})
    print(len(rooms),slug[:60],file=sys.stderr)
scans=sorted(glob.glob(D+'/*.png'))
for f in scans:
    n=os.path.basename(f);slug=n[:-4];im=Image.open(f).convert('L');im=ImageOps.invert(im);im=im.point(lambda v:min(255,int(28+v*0.78)));im.thumbnail((260,520));im.save(f'{OUT}/thumbs/{slug}.png',optimize=True)
    plans.append({'id':slug,'file':n,'kind':'scan','source':'resbim','project':'resbim scans','floor':slug.split('-')[-1].replace('_plan',''),'rooms':[],'types':{},'aspect':round(im.height/im.width,3)})
ifc=len(glob.glob(D+'/*.ifc'))
q=sum(1 for _ in open('/home/claude/greetjekoffie/hacktothefuture/questions.jsonl'))
json.dump({'built':'2026-11-15','models':ifc,'questions':q,'plans':plans},open(f'{OUT}/index.json','w'),ensure_ascii=False,separators=(',',':'))
print('done',len(plans),file=sys.stderr)
