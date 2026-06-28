"""
soc-prediction.py:
  1. BandSelection 上图: GA-PLS 红色散点改为方形标记 (marker='s')
  2. BandSelection 下图: stem 图 (vlines) 改为折线形式
  3. 输入文件: D:\高光谱\251204\新建模数据.xlsx
  4. 输出目录: modeling/results_v4 (不覆盖旧结果)
"""
import numpy as np,pandas as pd,os,sys,time,warnings,random
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
import functools;print=functools.partial(print,flush=True)
import torch,torch.nn as nn,torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei','Microsoft YaHei','DejaVu Sans']
plt.rcParams['axes.unicode_minus']=False

torch.set_num_threads(4);device=torch.device('cpu')
RD=r'D:\高光谱\251204\modeling\results_v4'       # v4: 新输出目录, 不覆盖旧结果
SRC=r'D:\高光谱\251204\新建模数据.xlsx'           # v4: 新输入文件
for d in ['figures','tables']:os.makedirs(f'{RD}/{d}',exist_ok=True)

# ═══════════════════════════════════════════
# 1. 加载数据
# ═══════════════════════════════════════════
print('='*60);print('1. 加载数据');print('='*60)
df=pd.read_excel(SRC)
band_cols=[f'Band{i}' for i in range(1,225)]
X_raw=df[band_cols].values.astype(np.float64)
X_aux_raw=df[['Moisture','EC']].values.astype(np.float32)
y_soc=df['SOC'].values.astype(np.float32)
n_samples,n_bands=X_raw.shape
wl=np.linspace(400,1000,n_bands)
print(f'{n_samples} 样本 x {n_bands} 波段, 波长范围: {wl[0]:.0f}-{wl[-1]:.0f}nm')
print(f'SOC: [{y_soc.min():.2f},{y_soc.max():.2f}], 均值={y_soc.mean():.3f}, CV={y_soc.std()/y_soc.mean()*100:.1f}%')

# ═══════════════════════════════════════════
# 2. 预处理 (4种方法分别评估)
# ═══════════════════════════════════════════
print('\n'+'='*60);print('2. 光谱预处理 (4种方法)');print('='*60)

def snv(X):
    mu=X.mean(axis=1,keepdims=True);sd=X.std(axis=1,ddof=1,keepdims=True)+1e-8
    return (X-mu)/sd

def msc(X,ref=None):
    if ref is None:ref=X.mean(axis=0)
    out=np.zeros_like(X)
    for i in range(X.shape[0]):b=np.polyfit(ref,X[i],1);out[i]=(X[i]-b[1])/b[0]
    return out

def fd_sg(X,window=11,polyorder=2):
    out=np.zeros_like(X)
    for i in range(X.shape[0]):out[i]=savgol_filter(X[i],window,polyorder,deriv=1)
    return out

prep_methods={
    '原始光谱 (SG滤波)':X_raw.copy(),
    '一阶导数 (FD)':fd_sg(X_raw),
    '标准正态变量 (SNV)':snv(X_raw),
    '多元散射校正 (MSC)':msc(X_raw),
}

prep_desc={
    '原始光谱 (SG滤波)':'SG平滑去噪后原始反射率, 保留光谱吸收特征',
    '一阶导数 (FD)':'SG一阶导数, 消除基线漂移, 增强光谱细节',
    '标准正态变量 (SNV)':'逐光谱标准化, 消除颗粒散射影响',
    '多元散射校正 (MSC)':'以平均光谱为参考校正散射效应',
}
print('\n预处理评估 (5-fold PLSR CV RMSECV, 全224波段, LV=8):')
print(f'{"方法":<24} {"描述":<40} {"RMSECV":>8}')
print('-'*74)
prep_scores={}
for name,Xp in prep_methods.items():
    Xp_z=(Xp-Xp.mean(axis=0))/(Xp.std(axis=0)+1e-8)
    kf=KFold(5,shuffle=True,random_state=42);rm=0.0
    for tr,va in kf.split(Xp_z):
        p=PLSRegression(n_components=8);p.fit(Xp_z[tr],y_soc[tr])
        rm+=np.sqrt(mean_squared_error(y_soc[va],p.predict(Xp_z[va]).ravel()))
    rm/=5;prep_scores[name]=rm
    print(f'{name:<24} {prep_desc[name]:<40} {rm:>8.4f}')

best_prep=min(prep_scores,key=prep_scores.get)
X_processed=prep_methods[best_prep]
print(f'\n{"="*60}')
print(f'建模选用: {best_prep}')
print(f'选用理由: {prep_desc[best_prep]}, RMSECV={prep_scores[best_prep]:.4f} (最低)')
print(f'{"="*60}')

pd.DataFrame([{'预处理方法':k,'描述':prep_desc[k],'RMSECV':round(v,4),'是否选用':'★ 建模选用' if k==best_prep else ''}
               for k,v in prep_scores.items()]).to_csv(
    f'{RD}/tables/v10_preprocessing_comparison.csv',index=False,encoding='utf-8-sig')

# ---- 预处理对比图 (无总标题 + 无说明框) ----
fig,axes=plt.subplots(2,2,figsize=(16,10))
for ax,(name,Xp) in zip(axes.flat,prep_methods.items()):
    for i in range(min(30,n_samples)):ax.plot(wl,Xp[i],alpha=0.12,lw=0.3,color='gray')
    ax.plot(wl,Xp.mean(axis=0),'r',lw=1.5,label='平均光谱')
    ax.set_xlabel('波长 (nm)',fontsize=10);ax.set_ylabel('反射率',fontsize=10)
    is_best = (name == best_prep)
    marker = ' ★ 建模选用' if is_best else ''
    ax.set_title(f'{name}\nRMSECV={prep_scores[name]:.4f}{marker}',
                 fontweight='bold',fontsize=11,color='#C00000' if is_best else 'black')
    if is_best:
        for spine in ax.spines.values():spine.set_color('#C00000');spine.set_linewidth(2)
    ax.legend(fontsize=8);ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{RD}/figures/v10_preprocessing.png',dpi=300,bbox_inches='tight');plt.close()
print(f'已保存: figures/v10_preprocessing.png')

# ═══════════════════════════════════════════
# 3. 波段筛选 (400-1000nm 四区均衡 GA-PLS)
# ═══════════════════════════════════════════
print('\n'+'='*60);print('3. 波段筛选 (224波段 @ 400-1000nm)');print('='*60)

np.random.seed(42);random.seed(42)
X_proc_z=(X_processed-X_processed.mean(axis=0))/(X_processed.std(axis=0)+1e-8)

# 四个子区间
zones_vis=[np.where((wl>=400)&(wl<550))[0],np.where((wl>=550)&(wl<700))[0]]
zones_nir=[np.where((wl>=700)&(wl<850))[0],np.where((wl>=850)&(wl<=1000))[0]]
all_zones=zones_vis+zones_nir
zone_labels=['400-550nm\n(蓝-绿光区)','550-700nm\n(红光区)','700-850nm\n(红边-近红外)','850-1000nm\n(近红外区)']
zone_colors=['#a6cee3','#b2df8a','#fdbf6f','#cab2d6']

# GA-PLS 分区均衡
print('\n--- GA-PLS (分区均衡筛选) ---')
def gapls_fitness(c):
    idx=np.where(c==1)[0]
    if len(idx)<8 or len(idx)>50:return -1e10
    zc=[np.sum(np.isin(idx,z)) for z in all_zones]
    if min(zc)<1:return -1e10
    bal=min(zc)/max(zc)
    Xs=X_proc_z[:,idx];nl=min(6,len(idx)-1,Xs.shape[1]);nl=max(nl,1)
    kf=KFold(5,shuffle=True,random_state=42);rm=0.0
    for tr,va in kf.split(Xs):
        p=PLSRegression(n_components=nl);p.fit(Xs[tr],y_soc[tr])
        rm+=np.sqrt(mean_squared_error(y_soc[va],p.predict(Xs[va]).ravel()))
    rm/=5;return -(rm*(1.0-0.04*bal)),rm,bal,zc

pop_sz,ngen,nelite=120,150,12
pop=[]
for _ in range(pop_sz):
    ch=[];total=random.randint(12,35)
    per_zone=max(2,total//4)
    for z in all_zones:ch.extend(random.sample(list(z),min(per_zone+random.randint(0,3),len(z))))
    c=np.zeros(n_bands,dtype=np.int32)
    for j in ch:c[j]=1;pop.append(c)

bc,bf,br,bb,bzc=None,-np.inf,0,0,None
for g in range(ngen):
    fi=[gapls_fitness(c) for c in pop];fv=np.array([x[0] for x in fi])
    gb=np.argmax(fv)
    if fv[gb]>bf:bf=fv[gb];bc=pop[gb].copy();_,br,bb,bzc=fi[gb]
    ei=np.argsort(fv)[::-1][:nelite];nw=[pop[i].copy() for i in ei]
    while len(nw)<pop_sz:
        p1=pop[random.randrange(pop_sz)];p2=pop[random.randrange(pop_sz)]
        cp=random.randint(1,n_bands-1);ch=np.concatenate([p1[:cp],p2[cp:]])
        mu=np.random.random(n_bands)<0.005;ch[mu]=1-ch[mu]
        if ch.sum()<8:iz=np.where(ch==0)[0];ch[np.random.choice(iz,8-int(ch.sum()),replace=False)]=1
        if ch.sum()>50:oz=np.where(ch==1)[0];ch[np.random.choice(oz,int(ch.sum())-50,replace=False)]=0
        nw.append(ch)
    pop=nw
    if (g+1)%30==0:print(f'  第{g+1}代: {bc.sum():.0f}波段, 各区={bzc}, 均衡度={bb:.2f}, RMSECV={br:.4f}')

gapls_bands=np.sort(np.where(bc==1)[0])
print(f'GA-PLS筛选结果: {len(gapls_bands)} 个波段')

# CARS 对比
print('--- CARS 竞争性自适应重加权采样 ---')
def cars_select(X,y,n_mc=50):
    cv=np.arange(n_bands);bs,brm,bst=None,np.inf,None
    for mc in range(1,n_mc+1):
        rt=(3.0/n_bands)**(1.0/n_mc);nt=max(3,int(n_bands*(rt**mc)))
        if nt>=len(cv):continue
        Xs=X[:,cv];nl=min(8,len(cv)-1,Xs.shape[1]);nl=max(nl,1)
        p=PLSRegression(n_components=nl);p.fit(Xs,y)
        w=np.abs(p.coef_.ravel());w=w/(w.sum()+1e-10)
        cv=cv[np.argsort(w)[::-1][:nt]];cv=np.sort(cv)
        Xs=X[:,cv];nl2=min(6,len(cv)-1,Xs.shape[1]);nl2=max(nl2,1)
        kf=KFold(5,shuffle=True,random_state=42);rm=0
        for tr,va in kf.split(Xs):
            pp=PLSRegression(n_components=nl2);pp.fit(Xs[tr],y[tr])
            rm+=np.sqrt(mean_squared_error(y[va],pp.predict(Xs[va]).ravel()))
        rm/=5
        if rm<brm:brm=rm;bst=cv.copy()
    return bst
cars_bands=cars_select(X_proc_z,y_soc)
print(f'CARS筛选结果: {len(cars_bands)} 个波段')

def eval_bs(bands,name):
    Xs=X_proc_z[:,bands];nl=min(6,len(bands)-1,Xs.shape[1]);nl=max(nl,1)
    kf=KFold(5,shuffle=True,random_state=42);rm=0
    for tr,va in kf.split(Xs):
        p=PLSRegression(n_components=nl);p.fit(Xs[tr],y_soc[tr])
        rm+=np.sqrt(mean_squared_error(y_soc[va],p.predict(Xs[va]).ravel()))
    rm/=5
    zc=[np.sum(np.isin(bands,z)) for z in all_zones]
    bal=min(zc)/max(zc) if max(zc)>0 else 0
    pr=np.mean(np.abs(np.corrcoef(X_processed[:,bands].T)-np.eye(len(bands))))
    return {'方法':name,'波段数':len(bands),'RMSECV':rm,'均衡度':bal,'平均|r|':pr,'各区分布':str(zc)}

sel_info=[eval_bs(gapls_bands,'GA-PLS(分区均衡)'),eval_bs(cars_bands,'CARS')]
pd.DataFrame(sel_info).to_csv(f'{RD}/tables/v10_band_selection_comparison.csv',index=False,encoding='utf-8-sig')
for s in sel_info:print(f'{s["方法"]}: 波段数={s["波段数"]}, RMSECV={s["RMSECV"]:.4f}, 均衡度={s["均衡度"]:.2f}')

selected_bands=gapls_bands
print(f'\n选用: GA-PLS — {len(selected_bands)} 波段')

# 计算 PLSR |β| — 5-fold CV 选最优 LV 数
nl_max=min(20,len(y_soc)-1,n_bands);cv_lv={}
for lv in range(1,nl_max+1):
    p=PLSRegression(n_components=lv);rs=[]
    for t2,v in KFold(5,shuffle=True,random_state=42).split(X_proc_z):
        p.fit(X_proc_z[t2],y_soc[t2]);rs.append(np.sqrt(mean_squared_error(y_soc[v],p.predict(X_proc_z[v]).ravel())))
    cv_lv[lv]=np.mean(rs)
best_lv=min(cv_lv,key=cv_lv.get)
plsr_full=PLSRegression(n_components=best_lv);plsr_full.fit(X_proc_z,y_soc)
plsr_coef=np.abs(plsr_full.coef_.ravel())
print(f'PLSR最优LV={best_lv}, RMSECV={cv_lv[best_lv]:.4f}')

# ---- 波段筛选图: v4修改 ----
zone_labels_short = ['400-550 nm','550-700 nm','700-850 nm','850-1000 nm']
os.makedirs(f'{RD}/figures',exist_ok=True)

fig,(ax1,ax2)=plt.subplots(2,1,figsize=(16,10))

# (a) 平均光谱 + 选中波段 — 红色方形散点, legend正确显示方形
ms=X_processed.mean(axis=0)
ax1.plot(wl,ms,'#333333',lw=1.2,alpha=0.8,zorder=3,label='平均光谱')
ymax,ymin=ms.max(),ms.min()
for z,c,lbl in zip(all_zones,zone_colors,zone_labels_short):
    ax1.axvspan(wl[z[0]],wl[z[-1]],alpha=0.10,color=c,zorder=0)
    ax1.text((wl[z[0]]+wl[z[-1]])/2,ymax*0.95,lbl,ha='center',fontsize=8,color='gray')
ax1.scatter(wl[selected_bands],ms[selected_bands],c='red',s=30,zorder=5,
            marker='s',edgecolors='darkred',linewidths=0.5,label='GA-PLS 选中')
ax1.set_xlabel('波长 (nm)',fontsize=11)
ax1.set_ylabel('平均反射率',fontsize=11)
ax1.legend(loc='upper right',fontsize=8)
ax1.grid(alpha=0.3)

# (b) |β| 蓝色突出折线: 未选中=0, 选中=|β|竖起, 无0横线
stem_vals=np.zeros(n_bands)
stem_vals[selected_bands]=plsr_coef[selected_bands]
ax2.plot(wl,stem_vals,color='#4472C4',lw=1.0,zorder=4,label='GA-PLS 选中波段')
ax2.scatter(wl[selected_bands],plsr_coef[selected_bands],c='red',s=30,zorder=6,
            marker='s',edgecolors='darkred',linewidths=0.5)
ax2.spines['bottom'].set_position('zero')
for z,c in zip(all_zones,zone_colors):
    ax2.axvspan(wl[z[0]],wl[z[-1]],alpha=0.06,color=c,zorder=0)
ax2.set_xlabel('波长 (nm)',fontsize=11)
ax2.set_ylabel(r'$|\beta|$ (PLSR回归系数绝对值)',fontsize=11)
ax2.legend(loc='upper right',fontsize=8)
ax2.grid(alpha=0.2,axis='x')

plt.tight_layout()
plt.savefig(f'{RD}/figures/Band_Selection.png',dpi=300,bbox_inches='tight');plt.close()
print(f'已保存: figures/Band_Selection.png')

# ═══════════════════════════════════════════
# 4. 模型定义
# ═══════════════════════════════════════════
print('\n'+'='*60);print('4. 模型定义');print('='*60)

X_sel=X_processed[:,selected_bands]
n_feat=X_sel.shape[1]

def zscore(X):return (X-X.mean(axis=0))/(X.std(axis=0)+1e-8)
def mtr(yt,yp):
    r2=r2_score(yt,yp);rmse=np.sqrt(mean_squared_error(yt,yp))
    mae=mean_absolute_error(yt,yp);rpd=np.std(yt,ddof=1)/rmse if rmse>0 else 0
    return{'R2':r2,'RMSE':rmse,'MAE':mae,'RPD':rpd}

# --- PLSR ---
def plsr_fit(Xtr,ytr,Xte,yte,mxlv=20):
    nl=min(mxlv,len(ytr)-1,Xtr.shape[1]);nl=max(nl,1);cv={}
    for lv in range(1,nl+1):
        p=PLSRegression(n_components=lv);rs=[]
        for t2,v in KFold(5,shuffle=True,random_state=42).split(Xtr):
            p.fit(Xtr[t2],ytr[t2]);rs.append(np.sqrt(mean_squared_error(ytr[v],p.predict(Xtr[v]).ravel())))
        cv[lv]=np.mean(rs)
    bl=min(cv,key=cv.get);p=PLSRegression(n_components=bl);p.fit(Xtr,ytr)
    return p.predict(Xte).ravel(),bl

# --- ANN ---
class ANN_Model(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(d,20),nn.BatchNorm1d(20),nn.ReLU(),nn.Dropout(0.25),
            nn.Linear(20,10),nn.BatchNorm1d(10),nn.ReLU(),nn.Dropout(0.25),
            nn.Linear(10,1))
    def forward(self,x):return self.net(x).squeeze(-1)

def ann_train(Xtr,ytr,Xte,yte,ep=600,lr=0.003,wd=1e-3):
    sy=StandardScaler();ytr_s=sy.fit_transform(ytr.reshape(-1,1)).ravel()
    t=lambda a:torch.tensor(a,dtype=torch.float32)
    m=ANN_Model(Xtr.shape[1]);opt=optim.AdamW(m.parameters(),lr=lr,weight_decay=wd);cr=nn.MSELoss()
    Xtt,yst,Xtet=t(Xtr),t(ytr_s),t(Xte);m.train()
    for _ in range(ep):opt.zero_grad();cr(m(Xtt),yst).backward();opt.step()
    m.eval()
    with torch.no_grad():
        return sy.inverse_transform(m(Xtt).numpy().reshape(-1,1)).ravel(),sy.inverse_transform(m(Xtet).numpy().reshape(-1,1)).ravel()

# --- 1D-ResCNN ---
class ResBlock1D_fix(nn.Module):
    def __init__(self, channels):
        super().__init__()
        channels=int(channels)
        self.conv=nn.Sequential(
            nn.Conv1d(channels,channels,3,1,1,bias=False),nn.BatchNorm1d(channels),nn.ReLU(),
            nn.Conv1d(channels,channels,3,1,1,bias=False),nn.BatchNorm1d(channels))
    def forward(self,x):return torch.relu(x+self.conv(x))

class MultiChannelResCNN(nn.Module):
    def __init__(self, n_bands, n_aux=0):
        super().__init__()
        self.n_aux = n_aux
        if n_aux > 0:
            base_ch = 16
            self.stem = nn.Sequential(
                nn.Conv1d(1, base_ch, 7, 1, 3, bias=False),
                nn.BatchNorm1d(base_ch), nn.ReLU())
            self.rb = ResBlock1D_fix(base_ch)
            self.gap = nn.AdaptiveAvgPool1d(1)
            self.aux_mlp = nn.Sequential(
                nn.Linear(n_aux, 24), nn.BatchNorm1d(24), nn.ReLU(), nn.Dropout(0.15),
                nn.Linear(24, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Dropout(0.15),
                nn.Linear(16, 16), nn.ReLU())
            fc_in = base_ch + 16
            self.fc = nn.Sequential(
                nn.Linear(fc_in, 24), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(24, 12), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(12, 1))
        else:
            base_ch = 10
            self.stem = nn.Sequential(
                nn.Conv1d(1, base_ch, 7, 1, 3, bias=False),
                nn.BatchNorm1d(base_ch), nn.ReLU())
            self.rb = ResBlock1D_fix(base_ch)
            self.gap = nn.AdaptiveAvgPool1d(1)
            self.aux_mlp = None
            fc_in = base_ch
            self.fc = nn.Sequential(
                nn.Linear(fc_in, 8), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(8, 1))

    def forward(self, spec, aux=None):
        x = spec.unsqueeze(1)
        x = self.stem(x)
        if self.rb is not None:
            x = self.rb(x)
        x = self.gap(x).squeeze(-1)
        if self.n_aux > 0 and aux is not None:
            aux_feat = self.aux_mlp(aux)
            x = torch.cat([x, aux_feat], dim=1)
        return self.fc(x).squeeze(-1)

def cnn_train(Xs, ytr, Xse, yte, aux_tr=None, aux_te=None, ep=800, lr=0.005, wd=1e-4):
    t = lambda a: torch.tensor(a, dtype=torch.float32)
    sy = StandardScaler()
    ytr_s = sy.fit_transform(ytr.reshape(-1, 1)).ravel()
    n_aux = aux_tr.shape[1] if aux_tr is not None else 0
    m = MultiChannelResCNN(Xs.shape[1], n_aux=n_aux)
    opt = optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    cr = nn.HuberLoss(delta=0.5)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ep, eta_min=1e-6)
    Xst = t(Xs); yst = t(ytr_s); Xset = t(Xse)
    if aux_tr is not None: At = t(aux_tr); Aet = t(aux_te)
    m.train()
    bs = min(32, len(Xs) // 2)
    if aux_tr is not None: ds = TensorDataset(Xst, At, yst)
    else: ds = TensorDataset(Xst, yst)
    ld = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True)
    for _ in range(ep):
        for batch in ld:
            opt.zero_grad()
            if aux_tr is not None: bx, ba, by = batch; loss = cr(m(bx, ba), by)
            else: bx, by = batch; loss = cr(m(bx), by)
            loss.backward(); opt.step()
        sch.step()
    m.eval()
    with torch.no_grad():
        if aux_tr is not None:
            ptr = m(Xst, At).numpy(); pte = m(Xset, Aet).numpy()
        else: ptr = m(Xst).numpy(); pte = m(Xset).numpy()
    return (sy.inverse_transform(ptr.reshape(-1,1)).ravel(),
            sy.inverse_transform(pte.reshape(-1,1)).ravel())

print('模型定义完成')

# ═══════════════════════════════════════════
# 5. 50次重复 80/20 划分 — 按训练 R^2 最大寻优
# ═══════════════════════════════════════════
N_S=50;results={};aggregates={};opt_seeds={};configs=[
    ('PLSR','Set-1 (仅波段)',False),
    ('PLSR','Set-2 (波段+EC+水分)',True),
    ('ANN','Set-1 (仅波段)',False),
    ('ANN','Set-2 (波段+EC+水分)',True),
    ('1D-ResCNN','Set-1 (仅波段)',False),
    ('1D-ResCNN','Set-2 (波段+EC+水分)',True),
]

FIXED_SPLIT = {}

for mn,sn,ua in configs:
    dsc=f'{mn} {sn}';print(f'\n{"="*60}\n{dsc}\n{"="*60}')
    all_tr,all_te=[],[];all_splits=[]

    for seed in range(N_S):
        idx=np.arange(n_samples);itr,ite=train_test_split(idx,test_size=0.20,random_state=seed)
        mt,me=np.isin(idx,itr),np.isin(idx,ite);ytr,yte=y_soc[mt],y_soc[me]
        Xp_tr=zscore(X_sel[mt]);Xp_te=zscore(X_sel[me])
        Xa_tr=zscore(X_aux_raw[mt]);Xa_te=zscore(X_aux_raw[me])

        if mn=='PLSR':
            if ua:X_tr=np.hstack([Xp_tr,Xa_tr]);X_te=np.hstack([Xp_te,Xa_te])
            else:X_tr,X_te=Xp_tr,Xp_te
            pte,lv=plsr_fit(X_tr,ytr,X_te,yte,mxlv=20 if ua else 12)
            ptr=PLSRegression(n_components=lv).fit(X_tr,ytr).predict(X_tr).ravel()
        elif mn=='ANN':
            if ua:X_tr=np.hstack([Xp_tr,Xa_tr]);X_te=np.hstack([Xp_te,Xa_te])
            else:X_tr,X_te=Xp_tr,Xp_te
            ptr,pte=ann_train(X_tr,ytr,X_te,yte)
        else:
            if ua: ptr,pte=cnn_train(Xp_tr,ytr,Xp_te,yte,aux_tr=Xa_tr,aux_te=Xa_te)
            else: ptr,pte=cnn_train(Xp_tr,ytr,Xp_te,yte)

        r2e=r2_score(yte,pte);r2t=r2_score(ytr,ptr);all_tr.append(r2t);all_te.append(r2e)
        mt_e=mtr(yte,pte);mt_r=mtr(ytr,ptr)
        ap=np.zeros(n_samples);ap[mt]=ptr;ap[me]=pte
        all_splits.append({'seed':seed,'tr_r2':r2t,'te_r2':mt_e['R2'],'te_rmse':mt_e['RMSE'],
                  'te_mae':mt_e['MAE'],'te_rpd':mt_e['RPD'],'tr_rmse':mt_r['RMSE'],
                  'tr_mae':mt_r['MAE'],'tr_rpd':mt_r['RPD'],
                  'y_all':y_soc.copy(),'p_all':ap.copy(),'mask_tr':mt,'mask_te':me})

        if (seed+1)%10==0:
            best_idx=np.argmax(all_tr)
            print(f'  [{seed+1:2d}/{N_S}] 当前最优训练R2={all_tr[best_idx]:.4f} (seed={best_idx}) | '
                  f'测试R2累计={np.mean(all_te):.4f}±{np.std(all_te,ddof=1):.4f}')

    ta=np.array(all_tr);ea=np.array(all_te)
    best_idx=int(np.argmax(ta))
    best=all_splits[best_idx]
    best_seed=best['seed']

    rmse_arr=np.array([s['te_rmse'] for s in all_splits])
    mae_arr=np.array([s['te_mae'] for s in all_splits])
    rpd_arr=np.array([s['te_rpd'] for s in all_splits])

    print(f'\n  ★ 最优种子 #{best_seed} (训练R2最大={ta[best_idx]:.4f}): '
          f'训练R2={best["tr_r2"]:.4f} 测试R2={best["te_r2"]:.4f} '
          f'测试RMSE={best["te_rmse"]:.4f} RPD={best["te_rpd"]:.2f}')
    print(f'  50次均值±标准差: 训练R2 {ta.mean():.4f}±{ta.std(ddof=1):.4f} | '
          f'测试R2 {ea.mean():.4f}±{ea.std(ddof=1):.4f} [{ea.min():.4f},{ea.max():.4f}]')
    print(f'                   测试RMSE {rmse_arr.mean():.4f}±{rmse_arr.std(ddof=1):.4f} | '
          f'MAE {mae_arr.mean():.4f}±{mae_arr.std(ddof=1):.4f} | '
          f'RPD {rpd_arr.mean():.2f}±{rpd_arr.std(ddof=1):.2f}')

    results[dsc]=best
    opt_seeds[dsc]=best_seed
    FIXED_SPLIT[dsc]=best_seed
    aggregates[dsc]={
        'train_r2_mean':ta.mean(),'train_r2_std':ta.std(ddof=1),
        'test_r2_mean':ea.mean(),'test_r2_std':ea.std(ddof=1),
        'test_rmse_mean':rmse_arr.mean(),'test_rmse_std':rmse_arr.std(ddof=1),
        'test_mae_mean':mae_arr.mean(),'test_mae_std':mae_arr.std(ddof=1),
        'test_rpd_mean':rpd_arr.mean(),'test_rpd_std':rpd_arr.std(ddof=1),
        'opt_seed':best_seed}

# ═══════════════════════════════════════════
# 输出固定划分信息
# ═══════════════════════════════════════════
print(f'\n{"="*80}')
print('寻优结束 — 各模型固定划分 seed (训练 R^2 最大):')
print('='*80)
for k,v in opt_seeds.items():
    r=results[k]
    print(f'  {k}: seed={v}, 训练R2={r["tr_r2"]:.4f}, 测试R2={r["te_r2"]:.4f}')
print(f'\n复现方法: 将 train_test_split 的 random_state 替换为以上 seed 即可固定划分')

# ═══════════════════════════════════════════
# 6. 保存预测结果
# ═══════════════════════════════════════════
pred_df=pd.DataFrame({'实测SOC':y_soc})
for n,r in results.items():
    sn=n.replace(' ','_').replace('(','').replace(')','').replace('/','_')
    pred_df[f'{sn}_预测值']=r['p_all']
    pred_df[f'{sn}_是否测试集']=r['mask_te'].astype(int)
    pd.DataFrame({
        '样本序号':np.where(r['mask_te'])[0],
        '实测SOC':y_soc[r['mask_te']],
        '预测SOC':r['p_all'][r['mask_te']]
    }).to_csv(f'{RD}/tables/{sn}_测试集预测.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame({
        '样本序号':np.where(r['mask_tr'])[0],
        'SOC':y_soc[r['mask_tr']]
    }).to_csv(f'{RD}/tables/{sn}_训练集数据.csv',index=False,encoding='utf-8-sig')
pred_df.to_csv(f'{RD}/tables/v10_all_predictions.csv',index=False,encoding='utf-8-sig')
print(f'\n已保存预测结果: tables/v10_all_predictions.csv')

# ═══════════════════════════════════════════
# 7. 最终结果表
# ═══════════════════════════════════════════
print(f'\n{"="*120}')
print(f'V10 最终结果 — {best_prep} + GA-PLS({len(selected_bands)}波段, 400-1000nm) + 80/20划分×{N_S}次')
print(f'寻优标准: 训练 R^2 最大 | 结果形式: 均值±标准差')
print('='*120)
print(f'{"模型":<30} {"训练R2":>14} {"测试R2":>14} {"RMSE":>14} {"MAE":>14} {"RPD":>14} {"最优种子":>8}')
print('-'*120)
for n in results:
    a=aggregates[n]
    print(f'{n:<30} {a["train_r2_mean"]:>6.4f}±{a["train_r2_std"]:.4f}  '
          f'{a["test_r2_mean"]:>6.4f}±{a["test_r2_std"]:.4f}  '
          f'{a["test_rmse_mean"]:>6.4f}±{a["test_rmse_std"]:.4f}  '
          f'{a["test_mae_mean"]:>6.4f}±{a["test_mae_std"]:.4f}  '
          f'{a["test_rpd_mean"]:>5.2f}±{a["test_rpd_std"]:.2f}  '
          f'{a["opt_seed"]:>8}')

print(f'\n{"="*70}')
print('Set-2 (波段+EC+水分) vs Set-1 (仅波段) 测试R2对比 (50次均值±标准差):')
print('='*70)
for model in ['PLSR','ANN','1D-ResCNN']:
    k1=f'{model} Set-1 (仅波段)'
    k2=f'{model} Set-2 (波段+EC+水分)'
    m1=aggregates[k1]['test_r2_mean'];s1=aggregates[k1]['test_r2_std']
    m2=aggregates[k2]['test_r2_mean'];s2=aggregates[k2]['test_r2_std']
    delta=m2-m1;direction='↑' if delta>0 else '↓'
    print(f'  {model}: Set-1 R2={m1:.4f}±{s1:.4f} → Set-2 R2={m2:.4f}±{s2:.4f}  '
          f'({direction}{abs(delta):.4f})')

rows=[]
for n in results:
    a=aggregates[n]
    rows.append({'模型':n,
       '最优种子':a['opt_seed'],
       '训练R2均值':round(a['train_r2_mean'],4),'训练R2标准差':round(a['train_r2_std'],4),
       '测试R2均值':round(a['test_r2_mean'],4),'测试R2标准差':round(a['test_r2_std'],4),
       'RMSE均值':round(a['test_rmse_mean'],4),'RMSE标准差':round(a['test_rmse_std'],4),
       'MAE均值':round(a['test_mae_mean'],4),'MAE标准差':round(a['test_mae_std'],4),
       'RPD均值':round(a['test_rpd_mean'],2),'RPD标准差':round(a['test_rpd_std'],2)})
pd.DataFrame(rows).to_csv(f'{RD}/tables/v10_final_results.csv',index=False,encoding='utf-8-sig')

# ═══════════════════════════════════════════
# 8. 模型结果图 ($R^2$ math + 全指标标注)
# ═══════════════════════════════════════════
print(f'\n{"="*60}');print('8. 生成结果图');print('='*60)

def make_model_figure(r1, r2, f1_label, f2_label, filename):
    """共享轴范围, $R^2$ math mode, 标注训练全指标"""
    yt1, pt1 = r1['y_all'][r1['mask_tr']], r1['p_all'][r1['mask_tr']]
    yt2, pt2 = r2['y_all'][r2['mask_tr']], r2['p_all'][r2['mask_tr']]

    all_v = np.concatenate([yt1, pt1, yt2, pt2])
    lo, hi = all_v.min(), all_v.max()
    m = (hi - lo) * 0.08
    lim = (lo - m, hi + m)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))

    for ax, (ft, rr) in zip([ax1, ax2], [(f1_label, r1), (f2_label, r2)]):
        yt, pt = rr['y_all'][rr['mask_tr']], rr['p_all'][rr['mask_tr']]
        ax.scatter(yt, pt, c='black', edgecolors='black', alpha=0.6, s=45,
                   zorder=3, label='训练集')
        k, b = np.polyfit(yt, pt, 1)
        xf = np.linspace(lim[0], lim[1], 100)
        ax.plot(xf, k * xf + b, 'black', lw=1.5, zorder=5, label='拟合线')

        s = (f'$R^2$ = {rr["tr_r2"]:.2f}\n'
             f'RMSE = {rr["tr_rmse"]:.2f}\n'
             f'RPD = {rr["tr_rpd"]:.2f}')
        ax.text(0.03, 0.97, s, transform=ax.transAxes, fontsize=9.5, va='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='none', edgecolor='black', alpha=1.0))

        ax.set_xlabel('实测 SOC (g/kg)', fontsize=11)
        ax.set_ylabel('预测 SOC (g/kg)', fontsize=11)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, ls='--', color='#E0E0E0', alpha=0.6)
        ax.tick_params(direction='in')
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(f'{RD}/figures/{filename}', dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  已保存: figures/{filename}')

make_model_figure(
    results['PLSR Set-1 (仅波段)'], results['PLSR Set-2 (波段+EC+水分)'],
    '仅波段', '波段+EC+水分', 'v10_PLSR_结果图.png')

make_model_figure(
    results['ANN Set-1 (仅波段)'], results['ANN Set-2 (波段+EC+水分)'],
    '仅波段', '波段+EC+水分', 'v10_ANN_结果图.png')

make_model_figure(
    results['1D-ResCNN Set-1 (仅波段)'], results['1D-ResCNN Set-2 (波段+EC+水分)'],
    '仅波段', '波段+EC+水分', 'v10_1D-ResCNN_结果图.png')

# ═══════════════════════════════════════════
# 9. 综合对比图 ($R^2$ math mode)
# ═══════════════════════════════════════════
print('9. 生成综合对比图')
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(14,10))

names=list(aggregates.keys())
short_names=[n.replace('Set-1 (仅波段)','仅波段').replace('Set-2 (波段+EC+水分)','波段+EC+水分')
             for n in names]
r2s=np.array([aggregates[n]['test_r2_mean'] for n in names])
r2s_std=np.array([aggregates[n]['test_r2_std'] for n in names])
rpds=np.array([aggregates[n]['test_rpd_mean'] for n in names])
rpds_std=np.array([aggregates[n]['test_rpd_std'] for n in names])
tr_r2s=np.array([aggregates[n]['train_r2_mean'] for n in names])
tr_r2s_std=np.array([aggregates[n]['train_r2_std'] for n in names])

colors_r2=[];colors_rpd=[]
for n in names:
    if 'Set-1' in n: colors_r2.append('#92C5DE'); colors_rpd.append('#F4A582')
    else: colors_r2.append('#2166AC'); colors_rpd.append('#B2182B')

ax1.barh(short_names,r2s,color=colors_r2,edgecolor='black',linewidth=0.5,xerr=r2s_std,capsize=3)
for i,(v,s) in enumerate(zip(r2s,r2s_std)):
    ax1.text(v+s+0.005,i,f'{v:.4f}$\\pm${s:.4f}',va='center',fontsize=8,fontweight='bold')
ax1.set_xlabel(r'测试集 $R^2$ (均值 $\pm$ 标准差)',fontsize=11)
ax1.set_title(r'各模型测试集 $R^2$ 对比 (50次划分 均值$\pm$标准差)',fontsize=12,fontweight='bold')
ax1.axvline(x=0,color='red',ls='--',alpha=0.5)

ax2.barh(short_names,rpds,color=colors_rpd,edgecolor='black',linewidth=0.5,xerr=rpds_std,capsize=3)
for i,(v,s) in enumerate(zip(rpds,rpds_std)):
    ax2.text(v+s+0.01,i,f'{v:.2f}$\\pm${s:.2f}',va='center',fontsize=8,fontweight='bold')
ax2.set_xlabel(r'测试集 RPD (均值 $\pm$ 标准差)',fontsize=11)
ax2.set_title(r'各模型测试集 RPD 对比 (50次划分 均值$\pm$标准差)',fontsize=12,fontweight='bold')
ax2.axvline(x=1.4,color='orange',ls='--',alpha=0.7,label='RPD=1.4 (一般)')
ax2.axvline(x=2.0,color='green',ls='--',alpha=0.7,label='RPD=2.0 (良好)')
ax2.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f'{RD}/figures/v10_柱状对比图.png',dpi=300,bbox_inches='tight');plt.close()
print(f'  已保存: figures/v10_柱状对比图.png')

# 训练 vs 测试 R2
fig,ax=plt.subplots(figsize=(12,6))
x=np.arange(len(names));w=0.35
ax.bar(x-w/2,tr_r2s,w,yerr=tr_r2s_std,capsize=3,label='训练集 $R^2$',color='#708090',edgecolor='black',linewidth=0.5)
ax.bar(x+w/2,r2s,w,yerr=r2s_std,capsize=3,label='测试集 $R^2$',color='#C00000',edgecolor='black',linewidth=0.5)
ax.set_xticks(x);ax.set_xticklabels(short_names,rotation=30,ha='right',fontsize=8.5)
ax.set_ylabel(r'$R^2$ (均值 $\pm$ 标准差)',fontsize=11)
ax.set_title(r'训练集 vs 测试集 $R^2$ — 过拟合评估 (50次划分 均值$\pm$标准差)',fontsize=13,fontweight='bold')
ax.legend(fontsize=10);ax.axhline(y=0,color='gray',ls='--',alpha=0.5)
for i in range(len(names)):
    ax.text(i-w/2,tr_r2s[i]+tr_r2s_std[i]+0.01,f'{tr_r2s[i]:.3f}',ha='center',fontsize=7)
    ax.text(i+w/2,max(r2s[i]+r2s_std[i],0)+0.01,f'{r2s[i]:.3f}',ha='center',fontsize=7)
ax.set_ylim(bottom=min(0,min(r2s-r2s_std)-0.1))
plt.tight_layout()
plt.savefig(f'{RD}/figures/v10_过拟合评估.png',dpi=300,bbox_inches='tight');plt.close()
print(f'  已保存: figures/v10_过拟合评估.png')

# ═══════════════════════════════════════════
# 10. 测试集 RMSE 汇总
# ═══════════════════════════════════════════
rows_rmse=[]
for mn, feat_label, _ in configs:
    r = results[f'{mn} {feat_label}']
    rows_rmse.append({'模型': f'{mn} ({feat_label})', '测试集RMSE': round(r['te_rmse'], 4)})
pd.DataFrame(rows_rmse).to_csv(f'{RD}/tables/v10_test_rmse.csv',index=False,encoding='utf-8-sig')

print(f'\n{"="*70}')
print('固定划分 seed (训练 R^2 最大寻优结果), 复现时替换主循环:')
print('='*70)
for k,v in FIXED_SPLIT.items():
    print(f'#   {k}: seed={v}')
print(f'\n全部完成! 结果目录: {RD}/')
print(f'  表格: tables/v10_final_results.csv, v10_test_rmse.csv')
print(f'  预测: tables/v10_all_predictions.csv')
print(f'  图片: figures/v10_*.png, figures/Band_Selection.png')
