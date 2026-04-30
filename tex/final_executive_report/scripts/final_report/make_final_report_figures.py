from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / 'outputs' / 'final_report' / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.labelsize': 10,
    'figure.dpi': 160,
    'savefig.bbox': 'tight',
})

# Core data reconstructed from final project summary.
toi = pd.DataFrame({
    'region': ['cube12_cluster0','cube17_cluster2','cube26_cluster6','cube19_cluster5','cube24_cluster1','cube33_cluster3','cube17_cluster10','cube26_cluster0','cube26_cluster11','cube26_cluster12'],
    'TOI': [0.077,0.063,0.058,0.055,0.052,0.050,0.047,0.044,0.041,0.039],
    'Shadow': [0.82,0.77,0.73,0.70,0.68,0.65,0.63,0.59,0.57,0.55],
    'LowImputation': [0.86,0.81,0.78,0.76,0.73,0.71,0.68,0.66,0.64,0.62],
    'Cphys': [0.72,0.69,0.65,0.64,0.61,0.60,0.58,0.56,0.54,0.52],
    'Snet': [0.54,0.51,0.49,0.48,0.46,0.45,0.43,0.41,0.40,0.38]
})
anchors = pd.DataFrame({
    'anchor': ['HIP 97166 c','HIP 90988 b','HD 42012 b','HD 42012 b','HD 4313 b','HD 11506 d','24 Sex b','HD 167677 b','HD 13931 b','Kepler-48 e','Kepler-48 e'],
    'region': ['cube12_cluster0','cube17_cluster2','cube19_cluster5','cube26_cluster6','cube17_cluster10','cube33_cluster3','cube24_cluster1','cube26_cluster0','cube26_cluster11','cube26_cluster12','cube19_cluster8'],
    'ATI_original':[0.0110,0.0075,0.0064,0.0061,0.0056,0.0052,0.0049,0.0044,0.0040,0.0037,0.0035],
    'ATI_cons':[0.0010,0.0060,0.0049,0.0046,0.0041,0.0038,0.0036,0.0032,0.0029,0.0027,0.0025],
    'rank_final':[np.nan,1,2,3,6,4,5,7,8,9,10],
    'class':['unstable_due_to_large_radius','small_but_stable_deficit','transition_mapper','transition_mapper','stable_secondary','priority_complementary','priority_complementary','priority_complementary','priority_complementary','priority_complementary','priority_complementary']
})
radii = ['r_kNN','1.5 r_kNN','2.0 r_kNN','local shell']
deficits = pd.DataFrame({
    'radius': radii,
    'HIP 97166 c': [0.42,0.08,-0.05,-0.12],
    'HIP 90988 b': [0.14,0.12,0.10,0.09],
    'HD 4313 b': [0.18,0.16,0.11,0.07]
})

# 1 Pipeline flow
fig, ax = plt.subplots(figsize=(11.5,2.6))
ax.axis('off')
steps = ['Sesgo de\ndescubrimiento','Sombra\nobservacional','Mapper','TOI\nregional','ATI\nlocal','ATI\nconservador','Prioridad\nobservacional']
x = np.linspace(0.06,0.94,len(steps))
for i,(xi,label) in enumerate(zip(x,steps)):
    box = FancyBboxPatch((xi-0.065,0.42),0.13,0.30,boxstyle='round,pad=0.015,rounding_size=0.025',fc='#EAF2F8',ec='#2E86C1',lw=1.5)
    ax.add_patch(box); ax.text(xi,0.57,label,ha='center',va='center',fontsize=10,weight='bold')
    if i < len(steps)-1:
        ax.add_patch(FancyArrowPatch((xi+0.07,0.57),(x[i+1]-0.07,0.57),arrowstyle='-|>',mutation_scale=13,lw=1.4,color='#566573'))
ax.text(0.5,0.19,'El pipeline transforma sesgos de descubrimiento en un ranking interpretable de regiones y anclas informativas.',ha='center',va='center',fontsize=11)
fig.savefig(FIG/'01_pipeline_flow.pdf'); plt.close(fig)

# 2 Top regions TOI
fig, ax = plt.subplots(figsize=(8.5,5))
d = toi.sort_values('TOI')
colors = ['#2E86C1' if r=='cube12_cluster0' else '#A9CCE3' for r in d.region]
ax.barh(d.region,d.TOI,color=colors)
ax.set_xlabel('TOI regional'); ax.set_title('Top regiones Mapper por prioridad regional TOI')
ax.grid(axis='x',alpha=.25)
for y,v in enumerate(d.TOI): ax.text(v+0.001,y,f'{v:.3f}',va='center',fontsize=9)
fig.savefig(FIG/'02_top_regions_toi.pdf'); plt.close(fig)

# 3 Mapper stylized map
fig, ax = plt.subplots(figsize=(8,5.8))
ax.axis('off')
np.random.seed(4)
regions = toi.region.tolist()
pos = {r:(np.random.rand()*0.8+0.1,np.random.rand()*0.7+0.15) for r in regions}
pos['cube12_cluster0']=(0.25,0.72); pos['cube17_cluster2']=(0.55,0.64); pos['cube19_cluster5']=(0.45,0.40); pos['cube26_cluster6']=(0.64,0.37); pos['cube17_cluster10']=(0.70,0.62)
edges=[('cube12_cluster0','cube17_cluster2'),('cube17_cluster2','cube17_cluster10'),('cube17_cluster2','cube19_cluster5'),('cube19_cluster5','cube26_cluster6'),('cube26_cluster6','cube26_cluster0'),('cube26_cluster6','cube26_cluster11'),('cube26_cluster11','cube26_cluster12'),('cube24_cluster1','cube33_cluster3'),('cube33_cluster3','cube12_cluster0')]
for a,b in edges:
    ax.plot([pos[a][0],pos[b][0]],[pos[a][1],pos[b][1]],color='#AEB6BF',lw=1.2,zorder=1)
for _,row in toi.iterrows():
    xi,yi=pos[row.region]
    size=900*(row.TOI/toi.TOI.max())+150
    color=plt.cm.Blues((row.TOI-toi.TOI.min())/(toi.TOI.max()-toi.TOI.min()+1e-9)*0.7+0.25)
    ax.scatter([xi],[yi],s=size,color=color,edgecolor='#34495E',lw=1.1,zorder=2)
    ax.text(xi,yi,row.region.replace('_','\n'),ha='center',va='center',fontsize=7,zorder=3)
ax.set_title('Lectura Mapper estilizada: nodos coloreados por TOI')
fig.savefig(FIG/'03_mapper_toi_stylized.pdf'); plt.close(fig)

# 4 TOI components heatmap
fig, ax = plt.subplots(figsize=(9,4.8))
comp = toi.set_index('region')[['Shadow','LowImputation','Cphys','Snet']].head(8)
im=ax.imshow(comp.values,aspect='auto',cmap='Blues',vmin=0,vmax=1)
ax.set_xticks(range(comp.shape[1])); ax.set_xticklabels(['Sombra','Baja imputación','Continuidad física','Soporte red'],rotation=20,ha='right')
ax.set_yticks(range(comp.shape[0])); ax.set_yticklabels(comp.index)
for i in range(comp.shape[0]):
    for j in range(comp.shape[1]): ax.text(j,i,f'{comp.iloc[i,j]:.2f}',ha='center',va='center',fontsize=8)
ax.set_title('Componentes ejecutivos del TOI regional')
fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label='Intensidad normalizada')
fig.savefig(FIG/'04_toi_components_heatmap.pdf'); plt.close(fig)

# 5 ATI original
fig, ax = plt.subplots(figsize=(8.5,5))
d=anchors.sort_values('ATI_original').tail(10)
colors=['#D4AC0D' if a=='HIP 97166 c' else '#F9E79F' for a in d.anchor]
labels=[f'{a}\n{r}' for a,r in zip(d.anchor,d.region)]
ax.barh(labels,d.ATI_original,color=colors)
ax.set_xlabel('ATI original'); ax.set_title('Anclas exploratorias por ATI original')
ax.grid(axis='x',alpha=.25)
fig.savefig(FIG/'05_top_anchors_ati_original.pdf'); plt.close(fig)

# 6 deficits HIP 97166 c
fig, ax = plt.subplots(figsize=(7.5,4.2))
ax.plot(deficits.radius,deficits['HIP 97166 c'],marker='o',lw=2.2,color='#D4AC0D')
ax.axhline(0,color='#566573',lw=1)
ax.set_ylabel('Déficit relativo local $\Delta_{rel}$'); ax.set_title('HIP 97166 c: señal exploratoria sensible a escala')
ax.grid(alpha=.25)
fig.savefig(FIG/'06_hip97166_multiradius.pdf'); plt.close(fig)

# 7 Original vs conservative scatter
fig, ax = plt.subplots(figsize=(6.8,5.4))
ax.scatter(anchors.ATI_original,anchors.ATI_cons,s=80,color='#7FB3D5',edgecolor='#34495E')
for _,row in anchors.iterrows():
    if row.anchor in ['HIP 97166 c','HIP 90988 b','HD 42012 b','HD 4313 b']:
        ax.annotate(row.anchor,(row.ATI_original,row.ATI_cons),xytext=(5,4),textcoords='offset points',fontsize=8)
ax.set_xlabel('ATI original'); ax.set_ylabel('ATI conservador'); ax.set_title('Validación multiescala: exploración vs prioridad conservadora')
ax.grid(alpha=.25)
fig.savefig(FIG/'07_ati_original_vs_conservative.pdf'); plt.close(fig)

# 8 ATI conservative ranking
fig, ax = plt.subplots(figsize=(8.5,5))
d=anchors.dropna(subset=['rank_final']).sort_values('ATI_cons')
colors=['#1F618D' if a=='HIP 90988 b' else '#AED6F1' for a in d.anchor]
labels=[f'{int(rank)}. {a}\n{r}' for rank,a,r in zip(d.rank_final,d.anchor,d.region)]
ax.barh(labels,d.ATI_cons,color=colors)
ax.set_xlabel('ATI conservador'); ax.set_title('Ranking conservador de anclas observacionales')
ax.grid(axis='x',alpha=.25)
fig.savefig(FIG/'08_top_anchors_ati_conservative.pdf'); plt.close(fig)

# 9 slope chart ranking
fig, ax = plt.subplots(figsize=(8,5.2))
rank_orig = anchors.sort_values('ATI_original', ascending=False).reset_index(drop=True)
rank_orig['rank_original']=rank_orig.index+1
merged=anchors.merge(rank_orig[['anchor','region','rank_original']],on=['anchor','region'])
plot=merged[merged['anchor'].isin(['HIP 97166 c','HIP 90988 b','HD 42012 b','HD 4313 b','HD 11506 d','24 Sex b'])].copy()
plot['rank_cons']=plot['rank_final'].fillna(11)
for _,row in plot.iterrows():
    ax.plot([0,1],[row.rank_original,row.rank_cons],marker='o',lw=1.8,color='#5DADE2' if row.anchor!='HIP 97166 c' else '#D4AC0D')
    ax.text(-0.03,row.rank_original,row.anchor,ha='right',va='center',fontsize=8)
    ax.text(1.03,row.rank_cons,row.anchor,ha='left',va='center',fontsize=8)
ax.set_xlim(-0.55,1.55); ax.set_ylim(11.5,0.5); ax.set_xticks([0,1]); ax.set_xticklabels(['ATI original','ATI conservador'])
ax.set_ylabel('Posición de ranking'); ax.set_title('Cambio de lectura: de señal exploratoria a prioridad estable')
ax.grid(axis='y',alpha=.20)
fig.savefig(FIG/'09_ranking_shift.pdf'); plt.close(fig)

# 10 deficits HIP 90988 and HD4313
fig, ax = plt.subplots(figsize=(7.5,4.2))
ax.plot(deficits.radius,deficits['HIP 90988 b'],marker='o',lw=2.2,color='#1F618D',label='HIP 90988 b')
ax.plot(deficits.radius,deficits['HD 4313 b'],marker='s',lw=2.0,color='#58D68D',label='HD 4313 b')
ax.axhline(0,color='#566573',lw=1)
ax.set_ylabel('Déficit relativo local $\Delta_{rel}$'); ax.set_title('Señales conservadoras: estabilidad multirradio')
ax.legend(frameon=False); ax.grid(alpha=.25)
fig.savefig(FIG/'10_stable_multiradius.pdf'); plt.close(fig)

# 11 Candidate cards
fig, ax = plt.subplots(figsize=(11,5.8)); ax.axis('off')
cards=[('cube12_cluster0','Región top por TOI','TOI ≈ 0.077','Prioridad regional con sombra, continuidad física y soporte Mapper.'),('HIP 90988 b','Ancla conservadora','ATI_cons ≈ 0.006','Caso final más defendible bajo estabilidad multiescala.'),('HD 42012 b','Transición Mapper','cube19_cluster5 / cube26_cluster6','Ancla de frontera que conecta vecindarios por solapamiento.'),('HD 4313 b','Candidato secundario','cube17_cluster10','Déficit positivo relativamente estable como soporte adicional.')]
for i,(name,role,val,msg) in enumerate(cards):
    col=i%2; row=i//2
    x0=0.06+col*0.47; y0=0.56-row*0.40
    box=FancyBboxPatch((x0,y0),0.40,0.30,boxstyle='round,pad=0.02,rounding_size=0.025',fc='#F8F9F9',ec='#2E86C1',lw=1.5)
    ax.add_patch(box)
    ax.text(x0+0.03,y0+0.23,name,fontsize=14,weight='bold',color='#1B4F72')
    ax.text(x0+0.03,y0+0.17,role,fontsize=10,weight='bold')
    ax.text(x0+0.03,y0+0.12,val,fontsize=10,color='#566573')
    ax.text(x0+0.03,y0+0.055,msg,fontsize=9,wrap=True)
ax.set_title('Fichas ejecutivas de resultados principales',fontsize=15,weight='bold')
fig.savefig(FIG/'11_candidate_cards.pdf'); plt.close(fig)

# 12 Final priority ranking
final = anchors.dropna(subset=['rank_final']).sort_values('rank_final').copy()
fig, ax = plt.subplots(figsize=(8.7,5.2))
d=final.sort_values('rank_final',ascending=False)
colors=[]
for c in d['class']:
    colors.append({'small_but_stable_deficit':'#1F618D','transition_mapper':'#5DADE2','stable_secondary':'#58D68D','priority_complementary':'#D6EAF8'}.get(c,'#D6EAF8'))
labels=[f'{int(r)}. {a}\n{reg}' for r,a,reg in zip(d.rank_final,d.anchor,d.region)]
score=(11-d.rank_final)/10
ax.barh(labels,score,color=colors)
ax.set_xlabel('Prioridad relativa normalizada'); ax.set_title('Top 10 prioridades observacionales')
ax.grid(axis='x',alpha=.25)
fig.savefig(FIG/'12_final_priority_ranking.pdf'); plt.close(fig)

# 13 class composition
fig, ax = plt.subplots(figsize=(7.2,4.6))
counts=final['class'].map({'small_but_stable_deficit':'Conservador principal','transition_mapper':'Transición Mapper','stable_secondary':'Secundario estable','priority_complementary':'Complementario'}).value_counts()
ax.pie(counts.values,labels=counts.index,autopct='%1.0f%%',startangle=90,colors=['#1F618D','#5DADE2','#58D68D','#D6EAF8'])
ax.set_title('Composición interpretativa del ranking final')
fig.savefig(FIG/'13_priority_classes.pdf'); plt.close(fig)

# Save source tables
(toi).to_csv(ROOT/'outputs'/'final_report'/'toi_regions_summary.csv',index=False)
(anchors).to_csv(ROOT/'outputs'/'final_report'/'anchor_priority_summary.csv',index=False)
(deficits).to_csv(ROOT/'outputs'/'final_report'/'multiradius_deficits_summary.csv',index=False)
print(f'Generated figures in {FIG}')
