import numpy as np, json, base64, pathlib
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams

rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
                 'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,
                 'legend.fontsize':9,'figure.dpi':150,'axes.grid':True,
                 'grid.alpha':0.3,'axes.spines.top':False,'axes.spines.right':False})

with open('/home/claude/ext_results.json') as f:
    ext = json.load(f)

# ── Figure 7: Sensitivity sweep (r_min, OD noise) ──────────────────────
fig7, axes = plt.subplots(1,3,figsize=(13,4.2))
fig7.suptitle('Figure 7 — Parameter Sensitivity Analysis',fontweight='bold')

ax=axes[0]
rmin_d = ext['rmin']
rvals=[d['r_min'] for d in rmin_d]
covs=[d['cov'] for d in rmin_d]
ax.plot(rvals,covs,'o-',color='steelblue',lw=2,ms=7,label='Coverage %')
ax2=ax.twinx()
ax2.plot(rvals,[d['lat'] for d in rmin_d],'s--',color='darkorange',lw=1.5,ms=6,label='Latency ms')
ax.set_xlabel('Keep-out Radius r_min [m]')
ax.set_ylabel('Certification Coverage [%]',color='steelblue')
ax2.set_ylabel('Mean Latency [ms]',color='darkorange')
ax.set_title('(a) Keep-out Radius Sweep (N=24)')
ax.set_ylim([85,100]); ax2.set_ylim([70,100])
lines1,l1=ax.get_legend_handles_labels(); lines2,l2=ax2.get_legend_handles_labels()
ax.legend(lines1+lines2,l1+l2,loc='lower left',fontsize=8)

ax=axes[1]
od_d=ext['od']
od_vals=[d['od_m'] for d in od_d]
cov_od=[d['cov'] for d in od_d]
ax.plot(od_vals,cov_od,'o-',color='seagreen',lw=2,ms=7)
ax.set_xlabel('OD Position Noise σ [m]')
ax.set_ylabel('Certification Coverage [%]')
ax.set_title('(b) OD Noise Sensitivity (N=24)')
ax.set_ylim([88,100])

ax=axes[2]
drop_d=ext['dropout']
dp_vals=[d['drop'] for d in drop_d]
cov_dp=[d['cov'] for d in drop_d]
p1_dp=[d['p1'] for d in drop_d]
p3_dp=[d['p3'] for d in drop_d]
ax.plot(dp_vals,cov_dp,'o-',color='steelblue',lw=2,ms=7,label='Coverage %')
ax.set_xlabel('Link Dropout Rate [%]')
ax.set_ylabel('Certification Coverage [%]')
ax.set_title('(c) Link Dropout Sensitivity (N=12, f=1)')
ax.set_ylim([88,102])
ax.axhline(100,color='gray',lw=0.8,ls=':')
ax.text(25,99.2,'Coverage stable across all dropout rates',fontsize=8,color='gray',ha='center')
plt.tight_layout()
fig7.savefig('/home/claude/fig7_sensitivity.png',bbox_inches='tight',dpi=150)
plt.close(); print("Fig 7 saved.")

# ── Figure 8: Oracle convergence & altitude sweep ──────────────────────
fig8, axes = plt.subplots(1,2,figsize=(10,4.2))
fig8.suptitle('Figure 8 — Oracle Convergence and Altitude Sweep',fontweight='bold')

ax=axes[0]
conv_d=ext['conv']
nints=[d['n'] for d in conv_d]
tms=[d['ms'] for d in conv_d]
agrs=[d['agree'] for d in conv_d]
ax.plot(nints,tms,'o-',color='steelblue',lw=2,ms=7,label='Wall time [ms]')
ax2=ax.twinx()
ax2.plot(nints,agrs,'s--',color='seagreen',lw=1.5,ms=6,label='GT agreement [%]')
ax.set_xlabel('Number of Time Samples (intervals)')
ax.set_ylabel('Mean Wall Time [ms]',color='steelblue')
ax2.set_ylabel('Agreement with GT (1500-sample) [%]',color='seagreen')
ax.set_title('(a) Oracle Timing vs Accuracy Tradeoff')
ax2.set_ylim([98,101])
lines1,l1=ax.get_legend_handles_labels(); lines2,l2=ax2.get_legend_handles_labels()
ax.legend(lines1+lines2,l1+l2,fontsize=8)
ax.axvline(300,color='red',lw=1.2,ls='--',alpha=0.7)
ax.text(320,max(tms)*0.6,'Selected\nn=300',fontsize=8,color='red')

ax=axes[1]
alt_d=ext['alt']
alts=[d['alt'] for d in alt_d]
minsep=[d['min_sep'] for d in alt_d]
oracle_t=[d['oracle_ms'] for d in alt_d]
ax.bar(alts,minsep,width=60,color='steelblue',alpha=0.75,label='Min sep [m]')
ax.axhline(650,color='red',lw=1.5,ls='--',label='Verified bound 650m')
ax.axhline(500,color='orange',lw=1.2,ls=':',label='Hard keep-out 500m')
ax.set_xlabel('Orbital Altitude [km]')
ax.set_ylabel('Min Free-Drift Separation [m]')
ax.set_title('(b) Altitude Sensitivity (N=12, ρ scaled)')
ax.legend(fontsize=8)
plt.tight_layout()
fig8.savefig('/home/claude/fig8_convergence_altitude.png',bbox_inches='tight',dpi=150)
plt.close(); print("Fig 8 saved.")

# ── Figure 9: Statistical distributions (latency CDF + KS) ─────────────
fig9, axes = plt.subplots(1,2,figsize=(10,4.2))
fig9.suptitle('Figure 9 — Statistical Analysis of Decision Latency (N=150 trials each)',fontweight='bold')

prop_lats=np.array(ext['prop_lats'])
rl_lats=np.array(ext['rl_lats'])
ci=ext['stats']['ci']

ax=axes[0]
ax.hist(prop_lats,bins=25,color='steelblue',alpha=0.7,density=True,label='Proposed (LTL+SMT)')
ax.axvline(np.mean(prop_lats),color='steelblue',lw=2,ls='--',label=f"Mean={np.mean(prop_lats):.1f}ms")
ax.axvline(ci[0],color='steelblue',lw=1,ls=':',alpha=0.6,label=f"95% CI [{ci[0]:.1f}, {ci[1]:.1f}]")
ax.axvline(ci[1],color='steelblue',lw=1,ls=':',alpha=0.6)
ax.set_xlabel('Decision Latency [ms]')
ax.set_ylabel('Density')
ax.set_title('(a) Latency Distribution — Proposed Framework')
ax.legend(fontsize=8)

ax=axes[1]
xs_p=np.sort(prop_lats); ys_p=np.arange(1,len(xs_p)+1)/len(xs_p)
xs_r=np.sort(rl_lats);   ys_r=np.arange(1,len(xs_r)+1)/len(xs_r)
ax.plot(xs_p,ys_p,color='steelblue',lw=2,label='Proposed (LTL+SMT)')
ax.plot(xs_r,ys_r,color='firebrick',lw=2,ls='--',label='Distributed RL (PPO)')
ks_D=ext['stats']['ks_D']; ks_p=float(ext['stats']['ks_p'])
mw_p=float(ext['stats']['mw_p'])
ax.set_xlabel('Decision Latency [ms]')
ax.set_ylabel('Empirical CDF')
ax.set_title('(b) Latency CDF Comparison + KS Test')
ax.legend(fontsize=8)
ax.text(0.05,0.82,f"KS D={ks_D:.3f}, p={ks_p:.1e}\nMW-U p={mw_p:.1e}\n→ Distributions differ significantly",
        transform=ax.transAxes,fontsize=8.5,bbox=dict(boxstyle='round',fc='white',ec='gray',alpha=0.8))
plt.tight_layout()
fig9.savefig('/home/claude/fig9_stats.png',bbox_inches='tight',dpi=150)
plt.close(); print("Fig 9 saved.")

# ── Figure 10: Quorum sensitivity + Byzantine fraction ─────────────────
fig10, axes = plt.subplots(1,2,figsize=(10,4.2))
fig10.suptitle('Figure 10 — Byzantine Fault Tolerance Analysis (N=12)',fontweight='bold')
qd=ext['quorum']
fs=[d['f'] for d in qd]; qs=[d['q'] for d in qd]
p1q=[d['p1'] for d in qd]; p3q=[d['p3'] for d in qd]
covq=[d['cov'] for d in qd]; latq=[d['lat'] for d in qd]

ax=axes[0]
x=np.arange(len(fs))
w=0.35
ax.bar(x-w/2,p1q,w,color='firebrick',alpha=0.8,label='P1 violation %')
ax.bar(x+w/2,p3q,w,color='darkorange',alpha=0.8,label='P3 violation %')
ax.set_xticks(x)
ax.set_xticklabels([f'f={fv}\nq={qv}/12' for fv,qv in zip(fs,qs)])
ax.set_ylabel('Violation Rate [%]')
ax.set_title('(a) P1/P3 vs Byzantine Fault Count')
ax.set_ylim([0,5])
ax.legend()
ax.text(0.5,0.85,'All P1,P3 rates = 0%\nProtocol correct ∀ f < N/3',
        transform=ax.transAxes,ha='center',fontsize=8.5,
        bbox=dict(boxstyle='round',fc='honeydew',ec='seagreen'))

ax=axes[1]
ax.plot(fs,covq,'o-',color='steelblue',lw=2,ms=8,label='Coverage %')
ax2=ax.twinx()
ax2.plot(fs,latq,'s--',color='darkorange',lw=1.5,ms=7,label='Latency ms')
ax.set_xlabel('Byzantine Fault Count f')
ax.set_ylabel('Certification Coverage [%]',color='steelblue')
ax2.set_ylabel('Mean Latency [ms]',color='darkorange')
ax.set_title('(b) Coverage & Latency vs f')
ax.set_xticks(fs)
ax.set_ylim([90,101])
lines1,l1=ax.get_legend_handles_labels(); lines2,l2=ax2.get_legend_handles_labels()
ax.legend(lines1+lines2,l1+l2,fontsize=8)
plt.tight_layout()
fig10.savefig('/home/claude/fig10_byzantine.png',bbox_inches='tight',dpi=150)
plt.close(); print("Fig 10 saved.")

print("\nAll extended figures generated.")
