"""保存済み40出力の固定結合で、0.1 msのスパイク相互相関を素材・結合別に計算する。

python f_run/run_liquid_readout_correlation.py
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '4')
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
from f_run.run_calibration import _simulate_states


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(2**20), b''):
            h.update(part)
    return h.hexdigest()


def choose_pairs(gains, seed=12345):
    # 80×40の全てのpre/post組を保持し、接続フラグで分類する。
    pre, post = np.indices(gains.shape)
    return np.column_stack((pre.ravel(), post.ravel(), (gains > 0).ravel())).astype(int)


def pair_correlograms(pre_packed, post_packed, pairs, max_lag, *, trial_batch=8, pair_batch=32):
    """各試行の線形相互相関を足す。正lagはpreが先、postが後。"""
    trials = len(pre_packed); steps = 5000; fft_size = 2*steps
    n_freq = fft_size//2+1
    pi, pj = pairs[:, 0], pairs[:, 1]
    counts = np.empty((len(pairs), 2*max_lag+1), np.int32)
    same_spectrum = np.zeros((len(pairs), n_freq), np.complex64)
    pre_total_time = np.zeros((pre_packed.shape[1],steps), np.int64)
    post_total_time = np.zeros((post_packed.shape[1],steps), np.int64)
    for begin in range(0,trials,trial_batch):
        pre = np.unpackbits(pre_packed[begin:begin+trial_batch],axis=-1,count=steps).astype(np.float32)
        post = np.unpackbits(post_packed[begin:begin+trial_batch],axis=-1,count=steps).astype(np.float32)
        pre_total_time += pre.sum(axis=0,dtype=np.int64)
        post_total_time += post.sum(axis=0,dtype=np.int64)
        fx = np.fft.rfft(pre,n=fft_size,axis=-1).astype(np.complex64)
        fy = np.fft.rfft(post,n=fft_size,axis=-1).astype(np.complex64)
        # 周波数方向に分割して、全pre×postをまとめて積算する。
        # ペア単位でFFT積を繰り返すより大幅に少ないカーネル呼び出しで処理できる。
        freq_batch = 512
        for f0 in range(0,n_freq,freq_batch):
            f1=min(f0+freq_batch,n_freq)
            cross=np.einsum('bif,bjf->ijf',fx[:,:,f0:f1].conj(),fy[:,:,f0:f1],optimize=True)
            same_spectrum[:,f0:f1] += cross.reshape(len(pairs),f1-f0)
    lag_steps = np.arange(-max_lag,max_lag+1)
    indices = lag_steps % fft_size
    rounding_error = 0.
    for start in range(0,len(pairs),pair_batch):
        end=min(start+pair_batch,len(pairs))
        selected=np.fft.irfft(same_spectrum[start:end],n=fft_size,axis=-1)[:,indices]
        rounding_error=max(rounding_error,float(np.max(np.abs(selected-np.rint(selected)))))
        counts[start:end]=np.rint(selected).astype(np.int32)
    if rounding_error > .1:
        raise ArithmeticError(f'FFT pair-count roundoff too large: {rounding_error}')
    del same_spectrum
    # 全ての異なる試行の組み合わせを用いたshift predictor。
    # sum_{i != j} corr(pre_i, post_j)/(n_trials-1)で同一試行の合計とスケールをそろえる。
    fpre = np.fft.rfft(pre_total_time,n=fft_size,axis=-1)
    fpost = np.fft.rfft(post_total_time,n=fft_size,axis=-1)
    shifted = np.empty(counts.shape,np.float32)
    for start in range(0,len(pairs),pair_batch):
        end=min(start+pair_batch,len(pairs))
        all_spectrum = fpre[pi[start:end]].conj()*fpost[pj[start:end]]
        total = np.fft.irfft(all_spectrum,n=fft_size,axis=-1)[:,indices]
        shifted[start:end] = ((np.rint(total)-counts[start:end])/(trials-1)).astype(np.float32)
    prefix = np.pad(np.cumsum(pre_total_time,axis=1),((0,0),(1,0)))
    exposure = np.empty((pre_packed.shape[1],len(lag_steps)),np.int64)
    for j,lag in enumerate(lag_steps):
        exposure[:,j] = prefix[:,steps-lag] if lag >= 0 else prefix[:,steps]-prefix[:,-lag]
    return {'pair_counts':counts,'shift_predictor_counts':shifted,
            'pre_exposure':exposure,'pre_spike_totals':pre_total_time.sum(axis=1),
            'post_spike_totals':post_total_time.sum(axis=1),
            'lag_ms':lag_steps*.1,'fft_rounding_error':np.array(rounding_error)}


def conditional_rates(counts, shifted, exposure, pre_ids, bin_s=.0001):
    denominator = exposure[pre_ids]*bin_s
    raw = np.full(counts.shape,np.nan,dtype=np.float32)
    corrected = np.full(counts.shape,np.nan,dtype=np.float32)
    np.divide(counts,denominator,out=raw,where=denominator>0)
    np.divide(counts-shifted,denominator,out=corrected,where=denominator>0)
    return raw,corrected


def window_scores(counts, lag_ms, duration_s, taus=(5.,10.,20.,40.), area_ms=.00819):
    """固定スパイク・加算型all-to-all STDPの、clip前の有限窓内更新量/秒。"""
    positive,negative = lag_ms > 1e-9, lag_ms < -1e-9
    ltp = np.stack([(area_ms/tau)*(counts[:,positive]@np.exp(-lag_ms[positive]/tau))/duration_s for tau in taus],axis=1)
    ltd = np.stack([-(area_ms/tau)*(counts[:,negative]@np.exp(lag_ms[negative]/tau))/duration_s for tau in taus],axis=1)
    return ltp,ltd,ltp[:,:,None]+ltd[:,None,:]


def self_test():
    # 実際の時間刻みで、pre(t=10ms), post(t=10.7ms)の符号を確認。
    x=np.zeros((2,1,5000),np.uint8); y=np.zeros_like(x)
    x[0,0,100]=1; y[0,0,107]=1
    x[1,0,200]=1; y[1,0,207]=1
    pairs=np.array([[0,0,1]])
    result=pair_correlograms(np.packbits(x,axis=-1),np.packbits(y,axis=-1),pairs,20,trial_batch=1)
    peak=int(np.argmax(result['pair_counts'][0]))
    assert np.isclose(result['lag_ms'][peak],.7)
    assert result['pair_counts'][0,peak]==2 and result['pair_counts'].sum()==2
    # 異なる試行間では±10ms以上離れるので、この短い範囲のshift predictorは0。
    np.testing.assert_allclose(result['shift_predictor_counts'],0,atol=1e-6)
    ltp,ltd,score=window_scores(result['pair_counts'],result['lag_ms'],1.)
    assert np.all(ltp>0) and np.all(ltd==0) and np.all(score>0)
    reverse=pair_correlograms(np.packbits(y,axis=-1),np.packbits(x,axis=-1),pairs,20,trial_batch=1)
    assert np.isclose(reverse['lag_ms'][np.argmax(reverse['pair_counts'][0])],-.7)
    # ゼロgain/無活動の相手に対して架空の相関を作らない。
    y[:]=0
    empty=pair_correlograms(np.packbits(x,axis=-1),np.packbits(y,axis=-1),pairs,20)
    assert not empty['pair_counts'].any()


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir',type=Path,default=ROOT/'g_tactile_results/calibration/output_calibration')
    parser.add_argument('--input-dir',type=Path,default=ROOT/'g_tactile_results/calibration/filtered_inputs')
    parser.add_argument('--out-dir',type=Path,default=ROOT/'g_tactile_results/liquid_readout_crosscorrelation_allpairs')
    parser.add_argument('--max-lag-ms',type=float,default=250.)
    parser.add_argument('--batch-size',type=int,default=24)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    self_test()
    if args.self_test:
        print('PASS: exact pair-count sign, shift predictor, STDP window and inactive controls');return
    source,out=args.source_dir.resolve(),args.out_dir.resolve()
    summary=json.loads((source/'output_summary.json').read_text(encoding='utf-8'))
    materials=[r['material'] for r in summary['materials']]
    wi=np.load(source/'weights/w_in1_liq1.npy')
    wr=np.load(source/'weights/w_liq1_intra.npy')
    wo=np.load(source/'weights/w_liq1_out1.npy')
    if wi.shape!=(6,100) or wr.shape!=(100,100) or wo.shape[0]!=100 or np.any(wo[80:]!=0):
        raise ValueError('This analysis requires the saved E-only output topology')
    pairs=choose_pairs(wo[:80])
    max_lag=int(round(args.max_lag_ms/.1))
    if not 1 <= max_lag < 5000 or args.batch_size < 1:parser.error('Invalid lag/batch size')
    condition={'source':str(source),'n_liquid_E':80,'n_output':wo.shape[1],
               'gain':summary['output_gain'],'source_probability':summary['output_probability'],
               'n_connected_pairs':int(np.count_nonzero(pairs[:,2])),
               'n_unconnected_pairs':int(np.count_nonzero(pairs[:,2]==0)),
               'bin_ms':.1,'max_lag_ms':max_lag*.1,'materials':materials,
               'positive_lag':'readout after liquid','state_policy':'reset_each_trial','learning':False,
               'code_sha256':sha(Path(__file__)),
               'simulator_sha256':sha(ROOT/'f_run/run_calibration.py'),
               'weight_sha256':{p.name:sha(p) for p in (source/'weights').glob('*.npy')},
               'input_sha256':{m:sha(args.input_dir/f'{m}.npy') for m in materials}}
    if out.exists():
        if json.loads((out/'plan.json').read_text(encoding='utf-8'))!=condition:
            parser.error('Saved analysis conditions differ; choose a new --out-dir')
    else:
        out.mkdir(parents=True);save_json(out/'plan.json',condition)
    np.savez_compressed(out/'connections.npz',pairs=pairs,gains=wo[:80],wi=wi,wr=wr,wo=wo)
    cfg_tree=ast.parse((ROOT/'c_configs/FIXED/cfg_learning_rule_models.py').read_text(encoding='utf-8'))
    cfg_node=next(n for n in cfg_tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CFG_LEARNING_RULE_MODELS' for t in n.targets))
    current_stdp=ast.literal_eval(cfg_node.value)['STDP']
    records=[]; started=time.perf_counter()
    print(f'[source] saved output{wo.shape[1]}, p={summary["output_probability"]}, gain={summary["output_gain"]}; '
          f'all {len(pairs)} pairs ({condition["n_connected_pairs"]} connected, '
          f'{condition["n_unconnected_pairs"]} unconnected)',flush=True)
    for material in materials:
        folder=out/material;folder.mkdir(exist_ok=True)
        spike_path=folder/'spikes.npz'
        if not spike_path.exists():
            pre=np.empty((324,80,625),np.uint8);post=np.empty((324,wo.shape[1],625),np.uint8)
            inputs=np.load(args.input_dir/f'{material}.npy',mmap_mode='r')
            with np.load(source/'all_materials'/material/'liquid/features.npz') as d:
                mapping={int(v):i for i,v in enumerate(d['trial_ids'])}
                liq_ref=d['counts_25ms'][[mapping[i] for i in range(1,325)]]
            with np.load(source/'all_materials'/material/'output/features.npz') as d:
                mapping={int(v):i for i,v in enumerate(d['trial_ids'])}
                out_ref=d['counts_25ms'][[mapping[i] for i in range(1,325)]]
            for begin in range(0,324,args.batch_size):
                end=min(begin+args.batch_size,324)
                px=np.empty((end-begin,80,5000),bool);py=np.empty((end-begin,wo.shape[1],5000),bool)
                def callback(t,fired,out_fired,v,ov):
                    px[:,:,t]=fired[:,:80];py[:,:,t]=out_fired
                bins,_,liquid=_simulate_states(inputs[begin:end],wi,wr,wo,1.,trace_callback=callback)
                np.testing.assert_array_equal(liquid,liq_ref[begin:end])
                np.testing.assert_array_equal(bins,out_ref[begin:end])
                np.testing.assert_array_equal(px.reshape(end-begin,80,20,250).sum(-1),liquid[:,:80])
                np.testing.assert_array_equal(py.reshape(end-begin,wo.shape[1],20,250).sum(-1),bins)
                pre[begin:end]=np.packbits(px,axis=-1);post[begin:end]=np.packbits(py,axis=-1)
            np.savez_compressed(spike_path,pre_packed=pre,post_packed=post,trial_ids=np.arange(1,325),dt_ms=.1)
            print(f'[spikes] {material}: all324 trials match saved liquid/output counts | {time.perf_counter()-started:.1f}s',flush=True)
        with np.load(spike_path) as d:
            pre,post=d['pre_packed'].copy(),d['post_packed'].copy()
        ccg_path=folder/'correlograms.npz'
        if not ccg_path.exists():
            result=pair_correlograms(pre,post,pairs,max_lag)
            taus=np.array([5.,10.,20.,40.]);area_ms=.0007*11.7
            ltp,ltd,score=window_scores(result['pair_counts'],result['lag_ms'],324*.5,taus=taus,area_ms=area_ms)
            lag=result['lag_ms'];positive=lag>1e-9;negative=lag< -1e-9
            current_kernel=np.zeros(len(lag))
            current_kernel[positive]=current_stdp['A_plus']*np.exp(-lag[positive]/current_stdp['tau_plus'])
            current_kernel[negative]=current_stdp['A_minus']*np.exp(lag[negative]/current_stdp['tau_minus'])
            result.update(taus_ms=taus,window_area_ms=np.array(area_ms),ltp_per_s=ltp,
                          ltd_per_s=ltd,net_update_per_s=score,
                          current_stdp_update_per_s=result['pair_counts']@current_kernel/(324*.5))
            np.savez_compressed(ccg_path,**result)
            print(f'[CCG] {material}: {len(pairs)} individual pairs at0.1ms, +/-{args.max_lag_ms:g}ms | {time.perf_counter()-started:.1f}s',flush=True)
        with np.load(ccg_path) as d:
            raw,corrected=conditional_rates(d['pair_counts'],d['shift_predictor_counts'],d['pre_exposure'],pairs[:,0])
            connected=pairs[:,2].astype(bool)
            # 生存ペアはpre/postの両方に少なくとも1回の発火があるもの。
            active=(d['pre_spike_totals'][pairs[:,0]]>0)&(d['post_spike_totals'][pairs[:,1]]>0)
            idx=np.flatnonzero(connected&active)
            average=np.nanmean(corrected[idx],axis=0)
            peak=int(np.nanargmax(average))
            records.append({'material':material,'trials':324,'connected_pairs':int(connected.sum()),
                            'active_connected_pairs':len(idx),'unconnected_pairs':int((~connected).sum()),
                            'mean_connected_excess_peak_lag_ms':float(d['lag_ms'][peak]),
                            'mean_connected_excess_peak_hz':float(average[peak]),
                            'mean_window_update_per_s':d['net_update_per_s'][connected].mean(axis=0).tolist(),
                            'fft_rounding_error':float(d['fft_rounding_error'])})
    save_json(out/'summary.json',{'condition':condition,'materials':records,'current_stdp':current_stdp,
                                 'candidate_taus_ms':[5,10,20,40],
                                 'zero_lag_policy':'excluded from STDP overlap; plotted and counted separately',
                                 'window_estimate':'fixed-spike all-to-all additive pair rule, before clipping, finite +/-lag range',
                                 'shift_predictor':'all different-trial pairs, scaled to same-trial total; descriptive correction, not a causal test'})
    print(f'[complete] {out}',flush=True)


if __name__=='__main__':main()
