"""Verify imported liquid weights and fresh-trial responses against calibration."""
from pathlib import Path
import sys
import json

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    import numpy as np
    import pandas as pd
    from brian2 import prefs,start_scope,defaultclock,ms,second,seed,TimedArray
    from f_run.run_training import build_cfg,build_network_cfg,build_input_current,load_tactile_data,make_training_network
    from f_run.run_liquid import make_liquid_network
    cfg=build_cfg();nc=build_network_cfg(cfg)
    assert nc['N_liq']==[100] and nc['synapse_model']=='double_exp'
    assert nc['tau_exc']==nc['tau_inh']==10 and nc['tau_r']==2 and nc['tau_d']==30
    assert not nc['poisson_input']['enabled']
    start_scope();prefs.codegen.target='numpy';defaultclock.dt=float(nc['dt_ms'])*ms;seed(12345)
    o=make_liquid_network(nc,6,np.random.default_rng(12345),cfg['run'])
    assert np.array_equal(o['G_liq'][0].typ[:],np.r_[np.ones(80),-np.ones(20)])
    wi=np.zeros((6,100));wr=np.zeros((100,100))
    for s in o['S_in']:wi[np.asarray(s.i[:]),np.asarray(s.j[:])]+=np.asarray(s.w[:])
    for s in o['S_intra']:
        sign=-1 if 'intra_I_' in s.name else 1
        wr[np.asarray(s.i[:]),np.asarray(s.j[:])]+=np.asarray(s.w[:])*sign
    ref=ROOT.parent/'LSM_liquid_cma_es/g_tactile_results/gain179_sequential_calibration_20261019'
    assert np.allclose(wi,np.load(ref/'debug_sid203/washi/weights/w_in1_liq1.npy'),atol=1e-12,rtol=1e-12)
    assert np.allclose(wr,np.load(ref/'debug_sid203/washi/weights/w_liq1_intra.npy'),atol=1e-12,rtol=1e-12)
    assert np.sum(np.any(wi!=0,axis=0))==25
    for m in o['M_liq_debug']:m.active=True
    o['net'].store('initial')
    results=[]
    for material in ['Al_board','buta_omote','buta_ura','cork','denim','rubber_board','washi','wood_board']:
        signal=pd.read_table(load_tactile_data(material,203),header=None).to_numpy().T[:3,3000:8000]
        dt=float(cfg['common']['dt_ms'])
        current=build_input_current(signal,np.arange(5000)*dt,dt,cfg['input_filter_map'],cfg['filter_funcs'],cfg['sensor_gain'],cfg['filter_gain'],cfg['opt_filter_gain'])
        o['net'].restore('initial',restore_random_state=True)
        input_ta=TimedArray(current.T,dt=float(nc['dt_ms'])*ms)
        o['G_in'].namespace['input_ta']=input_ta;o['G_in'].t_start=defaultclock.t
        o['net'].run(.5*second,namespace={'input_ta':input_ta})
        m=o['M_liq_debug'][0]
        b=np.zeros((100,20),int)
        np.add.at(b,(np.asarray(m.i,dtype=int),np.floor((np.asarray(m.t/ms)+1e-5)/25).astype(int)),1)
        with np.load(ref/'all_materials'/material/'features.npz') as d:expected=d['counts_25ms'][202]
        delta=int(np.max(abs(b-expected)))
        results.append(dict(material=material,max_bin_difference=delta,mean_E_rate_hz=float(b[:80].sum()/80/.5)))
        print(results[-1],flush=True)
    assert all(r['max_bin_difference']==0 for r in results)
    start_scope();seed(12345);defaultclock.dt=float(nc['dt_ms'])*ms
    training=make_training_network(nc,6,np.random.default_rng(12345),cfg['run'])
    training['G_in'].namespace['input_ta']=TimedArray(np.zeros((21,6)),dt=float(nc['dt_ms'])*ms)
    training['net'].run(1*ms)
    report=dict(status='complete',input_weight_matrix_matches=True,recurrent_weight_matrix_matches=True,all8_sid203_spike_bins_match=True,n_liquid=100,n_E=80,n_I=20,n_direct_input_E=25,all_material_checks=results,training_network_build_and_run_passed=True,output_neurons=nc['N_out'],learning_rule=nc['learning_rule'],validation_state_policy='Explicit store/restore for independent reference comparison; normal run state policy is unchanged')
    audit=ROOT/'g_tactile_results/calibrated_gain179_configuration_check.json'
    audit.parent.mkdir(parents=True,exist_ok=True);audit.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
