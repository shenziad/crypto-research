"""Save selected constructive gate sources/results without environments/weights."""
import datetime
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--version',choices=('v2','v3','v4','v5','v6','v7','v8'),default='v8');args=parser.parse_args()
    destination=ROOT/'sources/checkpoints'/f'20261029_gated_faithful_{args.version}'
    names=['dpf_reference.py','dpf_aes_native.cpp','dpf_aes_native.py','dpf_aes_native_check.py',
        'dcf_aes_native.py','dcf_aes_native_check.py','fused_gated_lut.py','gated_lut_strong_baselines.py',
        'faithful_shift_check.py','gated_input_quantizer.py','gated_input_quantizer_check.py',
        'qwen_gated_quality.py','qwen_gated_quality_faithful.py','qwen_gate_integer_certificate.py',
        'qwen_gated_strong_protocol.py','fetch_qwen_gate_model.py','checkpoint_gated_faithful.py']
    result_names=['dpf_reference_check.json','dpf_aes_native_check.json','dcf_aes_native_check.json',
        'faithful_shift_check.json','qwen_gate_ranges.json','qwen_gate_capture.npz',
        'qwen_gate_faithful_capture.npz','qwen_gate_fixed8_input_capture.npz','qwen_gate_integer_certificate.json',
        'qwen_layer_lut_tables.npz','qwen_gated_strong_faithful.json','gated_input_quantizer_pipeline.json',
        'gated_input_quantizer_explicit_checks.json','qwen_gated_quality_faithful.json',
        'qwen_gated_quality_fixed8_input.json','qwen_gated_quality_fixed8_input_stage.json',
        'qwen_gated_quality_fixed8_input_late.json']
    files=[ROOT/'experiments'/v for v in names]+[ROOT/'results'/v for v in result_names]
    files += [ROOT/v for v in ['toolchain/dpf_aes_native.so','data/manifest.json','data/wikitext2_test.txt',
        'notes/20261029_gated_strong_faithful.md','README.md','research_progress.json',
        'models/Qwen2.5-0.5B/manifest.json','models/Qwen2.5-0.5B/config.json']]
    files += list((ROOT/'sources/llama_reference').glob('*'))
    if args.version in ('v3','v4','v5','v6','v7','v8'):
        files += [ROOT/'experiments'/v for v in ['gated_contraction_protocol.py','qwen_gated_contraction_protocol.py',
            'gated_contraction_gpu_bridge.py','gated_wide_clamp_identity_check.py','sparse_cyclic_fss_check.py']]
        files += [ROOT/'results'/v for v in ['qwen_down_int12_selected.npz','qwen_down_int12_selected_manifest.json',
            'qwen_gate_fixed8_input_late_capture.npz','qwen_gate_fixed8_input_stage_capture.npz',
            'qwen_gate_fixed8_input_late_extended32_capture.npz','qwen_gated_quality_fixed8_input_late_extended32.json',
            'gated_contraction_synthetic.json','qwen_gated_contraction_first_checks.json','qwen_gated_contraction_protocol.json',
            'gated_wide_quantizer_check.json','gated_wide_clamp_identity.json','gated_input_quantizer_regression.json',
            'gated_contraction_gpu_bridge.json','sparse_cyclic_fss_check.json']]
        files.append(ROOT/'notes/20261029_gated_contraction.md')
    if args.version in ('v4','v5','v6','v7','v8'):
        files += [ROOT/'experiments'/v for v in ['dcf_sparse_native.cpp','dcf_sparse_native.py','dcf_sparse_native_check.py',
            'dcf_public_range.py','dcf_public_range_check.py']]
        files += [ROOT/'results'/v for v in ['dcf_sparse_native_check.json','dcf_public_range_check.json',
            'qwen_gated_quality_fixed8_input_late_g31_upfraction.json','qwen_gated_quality_fixed8_input_late_g63_upfraction.json',
            'qwen_gate_fixed8_input_late_g31_upfraction_capture.npz','qwen_gate_fixed8_input_late_g63_upfraction_capture.npz']]
        files += [ROOT/'toolchain/dcf_sparse_native.so',ROOT/'notes/20261029_public_spline_basis.md',
            ROOT/'sources/eurocrypt2021_mixed_mode/reading_note.json']
    if args.version in ('v5','v6','v7','v8'):
        for pattern in ('*public_spline*','public_silu*','public_polynomial*','dcf_wide*','dcf_spline*','dcf_selector*','qwen_norm_projection*'):
            files+=list((ROOT/'experiments').glob(pattern))
        for pattern in ('*public_spline*','public_silu*','public_polynomial*','dcf_wide*','qwen_norm_projection*'):
            files += [p for p in (ROOT/'results').glob(pattern) if p.suffix in ('.json','.npz','.png','.pdf')]
        files += list((ROOT/'notes').glob('20261029_public_*'))
        files += [ROOT/'toolchain'/v for v in ['dcf_wide_native.so','dcf_spline_native.so','dcf_selector_native.so']]
        files += [ROOT/'data'/v for v in ['ptb_test.txt','ptb_test_manifest.json']]
        files += list((ROOT/'sources/fusefss_reference').glob('*'))
    if args.version in ('v6','v7','v8'):
        files += list((ROOT/'experiments').glob('dcf_cuda*'))
        files += [p for p in (ROOT/'results').glob('dcf_cuda*') if p.suffix in ('.json','.npz')]
        files += [ROOT/'toolchain'/v for v in ['dcf_cuda.so','dcf_cuda_protocol.so']]
        files += [ROOT/'notes'/v for v in ['20261029_standalone_gate_simulation.md','20261029_finite_block_security_draft.md','20261029_claim_registry.json']]
        files += [ROOT/'sources/eurocrypt2021_mixed_mode/public_span_audit.json',ROOT/'experiments/reproduce_public_spline.sh']
    if args.version in ('v7','v8'):
        for pattern in ('*dither*','projective_norm*'):
            files += [p for p in (ROOT/'experiments').glob(pattern) if p.is_file()]
            files += [p for p in (ROOT/'results').glob(pattern) if p.suffix in ('.json','.npz','.png')]
        files += list((ROOT/'notes').glob('20261029_dither*'))
        files += [ROOT/'notes/20261029_projective_norm_screen.md',ROOT/'toolchain/dcf_dither_native.so']
        for name in ('primary_pdf_audit_20261029','google_spline_reference'):
            files += [p for p in (ROOT/'sources'/name).glob('*') if p.is_file()]
    if args.version=='v8':
        for pattern in ('*exact_coarse*','qwen15*','joint_swiglu*','weighted_swiglu*'):
            files += [p for p in (ROOT/'experiments').glob(pattern) if p.is_file()]
            files += [p for p in (ROOT/'results').glob(pattern) if p.suffix in ('.json','.npz','.png','.pdf')]
        files += [ROOT/'experiments/fetch_qwen15_model.py',ROOT/'notes/20261029_precision_side_screens.md']
        files += list((ROOT/'notes').glob('20261029_exact_coarse*'))
        files += [ROOT/'models/Qwen2.5-1.5B'/v for v in ['manifest.json','official_api_metadata.json','config.json','LICENSE']]
        files += [p for p in (ROOT/'sources/checkpoints/20261029_exact_coarse_out0_sources').rglob('*') if p.is_file()]
        files += [ROOT/'sources/checkpoints/20261029_gated_faithful_v7/experiments/dithered_silu.py']
    files=list(dict.fromkeys(files))
    assert not destination.exists(),'Checkpoint is immutable; choose a new version'
    assert all(f.is_file() for f in files),[str(f) for f in files if not f.is_file()]
    entries=[]
    for f in files:
        relative=f.relative_to(ROOT);target=destination/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,target)
        entries.append(dict(path=str(relative),bytes=target.stat().st_size,
            sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    manifest=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_root=str(ROOT),files=entries,
        purpose='Finite operator checkpoint. Environments and model weights excluded; use pinned downloader and model manifest.',
        current_note='notes/20261029_exact_coarse_closure.md' if args.version=='v8' else 'notes/20261029_dithered_protocol_closure.md' if args.version=='v7' else 'notes/20261029_public_spline_gpu_closure.md' if args.version=='v6' else 'notes/20261029_public_spline_frontier.md' if args.version=='v5' else 'notes/20261029_public_spline_basis.md' if args.version=='v4' else 'notes/20261029_gated_contraction.md' if args.version=='v3' else 'notes/20261029_gated_strong_faithful.md',
        security_status='One-shot fixed-input idealDCF joint-view proof draft for exactcoarse basis/weighted; onlinecomposition andimplementationsecurity unestablished.' if args.version=='v8' else 'Functional/numerical research; formal joint-view proof and concrete implementation security not established.',
        limits='Historical fused_gated_lut source supplies material generation. Main protocol runs use faithful FSS shift. Trusted parent/fork harness is not deployment.')
    (destination/'checkpoint_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for entry in entries:
        assert hashlib.sha256((destination/entry['path']).read_bytes()).hexdigest()==entry['sha256']
    print('verified',len(entries),'files',sum(e['bytes'] for e in entries),'bytes',str(destination))


if __name__=='__main__':main()
