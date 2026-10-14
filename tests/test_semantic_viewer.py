"""Semantic rendering must preserve label meaning and committed-file boundaries."""
import importlib.util
import hashlib
import io
import json
from pathlib import Path
import sys
import shutil

import numpy as np
from PIL import Image
import pytest

from test_live_viewer import running_server, request, write_frame

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT / 'scripts'))
import live_viewer as viewer
import render_semantic_gallery as gallery
from camera_delivery_evidence import run_identity


def test_legacy_ade_ids_are_resolved_by_class_name():
    registry = viewer.semantic_registry({'semantics': {'class_names': {'0':'wall','3':'floor','20':'car','42':'column'}}})
    classes = {c['id']:c for c in registry['classes']}
    assert classes[0]['color'] == [230,230,250]  # ADE zero is wall, not nuScenes noise.
    assert classes[3]['color'] == [255,0,255]
    assert classes[20]['color'] == [0,150,245]
    assert classes[42]['nuscenes_name'] == 'manmade'
    assert classes[255]['name'] == 'unknown'
    assert classes[255]['color'] != classes[3]['color']


def test_explicit_native_registry_wins_and_extensions_remain_distinct():
    native = json.loads((PROJECT/'taxonomy/garage.json').read_text())
    registry = viewer.semantic_registry({'semantic_registry': native, 'semantics': {'class_names':{'1':'building'}}})
    classes = {c['id']:c for c in registry['classes']}
    assert registry['schema_version'] == 'orbvi-garage-v1'
    assert classes[1]['name'] == 'floor'
    assert classes[2]['name'] == 'ceiling' and classes[2]['color'] == [255,190,0]
    assert classes[3]['name'] == 'column' and classes[3]['color'] == [0,190,190]
    assert classes[4]['color'] == [0,150,245]


def test_confidence_is_optional_validated_and_never_synthesized(tmp_path):
    write_frame(tmp_path)
    reader = viewer.RunReader(tmp_path)
    assert reader.frame()[0]['confidence'] is None
    path = tmp_path/'occupancy/000000.npz'
    with np.load(path,allow_pickle=False) as stored:
        data = {k:stored[k].copy() for k in stored.files}
    data['fused_confidence'] = np.array([0.,.8],np.float32)
    np.savez_compressed(path,**data)
    np.testing.assert_allclose(reader.frame()[0]['confidence'],[0.,.8])
    data['fused_confidence'][1] = 2.
    np.savez_compressed(path,**data)
    payload,code = reader.frame()
    assert code == 500 and 'confidence' in payload['status_detail']


def _view_fixture(run,prepared):
    prepared.mkdir()
    (prepared/'images').mkdir()
    Image.new('RGB',(4,3),(12,34,56)).save(prepared/'images/raw.jpg')
    raw = {'index':0,'images':{'3':{'path':'images/raw.jpg','stamp_ns':42}}}
    (prepared/'frames.jsonl').write_text(json.dumps(raw)+'\n')
    write_frame(run)
    registry = json.loads((PROJECT/'taxonomy/garage.json').read_text())
    metadata = {'route':'camera','expected_full_recording_frames':2,'config':{'prepared':str(prepared)},'semantic_registry':registry}
    (run/'run.json').write_text(json.dumps(metadata))
    for directory in ('views','semantic_images','semantic_maps','source_points'):
        (run/directory).mkdir()
    Image.new('RGB',(5,5),(90,100,110)).save(run/'semantic_images/000000_3_v0.jpg')
    np.savez_compressed(run/'semantic_maps/000000_3_v0.npz',labels=np.array([[0,1],[4,255]],np.uint8),registry='orbvi-garage-v1',pixel_domain='rectified_virtual_pinhole')
    np.savez_compressed(run/'source_points/000000.npz',source_uv=np.array([[0,0]],np.int32))
    view = {'view_id':'3_v0','physical_camera_id':'3','source_stamp_ns':42,'source_image_path':'images/raw.jpg',
            'image_path':'semantic_images/000000_3_v0.jpg','semantic_image_path':'semantic_images/000000_3_v0.jpg',
            'semantic_maps_path':'semantic_maps/000000_3_v0.npz','pixel_domain':'rectified_virtual_pinhole'}
    (run/'views/000000.json').write_text(json.dumps([view]))
    return view


def test_real_rectified_source_and_mask_keep_their_own_pixel_domain(tmp_path):
    run,prepared = tmp_path/'run',tmp_path/'prepared'
    _view_fixture(run,prepared)
    reader = viewer.RunReader(run)
    payload,code = reader.sources(0)
    assert code == 200 and payload['status']=='ready'
    source = payload['sources'][0]
    assert source['pixel_domain']=='rectified_virtual_pinhole'
    assert source['view_id']=='3_v0' and source['raw_url'].endswith('kind=raw')
    assert payload['point_provenance']=='source_points/000000.npz'
    raw,_ = reader.image(0,'3_v0','raw')
    rectified,_ = reader.image(0,'3_v0','rgb')
    assert Image.open(io.BytesIO(raw)).size == (4,3)
    assert Image.open(io.BytesIO(rectified)).size == (5,5)
    mask,kind = reader.image(0,'3_v0','semantic')
    image = np.asarray(Image.open(io.BytesIO(mask)))
    assert kind=='image/png'
    np.testing.assert_array_equal(image[0,0],[230,230,250])
    np.testing.assert_array_equal(image[0,1],[255,0,255])
    np.testing.assert_array_equal(image[1,0],[0,150,245])
    np.testing.assert_array_equal(image[1,1],[145,145,145])


def test_source_images_never_bypass_commit_or_expose_path_traversal(tmp_path):
    run,prepared = tmp_path/'run',tmp_path/'prepared'
    view = _view_fixture(run,prepared)
    (run/'frames.jsonl').write_text('')
    reader = viewer.RunReader(run)
    with pytest.raises(FileNotFoundError):
        reader.image(0,'3_v0','rgb')
    write_frame(run)
    (run/'run.json').write_text(json.dumps({'route':'camera','config':{'prepared':str(prepared)}}))
    view['image_path']=view['semantic_image_path']='../../secret.jpg'
    (run/'views/000000.json').write_text(json.dumps([view]))
    assert reader.sources(0)[0]['sources']==[]
    with running_server(run) as base:
        assert request(base,'/api/image?index=0&camera=..%2F..%2Fetc&kind=raw')[0]==400
        assert request(base,'/api/sources?index=-1')[0]==400


def test_gallery_indexes_every_commit_and_exports_full_geometry_with_disclosed_cut(tmp_path):
    run = tmp_path/'run'
    write_frame(run)
    write_frame(run,1)
    registry = json.loads((PROJECT/'taxonomy/garage.json').read_text())
    (run/'run.json').write_text(json.dumps({'route':'camera','expected_full_recording_frames':2,'semantic_registry':registry}))
    (run/'summary.json').write_text('{"complete":true}')
    path = run/'occupancy/000000.npz'
    with np.load(path,allow_pickle=False) as stored:
        data = {k:stored[k].copy() for k in stored.files}
    data['fused_labels']=np.array([2,4],np.uint8)
    np.savez_compressed(path,**data)
    output = tmp_path/'gallery'
    result = gallery.create_gallery(run,output,[0],gallery.resolve_pose('official'))
    assert result['frame_count']==2 and result['complete']
    manifest = json.loads((output/'manifest.json').read_text())
    assert [r['index'] for r in manifest['rows']]==[0,1]
    allheight = json.loads((output/'png/000000_allheight.view.json').read_text())
    cut = json.loads((output/'png/000000_ceiling_cut.view.json').read_text())
    assert allheight['displayed_voxels']==2 and allheight['filter']['hidden_semantic_ids']==[]
    assert cut['displayed_voxels']==1 and cut['filter']['hidden_semantic_ids']==[2]
    assert allheight['geometry_modified'] is cut['geometry_modified'] is False
    header = (output/'ply/000000_full.ply').read_bytes().split(b'end_header\n')[0]
    assert b'element vertex 2' in header
    np.testing.assert_array_equal(np.load(path)['fused_labels'],[2,4])


def test_final_gallery_requires_exact_matching_independent_acceptance(tmp_path):
    run = tmp_path/'run'
    write_frame(run)
    write_frame(run,1)
    metadata = {'route':'camera','expected_full_recording_frames':2,
                'config_sha256':'config','bag_sha256':'bag','inference_cache_key':'cache'}
    (run/'run.json').write_text(json.dumps(metadata))
    (run/'summary.json').write_text('{"complete":true}')
    output = tmp_path/'final-gallery'
    with pytest.raises(ValueError,match='matching independent full QA'):
        gallery.create_gallery(run,output,[0],gallery.resolve_pose('official'),require_complete=True)
    assert not output.exists(), 'A rejected final report must not publish images'
    qa = {'run_directory':str(run.resolve()),'accepted':True,'allow_partial':False,
          'expected_full_recording_frames':2,'statistics':{'verified_frames':2},
          'run_identity':run_identity(run,metadata)}
    (run/'verification.json').write_text(json.dumps(qa))
    result = gallery.create_gallery(run,output,[0],gallery.resolve_pose('official'),require_complete=True)
    assert result['complete'] and result['matching_independent_full_qa']
    published = json.loads((output/'manifest.json').read_text())
    assert published['run_identity'] == run_identity(run,metadata)
    # A different journal with identical rows is a different audited byte snapshot.
    with (run/'frames.jsonl').open('a') as journal:
        journal.write('\n')
    with pytest.raises(ValueError,match='matching independent full QA'):
        gallery.create_gallery(run,tmp_path/'stale-gallery',[0],gallery.resolve_pose('official'),require_complete=True)
    assert not (tmp_path/'stale-gallery').exists()


def test_real_saved_camera_frame_manifest_retains_run_identity_after_png_pair(tmp_path):
    """Copy one actual commit read-only; no neural inference or invented labels."""
    actual = PROJECT/'runs/camera_semantic_full'
    if not (actual/'occupancy/000000.npz').is_file():
        pytest.skip('Actual camera semantic artifacts unavailable')
    run = tmp_path/'real-frame'
    run.mkdir()
    run_bytes = (actual/'run.json').read_bytes()
    metadata = json.loads(run_bytes)
    assert all(metadata.get(key) for key in ('config_sha256','bag_sha256','inference_cache_key'))
    journal_line = (actual/'frames.jsonl').read_bytes().split(b'\n')[0]+b'\n'
    assert json.loads(journal_line)['index'] == 0
    (run/'run.json').write_bytes(run_bytes)
    (run/'frames.jsonl').write_bytes(journal_line)
    for relative in ('occupancy/000000.npz','traversability/000000.npz','source_points/000000.npz','views/000000.json'):
        target = run/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(actual/relative,target)
    for view in json.loads((actual/'views/000000.json').read_text()):
        for key in ('image_path','semantic_image_path','semantic_maps_path'):
            target = run/view[key]
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(actual/view[key],target)
    result = gallery.create_gallery(run,tmp_path/'gallery',[0],gallery.resolve_pose('official'))
    assert result['frame_count'] == 1 and result['complete'] is False
    manifest = json.loads((tmp_path/'gallery/manifest.json').read_text())
    assert len(manifest['samples']) == 2, 'Both real PNG sidecars must be rendered before identity is tested'
    expected = {key:metadata[key] for key in ('config_sha256','bag_sha256','inference_cache_key')}
    expected['run_json_sha256'] = hashlib.sha256(run_bytes).hexdigest()
    expected['frames_jsonl_sha256'] = hashlib.sha256(journal_line).hexdigest()
    assert manifest['run_identity'] == expected


def test_flat_lighting_changes_only_face_colors_and_preserves_true_geometry():
    indices = np.array([[0,0,0],[1,0,0],[0,1,1]],np.int32)
    labels = np.array([1,4,3],np.uint8)
    registry = viewer.semantic_registry({'semantic_registry':json.loads((PROJECT/'taxonomy/garage.json').read_text())})
    palette = gallery._registry_palette(registry)
    bounds = np.array([[0.,0.,0.],[3.,3.,3.]])
    eye = np.array([8.,-9.,12.])
    old = indices.copy(),labels.copy(),palette.copy()
    shaded_vertices,shaded_colors = gallery.semantic_mesh(indices,bounds,.3,palette,labels,eye)
    flat_vertices,flat_colors = gallery.semantic_mesh(indices,bounds,.3,palette,labels,eye,lighting='flat')
    np.testing.assert_array_equal(shaded_vertices,flat_vertices)
    assert np.any(shaded_colors != flat_colors)
    assert set(map(tuple,flat_colors)) <= set(map(tuple,palette[labels]))
    for actual,expected in zip((indices,labels,palette),old):
        np.testing.assert_array_equal(actual,expected)
    official = gallery.resolve_pose('official')
    assert official['projection']=='orthographic' and official['lighting']=='flat'
    assert gallery.resolve_pose('oblique')['pitch']==.72


def test_hybrid_source_variants_use_native_garage_fields_and_distinct_expert_palette(tmp_path):
    run,prepared=tmp_path/'run',tmp_path/'prepared'
    _view_fixture(run,prepared)
    source=run/'semantic_maps/000000_3_v0.npz'
    with np.load(source,allow_pickle=False) as stored:
        arrays={key:stored[key].copy() for key in stored.files}
    arrays.update(official_labels=np.array([[0,2],[4,255]],np.uint8),
                  b5_labels=np.array([[3,1],[4,255]],np.uint8),
                  b5_source_labels=np.full((2,2),149,np.uint8),
                  expert_id=np.array([[0,1],[2,255]],np.uint8))
    np.savez_compressed(source,**arrays)
    reader=viewer.RunReader(run)
    before=reader.frame(0)[0]
    payload,code=reader.sources(0)
    item=payload['sources'][0]
    assert code==200 and item['semantic_variants']==['final','official','indoor','expert']
    assert item['expert_counts']=={'0':1,'1':1,'2':1,'255':1}
    assert item['stamp_ns_text']=='42'
    official=np.asarray(Image.open(io.BytesIO(reader.image(0,'3_v0','official')[0])))
    indoor=np.asarray(Image.open(io.BytesIO(reader.image(0,'3_v0','indoor')[0])))
    expert=np.asarray(Image.open(io.BytesIO(reader.image(0,'3_v0','expert')[0])))
    np.testing.assert_array_equal(official[0,1],[255,190,0])
    np.testing.assert_array_equal(indoor[0,0],[0,190,190])  # Garage column 3, never ADE source149.
    np.testing.assert_array_equal(indoor[0,1],[255,0,255])
    for y,x,cid in [(0,0,0),(0,1,1),(1,0,2),(1,1,255)]:
        np.testing.assert_array_equal(expert[y,x],viewer.EXPERT_REGISTRY[cid]['color'])
    after=reader.frame(0)[0]
    assert before['labels']==after['labels'] and before['grid']==after['grid']
    assert before['centers']==after['centers']


def test_single_provider_hides_unavailable_experts_and_refuses_raw_ade_as_garage(tmp_path):
    run,prepared=tmp_path/'run',tmp_path/'prepared'
    _view_fixture(run,prepared)
    reader=viewer.RunReader(run)
    assert reader.sources(0)[0]['sources'][0]['semantic_variants']==['final']
    with pytest.raises(FileNotFoundError,match='unavailable'):
        reader.image(0,'3_v0','indoor')
    source=run/'semantic_maps/000000_3_v0.npz'
    with np.load(source,allow_pickle=False) as stored:
        arrays={key:stored[key].copy() for key in stored.files}
    arrays['b5_labels']=np.full((2,2),149,np.uint8)
    np.savez_compressed(source,**arrays)
    with pytest.raises(ValueError,match='raw ADE'):
        reader.image(0,'3_v0','indoor')


def test_hybrid_expert_endpoints_keep_the_same_query_path_guards(tmp_path):
    run,prepared=tmp_path/'run',tmp_path/'prepared'
    _view_fixture(run,prepared)
    with running_server(run) as base:
        assert request(base,'/api/image?index=0&camera=3_v0&kind=indoor')[0]==404
        assert request(base,'/api/image?index=0&camera=..%2Fetc&kind=expert')[0]==400
        assert request(base,'/api/image?index=0&camera=3_v0&kind=b5_source_labels')[0]==400
        assert request(base,'/api/image?index=0&camera=3_v0&kind=expert&path=../../x')[0]==400
