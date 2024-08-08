from mmdepth.apis import MMSegInferencer
models = MMSegInferencer.list_models('mmdepth')

# Load models into memory
inferencer = MMSegInferencer(model='scaledepth_clip_NYU_KITTI_352x512')
# Inference
inferencer('data/nyu/images/test/bathroom_00045.jpg', show=True)