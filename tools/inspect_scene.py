import sys
sys.path.insert(0, 'src')
from helios.components import scene
print('module:', scene)
print('Scene in module:', hasattr(scene, 'Scene'))
Scene = scene.Scene
# show type and repr
print('Scene type:', type(Scene))
print('Scene repr:', repr(Scene))
print('Scene has plot attribute:', hasattr(Scene, 'plot'))
print('Scene methods (subset):', [n for n in dir(Scene) if 'plot' in n or n.startswith('s')])
# instantiate
s = Scene()
print('instance has plot:', hasattr(s, 'plot'))
print('callable(plot):', callable(getattr(s, 'plot', None)))
print('Scene.__dict__ keys:', list(Scene.__dict__.keys()))
