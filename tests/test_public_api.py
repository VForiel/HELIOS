from astropy import units as u


def test_star_import_exports_public_api():
    namespace = {}

    exec("from helios import *", namespace)

    assert namespace["Pipeline"] is namespace["Context"]
    assert namespace["Scene"].__name__ == "PlanetarySystem"
    assert namespace["MMI"].__name__ == "MultiModeInterferometer"


def test_readme_style_pipeline_construction():
    import helios

    scene = helios.Scene(distance=10 * u.pc)
    scene.add(helios.Star(temperature=5700 * u.K, magnitude=5))

    pupil = helios.Pupil(diameter=8 * u.m)
    telescope = helios.TelescopeArray(
        pupil=pupil,
        size=8 * u.m,
        positions=[(0, 0)],
        latitude=0 * u.deg,
        longitude=0 * u.deg,
        altitude=2000 * u.m,
    )

    context = helios.Context()
    context.add_layer(scene)
    context.add_layer(telescope)
    context.add_layer(helios.Camera(pixels=(64, 64), ideal=True))

    description = context.description()

    assert "PlanetarySystem" in description
    assert telescope.is_interferometric() is False


def test_pipeline_roundtrip_current_component_layout():
    import helios

    scene = helios.Scene(distance=10 * u.pc)
    scene.add(helios.Star(temperature=5700 * u.K, magnitude=5))

    telescope = helios.TelescopeArray(
        pupil=helios.Pupil(diameter=8 * u.m),
        size=8 * u.m,
        positions=[(0, 0), (47, 0)],
    )
    camera = helios.Camera(pixels=(32, 32), ideal=True)

    pipeline = helios.Pipeline()
    pipeline.add_layer(scene)
    pipeline.add_layer(telescope)
    pipeline.add_layer(camera)

    restored = helios.Pipeline.from_dict(pipeline.to_dict())

    assert len(restored.layers) == 3
    assert restored.layers[0].distance == 10 * u.pc
    assert len(restored.layers[0].elements) == 1
    restored_telescope = restored.layers[1].elements[0]
    restored_camera = restored.layers[2].elements[0]

    assert restored_telescope.is_interferometric()
    assert restored_camera.pixels == (32, 32)
