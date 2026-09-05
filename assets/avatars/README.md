# Local test avatars

Run the following from the repository root:

```bash
uv run motioncapture-avatars fetch
uv run motioncapture-avatars verify
```

The fetched avatar binaries are local third-party assets and are not tracked by
this repository.

## 2D: Haru

- Format: Live2D Cubism `model3`
- Entrypoint: `live2d/haru/Haru.model3.json`
- Source: [Live2D Cubism Web Samples](https://github.com/Live2D/CubismWebSamples/tree/develop/Samples/Resources/Haru)
- License: [Live2D Free Material License](https://www.live2d.com/eula/live2d-free-material-license-agreement_en.html)
- Additional sample-model terms: [Live2D Sample Model Terms](https://www.live2d.com/eula/live2d-sample-model-terms_en.html)

Haru is for Live2D loading and parameter-mapping tests. Its model manifest
declares eye-blink and lip-sync parameter groups.

## 3D: Seed-san

- Format: VRM 1.0
- Entrypoint: `vrm/seed-san/Seed-san.vrm`
- Source: [VRM Specification Samples](https://github.com/vrm-c/vrm-specification/tree/master/samples/Seed-san)
- License: [VRM Public License 1.0](https://vrm.dev/en/licenses/1.0/)
- Author and required credit: VirtualCast, Inc.

The embedded metadata permits avatar use by everyone and redistribution, allows
modification and corporate commercial use, and requires credit. The validator
checks those fields, the VRM version, the full body/finger humanoid mapping, and
the face/gaze/viseme presets.
